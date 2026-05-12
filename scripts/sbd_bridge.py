"""
Sentence Boundary Detection bridge — spaCy-based.

Replaces the regex-only `split_complete_sentences` heuristic in
`crates/pipeline/src/v2.rs` with a semantic-aware boundary detector
that consults spaCy's dependency parser to decide whether a chunk of
accumulated speech text "looks like" a finished sentence before the
pipeline flushes it to the translator + TTS.

Why this layer is necessary (captured 2026-05-12 logs):
  - Whisper streaming commits words as they stabilise; without a
    semantic check, the accumulator flushes mid-clause whenever a
    word count threshold or a hold-duration ceiling triggers.
  - Documentary narration fragments produced ridiculous translations
    like "really looks like." being spoken alone, "is to" being its
    own sentence, and "out what that future" landing without the rest
    of the clause.
  - A boundary detector that also checks for a finite verb + a
    grammatical subject lets the pipeline wait until a clause is
    *complete enough to translate well*, instead of just *long
    enough to ship*.

Protocol (one JSON line per request, one JSON line per response):

  Startup (text line):
    {"status": "ready"}\\n

  Request:
    {"text": "...", "language": "pt"|"en"}\\n

  Response:
    {"complete": "...", "rest": "..."}\\n
      - `complete` is the longest prefix of `text` that ends at a
        sentence boundary AND every sentence in it has finite-verb +
        subject (or is an obvious imperative ending in punctuation).
      - `rest` is the tail that hasn't reached a complete sentence
        yet — the accumulator keeps holding it.
      - Both fields are present even when one is empty.

Cost per call: ~5-20 ms on CPU with the small models, dominated by
the parser. spaCy's tokenizer + parser is reentrant, so we keep a
single Doc-free pipeline per language preloaded at startup.

Requires:
  pip install spacy>=3.7
  python -m spacy download pt_core_news_sm
  python -m spacy download en_core_web_sm
"""

import io
import json
import sys
import traceback

import spacy


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_line(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ─── Sentence completeness rules ───────────────────────────────────────────
#
# A sentence is considered "complete enough to translate" when:
#
#   1. It ends with terminal punctuation (`.`, `!`, `?`, `;`, `…`).
#      Without this, even semantically well-formed clauses still
#      represent live speech the speaker might continue — wait.
#
#   2. It has a finite verb (a token with POS in {VERB, AUX}) somewhere
#      in the span. This rules out noun phrases like "the Statue of
#      Liberty." that pass rule 1 but aren't standalone sentences.
#
#   3. The verb has a grammatical subject (dependency relation
#      `nsubj`, `nsubj:pass`, `expl`) OR the verb itself sits at the
#      sentence root in imperative mood. Without this, the parser
#      occasionally produced "Alpha tinha 14.000 toneladas..." as a
#      single sentence — finite verb, no subject because "A plataforma
#      Piper" was orphaned by capitalisation.
#
# Rules are conservative on purpose: hold longer than flush
# prematurely. The accumulator has a hard MAX_HOLD ceiling that
# eventually forces a release, so even text that never satisfies all
# three rules cannot get stuck indefinitely.

TERMINAL_PUNCT = set(".!?;…")
SUBJECT_DEPS = {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}
FINITE_VERB_POS = {"VERB", "AUX"}


def has_finite_verb(span) -> bool:
    return any(token.pos_ in FINITE_VERB_POS for token in span)


def has_subject(span) -> bool:
    """True when any verb in the span has a child token marked as
    subject. Iterates every token because spaCy's `sent.root` only
    returns ONE root, and complex sentences can have multiple verbs
    (coordinated clauses, subordination)."""
    for token in span:
        if token.pos_ not in FINITE_VERB_POS:
            continue
        for child in token.children:
            if child.dep_ in SUBJECT_DEPS:
                return True
    return False


def looks_imperative(span) -> bool:
    """Imperatives in pt/en frequently lack an explicit subject. We
    accept them as complete when the very first content token is a
    verb (a heuristic; spaCy's morph features for imperative mood
    are unreliable on short clips)."""
    for token in span:
        if token.is_space or token.is_punct:
            continue
        return token.pos_ in FINITE_VERB_POS
    return False


def is_complete(span) -> bool:
    text = span.text.strip()
    if not text:
        return False
    if text[-1] not in TERMINAL_PUNCT:
        return False
    if not has_finite_verb(span):
        return False
    if has_subject(span):
        return True
    return looks_imperative(span)


# ─── Pipeline loading ───────────────────────────────────────────────────────

# The parser is the expensive component (~10-20 ms per call); the
# tokenizer and sentencizer alone are <1 ms but produce worse boundaries
# on speech-style text without punctuation. Keep the full pipeline.
LANGUAGE_TO_MODEL = {
    "pt": "pt_core_news_sm",
    "en": "en_core_web_sm",
}


def load_pipelines() -> dict:
    pipelines = {}
    for lang, model_name in LANGUAGE_TO_MODEL.items():
        log(f"[init] loading {model_name} …")
        pipelines[lang] = spacy.load(model_name)
        log(f"[init] {model_name} ready")
    return pipelines


# ─── Boundary splitting ────────────────────────────────────────────────────
#
# spaCy's built-in sentencizer is unreliable on speech-style text with
# capitalised proper nouns ("Piper Alpha" routinely got split between
# the two words). We bypass it: walk the raw text for terminal-punct
# boundaries ourselves, then submit each candidate prefix to the
# parser for structural validation. The parser still gives us the
# verb/subject signals we need; we just don't trust its boundary
# decisions.

def find_punct_boundaries(text: str) -> list[int]:
    """Return character indices (1-past-the-end) of every terminal
    punctuation position that looks like a real sentence break.

    Rules:
      - The character must be one of `TERMINAL_PUNCT`.
      - The character following must be whitespace or end-of-string
        (so "U.S.A.B." doesn't generate intermediate boundaries —
        only the last period qualifies).
      - The character preceding must NOT be a digit (pt-BR thousands
        separator "14.000" is not a sentence end; see ADR notes /
        memory `project-locale-punctuation-gotcha`).
    """
    boundaries: list[int] = []
    for i, ch in enumerate(text):
        if ch not in TERMINAL_PUNCT:
            continue
        next_idx = i + 1
        if next_idx < len(text):
            if not text[next_idx].isspace():
                # Run of punctuation ("?!", "…!") — let the LAST one win.
                if text[next_idx] in TERMINAL_PUNCT:
                    continue
                # Letter/digit immediately after period — not a real
                # boundary (acronyms, decimals, thousands separator).
                continue
        if i > 0 and text[i - 1].isdigit():
            continue
        boundaries.append(next_idx)
    return boundaries


def split_complete(text: str, language: str, pipelines: dict) -> tuple[str, str]:
    """Return `(complete, rest)`. `complete` is the longest prefix
    of `text` ending at a real terminal-punctuation boundary whose
    span passes the structural completeness rules; `rest` is the
    tail still being accumulated."""
    text = text.strip()
    if not text:
        return ("", "")
    nlp = pipelines.get(language)
    if nlp is None:
        # Unknown language: be conservative — return as pending so the
        # accumulator's safety nets (MAX_HOLD, MAX_WORDS) take over.
        return ("", text)

    boundaries = find_punct_boundaries(text)
    if not boundaries:
        # No terminal punctuation at all → nothing to release.
        return ("", text)

    # Walk candidates from earliest to latest, accepting each one as
    # long as the parsed span of its containing sentence has finite
    # verb + subject (or imperative). The first failing candidate
    # stops the walk — later candidates can't bypass an incomplete
    # earlier clause because they share the same prefix.
    last_complete_end = 0
    prev_start = 0
    for boundary_end in boundaries:
        candidate_span_text = text[prev_start:boundary_end].strip()
        if not candidate_span_text:
            continue
        # Parse just this candidate. spaCy will pick its own internal
        # sentence boundaries inside the candidate — we treat the
        # whole candidate as one logical sentence and check whether
        # any of the spans the parser identified has verb + subject.
        doc = nlp(candidate_span_text)
        if any(is_complete(span) for span in doc.sents):
            last_complete_end = boundary_end
            prev_start = boundary_end
        else:
            break

    if last_complete_end == 0:
        return ("", text)

    complete = text[:last_complete_end].strip()
    rest = text[last_complete_end:].strip()
    return (complete, rest)


# ─── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    pipelines = load_pipelines()
    write_line({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request.get("text", "")
            language = (request.get("language") or "en").strip().lower()
            complete, rest = split_complete(text, language, pipelines)
            write_line({"complete": complete, "rest": rest})
        except Exception as e:  # noqa: BLE001 — bridge must stay alive
            log(f"[sbd] error: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            # On any failure, return the whole input as "rest" so the
            # accumulator falls back to its safety nets without losing
            # text. Never crash the bridge — it serves both pipelines.
            try:
                fallback_rest = request.get("text", "") if isinstance(request, dict) else ""
            except Exception:
                fallback_rest = ""
            write_line({"complete": "", "rest": fallback_rest})


if __name__ == "__main__":
    main()
