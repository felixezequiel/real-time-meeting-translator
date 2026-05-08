"""
Streaming translation bridge — Qwen2.5-1.5B-Instruct via llama-cpp-python.

Replaces the previous NLLB-200-distilled-600M / CTranslate2 atomic
translator. Same wire shape on the request side (one JSON line in),
but the response is a *stream* of fragment lines: each commit-eligible
piece of the translation is emitted as soon as the LLM has produced it,
so the downstream TTS can start synthesising while the LLM is still
generating the rest of the sentence. This is the core lever for
sub-second TTFA — the friend's voiceMaster spike does the same trick
with cloud llama-3.3-70b on Groq; we do it locally with Qwen 1.5B Q4.

Why Qwen 2.5 1.5B Q4_K_M:
  - Multilingual training data includes solid PT-BR, unlike Phi or
    Llama-3.2 small variants which are English-heavy.
  - Q4_K_M fits in ~1 GB of VRAM (RTX 3050 6GB has plenty of room
    alongside whisper-small + Sepformer + Kokoro).
  - 60–80 tok/s on a 3050 = first-token at <100 ms, which is the
    only number that actually matters for TTFA.
  - Apache 2.0, mirrored on HuggingFace via TheBloke/QuantFactory.

Why a streaming protocol:
  - Atomic translate (NLLB) forced TTS to wait for the whole sentence.
    For a 4-second utterance the user heard nothing for ~500 ms after
    STT commit (translate 200 ms + TTS 300 ms).
  - Streaming + fragment commit lets TTS start on the first ~3 tokens
    (typically a comma-bound clause). Audio begins ~150–200 ms after
    STT commit; the rest of the sentence is synthesised while the
    earlier audio is already playing.

Protocol:

  Startup (text line):
    {"status": "ready"}\\n

  Request (text line on stdin):
    {"text": "...", "source_lang": "en"|"pt", "target_lang": "en"|"pt"}\\n

  Response (multiple text lines on stdout, one per fragment):
    {"fragment": "Olá,", "is_final": false}\\n
    {"fragment": " como vai", "is_final": false}\\n
    {"fragment": " você?", "is_final": false}\\n
    {"fragment": "", "is_final": true}\\n

  The Rust client reads lines until it sees `is_final: true`.

Fragment commit rules (mirrors voiceMaster's `_should_commit_fragment`):
  - Buffer non-empty.
  - Last char is one of `.!?;:,` → commit.
  - Buffer length ≥ 25 chars AND contains a space → commit on the last
    space (so we don't cut mid-word).
"""

import json
import os
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_line(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ─── Fragment commit policy ─────────────────────────────────────────────────
#
# Why we split sentence punctuation from clause punctuation: the TTS
# (Kokoro) synthesises each fragment in isolation. If we commit on
# every comma, every fragment is too short to carry a natural prosodic
# arc — pitch resets, stress resets, the result sounds choppy and
# robotic even though the words are correct. By holding clause-level
# punctuation (`,;:`) until we have enough buffered content to read
# as a phrase, the synthesised fragment gets a real intonation
# contour. Strong sentence punctuation always commits because that's
# already a natural prosodic boundary that a human speaker would
# pause at anyway.
SENTENCE_FINAL_PUNCTUATION = ".!?…"
CLAUSE_PUNCTUATION = ",;:"

# Minimum buffer size before a CLAUSE-level commit (comma / semicolon /
# colon) is allowed. Below this we keep accumulating across the comma
# so the resulting fragment is a full clause rather than a fragment of
# one — better prosody from Kokoro.
MIN_CLAUSE_FRAGMENT_CHARS = 35

# Hard cap: when we have this many characters of unpunctuated text,
# commit at the last space anyway. Prevents the worst case where a
# rambling speaker without punctuation pinned the queue indefinitely.
MAX_BUFFER_CHARS = 80


def split_commit_point(buffer: str) -> int:
    """Return the index *after which* `buffer` should be committed, or -1
    if it should keep accumulating. Three commit triggers:

      1. Sentence-final punctuation (`. ! ? …`) at the tail → commit
         everything. This is a natural prosodic boundary.
      2. Clause punctuation (`, ; :`) at the tail AND buffer ≥
         `MIN_CLAUSE_FRAGMENT_CHARS` → commit. Below the floor we keep
         going so single-word clauses don't fragment ("Bem,").
      3. Length cap (≥ `MAX_BUFFER_CHARS`) → commit at the last space
         to avoid runaway buffers in pathological inputs.
    """
    if not buffer:
        return -1

    last_char = buffer[-1]

    # Strong: sentence-final punctuation always commits.
    if last_char in SENTENCE_FINAL_PUNCTUATION:
        return len(buffer)

    # Weak: clause punctuation, but only if we have enough context for
    # the fragment to read as a clause rather than a fragment.
    if last_char in CLAUSE_PUNCTUATION and len(buffer) >= MIN_CLAUSE_FRAGMENT_CHARS:
        return len(buffer)

    # Length-based safety valve.
    if len(buffer) >= MAX_BUFFER_CHARS:
        last_space = buffer.rfind(" ")
        if last_space >= MAX_BUFFER_CHARS - 20:
            return last_space + 1

    return -1


# ─── Prompt: translation engine, no chatter ─────────────────────────────────

SYSTEM_PROMPT = """You are a professional simultaneous interpreter (the kind a TV news channel hires for live political speeches). Output ONLY the translation, nothing else.

Your job is NOT literal word-by-word translation. It is to convey what the speaker MEANS in fluent {target}, the way a human interpreter would in real time — concisely, naturally, without the filler.

Rules:
1. Translate from {source} to {target}.
2. **Preserve meaning, not words.** Drop filler words and verbal tics that carry no meaning ("uh", "um", "you know", "like", "I mean", "tipo", "sabe", "né", "é tipo"). The listener doesn't need them.
3. **Collapse repetition.** "yeah yeah yeah" → "sim". "no, no, wait wait" → "espera". "I I I think" → "Acho que".
4. **Compress disfluencies.** Restarts ("we should — we should do") and self-corrections ("the meeting, I mean, the call") become clean output that says the final intended thought.
5. **Be concise.** A verbose 15-word source with filler can legitimately become a clean 8-word translation. Real interpreters do this constantly. Do not invent content, but do trim what adds no information.
6. **Match the register.** Casual stays casual; formal stays formal. Don't elevate sloppy speech to polished prose, and don't dumb down formal speech.
7. **Keep proper nouns and brand names verbatim** — names of people, companies, products, show titles ("Huge Conversations", "Apple Vision Pro", "Snapchat").
8. **Keep technical jargon when already in the target language** (deploy, commit, pull request, headset, holograms).
9. **Numbers, code identifiers, version strings** — verbatim.
10. **Sentence fragments** — translate the fragment as-is, do not invent words to complete it.
11. **Empty or noise input** — output an empty string.
12. NEVER add commentary, explanations, or conversational responses. Output is ONLY the translation.
"""

LANG_NAMES = {"en": "English", "pt": "Portuguese (Brazilian)"}

# Few-shot examples seed the model with the "translation engine" behaviour
# even at small sizes. Keeping these short so the prompt fits comfortably
# in the 4k context window with room for the actual input.
FEWSHOT = [
    # PT -> EN (technical + everyday)
    ("Vou fazer o commit agora.", "I'll commit now.", "pt", "en"),
    ("ok", "ok", "pt", "en"),
    ("Tive um problema com o deploy ontem à noite.",
     "I had a problem with the deploy last night.", "pt", "en"),
    ("Bom dia, pessoal!", "Good morning, everyone!", "pt", "en"),
    ("Você pode repetir, por favor? Não consegui ouvir.",
     "Could you repeat that, please? I didn't catch it.", "pt", "en"),
    ("Acho que precisamos discutir isso na próxima reunião.",
     "I think we need to discuss this in the next meeting.", "pt", "en"),
    # EN -> PT (technical, casual, narration, questions, idiomatic)
    ("Let's merge the pull request.", "Vamos fazer o merge do pull request.", "en", "pt"),
    ("yeah", "sim", "en", "pt"),
    ("I think we should refactor this module.",
     "Acho que deveríamos refatorar este módulo.", "en", "pt"),
    ("Hey, good to meet you.", "Olá, prazer em conhecê-lo.", "en", "pt"),
    # Narrative / documentary tone
    ("By the time the war ended, the city had already begun to rebuild itself.",
     "Quando a guerra acabou, a cidade já tinha começado a se reconstruir.",
     "en", "pt"),
    ("What you're seeing here is the result of decades of careful planning.",
     "O que você está vendo aqui é o resultado de décadas de planejamento cuidadoso.",
     "en", "pt"),
    # Question with falling tag
    ("So you're saying we should just wait, right?",
     "Então você está dizendo que devemos só esperar, certo?",
     "en", "pt"),
    # Idiomatic / colloquial
    ("That makes sense, but I'm not sold on it yet.",
     "Faz sentido, mas ainda não estou convencido.",
     "en", "pt"),
    ("Long story short, the deal fell through.",
     "Resumindo, o negócio não foi adiante.",
     "en", "pt"),
    # Mid-clause fragment (common in streaming STT)
    ("which is exactly why we built the system this way",
     "que é exatamente por isso que construímos o sistema dessa forma",
     "en", "pt"),

    # ─── Interpreter-style compression (added 2026-05-07) ───────────────────
    # The prompt now asks the model to drop fillers, collapse
    # repetition, and trim disfluencies the way a live interpreter
    # would. These few-shots demonstrate the behaviour with realistic
    # casual speech taken from podcast transcripts. Without these
    # examples, a 1.5B model will default to literal word-for-word
    # output even with the verbal instruction.

    # EN -> PT: filler removal
    ("So, uh, yeah, I think, like, what we should do is, you know, just go ahead.",
     "Acho que devemos seguir em frente.", "en", "pt"),
    ("It's, uh, it's really nice to, you know, finally meet you in person.",
     "É um prazer finalmente te conhecer pessoalmente.", "en", "pt"),
    # EN -> PT: repetition collapse
    ("yeah yeah yeah, totally, yeah, sure, no problem at all.",
     "Claro, sem problemas.", "en", "pt"),
    ("Welcome, welcome to the first, the very first episode of our new series.",
     "Bem-vindo ao primeiro episódio da nossa nova série.", "en", "pt"),
    # EN -> PT: self-correction / restart
    ("I I I think we — we should — we should wait a bit longer.",
     "Acho que devemos esperar mais um pouco.", "en", "pt"),
    ("The meeting, I mean, the call — the call starts at three.",
     "A call começa às três.", "en", "pt"),
    # EN -> PT: the show-name / proper-noun case the user reported
    ("Welcome to the first episode of our new series, Huge Conversations.",
     "Bem-vindo ao primeiro episódio da nossa nova série, Huge Conversations.",
     "en", "pt"),
    ("Good to meet you. Yeah, looking forward to it.",
     "Prazer em te conhecer. Estou ansioso por isso.", "en", "pt"),

    # PT -> EN: filler removal
    ("Tipo, eu acho que, sabe, a gente devia, tipo, fazer isso assim mesmo.",
     "I think we should do it this way.", "pt", "en"),
    ("É, é, com certeza, sim, sim, sem problema.",
     "Sure, no problem.", "pt", "en"),
    # PT -> EN: self-correction
    ("A reunião, quer dizer, a chamada — a chamada é às três.",
     "The call is at three.", "pt", "en"),
    # PT -> EN: "né" / "tipo" tics
    ("Então, né, eu queria, tipo, falar sobre, sabe, esse projeto novo.",
     "I wanted to talk about this new project.", "pt", "en"),
]


def build_messages(text: str, source_lang: str, target_lang: str) -> list[dict]:
    """Compose the chat messages for one translation. Few-shot examples
    matching the requested direction go first, then the user input."""
    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)
    messages = [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(source=src_name, target=tgt_name)}
    ]
    for src_text, tgt_text, ex_src, ex_tgt in FEWSHOT:
        if ex_src != source_lang or ex_tgt != target_lang:
            continue
        messages.append({"role": "user", "content": src_text})
        messages.append({"role": "assistant", "content": tgt_text})
    messages.append({"role": "user", "content": text})
    return messages


# ─── Model loading ──────────────────────────────────────────────────────────

QWEN_MODEL_FILENAME = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"


def find_llm_model() -> str:
    """Locate the GGUF under models/ near the executable or cwd. Same
    search order whisper.cpp uses for ggml-*.bin."""
    candidates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates.append(os.path.join(project_root, "models", QWEN_MODEL_FILENAME))
    candidates.append(os.path.join(os.getcwd(), "models", QWEN_MODEL_FILENAME))

    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"GGUF model not found: {QWEN_MODEL_FILENAME}. "
        f"Run scripts/install.ps1 to download it. Tried: {candidates}"
    )


def load_llm():
    """Instantiate llama-cpp-python with CUDA offload when available.

    n_gpu_layers=-1 offloads every layer to GPU; on CPU-only systems
    this falls back gracefully (the wheel ships CPU + CUDA paths).
    n_ctx=4096 covers system prompt + few-shot + a long sentence with
    margin. Larger context costs RAM and slows attention.
    """
    import llama_cpp
    from llama_cpp import Llama

    model_path = find_llm_model()

    # Diagnostic: report whether the installed wheel actually has CUDA
    # support compiled in. The default `pip install llama-cpp-python`
    # ships a CPU-only wheel — even with n_gpu_layers=-1 the model
    # silently runs on CPU at ~10-15 tok/s instead of 60-80 tok/s on a
    # 3050. Symptom: GPU sitting at 0% utilisation while LLM stream
    # latency shows up as ~2 s for short sentences.
    cuda_supported = False
    try:
        cuda_supported = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        # Older llama-cpp-python versions don't expose the helper. Fall
        # back to inspecting the system info string the runtime prints.
        try:
            info = llama_cpp.llama_print_system_info().decode("utf-8", errors="replace")
            cuda_supported = "CUDA = 1" in info or "CUDA=1" in info
        except Exception:
            cuda_supported = False

    log(f"[init] llama-cpp CUDA support: {cuda_supported}")
    if not cuda_supported:
        log(
            "[init] WARNING: llama-cpp-python is the CPU build. Qwen will "
            "run at ~10-15 tok/s instead of 60-80 tok/s on a CUDA GPU. "
            "Reinstall the CUDA wheel:\n"
            "    python -m pip uninstall -y llama-cpp-python\n"
            "    python -m pip install --upgrade --force-reinstall --no-cache-dir "
            "--index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 "
            "llama-cpp-python"
        )

    log(f"Loading Qwen2.5-1.5B-Instruct from {model_path}…")
    n_gpu_layers = -1 if os.environ.get("QWEN_FORCE_CPU", "0") != "1" else 0
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=4096,
        n_threads=max(4, (os.cpu_count() or 4) - 1),
        verbose=False,
    )
    log(f"Qwen ready (n_gpu_layers={n_gpu_layers}, cuda_in_wheel={cuda_supported})")
    return llm


# ─── Streaming translation ──────────────────────────────────────────────────

def translate_streaming(llm, text: str, source_lang: str, target_lang: str) -> None:
    """Stream the translation, emitting one JSON-line per fragment as
    soon as the buffer hits a commit point. Closes with `is_final: true`."""
    if not text.strip():
        write_line({"fragment": "", "is_final": True})
        return

    messages = build_messages(text, source_lang, target_lang)

    buffer = ""
    emitted_any = False

    # `temperature=0` for deterministic output (translation, not creative
    # writing). `max_tokens=256` is plenty for a single utterance — if
    # the model produces more it's almost certainly looping.
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        token_text = delta.get("content", "")
        if not token_text:
            continue
        buffer += token_text
        commit_point = split_commit_point(buffer)
        while commit_point > 0:
            fragment = buffer[:commit_point]
            buffer = buffer[commit_point:]
            write_line({"fragment": fragment, "is_final": False})
            emitted_any = True
            commit_point = split_commit_point(buffer)

    # Final flush: anything left in the buffer is the tail of the
    # translation that didn't end on a commit point.
    if buffer:
        write_line({"fragment": buffer, "is_final": False})
        emitted_any = True

    # Always send a terminator so the Rust client knows the stream is
    # over — even when the model emitted nothing (rare but possible
    # when the input was just punctuation or noise).
    if not emitted_any:
        write_line({"fragment": "", "is_final": True})
    else:
        write_line({"fragment": "", "is_final": True})


# ─── Main loop ──────────────────────────────────────────────────────────────

def warmup_llm(llm) -> None:
    """Run a one-shot translation to compile CUDA kernels and prime the
    KV cache. Without this, the very first user-facing translation pays
    a 200-500 ms first-token spike that dominates the TTFA metric for
    the first sentence of every session.

    The result is discarded; we only care about the side effect of
    having the GPU/CPU kernel cache warm. Two passes (one per direction)
    so neither direction's first call carries cold-start overhead.
    """
    import time as _t

    warmup_pairs = [
        ("Hello.", "en", "pt"),
        ("Olá.", "pt", "en"),
    ]
    for text, src, tgt in warmup_pairs:
        t0 = _t.monotonic()
        # Drain the stream into a throwaway list — we don't write to
        # stdout because the bridge is still in the boot sequence.
        for _chunk in llm.create_chat_completion(
            messages=build_messages(text, src, tgt),
            temperature=0.0,
            max_tokens=32,
            stream=True,
        ):
            pass
        log(f"[init] warmup {src}→{tgt}: {(_t.monotonic() - t0) * 1000:.0f} ms")


def main() -> None:
    llm = load_llm()
    log("[init] warming up LLM kernels…")
    warmup_llm(llm)
    write_line({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request["text"]
            source_lang = request["source_lang"]
            target_lang = request["target_lang"]
            translate_streaming(llm, text, source_lang, target_lang)
        except Exception as e:
            log(f"Translation error: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
            write_line({"fragment": f"[translation error: {e}]", "is_final": False})
            write_line({"fragment": "", "is_final": True})


if __name__ == "__main__":
    main()
