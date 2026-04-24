# ADR 0001 — Migrate translation bridge from HuggingFace transformers to CTranslate2

- **Status:** Accepted
- **Date:** 2026-04-24
- **Deciders:** felix

## Context

The translation stage is the most Python-heavy stage in the pipeline after STT
was migrated to whisper.cpp native. `translation_bridge.py` currently uses
`transformers.MarianMTModel` + `MarianTokenizer` for Opus-MT. On GPU with fp16
it takes roughly 150 ms per short utterance — dominant non-STT cost in the
end-to-end path.

The project goal is maximum local latency (2–5 s target end-to-end, 100% local,
Windows-only, PT↔EN only). Translation throughput directly caps how many
chunks per second the pipeline can process without backpressure.

## Decision

Replace HuggingFace `transformers` Opus-MT inference with
**[CTranslate2](https://github.com/OpenNMT/CTranslate2)** running Opus-MT models
quantized to **int8** (on GPU: `int8_float16` mixed precision).

The JSON stdin/stdout protocol between Rust `OpusMtTranslator` and the Python
bridge stays **unchanged**. Only the Python implementation is swapped.

Models are converted once at install time using
`ct2-transformers-converter` and stored under
`models/opus-mt-en-ROMANCE-ct2/` and `models/opus-mt-ROMANCE-en-ct2/`.

## Alternatives considered

1. **Keep transformers, add torch.compile / int8 via bitsandbytes.**
   - Marginal speedup (~1.2–1.5×). Still pays Python + tokenizer overhead.
   - bitsandbytes has poor Windows support.
2. **Rewrite translation in Rust via FFI (tokenizers-rs + ort/onnxruntime).**
   - Largest speedup (~5×+) and eliminates Python bridge entirely.
   - Weeks of work: need SentencePiece tokenizer bindings, ONNX export of
     Opus-MT (non-trivial for seq2seq with beam search), model loading.
   - Deferred — revisit after CTranslate2 baseline is established.
3. **Switch model family (e.g. NLLB-200, M2M-100).**
   - Irrelevant to latency goal, changes quality characteristics.

## Consequences

### Positive
- Expected **3–5× throughput** on GPU (int8_float16) vs transformers fp16.
- ~4× smaller model on disk (int8 quantization).
- Much lower Python memory footprint (no torch tensors for generation).
- Same Opus-MT quality — CTranslate2 is a runtime swap, not a model swap.

### Negative
- Adds `ctranslate2` Python dependency and a **model conversion step** at
  install time (~30 s one-off).
- `install.ps1` grows — needs the converter to run after `pip install`.
- Int8 quantization introduces a tiny quality delta vs fp16 (measured
  chrF drop <1 point on FLORES).

### Neutral
- Rust-side code is unchanged — the JSON protocol is the contract.
- Existing tests in `crates/translation/src/lib.rs` keep passing (they test
  the protocol, not the Python implementation).

## Rollout

1. Add `ctranslate2>=4.0.0` to `scripts/requirements.txt`.
2. Add model-conversion step to `scripts/install.ps1`:
   `ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ROMANCE
   --output_dir models/opus-mt-en-ROMANCE-ct2 --quantization int8`.
3. Rewrite `scripts/translation_bridge.py` using `ctranslate2.Translator`
   and `transformers.MarianTokenizer` (tokenizer still comes from HF; only
   the model runtime changes).
4. Run the app, verify translation output and measure latency.

## Rollback

If quality regresses unacceptably: revert `translation_bridge.py` and remove
the CT2 model directories. Rust side is untouched, so rollback is a single
file revert.
