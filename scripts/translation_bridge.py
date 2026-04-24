"""
Translation bridge for Meeting Translator.
Runs as a persistent subprocess, communicating via JSON lines on stdin/stdout.

Protocol:
- Startup: prints {"status": "ready"} when the model is loaded
- Request:  {"text": "...", "source_lang": "en"|"pt", "target_lang": "en"|"pt"}
- Response: {"translated": "..."}

Backed by NLLB-200-distilled-600M running through CTranslate2 with int8
quantization. A single model handles both en<->pt directions, driven by
the `target_prefix` parameter. More natural, context-aware output than
Opus-MT at ~200ms/sentence on GPU int8_float16.

Requires: pip install ctranslate2 transformers sentencepiece
Model conversion done once at install time by scripts/install.ps1.
"""

import json
import sys
import io
import os
import site

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


def add_cuda_dll_dirs():
    """Expose NVIDIA pip-package DLL dirs to the loader before importing
    ctranslate2. Same strategy as stt_bridge.py — CT2 wheels on Windows
    depend on cuBLAS/cuDNN shipped by the `nvidia-*` pip packages."""
    if os.name != "nt":
        return
    candidate_dirs = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_dir):
            candidate_dirs.append(nvidia_dir)
    for path_entry in sys.path:
        nvidia_dir = os.path.join(path_entry, "nvidia")
        if os.path.isdir(nvidia_dir) and nvidia_dir not in candidate_dirs:
            candidate_dirs.append(nvidia_dir)
    for nvidia_dir in candidate_dirs:
        for subdir in ["cublas", "cudnn", "cuda_runtime", "cufft", "curand"]:
            bin_dir = os.path.join(nvidia_dir, subdir, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except (OSError, AttributeError):
                    pass
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


add_cuda_dll_dirs()

import ctranslate2
from transformers import AutoTokenizer

# One NLLB model covers every direction the app supports. Conversion is done
# once by install.ps1 via `python -m ctranslate2.converters.transformers`.
NLLB_MODEL_HF = "facebook/nllb-200-distilled-600M"
NLLB_MODEL_DIR = "nllb-200-distilled-600M-ct2"

# Wire-protocol ISO code (what Rust sends) -> NLLB FLORES-200 code.
# NLLB uses Latin-script codes like "eng_Latn", "por_Latn".
LANG_TO_NLLB = {
    "en": "eng_Latn",
    "pt": "por_Latn",
}


def find_ct2_model_dir(dir_name: str) -> str:
    """Locate the CT2 model directory under models/ near the executable or cwd."""
    candidates = [os.path.join("models", dir_name)]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates.append(os.path.join(project_root, "models", dir_name))

    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "model.bin")):
            return path
    raise FileNotFoundError(
        f"CT2 model directory not found: {dir_name}. "
        f"Run scripts/install.ps1 to convert the NLLB model."
    )


def pick_device():
    """Ask CTranslate2 directly — avoids a torch import just for CUDA probing,
    and the answer reflects what CT2 can actually use (DLLs already resolved)."""
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as e:
        sys.stderr.write(f"CUDA probe failed: {e}\n")
        sys.stderr.flush()
    return "cpu"


def load_model():
    device = pick_device()
    compute_type = "int8_float16" if device == "cuda" else "int8"
    sys.stderr.write(f"Translation device: {device} ({compute_type})\n")
    sys.stderr.flush()

    ct2_path = find_ct2_model_dir(NLLB_MODEL_DIR)
    sys.stderr.write(f"Loading CT2 model: {ct2_path}\n")
    sys.stderr.flush()

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_HF)
    translator = ctranslate2.Translator(
        ct2_path,
        device=device,
        compute_type=compute_type,
    )
    return tokenizer, translator


def translate(tokenizer, translator, text, source_lang, target_lang):
    src_nllb = LANG_TO_NLLB.get(source_lang)
    tgt_nllb = LANG_TO_NLLB.get(target_lang)
    if not src_nllb or not tgt_nllb:
        return f"[unsupported: {source_lang}->{target_lang}]"

    # NLLB tokenizer prepends src_lang as a header token when src_lang is set.
    tokenizer.src_lang = src_nllb
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))

    # target_prefix seeds the decoder with the desired output language.
    # beam_size=5 + length_penalty=1.0 + no_repeat_ngram_size=3 match the
    # NLLB paper's recommended generation settings for quality-weighted output.
    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[tgt_nllb]],
        beam_size=5,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        max_decoding_length=512,
    )
    output_tokens = results[0].hypotheses[0]

    # Strip the target-language prefix token from the output before decoding —
    # some transformers versions treat NLLB lang tokens as special and drop
    # them via skip_special_tokens, others don't. Removing explicitly is safe.
    if output_tokens and output_tokens[0] == tgt_nllb:
        output_tokens = output_tokens[1:]

    token_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def main():
    tokenizer, translator = load_model()
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request["text"]
            source_lang = request["source_lang"]
            target_lang = request["target_lang"]

            result = translate(tokenizer, translator, text, source_lang, target_lang)
            print(json.dumps({"translated": result}, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"translated": f"[translation error: {e}]"}), flush=True)


if __name__ == "__main__":
    main()
