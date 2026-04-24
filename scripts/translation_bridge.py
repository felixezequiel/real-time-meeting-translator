"""
Translation bridge for Meeting Translator.
Runs as a persistent subprocess, communicating via JSON lines on stdin/stdout.

Protocol:
- Startup: prints {"status": "ready"} when models are loaded
- Request:  {"text": "...", "source_lang": "en", "target_lang": "pt"}
- Response: {"translated": "..."}

Backed by CTranslate2 with int8 quantization for ~3-5x speedup over
transformers/MarianMT. See docs/adr/0001-ctranslate2-translation.md.

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
from transformers import MarianTokenizer

# HF model name → directory with the CT2-converted model.
# Tokenizers still come from the HF model (vocab + sentencepiece), only the
# generation runtime is swapped.
MODEL_MAP = {
    ("en", "pt"): ("Helsinki-NLP/opus-mt-en-ROMANCE", "opus-mt-en-ROMANCE-ct2"),
    ("pt", "en"): ("Helsinki-NLP/opus-mt-ROMANCE-en", "opus-mt-ROMANCE-en-ct2"),
}

ROMANCE_TARGET_PREFIX = {
    "pt": ">>pt<< ",
    "es": ">>es<< ",
    "fr": ">>fr<< ",
}


def find_ct2_model_dir(dir_name: str) -> str:
    """Locate a CT2 model directory under models/ near the executable or cwd."""
    candidates = [
        os.path.join("models", dir_name),
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates.append(os.path.join(project_root, "models", dir_name))

    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "model.bin")):
            return path
    raise FileNotFoundError(
        f"CT2 model directory not found: {dir_name}. "
        f"Run scripts/install.ps1 to convert models."
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


def load_models():
    device = pick_device()
    compute_type = "int8_float16" if device == "cuda" else "int8"
    sys.stderr.write(f"Translation device: {device} ({compute_type})\n")
    sys.stderr.flush()

    models = {}
    for (src, tgt), (hf_name, ct2_dir) in MODEL_MAP.items():
        ct2_path = find_ct2_model_dir(ct2_dir)
        sys.stderr.write(f"Loading CT2 model: {ct2_path}\n")
        sys.stderr.flush()

        tokenizer = MarianTokenizer.from_pretrained(hf_name)
        translator = ctranslate2.Translator(
            ct2_path,
            device=device,
            compute_type=compute_type,
        )
        models[(src, tgt)] = (tokenizer, translator, hf_name)
    return models


def translate(models, text, source_lang, target_lang):
    key = (source_lang, target_lang)
    if key not in models:
        return f"[unsupported: {source_lang}->{target_lang}]"

    tokenizer, translator, hf_name = models[key]

    # Opus-MT ROMANCE models need a >>lang<< prefix token to pick target.
    if "ROMANCE" in hf_name:
        prefix = ROMANCE_TARGET_PREFIX.get(target_lang, "")
        text = prefix + text

    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    # beam_size=5 + length_penalty~=1.0 is the Opus-MT paper default and gives
    # noticeably more coherent, idiomatic output than greedy (beam=1), at the
    # cost of ~80ms extra on GPU int8 — still inside the 2-5s latency target.
    # no_repeat_ngram_size prevents the occasional loop on ambiguous/short input.
    results = translator.translate_batch(
        [tokens],
        beam_size=5,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        max_decoding_length=512,
    )
    translated_tokens = results[0].hypotheses[0]
    token_ids = tokenizer.convert_tokens_to_ids(translated_tokens)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def main():
    models = load_models()
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

            result = translate(models, text, source_lang, target_lang)
            print(json.dumps({"translated": result}, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"translated": f"[translation error: {e}]"}), flush=True)


if __name__ == "__main__":
    main()
