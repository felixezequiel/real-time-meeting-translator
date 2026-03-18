"""
Translation bridge for Meeting Translator.
Runs as a persistent subprocess, communicating via JSON lines on stdin/stdout.

Protocol:
- Startup: prints {"status": "ready"} when models are loaded
- Request:  {"text": "...", "source_lang": "en", "target_lang": "pt"}
- Response: {"translated": "..."}

Requires: pip install transformers sentencepiece torch
"""

import json
import sys
import io
from transformers import MarianMTModel, MarianTokenizer

# Force UTF-8 on Windows pipes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

MODEL_MAP = {
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-ROMANCE",
    ("pt", "en"): "Helsinki-NLP/opus-mt-ROMANCE-en",
}

ROMANCE_TARGET_PREFIX = {
    "pt": ">>pt<< ",
    "es": ">>es<< ",
    "fr": ">>fr<< ",
}


def load_models():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stderr.write(f"Translation device: {device}\n")
    sys.stderr.flush()

    models = {}
    for (src, tgt), model_name in MODEL_MAP.items():
        sys.stderr.write(f"Loading model {model_name}...\n")
        sys.stderr.flush()
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        if device == "cuda":
            model = model.half()   # fp16 — ~40% faster on GPU
        model = model.to(device)
        model.eval()
        models[(src, tgt)] = (tokenizer, model, device)
    return models


def translate(models, text, source_lang, target_lang):
    key = (source_lang, target_lang)
    if key not in models:
        return f"[unsupported: {source_lang}->{target_lang}]"

    tokenizer, model, device = models[key]

    # For ROMANCE models, add target language prefix
    if target_lang in ROMANCE_TARGET_PREFIX and "ROMANCE" not in MODEL_MAP.get(key, ("", ""))[0]:
        pass
    if "ROMANCE" in MODEL_MAP.get(key, ""):
        prefix = ROMANCE_TARGET_PREFIX.get(target_lang, "")
        text = prefix + text

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    translated_tokens = model.generate(**inputs, max_length=512, num_beams=1)
    result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return result


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
            response = {"translated": result}
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as e:
            error_response = {"translated": f"[translation error: {e}]"}
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
