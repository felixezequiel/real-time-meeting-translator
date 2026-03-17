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
from transformers import MarianMTModel, MarianTokenizer

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
    models = {}
    for (src, tgt), model_name in MODEL_MAP.items():
        sys.stderr.write(f"Loading model {model_name}...\n")
        sys.stderr.flush()
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        models[(src, tgt)] = (tokenizer, model)
    return models


def translate(models, text, source_lang, target_lang):
    key = (source_lang, target_lang)
    if key not in models:
        return f"[unsupported: {source_lang}->{target_lang}]"

    tokenizer, model = models[key]

    # For ROMANCE models, add target language prefix
    if target_lang in ROMANCE_TARGET_PREFIX and "ROMANCE" not in MODEL_MAP.get(key, ("", ""))[0]:
        pass
    if "ROMANCE" in MODEL_MAP.get(key, ""):
        prefix = ROMANCE_TARGET_PREFIX.get(target_lang, "")
        text = prefix + text

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = model.generate(**inputs, max_length=512, num_beams=4)
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
