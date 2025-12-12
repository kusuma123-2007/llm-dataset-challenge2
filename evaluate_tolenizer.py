# evaluate_tokenizer.py
from tokenizers import Tokenizer
from pathlib import Path
import json

TOKENIZER_FILE = Path("tokenizer_out/tokenizer.json")
SAMPLES_FILE = Path("sample_tests.txt")
OUT_FILE = Path("sample_tokenization_results.txt")

def load_samples():
    if not SAMPLES_FILE.exists():
        return []
    return [l.strip() for l in SAMPLES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    if not TOKENIZER_FILE.exists():
        print("Tokenizer file not found. Run train_tokenizer.py first.")
        return
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    samples = load_samples()
    results = []
    for s in samples:
        enc = tokenizer.encode(s)
        results.append({"text": s, "tokens": enc.tokens, "ids": enc.ids})

    OUT_FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")
    print(f"Wrote results to {OUT_FILE}")

if __name__ == "__main__":
    main()
