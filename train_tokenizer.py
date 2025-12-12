# train_tokenizer.py
"""
Train a BPE tokenizer using Hugging Face tokenizers (Rust-backed).
Outputs:
 - tokenizer_out/tokenizer.json
 - tokenizer_out/vocab.json
 - tokenizer_out/merges.txt

Requirements: pip install tokenizers tqdm
"""

import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

INPUT_JSONL = Path("health_dataset_clean.jsonl")
OUT_DIR = Path("tokenizer_out")
VOCAB_SIZE = 20000     # adjust (5k-50k) depending on corpus size
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

def text_iterator(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            content = obj.get("content") or obj.get("text") or ""
            title = obj.get("title") or ""
            text = (title + " " + content).strip()
            if len(text) < 10:
                continue
            yield text

def train():
    if not INPUT_JSONL.exists():
        raise SystemExit(f"Missing {INPUT_JSONL}. Run cleaning step first.")

    # Create tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Normalizer & pre-tokenizer
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )

    print("Training tokenizer (this may take a while)...")
    tokenizer.train_from_iterator(text_iterator(INPUT_JSONL), trainer=trainer)

    # Save tokenizer native format
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print("Saved tokenizer.json")

    # Extract model vocab & merges
    model = tokenizer.get_model()
    vocab = model.get_vocab()
    merges = model.get_merges()

    # Save vocab.json (token -> id)
    with (OUT_DIR / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("Saved vocab.json")

    # Save merges.txt (HF-style)
    with (OUT_DIR / "merges.txt").open("w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")
    print("Saved merges.txt")

if __name__ == "__main__":
    train()
