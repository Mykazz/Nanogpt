#!/usr/bin/env python3
"""
Create char-level calibration token blocks for a nanoGPT char model
using raw OpenWebText documents.

Why this exists:
- data/openwebtext_small/train.bin contains GPT-2 BPE token ids
- a char model expects character ids from data/shakespeare_char/meta.pkl
- therefore we must re-tokenize raw text into char ids

This script:
1) loads OpenWebText raw documents from Hugging Face
2) keeps only characters that exist in the char vocabulary
3) concatenates the filtered text
4) samples random fixed-length blocks
5) saves a calibration .pt file compatible with quantize_nanogpt_gptq.py

Example:
    python make_calibration_tokens_char_from_openweb.py \
        --dataset_name openwebtext \
        --char_meta data/shakespeare_char/meta.pkl \
        --out calib_openweb_char.pt \
        --max_docs 3000 \
        --nsamples 128 \
        --block_size 256 \
        --seed 1337
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset


def load_char_meta(meta_path: str):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if "stoi" not in meta or "itos" not in meta:
        raise ValueError(f"meta.pkl at {meta_path} must contain 'stoi' and 'itos'.")

    stoi = meta["stoi"]
    itos = meta["itos"]

    if not isinstance(stoi, dict):
        raise TypeError("'stoi' must be a dict")
    if not isinstance(itos, (dict, list)):
        raise TypeError("'itos' must be a dict or list")

    vocab_chars = set(stoi.keys())
    return stoi, itos, vocab_chars


def normalize_text(text: str) -> str:
    """
    Light normalization only.
    Keep this conservative so calibration still reflects real text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = text.replace("\u00A0", " ")   # non-breaking space
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("…", "...")
    return text


def filter_to_vocab(text: str, vocab_chars: set[str]) -> str:
    """
    Keep only characters present in the char vocabulary.
    """
    return "".join(ch for ch in text if ch in vocab_chars)


def encode_char_text(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text]


def sample_blocks_from_token_stream(
    token_stream: np.ndarray,
    nsamples: int,
    block_size: int,
    seed: int,
) -> torch.Tensor:
    if token_stream.ndim != 1:
        raise ValueError("token_stream must be 1D")

    if len(token_stream) <= block_size:
        raise ValueError(
            f"Not enough tokens after filtering. Need > block_size={block_size}, "
            f"got {len(token_stream)}"
        )

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(token_stream) - block_size, size=nsamples)

    blocks = []
    for s in starts:
        chunk = token_stream[s : s + block_size]
        blocks.append(torch.from_numpy(chunk.astype(np.int64)))

    return torch.stack(blocks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="openwebtext")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--char_meta", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--max_docs", type=int, default=3000,
                        help="How many raw documents to read from OpenWebText")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--min_doc_chars_after_filter", type=int, default=32,
                        help="Drop docs that become too short after vocab filtering")

    args = parser.parse_args()

    stoi, itos, vocab_chars = load_char_meta(args.char_meta)
    print(f"Loaded char vocab from: {args.char_meta}")
    print(f"Vocab size: {len(stoi)}")

    print(f"Loading dataset: {args.dataset_name} [{args.split}]")
    ds = load_dataset(args.dataset_name, split=args.split)

    max_docs = min(args.max_docs, len(ds))
    ds = ds.select(range(max_docs))
    print(f"Using first {max_docs} documents")

    kept_docs = 0
    total_docs = 0
    total_raw_chars = 0
    total_filtered_chars = 0
    pieces: List[str] = []

    for ex in ds:
        total_docs += 1
        raw_text = ex["text"]
        total_raw_chars += len(raw_text)

        text = normalize_text(raw_text)
        text = filter_to_vocab(text, vocab_chars)

        if len(text) < args.min_doc_chars_after_filter:
            continue

        pieces.append(text)
        pieces.append("\n")  # separator between documents
        kept_docs += 1
        total_filtered_chars += len(text) + 1

    if not pieces:
        raise RuntimeError("No usable text remained after filtering to char vocabulary.")

    merged_text = "".join(pieces)
    token_ids = np.array(encode_char_text(merged_text, stoi), dtype=np.int64)

    print(f"Documents scanned              : {total_docs}")
    print(f"Documents kept                 : {kept_docs}")
    print(f"Raw chars scanned              : {total_raw_chars:,}")
    print(f"Filtered chars kept            : {total_filtered_chars:,}")
    print(f"Final char-token stream length : {len(token_ids):,}")

    calib = sample_blocks_from_token_stream(
        token_stream=token_ids,
        nsamples=args.nsamples,
        block_size=args.block_size,
        seed=args.seed,
    )

    out = {
        "tokens": calib,  # [nsamples, block_size]
        "meta": {
            "source": args.dataset_name,
            "split": args.split,
            "char_meta": args.char_meta,
            "max_docs": args.max_docs,
            "nsamples": args.nsamples,
            "block_size": args.block_size,
            "seed": args.seed,
            "kept_docs": kept_docs,
            "scanned_docs": total_docs,
            "raw_chars_scanned": total_raw_chars,
            "filtered_chars_kept": total_filtered_chars,
            "note": (
                "Raw OpenWebText was normalized and filtered to the target "
                "char vocabulary before encoding to char ids."
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    print(f"Saved calibration tokens to: {out_path}")
    print(f"tokens.shape = {tuple(calib.shape)}")


if __name__ == "__main__":
    main()