#!/usr/bin/env python3
"""
Create calibration token blocks for GPTQ-style post-training quantization.

Works with nanoGPT-style train.bin files, e.g.
    data/shakespeare/train.bin
    data/shakespeare_char/train.bin

Example:
    python make_calibration_tokens.py \
        --data_bin data/shakespeare/train.bin \
        --out calib_shakespeare.pt \
        --nsamples 128 \
        --block_size 128 \
        --seed 1337
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def detect_dtype(path: str) -> np.dtype:
    """
    nanoGPT commonly stores tokens as uint16.
    This function allows an override, but defaults to uint16.
    """
    return np.uint16


def load_train_bin(path: str, dtype: np.dtype) -> np.memmap:
    if not os.path.exists(path):
        raise FileNotFoundError(f"train.bin not found: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def sample_blocks(
    data: np.memmap,
    nsamples: int,
    block_size: int,
    seed: int,
) -> torch.Tensor:
    if len(data) <= block_size:
        raise ValueError(
            f"Dataset too short for block_size={block_size}. "
            f"len(data)={len(data)}"
        )

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - block_size - 1, size=nsamples)

    blocks = []
    for s in starts:
        chunk = np.array(data[s : s + block_size], dtype=np.int64)
        blocks.append(torch.from_numpy(chunk))

    return torch.stack(blocks, dim=0)  # [nsamples, block_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_bin", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    dtype = detect_dtype(args.data_bin)
    data = load_train_bin(args.data_bin, dtype=dtype)

    calib = sample_blocks(
        data=data,
        nsamples=args.nsamples,
        block_size=args.block_size,
        seed=args.seed,
    )

    out = {
        "tokens": calib,  # [nsamples, block_size]
        "meta": {
            "data_bin": args.data_bin,
            "nsamples": args.nsamples,
            "block_size": args.block_size,
            "seed": args.seed,
            "dtype": str(dtype),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved calibration tokens to: {out_path}")
    print(f"tokens.shape = {tuple(calib.shape)}")


if __name__ == "__main__":
    main()