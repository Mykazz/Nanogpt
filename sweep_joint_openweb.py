#!/usr/bin/env python3
"""
Sweep joint SparseGPT+GPTQ parameters using OpenWeb calibration,
evaluate perplexity/accuracy, measure checkpoint size, and plot results.

Run examples:

Non-char GPT-2 tokenizer model:
python sweep_joint_openweb.py --mode gpt2

Char model:
python sweep_joint_openweb.py --mode char

Both:
python sweep_joint_openweb.py --mode both
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

CONFIGS = {
    "gpt2": {
        "base_checkpoint": "out-shakespeare-gpt2-ft/ckpt_best.pt",
        "calib": "calib_openweb_gpt2.pt",
        "input_file": "data/shakespeare/input.txt",
        "dataset_dir": "data/shakespeare",
        "out_dir": "out-shakespeare-gpt2-ft/joint_openweb_sweep",
        "eval_block_size": 128,
        "batch_size": 8,
        "amp_dtype": "float16",
        "device": "cuda",
        "sparsities": [0.00, 0.10, 0.20, 0.30],
        "groupsizes": [32, 64, 128],
        "bits": [4],
    },
    "char": {
        "base_checkpoint": "out-shakespeare-char-gptqprep/ckpt_best.pt",
        "calib": "calib_openweb_char.pt",
        "input_file": "data/shakespeare_char/input.txt",
        "dataset_dir": "data/shakespeare_char",
        "out_dir": "out-shakespeare-char-gptqprep/joint_openweb_sweep",
        "eval_block_size": 128,
        "batch_size": 8,
        "amp_dtype": "float16",
        "device": "cuda",
        "sparsities": [0.00, 0.10, 0.20, 0.30, 0.40],
        "groupsizes": [32, 64, 128],
        "bits": [4],
    },
}


# ============================================================
# Helpers
# ============================================================

def run_cmd(cmd: List[str], log_path: Path) -> str:
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 100)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")

    print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed. See log: {log_path}")

    return proc.stdout


def parse_eval_metrics(text: str) -> Dict[str, Optional[float]]:
    def grab(pattern: str) -> Optional[float]:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "perplexity": grab(r"Perplexity\s*:\s*([0-9.eE+-]+)"),
        "loss": grab(r"Mean loss \(nats/token\):\s*([0-9.eE+-]+)"),
        "bits_per_token": grab(r"Bits per token\s*:\s*([0-9.eE+-]+)"),
        "top1_accuracy_percent": grab(r"Top-1 accuracy\s*:\s*([0-9.eE+-]+)%"),
        "tokens_evaluated": grab(r"Tokens evaluated\s*:\s*([0-9.eE+-]+)"),
    }


def checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def theoretical_selected_linear_bits(
    selected_dense_params: int,
    sparsity: float,
    bits: int,
) -> float:
    """
    Very simple estimate for selected compressed linear weights only.

    This ignores:
    - scales
    - zero_points
    - masks
    - embeddings
    - layer norms
    - biases
    - Python checkpoint overhead

    It is useful as a clean mathematical reference.
    """
    kept = selected_dense_params * (1.0 - sparsity)
    return kept * bits


def plot_metric(rows: List[Dict[str, str]], x_name: str, y_name: str, out_path: Path, title: str) -> None:
    groups = sorted(set(r["groupsize"] for r in rows))

    plt.figure(figsize=(10, 6))

    for g in groups:
        sub = [r for r in rows if r["groupsize"] == g]
        sub = sorted(sub, key=lambda r: float(r[x_name]))

        xs = [float(r[x_name]) for r in sub]
        ys = [float(r[y_name]) for r in sub]

        plt.plot(xs, ys, marker="o", label=f"groupsize={g}")

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main sweep
# ============================================================

def run_sweep(mode: str, dry_run: bool = False) -> None:
    cfg = CONFIGS[mode]

    root_out = Path(cfg["out_dir"])
    ckpt_dir = root_out / "checkpoints"
    log_dir = root_out / "logs"
    plot_dir = root_out / "plots"

    root_out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []

    # Baseline eval.
    baseline_log = log_dir / "baseline_eval.log"
    baseline_cmd = [
        "python", "eval_metrics.py",
        "--checkpoint", cfg["base_checkpoint"],
        "--input_file", cfg["input_file"],
        "--dataset_dir", cfg["dataset_dir"],
        "--device", cfg["device"],
        "--dtype", cfg["amp_dtype"],
        "--batch_size", str(cfg["batch_size"]),
        "--block_size", str(cfg["eval_block_size"]),
    ]

    if dry_run:
        print("DRY RUN baseline:", " ".join(baseline_cmd))
        baseline_metrics = {}
        baseline_size = 0.0
    else:
        baseline_text = run_cmd(baseline_cmd, baseline_log)
        baseline_metrics = parse_eval_metrics(baseline_text)
        baseline_size = checkpoint_size_mb(Path(cfg["base_checkpoint"]))

    results.append({
        "mode": mode,
        "kind": "baseline",
        "bits": 0,
        "sparsity": 0.0,
        "groupsize": 0,
        "checkpoint": cfg["base_checkpoint"],
        "checkpoint_size_mb": baseline_size,
        "perplexity": baseline_metrics.get("perplexity"),
        "loss": baseline_metrics.get("loss"),
        "bits_per_token": baseline_metrics.get("bits_per_token"),
        "top1_accuracy_percent": baseline_metrics.get("top1_accuracy_percent"),
        "tokens_evaluated": baseline_metrics.get("tokens_evaluated"),
    })

    # Joint sweep.
    for bits in cfg["bits"]:
        for sparsity in cfg["sparsities"]:
            for groupsize in cfg["groupsizes"]:
                tag = f"{mode}_joint_openweb_b{bits}_s{int(sparsity * 100):02d}_g{groupsize}"
                out_ckpt = ckpt_dir / f"{tag}.pt"

                compress_cmd = [
                    "python", "joint_sparsegpt_gptq_nanogpt.py",
                    "--checkpoint", cfg["base_checkpoint"],
                    "--calib", cfg["calib"],
                    "--out", str(out_ckpt),
                    "--bits", str(bits),
                    "--sparsity", str(sparsity),
                    "--pattern", "unstructured",
                    "--groupsize", str(groupsize),
                    "--blocksize", "128",
                    "--mask_blocksize", "128",
                    "--percdamp", "0.01",
                    "--batch_size", str(cfg["batch_size"]),
                    "--device", cfg["device"],
                    "--amp_dtype", cfg["amp_dtype"],
                    "--packing", "uint8",
                    "--skip_tied_lm_head",
                    "--keep_dequantized_state_dict",
                ]

                eval_cmd = [
                    "python", "eval_metrics.py",
                    "--checkpoint", str(out_ckpt),
                    "--input_file", cfg["input_file"],
                    "--dataset_dir", cfg["dataset_dir"],
                    "--device", cfg["device"],
                    "--dtype", cfg["amp_dtype"],
                    "--batch_size", str(cfg["batch_size"]),
                    "--block_size", str(cfg["eval_block_size"]),
                ]

                if dry_run:
                    print("\nDRY RUN compress:", " ".join(compress_cmd))
                    print("DRY RUN eval:", " ".join(eval_cmd))
                    continue

                run_cmd(compress_cmd, log_dir / f"{tag}_compress.log")
                eval_text = run_cmd(eval_cmd, log_dir / f"{tag}_eval.log")
                metrics = parse_eval_metrics(eval_text)

                size_mb = checkpoint_size_mb(out_ckpt)

                results.append({
                    "mode": mode,
                    "kind": "joint_sparsegpt_gptq",
                    "bits": bits,
                    "sparsity": sparsity,
                    "groupsize": groupsize,
                    "checkpoint": str(out_ckpt),
                    "checkpoint_size_mb": size_mb,
                    "perplexity": metrics.get("perplexity"),
                    "loss": metrics.get("loss"),
                    "bits_per_token": metrics.get("bits_per_token"),
                    "top1_accuracy_percent": metrics.get("top1_accuracy_percent"),
                    "tokens_evaluated": metrics.get("tokens_evaluated"),
                })

                csv_path = root_out / f"{mode}_joint_openweb_results.csv"
                write_csv(results, csv_path)

    if dry_run:
        return

    csv_path = root_out / f"{mode}_joint_openweb_results.csv"
    write_csv(results, csv_path)

    sweep_rows = [r for r in results if r["kind"] == "joint_sparsegpt_gptq"]

    plot_metric(
        sweep_rows,
        x_name="sparsity",
        y_name="perplexity",
        out_path=plot_dir / f"{mode}_perplexity_vs_sparsity.png",
        title=f"{mode}: Perplexity vs sparsity, OpenWeb calibration",
    )

    plot_metric(
        sweep_rows,
        x_name="sparsity",
        y_name="checkpoint_size_mb",
        out_path=plot_dir / f"{mode}_checkpoint_size_vs_sparsity.png",
        title=f"{mode}: checkpoint size vs sparsity, OpenWeb calibration",
    )

    plot_metric(
        sweep_rows,
        x_name="sparsity",
        y_name="top1_accuracy_percent",
        out_path=plot_dir / f"{mode}_accuracy_vs_sparsity.png",
        title=f"{mode}: top-1 accuracy vs sparsity, OpenWeb calibration",
    )

    print("\nDONE")
    print(f"CSV:   {csv_path}")
    print(f"PLOTS: {plot_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gpt2", "char", "both"], required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    modes = ["gpt2", "char"] if args.mode == "both" else [args.mode]

    for mode in modes:
        run_sweep(mode=mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()