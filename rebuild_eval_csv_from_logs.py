#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

def read_metric(text, name):
    m = re.search(rf"{re.escape(name)}\s*:\s*([0-9.+\-eE]+)", text)
    return m.group(1) if m else ""

def infer_params(path):
    name = path.name
    bits = re.search(r"_b(\d+)_", name)
    sparsity = re.search(r"_s(\d+)_", name)
    groupsize = re.search(r"_g(\d+)_", name)

    return {
        "bits": bits.group(1) if bits else "",
        "sparsity": f"{int(sparsity.group(1)) / 100:.2f}" if sparsity else "",
        "groupsize": groupsize.group(1) if groupsize else "",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    logs = sorted(ckpt_dir.glob("*_eval.log"))

    rows = []
    for log in logs:
        text = log.read_text(errors="ignore")
        ckpt = str(log).replace("_eval.log", ".pt")
        params = infer_params(Path(ckpt))

        rows.append({
            "checkpoint": ckpt,
            "bits": params["bits"],
            "sparsity": params["sparsity"],
            "groupsize": params["groupsize"],
            "perplexity": read_metric(text, "Perplexity"),
            "loss": read_metric(text, "Mean loss (nats/token)"),
            "bpt": read_metric(text, "Bits per token"),
            "accuracy": read_metric(text, "Top-1 accuracy").replace("%", ""),
        })

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "bits", "sparsity", "groupsize", "perplexity", "loss", "bpt", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.out_csv}")
    print(f"Parsed logs: {len(rows)}")

if __name__ == "__main__":
    main()