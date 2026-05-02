#!/usr/bin/env python3
"""
Create a pure compressed GPTQ checkpoint from a GPTQ checkpoint that may still contain
full dequantized float weights.

Input checkpoint may contain:
    ckpt["model"]       = full dequantized model state_dict
    ckpt["gptq_layers"] = compressed qweight/scales/zero_points

Output checkpoint contains:
    ckpt["model"]       = ONLY non-quantized parameters
                          embeddings, layer norms, biases, etc.
    ckpt["gptq_layers"] = compressed GPTQ tensors

This makes file size closer to true compressed deployable size.

Example:
    python make_pure_gptq_checkpoint.py \
      --original out-shakespeare-gpt2-ft/ckpt_best.pt \
      --quantized out-shakespeare-gpt2-ft/ckpt_best_gptq4_openweb_g64.pt \
      --out out-shakespeare-gpt2-ft/ckpt_best_gptq4_openweb_g64_PURE.pt
"""

from __future__ import annotations

import argparse
import os
import copy
from typing import Dict, Any, Set

import torch


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024:
            return f"{x:.3f} {u}"
        x /= 1024
    return f"{x:.3f} PB"


def strip_orig_mod_prefix_key(k: str) -> str:
    if k.startswith("_orig_mod."):
        return k[len("_orig_mod."):]
    return k


def normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        strip_orig_mod_prefix_key(k): v
        for k, v in sd.items()
        if torch.is_tensor(v)
    }


def get_quantized_weight_keys(gptq_layers: Dict[str, Any]) -> Set[str]:
    return {f"{layer_name}.weight" for layer_name in gptq_layers.keys()}


def state_dict_bytes(sd: Dict[str, torch.Tensor]) -> int:
    return sum(tensor_bytes(v) for v in sd.values())


def gptq_layers_bytes(gptq_layers: Dict[str, Any]) -> Dict[str, int]:
    qweight_bytes = 0
    scales_bytes = 0
    zero_bytes = 0

    for _, layer_state in gptq_layers.items():
        qweight_bytes += tensor_bytes(layer_state["qweight"])
        scales_bytes += tensor_bytes(layer_state["scales"])
        zero_bytes += tensor_bytes(layer_state["zero_points"])

    return {
        "qweight": qweight_bytes,
        "scales": scales_bytes,
        "zero_points": zero_bytes,
        "total": qweight_bytes + scales_bytes + zero_bytes,
    }


def make_pure_checkpoint(original_path: str, quantized_path: str, out_path: str) -> None:
    original_ckpt = torch.load(original_path, map_location="cpu")
    quant_ckpt = torch.load(quantized_path, map_location="cpu")

    if "model" not in original_ckpt:
        raise ValueError("Original checkpoint has no ckpt['model'].")

    if "model" not in quant_ckpt:
        raise ValueError("Quantized checkpoint has no ckpt['model'].")

    if "gptq_layers" not in quant_ckpt:
        raise ValueError("Quantized checkpoint has no ckpt['gptq_layers'].")

    original_model = normalize_state_dict_keys(original_ckpt["model"])
    quant_model = normalize_state_dict_keys(quant_ckpt["model"])
    gptq_layers = quant_ckpt["gptq_layers"]

    quant_weight_keys = get_quantized_weight_keys(gptq_layers)

    pure_model = {}

    for k, v in quant_model.items():
        if k in quant_weight_keys:
            continue
        pure_model[k] = v.detach().cpu()

    pure_ckpt = copy.deepcopy(quant_ckpt)
    pure_ckpt["model"] = pure_model

    if "gptq_meta" not in pure_ckpt:
        pure_ckpt["gptq_meta"] = {}

    pure_ckpt["gptq_meta"]["pure_compressed_checkpoint"] = True
    pure_ckpt["gptq_meta"]["model_field_contents"] = "non_quantized_parameters_only"
    pure_ckpt["gptq_meta"]["note_pure_checkpoint"] = (
        "This checkpoint removed dequantized float weights for GPTQ-quantized Linear layers. "
        "The compressed Linear weights are stored in ckpt['gptq_layers']."
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(pure_ckpt, out_path)

    original_tensor_bytes = state_dict_bytes(original_model)
    quant_full_tensor_bytes = state_dict_bytes(quant_model)
    pure_nonquant_bytes = state_dict_bytes(pure_model)
    gptq_b = gptq_layers_bytes(gptq_layers)
    pure_deployable_tensor_bytes = pure_nonquant_bytes + gptq_b["total"]

    original_file_bytes = os.path.getsize(original_path)
    quantized_file_bytes = os.path.getsize(quantized_path)
    pure_file_bytes = os.path.getsize(out_path)

    print("\n" + "=" * 100)
    print("PURE GPTQ CHECKPOINT CREATED")
    print("=" * 100)

    print(f"Original checkpoint       : {original_path}")
    print(f"Quantized input checkpoint: {quantized_path}")
    print(f"Pure output checkpoint    : {out_path}")

    print("\n" + "-" * 100)
    print("File sizes on disk")
    print("-" * 100)
    print(f"Original .pt file size              : {format_bytes(original_file_bytes)}")
    print(f"Quantized mixed .pt file size        : {format_bytes(quantized_file_bytes)}")
    print(f"Pure compressed .pt file size        : {format_bytes(pure_file_bytes)}")

    print("\n" + "-" * 100)
    print("Tensor storage analysis")
    print("-" * 100)
    print(f"Original model tensor bytes          : {format_bytes(original_tensor_bytes)}")
    print(f"Mixed quant ckpt['model'] tensor bytes: {format_bytes(quant_full_tensor_bytes)}")
    print(f"Pure non-quantized tensor bytes      : {format_bytes(pure_nonquant_bytes)}")
    print(f"GPTQ qweight bytes                   : {format_bytes(gptq_b['qweight'])}")
    print(f"GPTQ scales bytes                    : {format_bytes(gptq_b['scales'])}")
    print(f"GPTQ zero_points bytes               : {format_bytes(gptq_b['zero_points'])}")
    print(f"GPTQ compressed Linear bytes         : {format_bytes(gptq_b['total'])}")
    print(f"Pure deployable tensor bytes estimate: {format_bytes(pure_deployable_tensor_bytes)}")

    print("\n" + "-" * 100)
    print("Compression ratios")
    print("-" * 100)
    print(f"Pure .pt file compression ratio      : {original_file_bytes / pure_file_bytes:.3f}x")
    print(f"Tensor compression ratio             : {original_tensor_bytes / pure_deployable_tensor_bytes:.3f}x")

    print("\nUse this pure checkpoint with:")
    print(
        f"python eval_metrics.py "
        f"--checkpoint {out_path} "
        f"--input_file val_eval.txt "
        f"--device cuda "
        f"--dtype float16 "
        f"--batch_size 4 "
        f"--prefer_quantized"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--quantized", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    make_pure_checkpoint(
        original_path=args.original,
        quantized_path=args.quantized,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()