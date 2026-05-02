#!/usr/bin/env python3
import argparse
import os
import torch


def get_group_index(col_idx, cols, groupsize):
    if groupsize == -1 or groupsize >= cols:
        return 0
    return col_idx // groupsize


def unpack_4bit_rows(packed, original_cols):
    rows, packed_cols = packed.shape
    out = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8)

    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F

    return out[:, :original_cols]


def unpack_bitmask_rows(packed, original_cols):
    rows, packed_cols = packed.shape
    out = torch.zeros((rows, packed_cols * 8), dtype=torch.bool)

    for bit in range(8):
        out[:, bit::8] = ((packed >> bit) & 1).bool()

    return out[:, :original_cols]


def dequantize_joint_layer(layer_state):
    shape = tuple(layer_state["shape"])
    rows, cols = shape

    bits = int(layer_state["bits"])
    groupsize = int(layer_state["groupsize"])
    packing = str(layer_state["packing"])
    mask_packing = str(layer_state.get("mask_packing", "bool"))

    qweight = layer_state["qweight"].cpu()
    scales = layer_state["scales"].cpu().float()
    zero_points = layer_state["zero_points"].cpu().float()

    if packing == "packed4":
        q = unpack_4bit_rows(qweight, cols).float()
    elif packing == "uint8":
        q = qweight.float()
    else:
        raise ValueError(f"Unsupported qweight packing: {packing}")

    mask = layer_state["mask"].cpu()
    if mask_packing in ("packedbits", "bitpack", "packed"):
        mask = unpack_bitmask_rows(mask, cols)
    else:
        mask = mask.bool()

    col_group_idx = torch.tensor(
        [get_group_index(c, cols, groupsize) for c in range(cols)],
        dtype=torch.long,
    )

    scale_expanded = scales[:, col_group_idx]
    zero_expanded = zero_points[:, col_group_idx]

    w = (q - zero_expanded) * scale_expanded
    w = w * mask.float()

    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if "joint_sparsegpt_gptq_layers" not in ckpt:
        raise RuntimeError("Checkpoint has no joint_sparsegpt_gptq_layers.")

    if "model" not in ckpt:
        raise RuntimeError("Checkpoint has no model field.")

    model_sd = ckpt["model"]

    restored = 0
    for layer_name, layer_state in ckpt["joint_sparsegpt_gptq_layers"].items():
        weight_key = f"{layer_name}.weight"

        dense_w = dequantize_joint_layer(layer_state)
        model_sd[weight_key] = dense_w

        restored += 1

    ckpt["model"] = model_sd
    ckpt["eval_compatible_dense_from_joint_packed"] = True

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)

    print(f"Saved eval-compatible checkpoint: {args.out}")
    print(f"Restored dense compressed weights: {restored}")


if __name__ == "__main__":
    main()