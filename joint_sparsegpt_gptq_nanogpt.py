#!/usr/bin/env python3
"""
Jungtinis SparseGPT + GPTQ post-training suspaudimas nanoGPT checkpoint'ams.

"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig

# Checkpoint pagalbinės funkcijos


def strip_orig_mod_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod."):]] = v
        else:
            out[k] = v
    return out


def load_nanogpt_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[GPT, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_args" not in ckpt or "model" not in ckpt:
        raise ValueError("Expected checkpoint with keys 'model_args' and 'model'.")

    model = GPT(GPTConfig(**ckpt["model_args"]))
    state_dict = strip_orig_mod_prefix(ckpt["model"])

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    return model, ckpt


def get_joint_weight_keys(joint_layers: Dict[str, Any]) -> Set[str]:
    return {f"{layer_name}.weight" for layer_name in joint_layers.keys()}


def build_partial_noncompressed_state_dict(
    model: GPT,
    joint_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    full_sd = model.state_dict()
    compressed_weight_keys = get_joint_weight_keys(joint_layers)

    partial_sd: Dict[str, torch.Tensor] = {}
    for k, v in full_sd.items():
        if k in compressed_weight_keys:
            continue
        partial_sd[k] = v.detach().cpu()
    return partial_sd


def save_nanogpt_checkpoint(
    original_ckpt: Dict[str, Any],
    model: GPT,
    out_path: str,
    joint_meta: Dict[str, Any],
    joint_layers: Dict[str, Any],
    keep_dequantized_state_dict: bool,
) -> None:
    new_ckpt = copy.deepcopy(original_ckpt)

    if keep_dequantized_state_dict:
        new_ckpt["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        new_ckpt["model"] = build_partial_noncompressed_state_dict(
            model=model,
            joint_layers=joint_layers,
        )

    new_ckpt["joint_sparsegpt_gptq_meta"] = joint_meta
    new_ckpt["joint_sparsegpt_gptq_layers"] = joint_layers
    new_ckpt["compression_meta"] = joint_meta

    torch.save(new_ckpt, out_path)
    print(f"Saved joint SparseGPT+GPTQ checkpoint to: {out_path}")


# ============================================================
# Kalibracijos tokenų įkėlimas
# ============================================================

def load_calibration_tokens(calib_path: str) -> torch.Tensor:
    obj = torch.load(calib_path, map_location="cpu")
    if isinstance(obj, dict) and "tokens" in obj:
        tokens = obj["tokens"]
    elif torch.is_tensor(obj):
        tokens = obj
    else:
        raise ValueError("Calibration file must contain dict['tokens'] or be a tensor.")

    if tokens.ndim != 2:
        raise ValueError(f"Expected calibration tokens shape [N, T], got {tuple(tokens.shape)}")

    return tokens.long()


# ============================================================
# Duomenų konteineriai
# ============================================================

@dataclass
class QuantParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    symmetric: bool


@dataclass
class JointLayerResult:
    qweight_uint8: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    mask: torch.Tensor
    dequant_masked_weight: torch.Tensor
    bits: int
    groupsize: int
    packing: str
    mask_packing: str
    original_shape: Tuple[int, int]
    symmetric: bool
    sparsity: float
    target_sparsity: float
    pattern: str
    pruned_count: int
    total_count: int


# ============================================================
# Grupavimo pagalbinės funkcijos
# ============================================================

def get_group_bounds(col_idx: int, cols: int, groupsize: int) -> Tuple[int, int]:
    if groupsize == -1 or groupsize >= cols:
        return 0, cols
    g0 = (col_idx // groupsize) * groupsize
    g1 = min(g0 + groupsize, cols)
    return g0, g1


def get_num_groups(cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 1
    return math.ceil(cols / groupsize)


def get_group_index(col_idx: int, cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 0
    return col_idx // groupsize


def is_group_start(col_idx: int, groupsize: int) -> bool:
    if groupsize == -1:
        return col_idx == 0
    return (col_idx % groupsize) == 0


# ============================================================
# Kvantizacijos pagalbinės funkcijos
# ============================================================

def make_quant_params_for_slice(
    W_slice: torch.Tensor,
    bits: int,
    symmetric: bool,
    mask_slice: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> QuantParams:
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")

    qmin = 0
    qmax = (1 << bits) - 1

    if mask_slice is not None:
        if mask_slice.shape != W_slice.shape:
            raise ValueError("mask_slice shape must match W_slice")
        mask_bool = mask_slice.bool()
        has_kept = mask_bool.any(dim=1, keepdim=True)
    else:
        mask_bool = None
        has_kept = None

    if symmetric:
        if mask_slice is None:
            max_abs = W_slice.abs().amax(dim=1, keepdim=True).clamp(min=eps)
        else:
            masked_abs = torch.where(mask_bool, W_slice.abs(), torch.zeros_like(W_slice))
            max_abs_kept = masked_abs.amax(dim=1, keepdim=True)
            max_abs_all = W_slice.abs().amax(dim=1, keepdim=True)
            max_abs = torch.where(has_kept, max_abs_kept, max_abs_all).clamp(min=eps)

        mid = (qmin + qmax) / 2.0
        scale = max_abs / max(mid, 1.0)
        zero_point = torch.full_like(scale, fill_value=mid)
        return QuantParams(scale, zero_point, qmin, qmax, True)

    if mask_slice is None:
        w_min = W_slice.amin(dim=1, keepdim=True)
        w_max = W_slice.amax(dim=1, keepdim=True)
    else:
        inf = torch.tensor(float("inf"), device=W_slice.device, dtype=W_slice.dtype)
        ninf = torch.tensor(float("-inf"), device=W_slice.device, dtype=W_slice.dtype)

        w_min_kept = torch.where(mask_bool, W_slice, inf).amin(dim=1, keepdim=True)
        w_max_kept = torch.where(mask_bool, W_slice, ninf).amax(dim=1, keepdim=True)

        w_min_all = W_slice.amin(dim=1, keepdim=True)
        w_max_all = W_slice.amax(dim=1, keepdim=True)

        w_min = torch.where(has_kept, w_min_kept, w_min_all)
        w_max = torch.where(has_kept, w_max_kept, w_max_all)

    same = (w_max - w_min).abs() < eps
    w_max = torch.where(same, w_min + eps, w_max)

    scale = ((w_max - w_min) / float(qmax - qmin)).clamp(min=eps)
    zero_point = torch.round(qmin - w_min / scale).clamp(qmin, qmax)

    return QuantParams(scale, zero_point, qmin, qmax, False)


def quantize_with_given_params(
    w_col: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin = 0
    qmax = (1 << bits) - 1

    q = torch.round(w_col / scale + zero).clamp(qmin, qmax)
    q_int = q.to(torch.uint8)
    q_deq = (q.to(w_col.dtype) - zero) * scale
    return q_int, q_deq


# ============================================================
# qweight pakavimas
# ============================================================

def pack_4bit_rows(qweight_uint8: torch.Tensor) -> torch.Tensor:
    if qweight_uint8.dtype != torch.uint8:
        raise ValueError("qweight_uint8 must be uint8")
    if torch.any(qweight_uint8 > 15):
        raise ValueError("4-bit packing requires values <= 15")

    rows, cols = qweight_uint8.shape
    packed_cols = (cols + 1) // 2

    packed = torch.zeros((rows, packed_cols), dtype=torch.uint8, device=qweight_uint8.device)

    even = qweight_uint8[:, 0::2]
    odd = qweight_uint8[:, 1::2]

    packed[:, :even.size(1)] |= even
    if odd.numel() > 0:
        packed[:, :odd.size(1)] |= odd << 4

    return packed


def unpack_4bit_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8")

    rows, packed_cols = packed.shape
    out = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed.device)

    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F

    return out[:, :original_cols]


def maybe_pack_qweight(qweight_uint8: torch.Tensor, bits: int, packing: str) -> torch.Tensor:
    if packing == "uint8":
        return qweight_uint8.cpu()

    if packing == "packed4":
        if bits != 4:
            raise ValueError("--packing packed4 only works with --bits 4")
        return pack_4bit_rows(qweight_uint8).cpu()

    raise ValueError(f"Unsupported packing: {packing}")


def maybe_unpack_qweight(
    qweight_stored: torch.Tensor,
    bits: int,
    packing: str,
    original_shape: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    rows, cols = original_shape
    qweight_stored = qweight_stored.to(device)

    if packing == "uint8":
        if qweight_stored.shape != (rows, cols):
            raise ValueError(f"Expected qweight shape {(rows, cols)}, got {tuple(qweight_stored.shape)}")
        return qweight_stored

    if packing == "packed4":
        if bits != 4:
            raise ValueError("packed4 only works with bits=4")
        return unpack_4bit_rows(qweight_stored, original_cols=cols)

    raise ValueError(f"Unsupported packing: {packing}")


# ============================================================
# Maskės bitinis pakavimas
# ============================================================

def pack_bool_mask_rows(mask: torch.Tensor) -> torch.Tensor:
    """
    Supakuoja bool maskę [rows, cols] į uint8 [rows, ceil(cols / 8)].

    Bito tvarka:
        bit 0 = stulpelis 0 byte viduje
        bit 1 = stulpelis 1 byte viduje
        ...
        bit 7 = stulpelis 7 byte viduje

    True / keep  -> 1
    False / prune -> 0
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    rows, cols = mask.shape
    packed_cols = (cols + 7) // 8
    padded_cols = packed_cols * 8

    if padded_cols != cols:
        pad = torch.zeros((rows, padded_cols - cols), dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask, pad], dim=1)

    mask_u8 = mask.to(torch.uint8).view(rows, packed_cols, 8)
    shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=mask.device)

    packed = (mask_u8 * shifts.view(1, 1, 8)).sum(dim=2).to(torch.uint8)
    return packed


def unpack_bool_mask_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    """
    Išpakuoja uint8 [rows, ceil(cols / 8)] į bool [rows, cols].
    """
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)

    rows, packed_cols = packed.shape
    shifts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=packed.device)

    bits = ((packed.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 1).bool()
    mask = bits.view(rows, packed_cols * 8)

    return mask[:, :original_cols]


def maybe_pack_mask(mask: torch.Tensor, mask_packing: str) -> torch.Tensor:
    if mask_packing == "bool":
        return mask.cpu().bool()

    if mask_packing == "packedbits":
        return pack_bool_mask_rows(mask).cpu()

    raise ValueError(f"Unsupported mask_packing: {mask_packing}")


def maybe_unpack_mask(
    mask_stored: torch.Tensor,
    mask_packing: str,
    original_shape: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    rows, cols = original_shape
    mask_stored = mask_stored.to(device)

    if mask_packing == "bool":
        mask = mask_stored.bool()
        if mask.shape != (rows, cols):
            raise ValueError(f"Expected mask shape {(rows, cols)}, got {tuple(mask.shape)}")
        return mask

    if mask_packing == "packedbits":
        return unpack_bool_mask_rows(mask_stored.to(torch.uint8), original_cols=cols)

    raise ValueError(f"Unsupported mask_packing: {mask_packing}")


# ============================================================
# Sparse maskės pagalbinės funkcijos
# ============================================================

def parse_nm_pattern(pattern: str) -> Optional[Tuple[int, int]]:
    pattern = pattern.strip().lower()
    if pattern in ("", "unstructured", "none"):
        return None

    if ":" not in pattern:
        raise ValueError("Pattern must be 'unstructured' or N:M, e.g. '2:4'.")

    a, b = pattern.split(":")
    n = int(a)
    m = int(b)

    if n < 0 or m <= 0 or n > m:
        raise ValueError(f"Invalid N:M pattern {pattern}. Need 0 <= N <= M.")

    return n, m


@torch.no_grad()
def select_unstructured_mask_block_from_score(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    rows, block_cols = score.shape
    total = rows * block_cols
    n_prune_total = int(round(sparsity * total))

    if n_prune_total <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    if n_prune_total >= total:
        return torch.zeros_like(score, dtype=torch.bool)

    flat_score = score.reshape(-1)
    n_keep_total = total - n_prune_total

    keep_idx = torch.topk(flat_score, k=n_keep_total, largest=True, sorted=False).indices

    flat_mask = torch.zeros(total, dtype=torch.bool, device=score.device)
    flat_mask[keep_idx] = True
    return flat_mask.view(rows, block_cols)


@torch.no_grad()
def select_nm_mask_block_from_score(score: torch.Tensor, n_zero: int, m: int) -> torch.Tensor:
    rows, block_cols = score.shape
    mask = torch.ones_like(score, dtype=torch.bool)

    for g0 in range(0, block_cols, m):
        g1 = min(g0 + m, block_cols)
        group_cols = g1 - g0

        if group_cols == m:
            prune_count = n_zero
        else:
            prune_count = int(round((n_zero / float(m)) * group_cols))

        prune_count = max(0, min(prune_count, group_cols))

        if prune_count == 0:
            continue

        if prune_count == group_cols:
            mask[:, g0:g1] = False
            continue

        local_score = score[:, g0:g1]
        prune_idx = torch.topk(local_score, k=prune_count, largest=False, dim=1, sorted=False).indices

        row_idx = torch.arange(rows, device=score.device).view(-1, 1).expand_as(prune_idx)
        local_mask = mask[:, g0:g1]
        local_mask[row_idx, prune_idx] = False
        mask[:, g0:g1] = local_mask

    return mask


@torch.no_grad()
def compute_joint_mask_score_block(
    W_block: torch.Tensor,
    Hinv_diag_block: torch.Tensor,
    bits: int,
    symmetric: bool,
    quant_aware: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Sparse-only score:
        score = w^2 / Hinv_diag

    Joint quant-aware score:
        keep benefit = (w^2 - (w - q)^2) / Hinv_diag

    Didesnis score reiškia, kad svorį svarbiau palikti.
    """
    diag = Hinv_diag_block.to(W_block.device, dtype=W_block.dtype).abs().clamp(min=eps)

    if not quant_aware:
        return (W_block.float() ** 2) / diag.float().view(1, -1)

    qp = make_quant_params_for_slice(W_block, bits=bits, symmetric=symmetric, mask_slice=None)
    scale = qp.scale.to(W_block.dtype)
    zero = qp.zero_point.to(W_block.dtype)

    q = torch.round(W_block / scale + zero).clamp(qp.qmin, qp.qmax)
    q_deq = (q.to(W_block.dtype) - zero) * scale

    prune_err2 = W_block.float() ** 2
    quant_err2 = (W_block.float() - q_deq.float()) ** 2

    return (prune_err2 - quant_err2) / diag.float().view(1, -1)


# ============================================================
# Hessian stabilizavimas
# ============================================================

@torch.no_grad()
def stable_cholesky_inverse_info(
    H: torch.Tensor,
    percdamp: float,
    max_tries: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if H.ndim != 2 or H.size(0) != H.size(1):
        raise ValueError(f"H must be square, got {tuple(H.shape)}")

    H64 = H.to(torch.float64)
    H64 = 0.5 * (H64 + H64.T)

    diag = torch.diag(H64)
    if not torch.isfinite(diag).all():
        raise RuntimeError("Hessian diagonal contains non-finite values.")

    diag_abs = diag.abs()
    diag_mean = diag_abs.mean().item()
    diag_max = diag_abs.max().item()
    base = max(diag_mean, 1e-12)

    n = H64.size(0)
    ar = torch.arange(n, device=H64.device)
    last_exc: Optional[Exception] = None

    multipliers = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0][:max_tries]

    for mult in multipliers:
        used_damp = percdamp * mult * base
        H_try = H64.clone()
        H_try[ar, ar] += used_damp

        try:
            chol = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)
            return H_try, Hinv_chol_upper, used_damp
        except RuntimeError as exc:
            last_exc = exc

    max_base = max(diag_max, 1e-12)

    for mult in [1.0, 10.0, 100.0, 1000.0]:
        used_damp = percdamp * mult * max_base
        H_try = H64.clone()
        H_try[ar, ar] += used_damp

        try:
            chol = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)
            return H_try, Hinv_chol_upper, used_damp
        except RuntimeError as exc:
            last_exc = exc

    raise RuntimeError(
        "Cholesky failed after adaptive damping retries. "
        f"Initial percdamp={percdamp}. Last error: {last_exc}"
    )


# ============================================================
# Jungtinis SparseGPT + GPTQ vienam Linear sluoksniui
# ============================================================

@torch.no_grad()
def joint_sparsegpt_gptq_linear(
    layer: nn.Linear,
    H: torch.Tensor,
    bits: int,
    sparsity: float,
    pattern: str,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    groupsize: int,
    packing: str,
    mask_packing: str,
    symmetric: bool,
    act_order: bool,
    quant_aware_mask: bool,
) -> JointLayerResult:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    if packing == "packed4" and bits != 4:
        raise ValueError("--packing packed4 only valid with --bits 4")

    W_orig = layer.weight.data.float().clone()
    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch. Expected {(cols, cols)}, got {tuple(H.shape)}")

    nm = parse_nm_pattern(pattern)
    if nm is not None:
        n_zero, m = nm
        mask_blocksize_eff = m
        effective_sparsity = n_zero / float(m)
    else:
        n_zero, m = None, None
        mask_blocksize_eff = mask_blocksize
        effective_sparsity = sparsity

    print(f"      bits: {bits}")
    print(f"      target sparsity: {effective_sparsity:.4f}")
    print(f"      pattern: {pattern}")
    print(f"      groupsize: {groupsize}")
    print(f"      qweight packing: {packing}")
    print(f"      mask packing: {mask_packing}")
    print(f"      lazy update blocksize B: {blocksize}")
    print(f"      mask selection blocksize Bs: {mask_blocksize_eff}")
    print(f"      quant-aware mask: {quant_aware_mask}")

    H_damped, Hinv_chol_upper, used_damp = stable_cholesky_inverse_info(H, percdamp=percdamp)
    print(f"      used damping: {used_damp:.6e}")

    Hinv = Hinv_chol_upper.T @ Hinv_chol_upper
    Hinv = Hinv.to(W_orig.dtype)

    if act_order:
        perm = torch.argsort(torch.diag(H_damped), descending=True)
        invperm = torch.argsort(perm)
        W = W_orig[:, perm].contiguous()
        Hinv = Hinv[perm][:, perm].contiguous()
    else:
        invperm = None
        W = W_orig.clone()

    Q_deq_masked = torch.zeros_like(W)
    qweight_uint8 = torch.zeros((rows, cols), dtype=torch.uint8, device=W.device)
    M_global = torch.ones((rows, cols), dtype=torch.bool, device=W.device)

    ngroups = get_num_groups(cols, groupsize)
    scales = torch.zeros((rows, ngroups), dtype=torch.float32, device=W.device)
    zero_points = torch.zeros((rows, ngroups), dtype=torch.float32, device=W.device)

    selected_mask_until = -1
    Hinv_diag = torch.diag(Hinv)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2].contiguous()

        for local_i in range(count):
            global_col = i1 + local_i

            # Esminė dalis: maskė parenkama pagal dabartinį Hessian-koreguotą W.
            if global_col >= selected_mask_until:
                mb0 = global_col
                mb1 = min(mb0 + mask_blocksize_eff, cols)

                score = compute_joint_mask_score_block(
                    W_block=W[:, mb0:mb1],
                    Hinv_diag_block=Hinv_diag[mb0:mb1],
                    bits=bits,
                    symmetric=symmetric,
                    quant_aware=quant_aware_mask,
                )

                if nm is None:
                    M_block = select_unstructured_mask_block_from_score(score, sparsity)
                else:
                    M_block = select_nm_mask_block_from_score(score, n_zero=n_zero, m=m)

                M_global[:, mb0:mb1] = M_block
                selected_mask_until = mb1

            group_idx = get_group_index(global_col, cols, groupsize)

            # Esminė dalis: quant parametrai skaičiuojami iš dabartinės grupės ir keep maskės.
            if is_group_start(global_col, groupsize):
                g0, g1 = get_group_bounds(global_col, cols, groupsize)

                while selected_mask_until < g1:
                    mb0 = selected_mask_until
                    mb1 = min(mb0 + mask_blocksize_eff, cols)

                    score = compute_joint_mask_score_block(
                        W_block=W[:, mb0:mb1],
                        Hinv_diag_block=Hinv_diag[mb0:mb1],
                        bits=bits,
                        symmetric=symmetric,
                        quant_aware=quant_aware_mask,
                    )

                    if nm is None:
                        M_block = select_unstructured_mask_block_from_score(score, sparsity)
                    else:
                        M_block = select_nm_mask_block_from_score(score, n_zero=n_zero, m=m)

                    M_global[:, mb0:mb1] = M_block
                    selected_mask_until = mb1

                qp = make_quant_params_for_slice(
                    W_slice=W[:, g0:g1],
                    bits=bits,
                    symmetric=symmetric,
                    mask_slice=M_global[:, g0:g1],
                )

                scales[:, group_idx] = qp.scale.squeeze(1).to(torch.float32)
                zero_points[:, group_idx] = qp.zero_point.squeeze(1).to(torch.float32)

            d = Hinv1[local_i, local_i]
            if d.abs().item() < 1e-12:
                raise RuntimeError(f"Near-zero Hinv diagonal at column {global_col}: {d.item()}")

            scale = scales[:, group_idx].to(W.dtype)
            zero = zero_points[:, group_idx].to(W.dtype)

            w = W1[:, local_i]
            q_int, q_deq = quantize_with_given_params(w, scale, zero, bits)

            keep_mask_col = M_global[:, global_col]

            # Sujungtas operatorius: sparse maskė * GPTQ dequantized kvantizuotas stulpelis.
            compressed = torch.where(keep_mask_col, q_deq, torch.zeros_like(q_deq))

            Q1[:, local_i] = compressed
            Q_deq_masked[:, global_col] = compressed

            qweight_uint8[:, global_col] = torch.where(
                keep_mask_col,
                q_int,
                torch.zeros_like(q_int),
            )

            # Sujungta klaida: vienu metu kompensuojama pruning + quantization klaida.
            err = (w - compressed) / d
            Err1[:, local_i] = err

            if local_i + 1 < count:
                W1[:, local_i + 1:count] -= (
                    err.unsqueeze(1)
                    @ Hinv1[local_i, local_i + 1:count].unsqueeze(0)
                )

        W[:, i1:i2] = Q1

        if i2 < cols:
            W[:, i2:cols] -= Err1 @ Hinv[i1:i2, i2:cols]

    if act_order:
        assert invperm is not None

        Q_deq_masked = Q_deq_masked[:, invperm].contiguous()
        qweight_uint8 = qweight_uint8[:, invperm].contiguous()
        M_global = M_global[:, invperm].contiguous()

        # Metadata atstatoma originalia stulpelių tvarka.
        ngroups_orig = get_num_groups(cols, groupsize)
        scales_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=W_orig.device)
        zero_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=W_orig.device)
        q_re = torch.zeros_like(qweight_uint8)
        Q_re = torch.zeros_like(Q_deq_masked)

        for g in range(ngroups_orig):
            g0 = 0 if groupsize == -1 else g * groupsize
            g1 = cols if groupsize == -1 else min((g + 1) * groupsize, cols)

            qp = make_quant_params_for_slice(
                W_slice=Q_deq_masked[:, g0:g1],
                bits=bits,
                symmetric=symmetric,
                mask_slice=M_global[:, g0:g1],
            )

            scales_re[:, g] = qp.scale.squeeze(1).to(torch.float32)
            zero_re[:, g] = qp.zero_point.squeeze(1).to(torch.float32)

            scale = scales_re[:, g].to(Q_deq_masked.dtype)
            zero = zero_re[:, g].to(Q_deq_masked.dtype)

            for c in range(g0, g1):
                q_int, q_deq = quantize_with_given_params(Q_deq_masked[:, c], scale, zero, bits)
                q_re[:, c] = torch.where(M_global[:, c], q_int, torch.zeros_like(q_int))
                Q_re[:, c] = torch.where(M_global[:, c], q_deq, torch.zeros_like(q_deq))

        scales = scales_re
        zero_points = zero_re
        qweight_uint8 = q_re
        Q_deq_masked = Q_re

    Q_deq_masked = Q_deq_masked * M_global.to(Q_deq_masked.dtype)
    layer.weight.data.copy_(Q_deq_masked.to(layer.weight.data.dtype))

    total_count = rows * cols
    kept_count = int(M_global.sum().item())
    pruned_count = total_count - kept_count
    actual_sparsity = pruned_count / float(total_count)

    return JointLayerResult(
        qweight_uint8=qweight_uint8.detach(),
        scales=scales.detach().cpu(),
        zero_points=zero_points.detach().cpu(),
        mask=M_global.detach().cpu(),
        dequant_masked_weight=Q_deq_masked.detach().cpu(),
        bits=bits,
        groupsize=groupsize,
        packing=packing,
        mask_packing=mask_packing,
        original_shape=(rows, cols),
        symmetric=symmetric,
        sparsity=actual_sparsity,
        target_sparsity=effective_sparsity,
        pattern=pattern,
        pruned_count=pruned_count,
        total_count=total_count,
    )


# ============================================================
# Runtime wrapper
# ============================================================

class JointSparseQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        groupsize: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        mask: torch.Tensor,
        packing: str,
        mask_packing: str,
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize
        self.packing = packing
        self.mask_packing = mask_packing
        self.cache_dequantized = cache_dequantized

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous().to(torch.float32))
        self.register_buffer("zero_points", zero_points.contiguous().to(torch.float32))
        self.register_buffer("mask", mask.contiguous())

        col_group_idx = torch.tensor(
            [get_group_index(c, in_features, groupsize) for c in range(in_features)],
            dtype=torch.long,
        )
        self.register_buffer("col_group_idx", col_group_idx)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        device = self.qweight.device

        q = maybe_unpack_qweight(
            qweight_stored=self.qweight,
            bits=self.bits,
            packing=self.packing,
            original_shape=(self.out_features, self.in_features),
            device=device,
        ).to(torch.float32)

        mask = maybe_unpack_mask(
            mask_stored=self.mask,
            mask_packing=self.mask_packing,
            original_shape=(self.out_features, self.in_features),
            device=device,
        )

        scale_expanded = self.scales[:, self.col_group_idx]
        zero_expanded = self.zero_points[:, self.col_group_idx]

        w = (q - zero_expanded) * scale_expanded
        w = w * mask.to(dtype=w.dtype)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dequantized:
            if self._cached_weight is None or self._cached_weight.device != x.device:
                self._cached_weight = self.dequantize_weight().to(x.device)
            w = self._cached_weight.to(dtype=x.dtype)
        else:
            w = self.dequantize_weight().to(device=x.device, dtype=x.dtype)

        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        return F.linear(x, w, bias)

    @staticmethod
    def from_layer_state(
        layer_state: Dict[str, Any],
        bias: Optional[torch.Tensor],
        device: torch.device,
        cache_dequantized: bool = False,
    ) -> "JointSparseQuantLinear":
        shape = tuple(layer_state["shape"])
        out_features, in_features = shape

        return JointSparseQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bits=int(layer_state["bits"]),
            groupsize=int(layer_state["groupsize"]),
            qweight=layer_state["qweight"].to(device),
            scales=layer_state["scales"].to(device),
            zero_points=layer_state["zero_points"].to(device),
            mask=layer_state["mask"].to(device),
            packing=str(layer_state["packing"]),
            mask_packing=str(layer_state.get("mask_packing", "bool")),
            bias=bias,
            cache_dequantized=cache_dequantized,
        )


# ============================================================
# Modelio modulių pagalbinės funkcijos
# ============================================================

def get_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    obj = root
    for part in full_name.split("."):
        obj = getattr(obj, part)
    return obj


def set_module_by_name(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


@torch.no_grad()
def convert_model_to_joint_sparse_quant_linear(
    model: GPT,
    ckpt: Dict[str, Any],
    device: torch.device,
    cache_dequantized: bool = False,
) -> GPT:
    if "joint_sparsegpt_gptq_layers" not in ckpt:
        raise ValueError("Checkpoint has no 'joint_sparsegpt_gptq_layers' entry.")

    joint_layers = ckpt["joint_sparsegpt_gptq_layers"]

    for layer_name, layer_state in joint_layers.items():
        orig_layer = get_module_by_name(model, layer_name)
        if not isinstance(orig_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer_name}, got {type(orig_layer)}")

        bias = orig_layer.bias.detach().clone() if orig_layer.bias is not None else None

        qlayer = JointSparseQuantLinear.from_layer_state(
            layer_state=layer_state,
            bias=bias,
            device=device,
            cache_dequantized=cache_dequantized,
        )

        set_module_by_name(model, layer_name, qlayer)

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def load_joint_sparsegpt_gptq_checkpoint(
    ckpt_path: str,
    device: torch.device,
    prefer_joint_wrapper: bool = True,
    cache_dequantized: bool = False,
) -> Tuple[GPT, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_args" not in ckpt:
        raise ValueError("Checkpoint missing model_args")

    model = GPT(GPTConfig(**ckpt["model_args"]))

    if "model" in ckpt and ckpt["model"]:
        state_dict = strip_orig_mod_prefix(ckpt["model"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] missing keys when loading model state_dict: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys when loading model state_dict: {unexpected}")

    model.eval()
    model.to(device)

    if prefer_joint_wrapper and "joint_sparsegpt_gptq_layers" in ckpt:
        print("Replacing compressed linear layers with JointSparseQuantLinear runtime modules...")
        model = convert_model_to_joint_sparse_quant_linear(
            model=model,
            ckpt=ckpt,
            device=device,
            cache_dequantized=cache_dequantized,
        )

    return model, ckpt


# ============================================================
# nanoGPT forward pagalbinės funkcijos
# ============================================================

@torch.no_grad()
def model_has_tied_lm_head(model: GPT) -> bool:
    if not hasattr(model, "lm_head"):
        return False
    if not hasattr(model.transformer, "wte"):
        return False
    try:
        return model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    except Exception:
        return False


@torch.no_grad()
def compute_hidden_before_blocks(
    model: GPT,
    tokens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    tok_emb = model.transformer.wte(tokens.to(device))

    pos = torch.arange(0, tokens.size(1), dtype=torch.long, device=device)
    pos_emb = model.transformer.wpe(pos)[None, :, :]

    x = tok_emb + pos_emb

    if hasattr(model.transformer, "drop"):
        x = model.transformer.drop(x)

    return x.detach()


@torch.no_grad()
def run_block_on_hidden(
    block: nn.Module,
    hidden: torch.Tensor,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        out = block(hidden)
    return out.detach()


@torch.no_grad()
def get_block_module_names(model: GPT, block_idx: int) -> List[Tuple[str, nn.Linear]]:
    block = model.transformer.h[block_idx]
    names: List[Tuple[str, nn.Linear]] = []

    for subname, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            full_name = f"transformer.h.{block_idx}.{subname}"
            names.append((full_name, mod))

    return names


# ============================================================
# Hessian rinkimas
# ============================================================

class HessianCollector:
    def __init__(
        self,
        layer: nn.Linear,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        use_factor_2: bool = True,
    ):
        self.layer = layer
        self.device = device
        self.dtype = dtype
        self.use_factor_2 = use_factor_2
        self.in_features = layer.in_features
        self.H = torch.zeros((self.in_features, self.in_features), device=device, dtype=dtype)
        self.nsamples = 0
        self.handle = None

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        if not torch.is_tensor(x):
            return

        x = x.detach().reshape(-1, x.size(-1)).to(self.device, dtype=self.dtype)
        scale = 2.0 if self.use_factor_2 else 1.0

        self.H += scale * x.t().matmul(x)
        self.nsamples += x.size(0)

    def register(self) -> None:
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def collect_hessian_for_layer_from_block_inputs(
    block: nn.Module,
    layer: nn.Linear,
    block_inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, int]:
    collector = HessianCollector(
        layer=layer,
        device=device,
        dtype=torch.float64,
        use_factor_2=True,
    )

    collector.register()

    autocast_enabled = device.type == "cuda"
    n = block_inputs.size(0)

    for i in range(0, n, batch_size):
        batch_hidden = block_inputs[i:i + batch_size].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            _ = block(batch_hidden)

    collector.remove()
    return collector.H, collector.nsamples


# ============================================================
# Sluoksnių atranka
# ============================================================

def should_compress_layer_name(
    name: str,
    include: str,
    exclude: str,
    skip_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> bool:
    if include and include not in name:
        return False
    if exclude and exclude in name:
        return False
    if skip_lm_head and name == "lm_head":
        return False
    if skip_attn_out and name.endswith("attn.c_proj"):
        return False
    if skip_mlp_out and name.endswith("mlp.c_proj"):
        return False
    return True


# ============================================================
# Nuoseklus blokinis suspaudimas
# ============================================================

@torch.no_grad()
def compress_transformer_blocks_sequentially(
    model: GPT,
    calib_tokens: torch.Tensor,
    bits: int,
    sparsity: float,
    pattern: str,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    groupsize: int,
    include: str,
    exclude: str,
    packing: str,
    mask_packing: str,
    symmetric: bool,
    act_order: bool,
    quant_aware_mask: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
    joint_layers_out: Dict[str, Any],
    store_debug_dequant: bool,
) -> torch.Tensor:
    hidden = compute_hidden_before_blocks(model, calib_tokens.to(device), device=device)
    print(f"Initial hidden cache shape before blocks: {tuple(hidden.shape)}")

    nblocks = len(model.transformer.h)
    print(f"\nFound {nblocks} transformer blocks.")

    for bi in range(nblocks):
        block = model.transformer.h[bi]

        block_layers = get_block_module_names(model, bi)
        block_layers = [
            (name, layer)
            for name, layer in block_layers
            if should_compress_layer_name(
                name=name,
                include=include,
                exclude=exclude,
                skip_lm_head=True,
                skip_attn_out=skip_attn_out,
                skip_mlp_out=skip_mlp_out,
            )
        ]

        print(f"\n=== Joint SparseGPT+GPTQ compressing transformer block {bi}/{nblocks - 1} ===")

        if not block_layers:
            print("  no selected linear layers in this block.")
        else:
            for name, mod in block_layers:
                print(f"  - {name}: {tuple(mod.weight.shape)}")

        for idx, (name, layer) in enumerate(block_layers, start=1):
            print(f"\n    [{idx}/{len(block_layers)}] Collecting Hessian for: {name}")

            H, nsamples = collect_hessian_for_layer_from_block_inputs(
                block=block,
                layer=layer,
                block_inputs=hidden,
                batch_size=batch_size,
                device=device,
                amp_dtype=amp_dtype,
            )

            print(f"      samples: {nsamples}")
            print(f"      H shape: {tuple(H.shape)}")
            print(f"      joint compressing {name} ...")

            result = joint_sparsegpt_gptq_linear(
                layer=layer,
                H=H.to(layer.weight.device),
                bits=bits,
                sparsity=sparsity,
                pattern=pattern,
                percdamp=percdamp,
                blocksize=blocksize,
                mask_blocksize=mask_blocksize,
                groupsize=groupsize,
                packing=packing,
                mask_packing=mask_packing,
                symmetric=symmetric,
                act_order=act_order,
                quant_aware_mask=quant_aware_mask,
            )

            qweight_stored = maybe_pack_qweight(result.qweight_uint8.cpu(), bits=result.bits, packing=result.packing)
            mask_stored = maybe_pack_mask(result.mask.cpu(), mask_packing=result.mask_packing)

            layer_state: Dict[str, Any] = {
                "bits": int(result.bits),
                "groupsize": int(result.groupsize),
                "packing": str(result.packing),
                "mask_packing": str(result.mask_packing),
                "shape": list(result.original_shape),
                "qweight": qweight_stored,
                "scales": result.scales.cpu(),
                "zero_points": result.zero_points.cpu(),
                "mask": mask_stored,
                "symmetric": bool(result.symmetric),
                "sparsity": float(result.sparsity),
                "target_sparsity": float(result.target_sparsity),
                "pattern": str(result.pattern),
                "pruned_count": int(result.pruned_count),
                "total_count": int(result.total_count),
            }

            if store_debug_dequant:
                layer_state["dequant_masked_weight"] = result.dequant_masked_weight.cpu()

            joint_layers_out[name] = layer_state

            print(
                f"      saved joint tensors: "
                f"qweight={tuple(qweight_stored.shape)}, "
                f"scales={tuple(result.scales.shape)}, "
                f"zero_points={tuple(result.zero_points.shape)}, "
                f"mask={tuple(mask_stored.shape)}, "
                f"sparsity={100.0 * result.sparsity:.2f}%"
            )

        hidden = run_block_on_hidden(block, hidden, amp_dtype=amp_dtype, device=device)

    return hidden


@torch.no_grad()
def compress_nonblock_linears_after_blocks(
    model: GPT,
    hidden_after_blocks: torch.Tensor,
    bits: int,
    sparsity: float,
    pattern: str,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    groupsize: int,
    include: str,
    exclude: str,
    packing: str,
    mask_packing: str,
    symmetric: bool,
    act_order: bool,
    quant_aware_mask: bool,
    skip_lm_head: bool,
    skip_tied_lm_head: bool,
    joint_layers_out: Dict[str, Any],
    store_debug_dequant: bool,
) -> None:
    block_mod_ids = {
        id(m)
        for bi in range(len(model.transformer.h))
        for _, m in get_block_module_names(model, bi)
    }

    candidates: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and id(mod) not in block_mod_ids:
            candidates.append((name, mod))

    if not candidates:
        return

    tied = model_has_tied_lm_head(model)

    selected: List[Tuple[str, nn.Linear]] = []
    for name, mod in candidates:
        if not should_compress_layer_name(
            name=name,
            include=include,
            exclude=exclude,
            skip_lm_head=skip_lm_head,
            skip_attn_out=False,
            skip_mlp_out=False,
        ):
            continue

        if skip_tied_lm_head and tied and name == "lm_head":
            print("[info] skipping compression of tied lm_head")
            continue

        selected.append((name, mod))

    if not selected:
        return

    print("\n=== Joint SparseGPT+GPTQ compressing non-block linear layers ===")

    for idx, (name, layer) in enumerate(selected, start=1):
        print(f"\n    [{idx}/{len(selected)}] Collecting Hessian for: {name}")

        collector = HessianCollector(layer=layer, device=device, dtype=torch.float64, use_factor_2=True)
        collector.register()

        autocast_enabled = device.type == "cuda"
        n = hidden_after_blocks.size(0)

        for i in range(0, n, batch_size):
            batch_hidden = hidden_after_blocks[i:i + batch_size].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                x = model.transformer.ln_f(batch_hidden)
                _ = model.lm_head(x)

        collector.remove()
        H, nsamples = collector.H, collector.nsamples

        print(f"      samples: {nsamples}")
        print(f"      H shape: {tuple(H.shape)}")
        print(f"      joint compressing {name} ...")

        result = joint_sparsegpt_gptq_linear(
            layer=layer,
            H=H.to(layer.weight.device),
            bits=bits,
            sparsity=sparsity,
            pattern=pattern,
            percdamp=percdamp,
            blocksize=blocksize,
            mask_blocksize=mask_blocksize,
            groupsize=groupsize,
            packing=packing,
            mask_packing=mask_packing,
            symmetric=symmetric,
            act_order=act_order,
            quant_aware_mask=quant_aware_mask,
        )

        qweight_stored = maybe_pack_qweight(result.qweight_uint8.cpu(), bits=result.bits, packing=result.packing)
        mask_stored = maybe_pack_mask(result.mask.cpu(), mask_packing=result.mask_packing)

        layer_state: Dict[str, Any] = {
            "bits": int(result.bits),
            "groupsize": int(result.groupsize),
            "packing": str(result.packing),
            "mask_packing": str(result.mask_packing),
            "shape": list(result.original_shape),
            "qweight": qweight_stored,
            "scales": result.scales.cpu(),
            "zero_points": result.zero_points.cpu(),
            "mask": mask_stored,
            "symmetric": bool(result.symmetric),
            "sparsity": float(result.sparsity),
            "target_sparsity": float(result.target_sparsity),
            "pattern": str(result.pattern),
            "pruned_count": int(result.pruned_count),
            "total_count": int(result.total_count),
        }

        if store_debug_dequant:
            layer_state["dequant_masked_weight"] = result.dequant_masked_weight.cpu()

        joint_layers_out[name] = layer_state

        print(
            f"      saved joint tensors: "
            f"qweight={tuple(qweight_stored.shape)}, "
            f"scales={tuple(result.scales.shape)}, "
            f"zero_points={tuple(result.zero_points.shape)}, "
            f"mask={tuple(mask_stored.shape)}, "
            f"sparsity={100.0 * result.sparsity:.2f}%"
        )


@torch.no_grad()
def compress_model_joint_blockwise(
    model: GPT,
    calib_tokens: torch.Tensor,
    bits: int,
    sparsity: float,
    pattern: str,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    groupsize: int,
    include: str,
    exclude: str,
    packing: str,
    mask_packing: str,
    symmetric: bool,
    act_order: bool,
    quant_aware_mask: bool,
    skip_lm_head: bool,
    skip_tied_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
    store_debug_dequant: bool,
) -> Dict[str, Any]:
    joint_layers_out: Dict[str, Any] = {}

    hidden_after_blocks = compress_transformer_blocks_sequentially(
        model=model,
        calib_tokens=calib_tokens,
        bits=bits,
        sparsity=sparsity,
        pattern=pattern,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=percdamp,
        blocksize=blocksize,
        mask_blocksize=mask_blocksize,
        groupsize=groupsize,
        include=include,
        exclude=exclude,
        packing=packing,
        mask_packing=mask_packing,
        symmetric=symmetric,
        act_order=act_order,
        quant_aware_mask=quant_aware_mask,
        skip_attn_out=skip_attn_out,
        skip_mlp_out=skip_mlp_out,
        joint_layers_out=joint_layers_out,
        store_debug_dequant=store_debug_dequant,
    )

    compress_nonblock_linears_after_blocks(
        model=model,
        hidden_after_blocks=hidden_after_blocks,
        bits=bits,
        sparsity=sparsity,
        pattern=pattern,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=percdamp,
        blocksize=blocksize,
        mask_blocksize=mask_blocksize,
        groupsize=groupsize,
        include=include,
        exclude=exclude,
        packing=packing,
        mask_packing=mask_packing,
        symmetric=symmetric,
        act_order=act_order,
        quant_aware_mask=quant_aware_mask,
        skip_lm_head=skip_lm_head,
        skip_tied_lm_head=skip_tied_lm_head,
        joint_layers_out=joint_layers_out,
        store_debug_dequant=store_debug_dequant,
    )

    return joint_layers_out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--pattern", type=str, default="unstructured")
    parser.add_argument("--groupsize", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--mask_blocksize", type=int, default=128)

    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")

    parser.add_argument(
        "--packing",
        type=str,
        default="uint8",
        choices=["uint8", "packed4"],
        help="'uint8' stores one quantized value per byte. 'packed4' packs two 4-bit values per byte.",
    )

    parser.add_argument(
        "--mask_packing",
        type=str,
        default="packedbits",
        choices=["bool", "packedbits"],
        help="'bool' stores dense mask. 'packedbits' stores 8 mask values per byte.",
    )

    parser.add_argument(
        "--keep_dequantized_state_dict",
        action="store_true",
        help="Keep full dense dequantized masked weights in ckpt['model'] for simple eval compatibility.",
    )

    parser.add_argument(
        "--store_debug_dequant",
        action="store_true",
        help="Store dequant_masked_weight in every compressed layer. Do NOT use for compact checkpoints.",
    )

    parser.add_argument("--act_order", action="store_true")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--no_quant_aware_mask", action="store_true")

    parser.add_argument("--skip_lm_head", action="store_true")
    parser.add_argument("--skip_tied_lm_head", action="store_true")
    parser.add_argument("--skip_attn_out", action="store_true")
    parser.add_argument("--skip_mlp_out", action="store_true")

    args = parser.parse_args()

    if args.bits < 2 or args.bits > 8:
        raise ValueError("--bits must be in [2, 8]")

    if args.packing == "packed4" and args.bits != 4:
        raise ValueError("--packing packed4 is only valid with --bits 4")

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0, 1)")

    nm = parse_nm_pattern(args.pattern)
    if nm is not None:
        n_zero, m = nm
        print(f"[info] using semi-structured pattern {args.pattern}: {n_zero} zeros per {m} weights")
    else:
        print(f"[info] using unstructured sparsity={args.sparsity}")

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_nanogpt_checkpoint(args.checkpoint, device=device)

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")

    model_block_size = model.config.block_size
    if calib_tokens.size(1) > model_block_size:
        calib_tokens = calib_tokens[:, :model_block_size]
        print(f"Trimmed calibration sequence length to model block_size={model_block_size}")

    tied = model_has_tied_lm_head(model)
    if tied:
        print("[info] model appears to have tied token embedding and lm_head weights")

    joint_layers = compress_model_joint_blockwise(
        model=model,
        calib_tokens=calib_tokens,
        bits=args.bits,
        sparsity=args.sparsity,
        pattern=args.pattern,
        batch_size=args.batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        mask_blocksize=args.mask_blocksize,
        groupsize=args.groupsize,
        include=args.include,
        exclude=args.exclude,
        packing=args.packing,
        mask_packing=args.mask_packing,
        symmetric=bool(args.symmetric),
        act_order=bool(args.act_order),
        quant_aware_mask=not bool(args.no_quant_aware_mask),
        skip_lm_head=bool(args.skip_lm_head),
        skip_tied_lm_head=bool(args.skip_tied_lm_head),
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
        store_debug_dequant=bool(args.store_debug_dequant),
    )

    total_pruned = sum(int(v["pruned_count"]) for v in joint_layers.values())
    total_weights = sum(int(v["total_count"]) for v in joint_layers.values())
    actual_total_sparsity = total_pruned / float(total_weights) if total_weights > 0 else 0.0

    joint_meta = {
        "method": "joint_sparsegpt_gptq_blockwise_with_sequential_block_inputs_packed_mask",
        "bits": int(args.bits),
        "sparsity": float(args.sparsity),
        "pattern": str(args.pattern),
        "groupsize": int(args.groupsize),
        "percdamp": float(args.percdamp),
        "blocksize": int(args.blocksize),
        "mask_blocksize": int(args.mask_blocksize),
        "packing": str(args.packing),
        "mask_packing": str(args.mask_packing),
        "act_order": bool(args.act_order),
        "symmetric": bool(args.symmetric),
        "quant_aware_mask": not bool(args.no_quant_aware_mask),
        "hessian_form": "2 * X^T X + adaptive_damp * I",
        "calibration_source": args.calib,
        "keep_dequantized_state_dict": bool(args.keep_dequantized_state_dict),
        "store_debug_dequant": bool(args.store_debug_dequant),
        "model_field_contents": (
            "full_dense_dequantized_masked_state_dict"
            if args.keep_dequantized_state_dict
            else "non_compressed_parameters_only"
        ),
        "skip_lm_head": bool(args.skip_lm_head),
        "skip_tied_lm_head": bool(args.skip_tied_lm_head),
        "skip_attn_out": bool(args.skip_attn_out),
        "skip_mlp_out": bool(args.skip_mlp_out),
        "total_pruned_weights": int(total_pruned),
        "total_compressed_weights": int(total_weights),
        "actual_total_sparsity": float(actual_total_sparsity),
        "note": (
            "Compressed linear layers store qweight + scales + zero_points + packed/unpacked mask. "
            "Runtime reconstruction is W = mask * ((q - zero_point) * scale). "
            "Use --packing packed4 and --mask_packing packedbits for compact deployable storage."
        ),
    }

    save_nanogpt_checkpoint(
        original_ckpt=ckpt,
        model=model,
        out_path=args.out,
        joint_meta=joint_meta,
        joint_layers=joint_layers,
        keep_dequantized_state_dict=bool(args.keep_dequantized_state_dict),
    )

    print("\nDone.")
    print("Checkpoint now contains:")
    print("  - ckpt['joint_sparsegpt_gptq_meta']")
    print("  - ckpt['joint_sparsegpt_gptq_layers'][layer_name]['qweight'/'scales'/'zero_points'/'mask']")
    print(f"  - qweight packing: {args.packing}")
    print(f"  - mask packing: {args.mask_packing}")
    print(f"  - total joint-compressed layers: {len(joint_layers)}")
    print(f"  - total pruned weights: {total_pruned:,} / {total_weights:,}")
    print(f"  - actual sparsity over selected layers: {100.0 * actual_total_sparsity:.2f}%")

    if args.keep_dequantized_state_dict:
        print("  - ckpt['model'] with FULL dense dequantized masked weights")
    else:
        print("  - ckpt['model'] with ONLY NON-COMPRESSED parameters")
        print("    compressed linear weights are omitted from ckpt['model']")


if __name__ == "__main__":
    main()