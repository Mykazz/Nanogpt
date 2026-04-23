#!/usr/bin/env python3
"""
GPTQ-style post-training quantization for nanoGPT checkpoints.

This version is adapted for small GPT / nanoGPT use-cases such as Shakespeare,
while remaining close to the GPTQ paper where it matters most.

Key changes relative to the earlier script:
- true SEQUENTIAL BLOCKWISE quantization flow:
    * quantize block 0
    * run quantized block 0 on calibration activations
    * use resulting activations as inputs to block 1
    * repeat
- block input caching, instead of full-model Hessian collection per layer
- optional act-order
- optional symmetric quantization
- optional skipping of lm_head / tied weights
- grouped stats computed cleanly at group boundary from CURRENT updated weights
- keeps real quantized tensors + scales + zero_points
- supports uint8 storage for 2..8 bits
- optional packed 4-bit storage
- includes QuantLinear runtime wrapper for deployable loading

IMPORTANT CHECKPOINT CHANGE:
- If --keep_dequantized_state_dict is ON:
    ckpt["model"] stores the full dequantized model state_dict
- If --keep_dequantized_state_dict is OFF:
    ckpt["model"] stores ONLY NON-QUANTIZED parameters
    (embeddings, layer norms, biases, etc.)
    and quantized linear weights are stored in ckpt["gptq_layers"]
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig


# ============================================================
# Checkpoint utilities
# ============================================================

def strip_orig_mod_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
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

    model_args = ckpt["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = strip_orig_mod_prefix(ckpt["model"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    return model, ckpt


def get_quantized_weight_keys(gptq_layers: Dict[str, Any]) -> Set[str]:
    return {f"{layer_name}.weight" for layer_name in gptq_layers.keys()}


def build_partial_nonquantized_state_dict(
    model: GPT,
    gptq_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    full_sd = model.state_dict()
    quant_weight_keys = get_quantized_weight_keys(gptq_layers)

    partial_sd = {}
    for k, v in full_sd.items():
        if k in quant_weight_keys:
            continue
        partial_sd[k] = v.detach().cpu()

    return partial_sd


def save_nanogpt_checkpoint(
    original_ckpt: Dict[str, Any],
    model: GPT,
    out_path: str,
    quant_meta: Dict[str, Any],
    gptq_layers: Dict[str, Any],
    keep_dequantized_state_dict: bool = True,
) -> None:
    new_ckpt = copy.deepcopy(original_ckpt)

    if keep_dequantized_state_dict:
        new_ckpt["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        new_ckpt["model"] = build_partial_nonquantized_state_dict(
            model=model,
            gptq_layers=gptq_layers,
        )

    new_ckpt["gptq_meta"] = quant_meta
    new_ckpt["gptq_layers"] = gptq_layers

    torch.save(new_ckpt, out_path)
    print(f"Saved quantized checkpoint to: {out_path}")


# ============================================================
# Calibration loading
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


def iter_calibration_batches(tokens: torch.Tensor, batch_size: int, device: torch.device):
    n = tokens.size(0)
    for i in range(0, n, batch_size):
        yield tokens[i:i + batch_size].to(device)


# ============================================================
# Quantization helpers
# ============================================================

@dataclass
class QuantParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    symmetric: bool


@dataclass
class LayerQuantResult:
    qweight_uint8: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    dequant_weight: torch.Tensor
    bits: int
    groupsize: int
    packing: str
    original_shape: Tuple[int, int]
    symmetric: bool


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


def make_quant_params_for_slice(
    W_slice: torch.Tensor,
    bits: int,
    symmetric: bool,
) -> QuantParams:
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")

    qmin = 0
    qmax = (1 << bits) - 1

    if symmetric:
        # Use unsigned storage but symmetric dequantization around midpoint.
        max_abs = W_slice.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        mid = (qmin + qmax) / 2.0
        scale = max_abs / max(mid, 1.0)
        zero_point = torch.full_like(scale, fill_value=mid)
        return QuantParams(
            scale=scale,
            zero_point=zero_point,
            qmin=qmin,
            qmax=qmax,
            symmetric=True,
        )

    w_min = W_slice.amin(dim=1, keepdim=True)
    w_max = W_slice.amax(dim=1, keepdim=True)

    same = (w_max - w_min).abs() < 1e-8
    w_max = torch.where(same, w_min + 1e-8, w_max)

    scale = (w_max - w_min) / float(qmax - qmin)
    scale = scale.clamp(min=1e-8)

    zero_point = torch.round(qmin - w_min / scale).clamp(qmin, qmax)

    return QuantParams(
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        symmetric=False,
    )


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


def build_current_group_slice(
    W_work: torch.Tensor,
    Q_done: torch.Tensor,
    g0: int,
    g1: int,
) -> torch.Tensor:
    """
    Return current best view of a group.
    Q_done contains dequantized columns already finalized.
    W_work contains current updated values for not-yet-finalized columns.
    """
    out = W_work[:, g0:g1].clone()
    done_mask = torch.zeros((g1 - g0,), dtype=torch.bool, device=W_work.device)

    # If a column has been finalized, Q_done will differ from W_work conceptually.
    # We detect finalized columns via NaN-free marker logic by caller, or simply overwrite
    # after quantization in-place in W_work if desired. In this implementation Q_done is
    # zero for unfinished columns, but that may collide with legitimate zero values.
    # So caller should only use this function at group entry before any columns in that group
    # are quantized, which is what we do.
    _ = done_mask  # kept for readability / extension
    return out


# ============================================================
# Optional packing helpers
# ============================================================

def pack_4bit_rows(qweight_uint8: torch.Tensor) -> torch.Tensor:
    if qweight_uint8.dtype != torch.uint8:
        raise ValueError("qweight_uint8 must be torch.uint8")
    if qweight_uint8.numel() == 0:
        return qweight_uint8

    rows, cols = qweight_uint8.shape
    if torch.any(qweight_uint8 > 15):
        raise ValueError("4-bit packing requires values in [0, 15].")

    even = qweight_uint8[:, 0::2]
    odd = qweight_uint8[:, 1::2]

    packed_cols = (cols + 1) // 2
    packed = torch.zeros((rows, packed_cols), dtype=torch.uint8, device=qweight_uint8.device)

    packed[:, :even.size(1)] |= even
    if odd.numel() > 0:
        packed[:, :odd.size(1)] |= (odd << 4)

    return packed


def unpack_4bit_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be torch.uint8")

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
            raise ValueError("packed4 packing is only valid when bits == 4")
        return pack_4bit_rows(qweight_uint8).cpu()
    raise ValueError(f"Unsupported packing mode: {packing}")


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
            raise ValueError(
                f"Stored uint8 qweight has shape {tuple(qweight_stored.shape)}, expected {(rows, cols)}"
            )
        return qweight_stored

    if packing == "packed4":
        if bits != 4:
            raise ValueError("packed4 storage is only valid when bits == 4")
        return unpack_4bit_rows(qweight_stored, cols)

    raise ValueError(f"Unsupported packing mode: {packing}")


# ============================================================
# Robust Hessian stabilization
# ============================================================

@torch.no_grad()
def stable_cholesky_inverse_info(
    H: torch.Tensor,
    percdamp: float,
    max_tries: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
        H_damped: damped Hessian
        Hinv_chol_upper: U where H^{-1} = U^T U
        used_damp
    """
    if H.ndim != 2 or H.size(0) != H.size(1):
        raise ValueError(f"H must be square, got shape {tuple(H.shape)}")

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
    ar = torch.arange(n, device=H.device)
    last_exc = None

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
# GPTQ quantization for one linear layer
# ============================================================

@torch.no_grad()
def gptq_quantize_linear(
    layer: nn.Linear,
    H: torch.Tensor,
    bits: int = 4,
    percdamp: float = 0.01,
    blocksize: int = 128,
    groupsize: int = -1,
    packing: str = "uint8",
    act_order: bool = False,
    symmetric: bool = False,
) -> LayerQuantResult:
    W_orig = layer.weight.data.float().clone()
    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch. Expected {(cols, cols)}, got {tuple(H.shape)}")

    H_damped, Hinv_chol_upper, used_damp = stable_cholesky_inverse_info(H, percdamp=percdamp)
    print(f"      used damping: {used_damp:.6e}")

    # Reconstruct H^{-1} explicitly. This is okay for nanoGPT scale.
    Hinv = Hinv_chol_upper.T @ Hinv_chol_upper
    Hinv = Hinv.to(W_orig.dtype)

    # Optional act-order: sort columns by descending Hessian diagonal.
    if act_order:
        perm = torch.argsort(torch.diag(H_damped), descending=True)
        invperm = torch.argsort(perm)
        W = W_orig[:, perm].contiguous()
        Hinv = Hinv[perm][:, perm].contiguous()
    else:
        perm = None
        invperm = None
        W = W_orig.clone()

    Q = torch.zeros_like(W)
    qweight_uint8 = torch.zeros((rows, cols), dtype=torch.uint8, device=W.device)

    ngroups = get_num_groups(cols, groupsize)
    scales = torch.zeros((rows, ngroups), dtype=torch.float32, device=W.device)
    zero_points = torch.zeros((rows, ngroups), dtype=torch.float32, device=W.device)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Q1_int = torch.zeros((rows, count), dtype=torch.uint8, device=W.device)
        Err1 = torch.zeros_like(W1)

        Hinv1 = Hinv[i1:i2, i1:i2].contiguous()

        for i in range(count):
            global_col = i1 + i
            d = Hinv1[i, i]

            if d.abs().item() < 1e-12:
                raise RuntimeError(
                    f"Encountered near-zero Hinv diagonal at permuted column {global_col}: {d.item()}"
                )

            group_idx = get_group_index(global_col, cols, groupsize)

            # ------------------------------------------------------------
            # CRUCIAL PART:
            # Compute group quant params ONCE when ENTERING the group,
            # using CURRENT updated weights for the whole group.
            # ------------------------------------------------------------
            if is_group_start(global_col, groupsize):
                g0, g1 = get_group_bounds(global_col, cols, groupsize)

                # Current updated group slice.
                # At group entry, no columns in that group have yet been finalized,
                # so W[:, g0:g1] already represents the current updated weights.
                W_group_current = W[:, g0:g1]

                qp = make_quant_params_for_slice(
                    W_slice=W_group_current,
                    bits=bits,
                    symmetric=symmetric,
                )
                scales[:, group_idx] = qp.scale.squeeze(1).to(torch.float32)
                zero_points[:, group_idx] = qp.zero_point.squeeze(1).to(torch.float32)

            scale = scales[:, group_idx].to(W.dtype)
            zero = zero_points[:, group_idx].to(W.dtype)

            w = W1[:, i]
            q_int, q = quantize_with_given_params(
                w_col=w,
                scale=scale,
                zero=zero,
                bits=bits,
            )

            Q1[:, i] = q
            Q1_int[:, i] = q_int
            Q[:, global_col] = q
            qweight_uint8[:, global_col] = q_int

            err = (w - q) / d
            Err1[:, i] = err

            if i + 1 < count:
                W1[:, i + 1:count] -= err.unsqueeze(1) @ Hinv1[i, i + 1:count].unsqueeze(0)

        # Write finished block back
        W[:, i1:i2] = Q1

        # ------------------------------------------------------------
        # CRUCIAL PART:
        # Lazy batch-update of remaining columns, paper-style.
        # ------------------------------------------------------------
        if i2 < cols:
            W[:, i2:cols] -= Err1 @ Hinv[i1:i2, i2:cols]

    if act_order:
        Q = Q[:, invperm].contiguous()
        qweight_uint8 = qweight_uint8[:, invperm].contiguous()

        # Rebuild group stats in ORIGINAL column order for runtime simplicity.
        # This sacrifices exact per-permuted-group metadata alignment but makes loading cleaner.
        # For small nanoGPT models this is reasonable. If you want exact act-order deployment,
        # store permutation explicitly and use it in runtime dequantization.
        # Here we disable grouped-runtime mismatch by recomputing stats on final Q.
        ngroups_orig = get_num_groups(cols, groupsize)
        scales_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=W.device)
        zero_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=W.device)
        q_re = torch.zeros_like(qweight_uint8)

        for g in range(ngroups_orig):
            g0 = 0 if groupsize == -1 else g * groupsize
            g1 = cols if groupsize == -1 else min((g + 1) * groupsize, cols)

            qp = make_quant_params_for_slice(
                W_slice=Q[:, g0:g1],
                bits=bits,
                symmetric=symmetric,
            )
            scales_re[:, g] = qp.scale.squeeze(1).to(torch.float32)
            zero_re[:, g] = qp.zero_point.squeeze(1).to(torch.float32)

            for c in range(g0, g1):
                q_int, q_deq = quantize_with_given_params(
                    w_col=Q[:, c],
                    scale=scales_re[:, g].to(Q.dtype),
                    zero=zero_re[:, g].to(Q.dtype),
                    bits=bits,
                )
                q_re[:, c] = q_int
                Q[:, c] = q_deq

        scales = scales_re
        zero_points = zero_re
        qweight_uint8 = q_re

    layer.weight.data.copy_(Q.to(layer.weight.data.dtype))

    return LayerQuantResult(
        qweight_uint8=qweight_uint8.detach(),
        scales=scales.detach().cpu(),
        zero_points=zero_points.detach().cpu(),
        dequant_weight=Q.detach().cpu(),
        bits=bits,
        groupsize=groupsize,
        packing=packing,
        original_shape=(rows, cols),
        symmetric=symmetric,
    )


# ============================================================
# QuantLinear runtime wrapper
# ============================================================

class QuantLinear(nn.Module):
    """
    Pure-PyTorch quantized linear runtime wrapper.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        groupsize: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        packing: str = "uint8",
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize
        self.packing = packing
        self.cache_dequantized = cache_dequantized

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous().to(torch.float32))
        self.register_buffer("zero_points", zero_points.contiguous().to(torch.float32))

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

        scale_expanded = self.scales[:, self.col_group_idx]
        zero_expanded = self.zero_points[:, self.col_group_idx]

        w = (q - zero_expanded) * scale_expanded
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
    ) -> "QuantLinear":
        shape = tuple(layer_state["shape"])
        out_features, in_features = shape

        return QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bits=int(layer_state["bits"]),
            groupsize=int(layer_state["groupsize"]),
            qweight=layer_state["qweight"].to(device),
            scales=layer_state["scales"].to(device),
            zero_points=layer_state["zero_points"].to(device),
            packing=str(layer_state["packing"]),
            bias=bias,
            cache_dequantized=cache_dequantized,
        )


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
def convert_model_to_quant_linear(
    model: GPT,
    ckpt: Dict[str, Any],
    device: torch.device,
    cache_dequantized: bool = False,
) -> GPT:
    if "gptq_layers" not in ckpt:
        raise ValueError("Checkpoint has no 'gptq_layers' entry.")

    gptq_layers = ckpt["gptq_layers"]

    for layer_name, layer_state in gptq_layers.items():
        orig_layer = get_module_by_name(model, layer_name)
        if not isinstance(orig_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer_name}, got {type(orig_layer)}")

        bias = orig_layer.bias.detach().clone() if orig_layer.bias is not None else None

        qlayer = QuantLinear.from_layer_state(
            layer_state=layer_state,
            bias=bias,
            device=device,
            cache_dequantized=cache_dequantized,
        )
        set_module_by_name(model, layer_name, qlayer)

    model.eval()
    model.to(device)
    return model


# ============================================================
# Helpers for nanoGPT forward decomposition
# ============================================================

@torch.no_grad()
def infer_device_of_model(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def get_block_module_names(model: GPT, block_idx: int) -> List[Tuple[str, nn.Linear]]:
    block = model.transformer.h[block_idx]
    names = []
    for subname, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            full_name = f"transformer.h.{block_idx}.{subname}"
            names.append((full_name, mod))
    return names


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
    """
    Compute hidden states entering transformer block 0.
    Shape: [N, T, C]
    """
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
def run_tail_from_hidden(
    model: GPT,
    hidden_after_last_block: torch.Tensor,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Runs final ln_f + lm_head from hidden states after final transformer block.
    """
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        x = hidden_after_last_block
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
    return logits


# ============================================================
# Hessian collection from cached block inputs
# ============================================================

class HessianCollector:
    """
    Collects H = 2 * sum X^T X for one nn.Linear layer.
    """

    def __init__(self, layer: nn.Linear, device: torch.device, dtype: torch.dtype = torch.float64):
        self.layer = layer
        self.device = device
        self.dtype = dtype
        self.in_features = layer.in_features
        self.H = torch.zeros((self.in_features, self.in_features), device=device, dtype=dtype)
        self.nsamples = 0
        self.handle = None

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        x = x.detach().reshape(-1, x.size(-1)).to(self.device, dtype=self.dtype)
        self.H += 2.0 * x.t().matmul(x)
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
    collector = HessianCollector(layer=layer, device=device, dtype=torch.float64)
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
# Layer selection
# ============================================================

def should_quantize_layer_name(
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
# Sequential blockwise quantization driver
# ============================================================

@torch.no_grad()
def quantize_transformer_blocks_sequentially(
    model: GPT,
    calib_tokens: torch.Tensor,
    bits: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    groupsize: int,
    include: str,
    exclude: str,
    packing: str,
    act_order: bool,
    symmetric: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
    gptq_layers_out: Dict[str, Any],
) -> torch.Tensor:
    # ------------------------------------------------------------
    # CRUCIAL PART:
    # Cache activations entering block 0, then propagate through
    # each already-quantized block to get inputs for the next block.
    # ------------------------------------------------------------
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
            if should_quantize_layer_name(
                name=name,
                include=include,
                exclude=exclude,
                skip_lm_head=True,
                skip_attn_out=skip_attn_out,
                skip_mlp_out=skip_mlp_out,
            )
        ]

        print(f"\n=== Quantizing transformer block {bi}/{nblocks - 1} ===")
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
            print(f"      quantizing {name} ...")

            result = gptq_quantize_linear(
                layer=layer,
                H=H.to(layer.weight.device),
                bits=bits,
                percdamp=percdamp,
                blocksize=blocksize,
                groupsize=groupsize,
                packing=packing,
                act_order=act_order,
                symmetric=symmetric,
            )

            qweight_stored = maybe_pack_qweight(
                qweight_uint8=result.qweight_uint8.cpu(),
                bits=result.bits,
                packing=result.packing,
            )

            gptq_layers_out[name] = {
                "bits": result.bits,
                "groupsize": result.groupsize,
                "packing": result.packing,
                "shape": list(result.original_shape),
                "qweight": qweight_stored,
                "scales": result.scales.cpu(),
                "zero_points": result.zero_points.cpu(),
                "symmetric": bool(result.symmetric),
            }

            print(
                f"      saved quant tensors: qweight={tuple(gptq_layers_out[name]['qweight'].shape)}, "
                f"scales={tuple(gptq_layers_out[name]['scales'].shape)}, "
                f"zero_points={tuple(gptq_layers_out[name]['zero_points'].shape)}"
            )
            print("      done.")

        # ------------------------------------------------------------
        # CRUCIAL PART:
        # Propagate hidden states through the NOW-QUANTIZED block.
        # This is the most important paper-aligned fix.
        # ------------------------------------------------------------
        hidden = run_block_on_hidden(
            block=block,
            hidden=hidden,
            amp_dtype=amp_dtype,
            device=device,
        )

    return hidden


@torch.no_grad()
def quantize_nonblock_linears_after_blocks(
    model: GPT,
    hidden_after_blocks: torch.Tensor,
    bits: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    groupsize: int,
    include: str,
    exclude: str,
    packing: str,
    act_order: bool,
    symmetric: bool,
    skip_lm_head: bool,
    skip_tied_lm_head: bool,
    gptq_layers_out: Dict[str, Any],
) -> None:
    candidates: List[Tuple[str, nn.Linear]] = []

    block_mod_ids = {id(m) for bi in range(len(model.transformer.h)) for _, m in get_block_module_names(model, bi)}

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and id(mod) not in block_mod_ids:
            candidates.append((name, mod))

    if not candidates:
        return

    tied = model_has_tied_lm_head(model)

    selected = []
    for name, mod in candidates:
        if not should_quantize_layer_name(
            name=name,
            include=include,
            exclude=exclude,
            skip_lm_head=skip_lm_head,
            skip_attn_out=False,
            skip_mlp_out=False,
        ):
            continue

        if skip_tied_lm_head and tied and name == "lm_head":
            print("[info] skipping quantization of tied lm_head")
            continue

        selected.append((name, mod))

    if not selected:
        return

    print("\n=== Quantizing non-block linear layers ===")
    for name, mod in selected:
        print(f"  - {name}: {tuple(mod.weight.shape)}")

    for idx, (name, layer) in enumerate(selected, start=1):
        print(f"\n    [{idx}/{len(selected)}] Collecting Hessian for: {name}")

        collector = HessianCollector(layer=layer, device=device, dtype=torch.float64)
        collector.register()

        autocast_enabled = device.type == "cuda"
        n = hidden_after_blocks.size(0)
        for i in range(0, n, batch_size):
            batch_hidden = hidden_after_blocks[i:i + batch_size].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                _ = model.transformer.ln_f(batch_hidden)
                # lm_head will be invoked only if the layer is there and the module forward is used
                if name == "lm_head":
                    _ = model.lm_head(model.transformer.ln_f(batch_hidden))
                else:
                    # If another non-block linear exists in custom model variants,
                    # you may need a more exact path here.
                    _ = model.lm_head(model.transformer.ln_f(batch_hidden))

        collector.remove()
        H, nsamples = collector.H, collector.nsamples

        print(f"      samples: {nsamples}")
        print(f"      H shape: {tuple(H.shape)}")
        print(f"      quantizing {name} ...")

        result = gptq_quantize_linear(
            layer=layer,
            H=H.to(layer.weight.device),
            bits=bits,
            percdamp=percdamp,
            blocksize=blocksize,
            groupsize=groupsize,
            packing=packing,
            act_order=act_order,
            symmetric=symmetric,
        )

        qweight_stored = maybe_pack_qweight(
            qweight_uint8=result.qweight_uint8.cpu(),
            bits=result.bits,
            packing=result.packing,
        )

        gptq_layers_out[name] = {
            "bits": result.bits,
            "groupsize": result.groupsize,
            "packing": result.packing,
            "shape": list(result.original_shape),
            "qweight": qweight_stored,
            "scales": result.scales.cpu(),
            "zero_points": result.zero_points.cpu(),
            "symmetric": bool(result.symmetric),
        }

        print(
            f"      saved quant tensors: qweight={tuple(gptq_layers_out[name]['qweight'].shape)}, "
            f"scales={tuple(gptq_layers_out[name]['scales'].shape)}, "
            f"zero_points={tuple(gptq_layers_out[name]['zero_points'].shape)}"
        )
        print("      done.")


@torch.no_grad()
def quantize_model_blockwise(
    model: GPT,
    calib_tokens: torch.Tensor,
    bits: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    groupsize: int,
    include: str,
    exclude: str,
    packing: str,
    act_order: bool,
    symmetric: bool,
    skip_lm_head: bool,
    skip_tied_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> Dict[str, Any]:
    gptq_layers_out: Dict[str, Any] = {}

    hidden_after_blocks = quantize_transformer_blocks_sequentially(
        model=model,
        calib_tokens=calib_tokens,
        bits=bits,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=percdamp,
        blocksize=blocksize,
        groupsize=groupsize,
        include=include,
        exclude=exclude,
        packing=packing,
        act_order=act_order,
        symmetric=symmetric,
        skip_attn_out=skip_attn_out,
        skip_mlp_out=skip_mlp_out,
        gptq_layers_out=gptq_layers_out,
    )

    quantize_nonblock_linears_after_blocks(
        model=model,
        hidden_after_blocks=hidden_after_blocks,
        bits=bits,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=percdamp,
        blocksize=blocksize,
        groupsize=groupsize,
        include=include,
        exclude=exclude,
        packing=packing,
        act_order=act_order,
        symmetric=symmetric,
        skip_lm_head=skip_lm_head,
        skip_tied_lm_head=skip_tied_lm_head,
        gptq_layers_out=gptq_layers_out,
    )

    return gptq_layers_out


# ============================================================
# Optional: load a quantized checkpoint into QuantLinear runtime
# ============================================================

@torch.no_grad()
def load_quantized_nanogpt_checkpoint(
    ckpt_path: str,
    device: torch.device,
    prefer_quantlinear: bool = True,
    cache_dequantized: bool = False,
) -> Tuple[GPT, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_args" not in ckpt:
        raise ValueError("Checkpoint missing model_args")

    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)

    if "model" in ckpt and ckpt["model"]:
        state_dict = strip_orig_mod_prefix(ckpt["model"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] missing keys when loading model state_dict: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys when loading model state_dict: {unexpected}")

    model.eval()
    model.to(device)

    if prefer_quantlinear and "gptq_layers" in ckpt:
        print("Replacing quantized linear layers with QuantLinear runtime modules...")
        model = convert_model_to_quant_linear(
            model=model,
            ckpt=ckpt,
            device=device,
            cache_dequantized=cache_dequantized,
        )

    return model, ckpt


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--groupsize",
        type=int,
        default=64,
        help="-1 for full-row quantization, otherwise grouped quantization size (e.g. 32, 64, 128)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)

    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")

    parser.add_argument(
        "--packing",
        type=str,
        default="uint8",
        choices=["uint8", "packed4"],
        help=(
            "'uint8' stores one quantized value per byte (works for 2..8 bits). "
            "'packed4' packs two 4-bit values into one byte."
        ),
    )

    parser.add_argument(
        "--keep_dequantized_state_dict",
        action="store_true",
        help=(
            "Also keep full dequantized float weights in ckpt['model']. "
            "If omitted, ckpt['model'] will store only NON-QUANTIZED params, "
            "while quantized linear weights live in ckpt['gptq_layers']."
        ),
    )

    # Reasonable small-model / nanoGPT options
    parser.add_argument("--act_order", action="store_true",
                        help="Sort columns by Hessian diagonal importance before GPTQ.")
    parser.add_argument("--symmetric", action="store_true",
                        help="Use symmetric per-row/per-group quantization instead of asymmetric min-max.")
    parser.add_argument("--skip_lm_head", action="store_true",
                        help="Do not quantize lm_head.")
    parser.add_argument("--skip_tied_lm_head", action="store_true",
                        help="If lm_head is tied to token embedding, skip quantizing lm_head.")
    parser.add_argument("--skip_attn_out", action="store_true",
                        help="Skip attention output projections (attn.c_proj).")
    parser.add_argument("--skip_mlp_out", action="store_true",
                        help="Skip MLP output projections (mlp.c_proj).")

    args = parser.parse_args()

    if args.packing == "packed4" and args.bits != 4:
        raise ValueError("--packing packed4 is only valid with --bits 4")

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

    gptq_layers = quantize_model_blockwise(
        model=model,
        calib_tokens=calib_tokens,
        bits=args.bits,
        batch_size=args.batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        groupsize=args.groupsize,
        include=args.include,
        exclude=args.exclude,
        packing=args.packing,
        act_order=bool(args.act_order),
        symmetric=bool(args.symmetric),
        skip_lm_head=bool(args.skip_lm_head),
        skip_tied_lm_head=bool(args.skip_tied_lm_head),
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
    )

    quant_meta = {
        "method": "gptq_style_blockwise_grouped_quant_with_sequential_block_inputs",
        "bits": args.bits,
        "groupsize": args.groupsize,
        "percdamp": args.percdamp,
        "blocksize": args.blocksize,
        "packing": args.packing,
        "act_order": bool(args.act_order),
        "symmetric": bool(args.symmetric),
        "hessian_form": "2 * X^T X + adaptive_damp * I",
        "calibration_source": args.calib,
        "keep_dequantized_state_dict": bool(args.keep_dequantized_state_dict),
        "model_field_contents": (
            "full_dequantized_state_dict"
            if args.keep_dequantized_state_dict
            else "non_quantized_parameters_only"
        ),
        "skip_lm_head": bool(args.skip_lm_head),
        "skip_tied_lm_head": bool(args.skip_tied_lm_head),
        "skip_attn_out": bool(args.skip_attn_out),
        "skip_mlp_out": bool(args.skip_mlp_out),
        "note": (
            "Checkpoint contains quantized tensors in ckpt['gptq_layers'] "
            "(qweight + scales + zero_points). "
            "ckpt['model'] contains either the full dequantized state_dict, "
            "or only non-quantized parameters for compact but reconstructable loading. "
            "This implementation follows GPTQ blockwise quantization with sequential "
            "propagation of quantized block activations, but remains pure PyTorch and "
            "therefore does not include custom fast low-bit CUDA kernels."
        ),
    }

    save_nanogpt_checkpoint(
        original_ckpt=ckpt,
        model=model,
        out_path=args.out,
        quant_meta=quant_meta,
        gptq_layers=gptq_layers,
        keep_dequantized_state_dict=args.keep_dequantized_state_dict,
    )

    print("\nDone.")
    print("Checkpoint now contains:")
    print("  - ckpt['gptq_meta']")
    print("  - ckpt['gptq_layers'][layer_name]['qweight'/'scales'/'zero_points']")
    if args.keep_dequantized_state_dict:
        print("  - ckpt['model'] with FULL dequantized float weights")
    else:
        print("  - ckpt['model'] with ONLY NON-QUANTIZED parameters")
        print("    (quantized linear weights are omitted from ckpt['model'])")


if __name__ == "__main__":
    main()