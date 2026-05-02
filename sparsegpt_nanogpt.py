#!/usr/bin/env python3
"""
SparseGPT tipo vieno žingsnio post-training pruning nanoGPT checkpoint'ams.

Trumpai:
- surenka Hessian aproksimaciją iš kalibracijos aktyvacijų
- kiekvieną Linear sluoksnį praretina stulpelis po stulpelio
- naudoja H^{-1} kompensacijai po svorių pašalinimo
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

import torch
import torch.nn as nn

from model import GPT, GPTConfig


# ============================================================
# Checkpoint pagalbinės funkcijos
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


def get_sparse_weight_keys(sparsegpt_layers: Dict[str, Any]) -> Set[str]:
    return {f"{layer_name}.weight" for layer_name in sparsegpt_layers.keys()}


def build_partial_nonpruned_state_dict(
    model: GPT,
    sparsegpt_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Sukuria state_dict be tų Linear svorių, kurie saugomi sparsegpt_layers.
    """
    full_sd = model.state_dict()
    sparse_weight_keys = get_sparse_weight_keys(sparsegpt_layers)

    partial_sd = {}
    for k, v in full_sd.items():
        if k in sparse_weight_keys:
            continue
        partial_sd[k] = v.detach().cpu()

    return partial_sd


def save_nanogpt_checkpoint(
    original_ckpt: Dict[str, Any],
    model: GPT,
    out_path: str,
    sparse_meta: Dict[str, Any],
    sparsegpt_layers: Dict[str, Any],
    keep_pruned_state_dict: bool = True,
) -> None:
    new_ckpt = copy.deepcopy(original_ckpt)

    if keep_pruned_state_dict:
        new_ckpt["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        new_ckpt["model"] = build_partial_nonpruned_state_dict(
            model=model,
            sparsegpt_layers=sparsegpt_layers,
        )

    new_ckpt["sparsegpt_meta"] = sparse_meta
    new_ckpt["sparsegpt_layers"] = sparsegpt_layers

    torch.save(new_ckpt, out_path)
    print(f"Saved SparseGPT-pruned checkpoint to: {out_path}")


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
# SparseGPT rezultatų struktūra
# ============================================================

@dataclass
class SparseLayerResult:
    pruned_weight: torch.Tensor
    mask: torch.Tensor
    original_shape: Tuple[int, int]
    sparsity: float
    target_sparsity: float
    pattern: str
    pruned_count: int
    total_count: int


# ============================================================
# Stabilus Hessian inversijos skaičiavimas
# ============================================================

@torch.no_grad()
def stable_cholesky_inverse_info(
    H: torch.Tensor,
    percdamp: float,
    max_tries: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Stabiliai apskaičiuoja damped Hessian ir H^{-1} Cholesky faktorių.
    """
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
# Maskės parinkimas
# ============================================================

def parse_nm_pattern(pattern: str) -> Optional[Tuple[int, int]]:
    """
    Perskaito N:M šabloną, pvz. 2:4 arba 4:8.
    """
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
def select_unstructured_mask_block(
    W_block: torch.Tensor,
    Hinv_diag_block: torch.Tensor,
    sparsity: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Parenka unstructured maskę pagal SparseGPT saliency.
    True reiškia palikti svorį, False reiškia pašalinti.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1).")

    rows, block_cols = W_block.shape
    total = rows * block_cols
    n_prune_total = int(round(sparsity * total))

    if n_prune_total <= 0:
        return torch.ones_like(W_block, dtype=torch.bool)
    if n_prune_total >= total:
        return torch.zeros_like(W_block, dtype=torch.bool)

    diag = Hinv_diag_block.to(W_block.device, dtype=W_block.dtype).abs().clamp(min=eps)

    # Svarbumo matas: didesnis score => svorį verta palikti.
    score = (W_block.float() ** 2) / diag.float().view(1, -1)

    flat_score = score.reshape(-1)
    n_keep_total = total - n_prune_total

    topk_idx = torch.topk(flat_score, k=n_keep_total, largest=True, sorted=False).indices

    flat_mask = torch.zeros(total, dtype=torch.bool, device=W_block.device)
    flat_mask[topk_idx] = True

    return flat_mask.view(rows, block_cols)


@torch.no_grad()
def select_nm_mask_block(
    W_block: torch.Tensor,
    Hinv_diag_block: torch.Tensor,
    n_zero: int,
    m: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Parenka N:M maskę: kiekvienoje M grupėje pašalinama N svorių.
    """
    rows, block_cols = W_block.shape
    mask = torch.ones_like(W_block, dtype=torch.bool)

    diag = Hinv_diag_block.to(W_block.device, dtype=W_block.dtype).abs().clamp(min=eps)

    for g0 in range(0, block_cols, m):
        g1 = min(g0 + m, block_cols)
        group_cols = g1 - g0

        if group_cols <= 0:
            continue

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

        # Kiekvienoje eilutėje šalinami mažiausio score svoriai.
        score = (W_block[:, g0:g1].float() ** 2) / (
            diag[g0:g1].float().view(1, -1)
        )

        prune_idx = torch.topk(
            score,
            k=prune_count,
            largest=False,
            dim=1,
            sorted=False,
        ).indices

        row_idx = torch.arange(rows, device=W_block.device).view(-1, 1).expand_as(prune_idx)

        local_mask = mask[:, g0:g1]
        local_mask[row_idx, prune_idx] = False
        mask[:, g0:g1] = local_mask

    return mask


# ============================================================
# Vieno Linear sluoksnio SparseGPT pruning
# ============================================================

@torch.no_grad()
def sparsegpt_prune_linear(
    layer: nn.Linear,
    H: torch.Tensor,
    sparsity: float = 0.5,
    percdamp: float = 0.01,
    blocksize: int = 128,
    mask_blocksize: int = 128,
    pattern: str = "unstructured",
) -> SparseLayerResult:
    """
    Pagrindinis SparseGPT algoritmas vienam Linear sluoksniui.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1).")

    W_orig = layer.weight.data.float().clone()
    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch. Expected {(cols, cols)}, got {tuple(H.shape)}")

    nm = parse_nm_pattern(pattern)

    if nm is not None:
        n_zero, m = nm

        if mask_blocksize != m:
            print(
                f"      [info] N:M pattern {pattern} requested; "
                f"using mask_blocksize={m} for strict groups."
            )

        mask_blocksize_eff = m
        effective_sparsity = n_zero / float(m)
    else:
        n_zero, m = None, None
        mask_blocksize_eff = mask_blocksize
        effective_sparsity = sparsity

    if mask_blocksize_eff <= 0:
        raise ValueError("mask_blocksize must be positive.")
    if blocksize <= 0:
        raise ValueError("blocksize must be positive.")

    print(f"      target sparsity: {effective_sparsity:.4f}")
    print(f"      pattern: {pattern}")
    print(f"      lazy update blocksize B: {blocksize}")
    print(f"      mask selection blocksize Bs: {mask_blocksize_eff}")

    H_damped, Hinv_chol_upper, used_damp = stable_cholesky_inverse_info(
        H,
        percdamp=percdamp,
    )
    print(f"      used damping: {used_damp:.6e}")

    # Iš Cholesky faktoriaus atkuriamas pilnas H^{-1}.
    Hinv = Hinv_chol_upper.T @ Hinv_chol_upper
    Hinv = Hinv.to(W_orig.dtype)

    W = W_orig.clone()
    M_global = torch.ones((rows, cols), dtype=torch.bool, device=W.device)

    selected_mask_until = -1

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)

        Hinv1 = Hinv[i1:i2, i1:i2].contiguous()

        for local_i in range(count):
            global_col = i1 + local_i

            # Naujas maskės blokas parenkamas tik tada, kai pasiekiamas jo pradžios stulpelis.
            if global_col >= selected_mask_until:
                mb0 = global_col
                mb1 = min(mb0 + mask_blocksize_eff, cols)

                W_mask_block = W[:, mb0:mb1]
                Hdiag_mask_block = torch.diag(Hinv)[mb0:mb1]

                if nm is None:
                    M_block = select_unstructured_mask_block(
                        W_block=W_mask_block,
                        Hinv_diag_block=Hdiag_mask_block,
                        sparsity=sparsity,
                    )
                else:
                    M_block = select_nm_mask_block(
                        W_block=W_mask_block,
                        Hinv_diag_block=Hdiag_mask_block,
                        n_zero=n_zero,
                        m=m,
                    )

                M_global[:, mb0:mb1] = M_block
                selected_mask_until = mb1

            d = Hinv1[local_i, local_i]

            if d.abs().item() < 1e-12:
                raise RuntimeError(
                    f"Encountered near-zero Hinv diagonal at column {global_col}: {d.item()}"
                )

            w = W1[:, local_i]
            keep_mask_col = M_global[:, global_col]

            # Pašalintų svorių klaida, padalinta iš H^{-1} diagonalės.
            err = w / d
            err = torch.where(
                keep_mask_col,
                torch.zeros_like(err),
                err,
            )

            Err1[:, local_i] = err

            # Dabartinis stulpelis užfiksuojamas: palikti lieka, pašalinti tampa 0.
            W1[:, local_i] = torch.where(
                keep_mask_col,
                w,
                torch.zeros_like(w),
            )

            # Klaida kompensuojama vėlesniuose to paties bloko stulpeliuose.
            if local_i + 1 < count:
                W1[:, local_i + 1:count] -= (
                    err.unsqueeze(1)
                    @ Hinv1[local_i, local_i + 1:count].unsqueeze(0)
                )

        W[:, i1:i2] = W1

        # Klaida kompensuojama visuose likusiuose stulpeliuose už bloko ribų.
        if i2 < cols:
            W[:, i2:cols] -= Err1 @ Hinv[i1:i2, i2:cols]

    W_pruned = W * M_global.to(W.dtype)

    layer.weight.data.copy_(W_pruned.to(layer.weight.data.dtype))

    total_count = rows * cols
    kept_count = int(M_global.sum().item())
    pruned_count = total_count - kept_count
    actual_sparsity = pruned_count / float(total_count)

    return SparseLayerResult(
        pruned_weight=W_pruned.detach().cpu(),
        mask=M_global.detach().cpu(),
        original_shape=(rows, cols),
        sparsity=actual_sparsity,
        target_sparsity=effective_sparsity,
        pattern=pattern,
        pruned_count=pruned_count,
        total_count=total_count,
    )


# ============================================================
# Paprastas sparse Linear wrapper
# ============================================================

class SparseLinear(nn.Module):
    """
    Sparse Linear sluoksnis korektiškam įkėlimui.
    Greičio automatiškai neduoda, nes naudoja PyTorch dense linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pruned_weight: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(pruned_weight.detach().clone())
        self.register_buffer("mask", mask.detach().clone().bool())

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask.to(dtype=self.weight.dtype)
        return nn.functional.linear(x, w.to(dtype=x.dtype), self.bias)

    @staticmethod
    def from_layer_state(
        layer_state: Dict[str, Any],
        bias: Optional[torch.Tensor],
        device: torch.device,
    ) -> "SparseLinear":
        shape = tuple(layer_state["shape"])
        out_features, in_features = shape

        return SparseLinear(
            in_features=in_features,
            out_features=out_features,
            pruned_weight=layer_state["pruned_weight"].to(device),
            mask=layer_state["mask"].to(device),
            bias=bias.to(device) if bias is not None else None,
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
def convert_model_to_sparse_linear(
    model: GPT,
    ckpt: Dict[str, Any],
    device: torch.device,
) -> GPT:
    if "sparsegpt_layers" not in ckpt:
        raise ValueError("Checkpoint has no 'sparsegpt_layers' entry.")

    sparsegpt_layers = ckpt["sparsegpt_layers"]

    for layer_name, layer_state in sparsegpt_layers.items():
        orig_layer = get_module_by_name(model, layer_name)

        if not isinstance(orig_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer_name}, got {type(orig_layer)}")

        bias = orig_layer.bias.detach().clone() if orig_layer.bias is not None else None

        sparse_layer = SparseLinear.from_layer_state(
            layer_state=layer_state,
            bias=bias,
            device=device,
        )

        set_module_by_name(model, layer_name, sparse_layer)

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def load_sparsegpt_nanogpt_checkpoint(
    ckpt_path: str,
    device: torch.device,
    prefer_sparse_wrapper: bool = False,
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

    if prefer_sparse_wrapper and "sparsegpt_layers" in ckpt:
        print("Replacing pruned linear layers with SparseLinear wrapper modules...")
        model = convert_model_to_sparse_linear(
            model=model,
            ckpt=ckpt,
            device=device,
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
    """
    Apskaičiuoja hidden būsenas prieš pirmą transformer bloką.
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

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=autocast_enabled,
    ):
        out = block(hidden)

    return out.detach()


@torch.no_grad()
def run_tail_from_hidden(
    model: GPT,
    hidden_after_last_block: torch.Tensor,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    autocast_enabled = device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=autocast_enabled,
    ):
        x = hidden_after_last_block
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)

    return logits


@torch.no_grad()
def get_block_module_names(model: GPT, block_idx: int) -> List[Tuple[str, nn.Linear]]:
    block = model.transformer.h[block_idx]
    names = []

    for subname, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            full_name = f"transformer.h.{block_idx}.{subname}"
            names.append((full_name, mod))

    return names


# ============================================================
# Hessian rinkimas
# ============================================================

class HessianCollector:
    """
    Surenka H = 2 * X^T X vienam Linear sluoksniui.
    """

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
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=device,
            dtype=dtype,
        )
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
    use_factor_2: bool = True,
) -> Tuple[torch.Tensor, int]:
    collector = HessianCollector(
        layer=layer,
        device=device,
        dtype=torch.float64,
        use_factor_2=use_factor_2,
    )

    collector.register()

    autocast_enabled = device.type == "cuda"
    n = block_inputs.size(0)

    for i in range(0, n, batch_size):
        batch_hidden = block_inputs[i:i + batch_size].to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            _ = block(batch_hidden)

    collector.remove()

    return collector.H, collector.nsamples


# ============================================================
# Sluoksnių atranka
# ============================================================

def should_prune_layer_name(
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
# Transformer blokų pruning iš eilės
# ============================================================

@torch.no_grad()
def sparsegpt_prune_transformer_blocks_sequentially(
    model: GPT,
    calib_tokens: torch.Tensor,
    sparsity: float,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    pattern: str,
    include: str,
    exclude: str,
    skip_attn_out: bool,
    skip_mlp_out: bool,
    sparsegpt_layers_out: Dict[str, Any],
) -> torch.Tensor:
    # Pirmiausia gaunamos aktyvacijos prieš 0 bloką.
    hidden = compute_hidden_before_blocks(
        model=model,
        tokens=calib_tokens.to(device),
        device=device,
    )

    print(f"Initial hidden cache shape before blocks: {tuple(hidden.shape)}")

    nblocks = len(model.transformer.h)
    print(f"\nFound {nblocks} transformer blocks.")

    for bi in range(nblocks):
        block = model.transformer.h[bi]

        block_layers = get_block_module_names(model, bi)

        block_layers = [
            (name, layer)
            for name, layer in block_layers
            if should_prune_layer_name(
                name=name,
                include=include,
                exclude=exclude,
                skip_lm_head=True,
                skip_attn_out=skip_attn_out,
                skip_mlp_out=skip_mlp_out,
            )
        ]

        print(f"\n=== SparseGPT pruning transformer block {bi}/{nblocks - 1} ===")

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
                use_factor_2=True,
            )

            print(f"      samples: {nsamples}")
            print(f"      H shape: {tuple(H.shape)}")
            print(f"      pruning {name} ...")

            result = sparsegpt_prune_linear(
                layer=layer,
                H=H.to(layer.weight.device),
                sparsity=sparsity,
                percdamp=percdamp,
                blocksize=blocksize,
                mask_blocksize=mask_blocksize,
                pattern=pattern,
            )

            sparsegpt_layers_out[name] = {
                "shape": list(result.original_shape),
                "sparsity": float(result.sparsity),
                "target_sparsity": float(result.target_sparsity),
                "pattern": result.pattern,
                "pruned_count": int(result.pruned_count),
                "total_count": int(result.total_count),
                "mask": result.mask.cpu().to(torch.bool),
                "pruned_weight": result.pruned_weight.cpu(),
            }

            print(
                f"      saved sparse tensors: "
                f"pruned_weight={tuple(result.pruned_weight.shape)}, "
                f"mask={tuple(result.mask.shape)}, "
                f"sparsity={100.0 * result.sparsity:.2f}%"
            )
            print("      done.")

        # Po praretinimo aktyvacijos paleidžiamos per jau praretintą bloką.
        hidden = run_block_on_hidden(
            block=block,
            hidden=hidden,
            amp_dtype=amp_dtype,
            device=device,
        )

    return hidden


@torch.no_grad()
def sparsegpt_prune_nonblock_linears_after_blocks(
    model: GPT,
    hidden_after_blocks: torch.Tensor,
    sparsity: float,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    pattern: str,
    include: str,
    exclude: str,
    skip_lm_head: bool,
    skip_tied_lm_head: bool,
    sparsegpt_layers_out: Dict[str, Any],
) -> None:
    candidates: List[Tuple[str, nn.Linear]] = []

    block_mod_ids = {
        id(m)
        for bi in range(len(model.transformer.h))
        for _, m in get_block_module_names(model, bi)
    }

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and id(mod) not in block_mod_ids:
            candidates.append((name, mod))

    if not candidates:
        return

    tied = model_has_tied_lm_head(model)

    selected = []

    for name, mod in candidates:
        if not should_prune_layer_name(
            name=name,
            include=include,
            exclude=exclude,
            skip_lm_head=skip_lm_head,
            skip_attn_out=False,
            skip_mlp_out=False,
        ):
            continue

        if skip_tied_lm_head and tied and name == "lm_head":
            print("[info] skipping pruning of tied lm_head")
            continue

        selected.append((name, mod))

    if not selected:
        return

    print("\n=== SparseGPT pruning non-block linear layers ===")

    for name, mod in selected:
        print(f"  - {name}: {tuple(mod.weight.shape)}")

    for idx, (name, layer) in enumerate(selected, start=1):
        print(f"\n    [{idx}/{len(selected)}] Collecting Hessian for: {name}")

        collector = HessianCollector(
            layer=layer,
            device=device,
            dtype=torch.float64,
            use_factor_2=True,
        )

        collector.register()

        autocast_enabled = device.type == "cuda"
        n = hidden_after_blocks.size(0)

        for i in range(0, n, batch_size):
            batch_hidden = hidden_after_blocks[i:i + batch_size].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=autocast_enabled,
            ):
                x = model.transformer.ln_f(batch_hidden)
                _ = model.lm_head(x)

        collector.remove()

        H, nsamples = collector.H, collector.nsamples

        print(f"      samples: {nsamples}")
        print(f"      H shape: {tuple(H.shape)}")
        print(f"      pruning {name} ...")

        result = sparsegpt_prune_linear(
            layer=layer,
            H=H.to(layer.weight.device),
            sparsity=sparsity,
            percdamp=percdamp,
            blocksize=blocksize,
            mask_blocksize=mask_blocksize,
            pattern=pattern,
        )

        sparsegpt_layers_out[name] = {
            "shape": list(result.original_shape),
            "sparsity": float(result.sparsity),
            "target_sparsity": float(result.target_sparsity),
            "pattern": result.pattern,
            "pruned_count": int(result.pruned_count),
            "total_count": int(result.total_count),
            "mask": result.mask.cpu().to(torch.bool),
            "pruned_weight": result.pruned_weight.cpu(),
        }

        print(
            f"      saved sparse tensors: "
            f"pruned_weight={tuple(result.pruned_weight.shape)}, "
            f"mask={tuple(result.mask.shape)}, "
            f"sparsity={100.0 * result.sparsity:.2f}%"
        )
        print("      done.")


@torch.no_grad()
def sparsegpt_prune_model_blockwise(
    model: GPT,
    calib_tokens: torch.Tensor,
    sparsity: float,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    pattern: str,
    include: str,
    exclude: str,
    skip_lm_head: bool,
    skip_tied_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> Dict[str, Any]:
    sparsegpt_layers_out: Dict[str, Any] = {}

    hidden_after_blocks = sparsegpt_prune_transformer_blocks_sequentially(
        model=model,
        calib_tokens=calib_tokens,
        sparsity=sparsity,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=percdamp,
        blocksize=blocksize,
        mask_blocksize=mask_blocksize,
        pattern=pattern,
        include=include,
        exclude=exclude,
        skip_attn_out=skip_attn_out,
        skip_mlp_out=skip_mlp_out,
        sparsegpt_layers_out=sparsegpt_layers_out,
    )

    sparsegpt_prune_nonblock_linears_after_blocks(
        model=model,
        hidden_after_blocks=hidden_after_blocks,
        sparsity=sparsity,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=percdamp,
        blocksize=blocksize,
        mask_blocksize=mask_blocksize,
        pattern=pattern,
        include=include,
        exclude=exclude,
        skip_lm_head=skip_lm_head,
        skip_tied_lm_head=skip_tied_lm_head,
        sparsegpt_layers_out=sparsegpt_layers_out,
    )

    return sparsegpt_layers_out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Target unstructured sparsity fraction, e.g. 0.5 for 50%. Ignored for strict N:M pattern except metadata.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="unstructured",
        help="'unstructured' or N:M pattern like '2:4' or '4:8'. N means zeros per M consecutive weights.",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )

    parser.add_argument("--percdamp", type=float, default=0.01)

    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Lazy update blocksize B from SparseGPT/GPTQ-style algorithm.",
    )

    parser.add_argument(
        "--mask_blocksize",
        type=int,
        default=128,
        help="Adaptive mask selection blocksize Bs. For N:M, it is automatically replaced by M.",
    )

    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")

    parser.add_argument(
        "--keep_pruned_state_dict",
        action="store_true",
        help=(
            "Also keep full dense pruned weights in ckpt['model']. "
            "If omitted, ckpt['model'] stores only non-pruned parameters, "
            "while pruned linear weights and masks live in ckpt['sparsegpt_layers']."
        ),
    )

    parser.add_argument(
        "--skip_lm_head",
        action="store_true",
        help="Do not prune lm_head.",
    )

    parser.add_argument(
        "--skip_tied_lm_head",
        action="store_true",
        help="If lm_head is tied to token embedding, skip pruning lm_head.",
    )

    parser.add_argument(
        "--skip_attn_out",
        action="store_true",
        help="Skip attention output projections, usually attn.c_proj.",
    )

    parser.add_argument(
        "--skip_mlp_out",
        action="store_true",
        help="Skip MLP output projections, usually mlp.c_proj.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0, 1).")

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

    sparsegpt_layers = sparsegpt_prune_model_blockwise(
        model=model,
        calib_tokens=calib_tokens,
        sparsity=args.sparsity,
        batch_size=args.batch_size,
        device=device,
        amp_dtype=amp_dtype,
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        mask_blocksize=args.mask_blocksize,
        pattern=args.pattern,
        include=args.include,
        exclude=args.exclude,
        skip_lm_head=bool(args.skip_lm_head),
        skip_tied_lm_head=bool(args.skip_tied_lm_head),
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
    )

    total_pruned = sum(int(v["pruned_count"]) for v in sparsegpt_layers.values())
    total_weights = sum(int(v["total_count"]) for v in sparsegpt_layers.values())

    total_sparsity = total_pruned / float(total_weights) if total_weights > 0 else 0.0

    sparse_meta = {
        "method": "sparsegpt_style_blockwise_pruning_with_sequential_block_inputs",
        "sparsity": args.sparsity,
        "pattern": args.pattern,
        "percdamp": args.percdamp,
        "blocksize": args.blocksize,
        "mask_blocksize": args.mask_blocksize,
        "hessian_form": "2 * X^T X + adaptive_damp * I",
        "calibration_source": args.calib,
        "keep_pruned_state_dict": bool(args.keep_pruned_state_dict),
        "model_field_contents": (
            "full_dense_pruned_state_dict"
            if args.keep_pruned_state_dict
            else "non_pruned_parameters_only"
        ),
        "skip_lm_head": bool(args.skip_lm_head),
        "skip_tied_lm_head": bool(args.skip_tied_lm_head),
        "skip_attn_out": bool(args.skip_attn_out),
        "skip_mlp_out": bool(args.skip_mlp_out),
        "total_pruned_weights": int(total_pruned),
        "total_prunable_weights": int(total_weights),
        "actual_total_sparsity": float(total_sparsity),
        "note": (
            "Checkpoint contains sparse tensors in ckpt['sparsegpt_layers'] "
            "(pruned_weight + binary mask). "
            "ckpt['model'] contains either the full dense pruned state_dict "
            "or only non-pruned parameters for compact reconstructable loading."
        ),
    }

    save_nanogpt_checkpoint(
        original_ckpt=ckpt,
        model=model,
        out_path=args.out,
        sparse_meta=sparse_meta,
        sparsegpt_layers=sparsegpt_layers,
        keep_pruned_state_dict=bool(args.keep_pruned_state_dict),
    )

    print("\nDone.")
    print("Checkpoint now contains:")
    print("  - ckpt['sparsegpt_meta']")
    print("  - ckpt['sparsegpt_layers'][layer_name]['pruned_weight'/'mask']")
    print(f"  - total sparse layers: {len(sparsegpt_layers)}")
    print(f"  - total pruned weights: {total_pruned:,} / {total_weights:,}")
    print(f"  - actual sparsity over selected layers: {100.0 * total_sparsity:.2f}%")

    if args.keep_pruned_state_dict:
        print("  - ckpt['model'] with FULL dense pruned weights")
    else:
        print("  - ckpt['model'] with ONLY NON-PRUNED parameters")
        print("    selected sparse linear weights are omitted from ckpt['model']")


if __name__ == "__main__":
    main()