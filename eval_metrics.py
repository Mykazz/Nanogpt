import os
import math
import pickle
import argparse
from contextlib import nullcontext
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from model import GPT, GPTConfig


IGNORE_INDEX = -1


# ============================================================
# Quantized layer helpers
# ============================================================

def strip_orig_mod_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod."):]] = v
        else:
            out[k] = v
    return out


def get_group_index(col_idx: int, cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 0
    return col_idx // groupsize


def unpack_4bit_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    """
    Unpack [rows, ceil(cols/2)] uint8 packed 4-bit values
    into [rows, cols] uint8.
    """
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be torch.uint8")

    rows, packed_cols = packed.shape
    out = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed.device)

    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F

    return out[:, :original_cols]


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


class QuantLinear(nn.Module):
    """
    Pure-PyTorch quantized linear runtime wrapper.

    This is correct but not kernel-optimized:
      unpack -> expand group params -> dequantize -> F.linear
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
    """
    Replace nn.Linear layers listed in ckpt['gptq_layers'] with QuantLinear.
    """
    if "gptq_layers" not in ckpt:
        raise ValueError("Checkpoint has no 'gptq_layers' entry.")

    gptq_layers = ckpt["gptq_layers"]

    for layer_name, layer_state in gptq_layers.items():
        orig_layer = get_module_by_name(model, layer_name)
        if not isinstance(orig_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer_name}, got {type(orig_layer)}")

        bias = orig_layer.bias.detach().clone() if orig_layer.bias is not None else None
        shape = tuple(layer_state["shape"])
        out_features, in_features = shape

        qlayer = QuantLinear(
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
        set_module_by_name(model, layer_name, qlayer)

    model.eval()
    model.to(device)
    return model


def checkpoint_has_usable_model_state(checkpoint: Dict[str, Any]) -> bool:
    state_dict = checkpoint.get("model", None)
    return isinstance(state_dict, dict) and len(state_dict) > 0


def checkpoint_has_quantized_layers(checkpoint: Dict[str, Any]) -> bool:
    qlayers = checkpoint.get("gptq_layers", None)
    return isinstance(qlayers, dict) and len(qlayers) > 0


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path: str | None,
    init_from: str | None,
    device: str,
    prefer_quantized: bool = True,
    cache_dequantized: bool = False,
):
    """
    Load either:
      - normal nanoGPT checkpoint
      - quantized checkpoint with gptq_layers
      - pretrained GPT-2 via init_from=...

    Returns:
      model, checkpoint_info
    """
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)

        has_model_state = checkpoint_has_usable_model_state(checkpoint)
        has_quant = checkpoint_has_quantized_layers(checkpoint)

        if has_model_state:
            state_dict = strip_orig_mod_prefix(checkpoint["model"])

            # strict=False because if later we replace linears with QuantLinear,
            # we mainly need the non-quantized tensors + biases loaded.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if unexpected:
                raise RuntimeError(f"Unexpected keys in checkpoint model state_dict: {unexpected}")

            # If there are missing keys while model state exists, report them clearly.
            # Usually this should be empty for standard checkpoints.
            if missing and not has_quant:
                raise RuntimeError(f"Missing keys in checkpoint model state_dict: {missing}")

        elif has_quant:
            # CRUCIAL:
            # A quant-only checkpoint with empty checkpoint["model"] does NOT contain
            # embeddings / layernorms / biases / non-quantized params, so it cannot be
            # fully reconstructed from eval_metrics.py alone.
            raise RuntimeError(
                "This checkpoint contains 'gptq_layers' but checkpoint['model'] is empty.\n"
                "That means the checkpoint does NOT include required non-quantized parameters "
                "(e.g. embeddings, layer norms, biases).\n\n"
                "Re-create the checkpoint with:\n"
                "  --keep_dequantized_state_dict\n"
                "or modify your quantization script to save all non-quantized parameters."
            )

        else:
            raise RuntimeError(
                "Checkpoint has neither a usable checkpoint['model'] nor checkpoint['gptq_layers']."
            )

        model.eval()
        model.to(device)

        if has_quant and prefer_quantized:
            print("Detected quantized checkpoint. Replacing quantized nn.Linear layers with QuantLinear...")
            model = convert_model_to_quant_linear(
                model=model,
                ckpt=checkpoint,
                device=torch.device(device),
                cache_dequantized=cache_dequantized,
            )

        return model, checkpoint

    if init_from is not None:
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
        model.eval()
        model.to(device)
        return model, {
            "model_args": {
                "block_size": model.config.block_size,
                "vocab_size": model.config.vocab_size,
                "n_layer": model.config.n_layer,
                "n_head": model.config.n_head,
                "n_embd": model.config.n_embd,
                "bias": model.config.bias,
            },
            "config": {},
        }

    raise ValueError("Provide either --checkpoint or --init_from.")


# ============================================================
# Tokenizer / data helpers
# ============================================================

def get_tokenizer(checkpoint_info, dataset_dir: str | None):
    """
    Returns:
      encode(text) -> list[int]
      decode(ids) -> str
      tokenizer_name: str

    Priority:
      1) meta.pkl from dataset_dir, if available
      2) meta.pkl inferred from checkpoint config dataset, if available
      3) GPT-2 tokenizer
    """
    candidate_meta_paths = []

    if dataset_dir is not None:
        candidate_meta_paths.append(os.path.join(dataset_dir, "meta.pkl"))

    if "config" in checkpoint_info and isinstance(checkpoint_info["config"], dict):
        ds = checkpoint_info["config"].get("dataset", None)
        if ds:
            candidate_meta_paths.append(os.path.join("data", ds, "meta.pkl"))

    for meta_path in candidate_meta_paths:
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            if "stoi" in meta and "itos" in meta:
                stoi, itos = meta["stoi"], meta["itos"]

                def encode_char(s: str):
                    unknown = sorted(set(c for c in s if c not in stoi))
                    if unknown:
                        preview = "".join(unknown[:20])
                        raise ValueError(
                            f"Found {len(unknown)} character(s) not in vocabulary. "
                            f"First few: {repr(preview)}"
                        )
                    return [stoi[c] for c in s]

                def decode_char(ids):
                    return "".join(itos[i] for i in ids)

                return encode_char, decode_char, f"meta.pkl ({meta_path})"

    enc = tiktoken.get_encoding("gpt2")

    def encode_bpe(s: str):
        return enc.encode(s, allowed_special={"<|endoftext|>"})

    def decode_bpe(ids):
        return enc.decode(ids)

    return encode_bpe, decode_bpe, "gpt2"


def load_tokens_from_input(input_file: str, encode_fn):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    ids = encode_fn(text)
    if len(ids) < 2:
        raise ValueError("Input text is too short after tokenization; need at least 2 tokens.")

    return np.array(ids, dtype=np.int64), text


def iterate_eval_windows(token_ids: np.ndarray, block_size: int, stride: int | None):
    """
    Generate evaluation windows for perplexity / loss computation.

    IMPORTANT:
    - If stride == block_size: non-overlapping windows
    - If stride < block_size: overlapping windows
    - For overlapping windows, overlap is used only as CONTEXT, not scored again.

    Yields:
      x          : input ids for this window
      y          : target ids for this window
      score_from : first target position INSIDE THIS WINDOW that should count
                   toward metrics
      start      : starting token index of this window

    CRUCIAL FIX:
    - The overlap to ignore must be computed from absolute positions using
      the previous window end, NOT from len(x) - stride.
    - This matters especially for the final truncated window.
    """
    if stride is None:
        stride = block_size

    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    if stride > block_size:
        raise ValueError("stride must be <= block_size, otherwise some tokens are skipped.")

    n = len(token_ids)
    if n < 2:
        return

    prev_end = None  # absolute end index in x-space of previous window

    for start in range(0, n - 1, stride):
        end = min(start + block_size, n - 1)

        x = token_ids[start:end]
        y = token_ids[start + 1:end + 1]

        if len(x) == 0:
            continue

        if prev_end is None:
            score_from = 0
        else:
            # Number of target positions in this window already covered by the
            # previous window. This is the amount of overlap to ignore.
            #
            # Current y positions correspond to absolute target indices:
            #   start+1, start+2, ..., end
            #
            # Previous window already covered targets up to absolute index prev_end.
            # Therefore ignore the first (prev_end - start) positions in this y.
            score_from = max(0, prev_end - start)

        yield x, y, score_from, start

        prev_end = end

        if end >= n - 1:
            break


def make_autocast_context(device: str, dtype_str: str):
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]

    if device_type == "cpu":
        return nullcontext()

    return torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def maybe_crop_model_for_eval(model, eval_block_size: int):
    """
    Crop model block size for eval only if requested block size is smaller.
    """
    if eval_block_size < model.config.block_size:
        model.crop_block_size(eval_block_size)
    return model


def safe_perplexity(mean_loss: float) -> float:
    return math.exp(mean_loss) if mean_loss < 20 else float("inf")


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_metrics(
    model,
    token_ids,
    block_size,
    batch_size,
    device,
    dtype_str="float16",
    stride=None,
):
    """
    Computes:
      - mean cross-entropy loss (nats/token)
      - perplexity
      - bits per token
      - top-1 accuracy
      - token/batch counts

    CRUCIAL:
    - Uses IGNORE_INDEX = -1, matching nanoGPT's model.py
    - When stride < block_size, overlapping tokens are NOT double-counted
    - x and y are already externally shifted, so no extra "subtract batch_size"
      correction is needed here
    """
    ctx = make_autocast_context(device, dtype_str)

    if stride is None:
        stride = block_size

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    total_batches = 0
    total_windows = 0

    batch_x = []
    batch_y = []
    batch_score_from = []

    def process_batch(batch_x_local, batch_y_local, batch_score_from_local):
        nonlocal total_nll, total_tokens, total_correct, total_batches

        if not batch_x_local:
            return

        max_len = max(len(arr) for arr in batch_x_local)

        xb = torch.full((len(batch_x_local), max_len), 0, dtype=torch.long)
        yb = torch.full((len(batch_y_local), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, (x_arr, y_arr, score_from) in enumerate(
            zip(batch_x_local, batch_y_local, batch_score_from_local)
        ):
            L = len(x_arr)
            xb[i, :L] = torch.tensor(x_arr, dtype=torch.long)
            yb[i, :L] = torch.tensor(y_arr, dtype=torch.long)

            # Crucial masking step:
            # everything before score_from is overlap/context only
            if score_from > 0:
                yb[i, :score_from] = IGNORE_INDEX

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with ctx:
            logits, loss = model(xb, yb)

        valid_mask = (yb != IGNORE_INDEX)
        scored_tokens = valid_mask.sum().item()

        if scored_tokens == 0:
            return

        # nanoGPT loss is mean CE over valid tokens, so multiply back to recover
        # total negative log-likelihood contribution for this batch.
        total_nll += loss.item() * scored_tokens
        total_tokens += scored_tokens
        total_batches += 1

        preds = logits.argmax(dim=-1)
        total_correct += ((preds == yb) & valid_mask).sum().item()

    for x, y, score_from, _start in iterate_eval_windows(token_ids, block_size, stride):
        batch_x.append(x)
        batch_y.append(y)
        batch_score_from.append(score_from)
        total_windows += 1

        if len(batch_x) == batch_size:
            process_batch(batch_x, batch_y, batch_score_from)
            batch_x.clear()
            batch_y.clear()
            batch_score_from.clear()

    process_batch(batch_x, batch_y, batch_score_from)

    if total_tokens == 0:
        raise RuntimeError("No tokens evaluated. Reduce block_size or use a longer file.")

    mean_loss = total_nll / total_tokens
    perplexity = safe_perplexity(mean_loss)
    bits_per_token = mean_loss / math.log(2.0)
    accuracy = total_correct / total_tokens

    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity,
        "bits_per_token": bits_per_token,
        "top1_accuracy": accuracy,
        "tokens_evaluated": total_tokens,
        "batches_evaluated": total_batches,
        "windows_evaluated": total_windows,
        "stride_used": stride,
        "evaluation_mode": "non-overlapping" if stride == block_size else "overlapping",
    }


@torch.no_grad()
def show_prediction_examples(
    model,
    token_ids,
    decode_fn,
    block_size,
    batch_size,
    device,
    dtype_str="float16",
    stride=None,
    num_examples=2,
    show_tokens=12,
    top_k=5,
):
    """
    Print a few concrete prediction examples:
      - context window text
      - real next token
      - predicted next token
      - top-k candidates
    """
    ctx = make_autocast_context(device, dtype_str)

    if stride is None:
        stride = block_size

    printed = 0
    batch_x = []
    batch_y = []
    batch_score_from = []
    batch_starts = []

    def process_examples(batch_x_local, batch_y_local, batch_score_from_local, batch_starts_local):
        nonlocal printed

        if not batch_x_local or printed >= num_examples:
            return

        max_len = max(len(arr) for arr in batch_x_local)

        xb = torch.full((len(batch_x_local), max_len), 0, dtype=torch.long)
        yb = torch.full((len(batch_y_local), max_len), IGNORE_INDEX, dtype=torch.long)

        lengths = []
        for i, (x_arr, y_arr, score_from) in enumerate(
            zip(batch_x_local, batch_y_local, batch_score_from_local)
        ):
            L = len(x_arr)
            lengths.append(L)
            xb[i, :L] = torch.tensor(x_arr, dtype=torch.long)
            yb[i, :L] = torch.tensor(y_arr, dtype=torch.long)

            if score_from > 0:
                yb[i, :score_from] = IGNORE_INDEX

        xb_dev = xb.to(device, non_blocking=True)
        yb_dev = yb.to(device, non_blocking=True)

        with ctx:
            logits, _ = model(xb_dev, yb_dev)

        probs = F.softmax(logits.float(), dim=-1)
        preds = logits.argmax(dim=-1).cpu()

        for row in range(xb.size(0)):
            if printed >= num_examples:
                return

            start_idx = batch_starts_local[row]
            x_row = xb[row, :lengths[row]].cpu().tolist()
            y_row = yb[row, :lengths[row]].cpu().tolist()
            pred_row = preds[row, :lengths[row]].tolist()
            probs_row = probs[row, :lengths[row]].cpu()
            score_from = batch_score_from_local[row]

            print("\n" + "=" * 100)
            print(f"EXAMPLE {printed + 1}")
            print(f"Token window starts at token index: {start_idx}")
            print(f"Scored positions start at        : {score_from}")
            print("-" * 100)

            context_text = decode_fn(x_row)
            print("CONTEXT WINDOW TEXT:")
            print(repr(context_text[:500]))
            if len(context_text) > 500:
                print("... [truncated]")
            print("-" * 100)

            valid_positions = [t for t in range(len(y_row)) if y_row[t] != IGNORE_INDEX]
            n_show = min(show_tokens, len(valid_positions))

            print(f"FIRST {n_show} SCORED TOKEN PREDICTIONS IN THIS WINDOW:\n")

            for idx_in_valid in range(n_show):
                t = valid_positions[idx_in_valid]

                real_id = y_row[t]
                pred_id = pred_row[t]
                is_correct = real_id == pred_id

                real_text = decode_fn([real_id])
                pred_text = decode_fn([pred_id])

                top_vals, top_ids = torch.topk(
                    probs_row[t], k=min(top_k, probs_row[t].shape[-1])
                )
                top_vals = top_vals.tolist()
                top_ids = top_ids.tolist()

                prefix_ids = x_row[max(0, t - 12): t + 1]
                prefix_text = decode_fn(prefix_ids)

                print(f"Position {t:3d}")
                print(f"  prefix (last {len(prefix_ids)} input tokens): {repr(prefix_text)}")
                print(f"  REAL token id : {real_id}")
                print(f"  REAL token    : {repr(real_text)}")
                print(f"  PRED token id : {pred_id}")
                print(f"  PRED token    : {repr(pred_text)}")
                print(f"  Correct       : {is_correct}")

                print("  Top candidates:")
                for rank, (cand_id, cand_prob) in enumerate(zip(top_ids, top_vals), start=1):
                    cand_text = decode_fn([cand_id])
                    print(
                        f"    {rank:>2d}. id={cand_id:<6d} prob={cand_prob:>10.6f} "
                        f"token={repr(cand_text)}"
                    )
                print()

            printed += 1

    for x, y, score_from, start in iterate_eval_windows(token_ids, block_size, stride):
        batch_x.append(x)
        batch_y.append(y)
        batch_score_from.append(score_from)
        batch_starts.append(start)

        if len(batch_x) == batch_size:
            process_examples(batch_x, batch_y, batch_score_from, batch_starts)
            if printed >= num_examples:
                return
            batch_x.clear()
            batch_y.clear()
            batch_score_from.clear()
            batch_starts.clear()

    process_examples(batch_x, batch_y, batch_score_from, batch_starts)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nanoGPT model metrics on a text file."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Raw text file to evaluate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to ckpt.pt / ckpt_best.pt / ckpt_last.pt")
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Pretrained model name: gpt2 | gpt2-medium | gpt2-large | gpt2-xl",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Optional dataset dir containing meta.pkl (useful for char models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Override eval block size; must be <= model checkpoint block size",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Window stride. Default uses non-overlapping windows (stride=block_size). "
            "If stride < block_size, windows overlap, but overlapping targets are NOT "
            "double-counted in metrics."
        ),
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="If > 0, print concrete prediction examples",
    )
    parser.add_argument(
        "--show_tokens",
        type=int,
        default=12,
        help="How many scored token positions to display per example window",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many top candidate predictions to show per token position",
    )
    parser.add_argument(
        "--prefer_quantized",
        action="store_true",
        help="If checkpoint contains gptq_layers, replace quantized nn.Linear with QuantLinear.",
    )
    parser.add_argument(
        "--cache_dequantized",
        action="store_true",
        help="Cache dequantized QuantLinear weights after first use. Uses more memory.",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.init_from is None:
        raise ValueError("Provide either --checkpoint or --init_from.")

    if args.checkpoint is not None and args.init_from is not None:
        raise ValueError("Provide only one of --checkpoint or --init_from, not both.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model, info = load_model(
        args.checkpoint,
        args.init_from,
        args.device,
        prefer_quantized=args.prefer_quantized,
        cache_dequantized=args.cache_dequantized,
    )

    ckpt_block_size = model.config.block_size
    eval_block_size = args.block_size if args.block_size is not None else ckpt_block_size

    if eval_block_size <= 0:
        raise ValueError("block_size must be positive.")

    if eval_block_size > ckpt_block_size:
        raise ValueError(
            f"Requested block_size={eval_block_size}, but model supports at most {ckpt_block_size}."
        )

    if args.stride is not None and args.stride > eval_block_size:
        raise ValueError(
            f"Requested stride={args.stride}, but stride must be <= block_size={eval_block_size}."
        )

    model = maybe_crop_model_for_eval(model, eval_block_size)

    encode, decode, tokenizer_name = get_tokenizer(info, args.dataset_dir)
    token_ids, raw_text = load_tokens_from_input(args.input_file, encode)

    print(f"Model source      : {args.checkpoint if args.checkpoint else args.init_from}")
    print(f"Tokenizer         : {tokenizer_name}")
    print(f"Device            : {args.device}")
    print(f"Dtype             : {args.dtype}")
    print(f"Model block size  : {ckpt_block_size}")
    print(f"Eval block size   : {eval_block_size}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Input file        : {args.input_file}")
    print(f"Raw chars         : {len(raw_text):,}")
    print(f"Tokenized length  : {len(token_ids):,}")
    print(f"Prefer quantized  : {args.prefer_quantized}")
    print(f"Cache dequantized : {args.cache_dequantized}")

    metrics = evaluate_metrics(
        model=model,
        token_ids=token_ids,
        block_size=eval_block_size,
        batch_size=args.batch_size,
        device=args.device,
        dtype_str=args.dtype,
        stride=args.stride,
    )

    print("\n=== Evaluation Metrics ===")
    print(f"Perplexity            : {metrics['perplexity']:.6f}")
    print(f"Mean loss (nats/token): {metrics['mean_loss']:.6f}")
    print(f"Bits per token        : {metrics['bits_per_token']:.6f}")
    print(f"Top-1 accuracy        : {metrics['top1_accuracy']:.6%}")
    print(f"Tokens evaluated      : {metrics['tokens_evaluated']:,}")
    print(f"Batches evaluated     : {metrics['batches_evaluated']:,}")
    print(f"Windows evaluated     : {metrics['windows_evaluated']:,}")
    print(f"Stride used           : {metrics['stride_used']}")
    print(f"Evaluation mode       : {metrics['evaluation_mode']}")

    if args.show_examples > 0:
        show_prediction_examples(
            model=model,
            token_ids=token_ids,
            decode_fn=decode,
            block_size=eval_block_size,
            batch_size=args.batch_size,
            device=args.device,
            dtype_str=args.dtype,
            stride=args.stride,
            num_examples=args.show_examples,
            show_tokens=args.show_tokens,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()