import os
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

from model import GPT, GPTConfig


IGNORE_INDEX = -1


def load_model(checkpoint_path: str | None, init_from: str | None, device: str):
    """
    Load either:
      - a nanoGPT checkpoint (ckpt.pt), or
      - a pretrained GPT-2 model via init_from=gpt2|gpt2-medium|...
    Returns:
      model, model_info
    """
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)

        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
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
    - For overlapping windows, we use overlap only as CONTEXT, not score
      the overlapped targets again.

    Yields:
      x          : input ids for this window
      y          : target ids for this window
      score_from : first target position INSIDE THIS WINDOW that should count
                   toward metrics
      start      : starting token index of this window
    """
    if stride is None:
        stride = block_size

    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    n = len(token_ids)
    if n < 2:
        return

    first_window = True

    for start in range(0, n - 1, stride):
        end = min(start + block_size, n - 1)

        x = token_ids[start:end]
        y = token_ids[start + 1:end + 1]

        if len(x) == 0:
            continue

        if first_window:
            score_from = 0
            first_window = False
        else:
            # Score only the newly introduced suffix when windows overlap.
            score_from = max(0, len(x) - stride)

        yield x, y, score_from, start

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

        # CRUCIAL PART:
        # Use IGNORE_INDEX = -1, because nanoGPT model.py expects ignore_index=-1.
        xb = torch.full((len(batch_x_local), max_len), 0, dtype=torch.long)
        yb = torch.full((len(batch_y_local), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, (x_arr, y_arr, score_from) in enumerate(
            zip(batch_x_local, batch_y_local, batch_score_from_local)
        ):
            L = len(x_arr)
            xb[i, :L] = torch.tensor(x_arr, dtype=torch.long)
            yb[i, :L] = torch.tensor(y_arr, dtype=torch.long)

            # Ignore overlapped prefix tokens so they do not affect perplexity twice
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

        # loss returned by nanoGPT model is mean over valid positions only
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

    # final partial batch
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
    args = parser.parse_args()

    if args.checkpoint is None and args.init_from is None:
        raise ValueError("Provide either --checkpoint or --init_from.")

    if args.checkpoint is not None and args.init_from is not None:
        raise ValueError("Provide only one of --checkpoint or --init_from, not both.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model, info = load_model(args.checkpoint, args.init_from, args.device)

    ckpt_block_size = model.config.block_size
    eval_block_size = args.block_size if args.block_size is not None else ckpt_block_size

    if eval_block_size <= 0:
        raise ValueError("block_size must be positive.")

    if eval_block_size > ckpt_block_size:
        raise ValueError(
            f"Requested block_size={eval_block_size}, but model supports at most {ckpt_block_size}."
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