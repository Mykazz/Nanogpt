"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import csv
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval

# checkpoint behavior
always_save_checkpoint = True   # save latest checkpoint every eval
save_best_checkpoint = True     # save separate best checkpoint
save_last_checkpoint = True     # save separate last checkpoint

# metrics / plotting
save_metrics_csv = True
save_plots = True

# initialization
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
resume_checkpoint = None  # if None, defaults to out_dir/ckpt_last.pt or out_dir/ckpt.pt

# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', or 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank

    # world_size processes train simultaneously, scale down accumulation per process
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # Recreate np.memmap every batch to avoid a memory leak
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9
best_val_iter = -1

# history for graphs
metrics_history = {
    'iter': [],
    'train_loss': [],
    'val_loss': [],
    'train_ppl': [],
    'val_ppl': [],
    'train_acc': [],
    'val_acc': [],
}

# derive vocab_size from dataset if possible
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

def strip_unwanted_prefix_from_state_dict(state_dict, unwanted_prefix='_orig_mod.'):
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def resolve_resume_path():
    if resume_checkpoint is not None:
        return resume_checkpoint

    preferred = os.path.join(out_dir, 'ckpt_last.pt')
    legacy = os.path.join(out_dir, 'ckpt.pt')

    if os.path.exists(preferred):
        return preferred
    return legacy

def safe_exp_loss(loss_value: float) -> float:
    return math.exp(loss_value) if loss_value < 20 else float('inf')

def save_checkpoint(path, model_state, optimizer_state, model_args, iter_num, best_val_loss, best_val_iter, config, history):
    checkpoint = {
        'model': model_state,
        'optimizer': optimizer_state,
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'best_val_iter': best_val_iter,
        'config': config,
        'metrics_history': history,
    }
    torch.save(checkpoint, path)

def save_metrics_history_csv(history, csv_path):
    rows = zip(
        history['iter'],
        history['train_loss'],
        history['val_loss'],
        history['train_ppl'],
        history['val_ppl'],
        history['train_acc'],
        history['val_acc'],
    )
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iter',
            'train_loss',
            'val_loss',
            'train_ppl',
            'val_ppl',
            'train_acc',
            'val_acc',
        ])
        writer.writerows(rows)

def plot_metrics(history, out_dir):
    # CRUCIAL PART:
    # Matplotlib import is inside function, and backend is forced to Agg
    # so plotting works even on headless servers / WSL terminals.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(history['iter']) == 0:
        return

    x = history['iter']

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, history['train_loss'], label='Train Loss')
    plt.plot(x, history['val_loss'], label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Training Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150)
    plt.close()

    # Perplexity plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, history['train_ppl'], label='Train Perplexity')
    plt.plot(x, history['val_ppl'], label='Val Perplexity')
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs Training Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'perplexity_curve.png'), dpi=150)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, history['train_acc'], label='Train Accuracy')
    plt.plot(x, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'), dpi=150)
    plt.close()

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    ckpt_path = resolve_resume_path()
    print(f"Resuming training from {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # force these config attributes to match checkpoint
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    state_dict = strip_unwanted_prefix_from_state_dict(state_dict)
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
    best_val_iter = checkpoint.get('best_val_iter', best_val_iter)

    if 'metrics_history' in checkpoint:
        metrics_history = checkpoint['metrics_history']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

else:
    raise ValueError(f"Unknown init_from value: {init_from}")

# crop down the model block size if desired
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# initialize GradScaler
scaler = torch.amp.GradScaler(device='cuda', enabled=(device_type == 'cuda' and dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# wrap model into DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# unwrap DDP container if needed
raw_model = model.module if ddp else model

@torch.no_grad()
def estimate_metrics():
    """
    Estimate metrics over eval_iters random batches for train/val.

    Returns:
      {
        'train': {'loss': ..., 'ppl': ..., 'acc': ...},
        'val':   {'loss': ..., 'ppl': ..., 'acc': ...},
      }
    """
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        correct = 0
        total = 0

        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)

            losses[k] = loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == Y).sum().item()
            total += Y.numel()

        mean_loss = losses.mean().item()
        out[split] = {
            'loss': mean_loss,
            'ppl': safe_exp_loss(mean_loss),
            'acc': correct / total,
        }

    model.train()
    return out

def get_lr(it):
    """Cosine decay with warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # MODIFIKUOTOS METRIKOS, CHECKPOINTAI IR PLOTAI
    if iter_num % eval_interval == 0 and master_process:
        metrics = estimate_metrics()

        train_loss = metrics['train']['loss']
        val_loss = metrics['val']['loss']
        train_ppl = metrics['train']['ppl']
        val_ppl = metrics['val']['ppl']
        train_acc = metrics['train']['acc']
        val_acc = metrics['val']['acc']

        print(
            f"step {iter_num}: "
            f"train loss {train_loss:.4f}, train ppl {train_ppl:.2f}, train acc {train_acc:.4%}, "
            f"val loss {val_loss:.4f}, val ppl {val_ppl:.2f}, val acc {val_acc:.4%}"
        )

       
        # SAUGOMOS METRIKOS 
        metrics_history['iter'].append(iter_num)
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_ppl'].append(train_ppl)
        metrics_history['val_ppl'].append(val_ppl)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_acc'].append(val_acc)

        if save_metrics_csv:
            save_metrics_history_csv(metrics_history, os.path.join(out_dir, 'metrics_history.csv'))

        if save_plots:
            plot_metrics(metrics_history, out_dir)

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": train_loss,
                "train/ppl": train_ppl,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/ppl": val_ppl,
                "val/acc": val_acc,
                "lr": lr,
                "mfu": running_mfu * 100,
                "best_val_loss": float(best_val_loss),
            })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_iter = iter_num

        if iter_num > 0:
            model_state = raw_model.state_dict()
            optimizer_state = optimizer.state_dict()

            if always_save_checkpoint and save_last_checkpoint:
                last_ckpt_path = os.path.join(out_dir, 'ckpt_last.pt')
                print(f"saving last checkpoint to {last_ckpt_path}")
                save_checkpoint(
                    last_ckpt_path,
                    model_state,
                    optimizer_state,
                    model_args,
                    iter_num,
                    best_val_loss,
                    best_val_iter,
                    config,
                    metrics_history,
                )

            if save_best_checkpoint and is_best:
                best_ckpt_path = os.path.join(out_dir, 'ckpt_best.pt')
                print(f"saving BEST checkpoint to {best_ckpt_path}")
                save_checkpoint(
                    best_ckpt_path,
                    model_state,
                    optimizer_state,
                    model_args,
                    iter_num,
                    best_val_loss,
                    best_val_iter,
                    config,
                    metrics_history,
                )

            # optional legacy path for compatibility with older scripts
            if always_save_checkpoint:
                legacy_ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                save_checkpoint(
                    legacy_ckpt_path,
                    model_state,
                    optimizer_state,
                    model_args,
                    iter_num,
                    best_val_loss,
                    best_val_iter,
                    config,
                    metrics_history,
                )

    if iter_num == 0 and eval_only:
        break

    # forward/backward/update with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # sync gradients only at last micro-step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # async prefetch next batch while GPU is busy
        X, Y = get_batch('train')

        # backward
        scaler.scale(loss).backward()

    # clip gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # optimizer step
    scaler.step(optimizer)
    scaler.update()

    # flush gradients
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps

        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # termination
    if iter_num > max_iters:
        break

# GALINIS VERTINIMAS IR CHECKPOINTAI
if master_process:
    metrics = estimate_metrics()

    train_loss = metrics['train']['loss']
    val_loss = metrics['val']['loss']
    train_ppl = metrics['train']['ppl']
    val_ppl = metrics['val']['ppl']
    train_acc = metrics['train']['acc']
    val_acc = metrics['val']['acc']

    print("\n=== FINAL EVALUATION ===")
    print(f"final train loss: {train_loss:.4f}")
    print(f"final val loss:   {val_loss:.4f}")
    print(f"final train ppl:  {train_ppl:.2f}")
    print(f"final val ppl:    {val_ppl:.2f}")
    print(f"final train acc:  {train_acc:.4%}")
    print(f"final val acc:    {val_acc:.4%}")
    print(f"best val loss:    {best_val_loss:.4f}")
    print(f"best val ppl:     {safe_exp_loss(best_val_loss):.2f}")
    print(f"best val iter:    {best_val_iter}")

    # Save final metrics point as well
    if len(metrics_history['iter']) == 0 or metrics_history['iter'][-1] != iter_num:
        metrics_history['iter'].append(iter_num)
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_ppl'].append(train_ppl)
        metrics_history['val_ppl'].append(val_ppl)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_acc'].append(val_acc)

    if save_metrics_csv:
        save_metrics_history_csv(metrics_history, os.path.join(out_dir, 'metrics_history.csv'))

    if save_plots:
        plot_metrics(metrics_history, out_dir)
        print(f"\nSaved plots to {out_dir}/")
        print(f" - {os.path.join(out_dir, 'loss_curve.png')}")
        print(f" - {os.path.join(out_dir, 'perplexity_curve.png')}")
        print(f" - {os.path.join(out_dir, 'accuracy_curve.png')}")
        print(f" - {os.path.join(out_dir, 'metrics_history.csv')}")

if ddp:
    destroy_process_group()


