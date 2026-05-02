#!/usr/bin/env bash
set -e

DEVICE="cuda"
DTYPE="float16"
BATCH_SIZE=8
BLOCK_SIZE=128

eval_dir () {
  CKPTDIR="$1"
  INPUT="$2"
  DATASET="$3"
  OUTCSV="$4"

  echo "checkpoint,bits,sparsity,groupsize,perplexity,loss,bpt,accuracy" > "$OUTCSV"

  for CKPT in "$CKPTDIR"/*.pt; do
    BASE=$(basename "$CKPT")

    BITS=$(echo "$BASE" | sed -n 's/.*_b\([0-9]*\)_.*/\1/p')
    S_INT=$(echo "$BASE" | sed -n 's/.*_s\([0-9]*\)_.*/\1/p')
    GROUPSIZE=$(echo "$BASE" | sed -n 's/.*_g\([0-9]*\)_.*/\1/p')
    SPARSITY=$(python - <<PY
print(f"{int('$S_INT')/100:.2f}")
PY
)

    LOG="${CKPT%.pt}_eval.log"

    echo "Evaluating $CKPT"

    python eval_metrics.py \
      --checkpoint "$CKPT" \
      --input_file "$INPUT" \
      --dataset_dir "$DATASET" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --batch_size "$BATCH_SIZE" \
      --block_size "$BLOCK_SIZE" | tee "$LOG"

    PPL=$(grep "Perplexity" "$LOG" | awk '{print $3}')
    LOSS=$(grep "Mean loss" "$LOG" | awk '{print $5}')
    BPT=$(grep "Bits per token" "$LOG" | awk '{print $5}')
    ACC=$(grep "Top-1 accuracy" "$LOG" | awk '{print $4}' | tr -d '%')

    echo "$CKPT,$BITS,$SPARSITY,$GROUPSIZE,$PPL,$LOSS,$BPT,$ACC" >> "$OUTCSV"
  done
}

eval_dir \
  "out-shakespeare-gpt2-ft/joint_packed_sweep/checkpoints" \
  "data/shakespeare/input.txt" \
  "data/shakespeare" \
  "out-shakespeare-gpt2-ft/joint_packed_sweep/eval_results_fixed.csv"

eval_dir \
  "out-shakespeare-char-gptqprep/joint_packed_sweep/checkpoints" \
  "data/shakespeare_char/input.txt" \
  "data/shakespeare_char" \
  "out-shakespeare-char-gptqprep/joint_packed_sweep/eval_results_fixed.csv"