#!/usr/bin/env bash
set -e

DEVICE="cuda"
DTYPE="float16"
BATCH_SIZE=8
BLOCK_SIZE=128

COMPRESS_SCRIPT="joint_sparsegpt_gptq_nanogpt.py"

# ============================================================
# GPT-2 Shakespeare model
# ============================================================

GPT2_ORIG="out-shakespeare-gpt2-ft/ckpt_best.pt"
GPT2_CALIB="calib_openweb_gpt2.pt"
GPT2_INPUT="data/shakespeare/input.txt"
GPT2_DATASET="data/shakespeare"
GPT2_OUTDIR="out-shakespeare-gpt2-ft/joint_packed_sweep"
GPT2_CKPTDIR="${GPT2_OUTDIR}/checkpoints"
GPT2_EVAL_CSV="${GPT2_OUTDIR}/eval_results.csv"
GPT2_SIZE_CSV="${GPT2_OUTDIR}/size_results.csv"

mkdir -p "$GPT2_CKPTDIR"
echo "checkpoint,bits,sparsity,groupsize,perplexity,loss,bpt,accuracy" > "$GPT2_EVAL_CSV"

for BITS in 4; do
  for SPARSITY in 0.00 0.10 0.20 0.30; do
    for GROUPSIZE in 32 64 128; do

      S_TAG=$(printf "%.0f" "$(echo "$SPARSITY * 100" | bc -l)")
      OUT="${GPT2_CKPTDIR}/gpt2_joint_b${BITS}_s${S_TAG}_g${GROUPSIZE}_packed.pt"
      LOG="${OUT%.pt}_eval.log"

      echo ""
      echo "============================================================"
      echo "Compressing GPT-2: bits=${BITS}, sparsity=${SPARSITY}, groupsize=${GROUPSIZE}"
      echo "============================================================"

      python "$COMPRESS_SCRIPT" \
        --checkpoint "$GPT2_ORIG" \
        --calib "$GPT2_CALIB" \
        --out "$OUT" \
        --bits "$BITS" \
        --sparsity "$SPARSITY" \
        --pattern unstructured \
        --groupsize "$GROUPSIZE" \
        --blocksize 128 \
        --mask_blocksize 128 \
        --percdamp 0.01 \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --amp_dtype "$DTYPE" \
        --packing packed4 \
        --mask_packing packedbits \
        --skip_tied_lm_head

      python eval_metrics.py \
        --checkpoint "$OUT" \
        --input_file "$GPT2_INPUT" \
        --dataset_dir "$GPT2_DATASET" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --batch_size "$BATCH_SIZE" \
        --block_size "$BLOCK_SIZE" | tee "$LOG"

      PPL=$(grep "Perplexity" "$LOG" | awk '{print $3}')
      LOSS=$(grep "Mean loss" "$LOG" | awk '{print $5}')
      BPT=$(grep "Bits per token" "$LOG" | awk '{print $5}')
      ACC=$(grep "Top-1 accuracy" "$LOG" | awk '{print $4}' | tr -d '%')

      echo "$OUT,$BITS,$SPARSITY,$GROUPSIZE,$PPL,$LOSS,$BPT,$ACC" >> "$GPT2_EVAL_CSV"

    done
  done
done

python analyze_joint_checkpoint_sizes.py \
  --original "$GPT2_ORIG" \
  --checkpoints "$GPT2_CKPTDIR"/*.pt \
  --csv "$GPT2_SIZE_CSV"


# ============================================================
# Char Shakespeare model
# ============================================================

CHAR_ORIG="out-shakespeare-char-gptqprep/ckpt_best.pt"
CHAR_CALIB="calib_openweb_char.pt"
CHAR_INPUT="data/shakespeare_char/input.txt"
CHAR_DATASET="data/shakespeare_char"
CHAR_OUTDIR="out-shakespeare-char-gptqprep/joint_packed_sweep"
CHAR_CKPTDIR="${CHAR_OUTDIR}/checkpoints"
CHAR_EVAL_CSV="${CHAR_OUTDIR}/eval_results.csv"
CHAR_SIZE_CSV="${CHAR_OUTDIR}/size_results.csv"

mkdir -p "$CHAR_CKPTDIR"
echo "checkpoint,bits,sparsity,groupsize,perplexity,loss,bpt,accuracy" > "$CHAR_EVAL_CSV"

for BITS in 4; do
  for SPARSITY in 0.00 0.10 0.20 0.30 0.40; do
    for GROUPSIZE in 32 64 128; do

      S_TAG=$(printf "%.0f" "$(echo "$SPARSITY * 100" | bc -l)")
      OUT="${CHAR_CKPTDIR}/char_joint_b${BITS}_s${S_TAG}_g${GROUPSIZE}_packed.pt"
      LOG="${OUT%.pt}_eval.log"

      echo ""
      echo "============================================================"
      echo "Compressing CHAR: bits=${BITS}, sparsity=${SPARSITY}, groupsize=${GROUPSIZE}"
      echo "============================================================"

      python "$COMPRESS_SCRIPT" \
        --checkpoint "$CHAR_ORIG" \
        --calib "$CHAR_CALIB" \
        --out "$OUT" \
        --bits "$BITS" \
        --sparsity "$SPARSITY" \
        --pattern unstructured \
        --groupsize "$GROUPSIZE" \
        --blocksize 128 \
        --mask_blocksize 128 \
        --percdamp 0.01 \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --amp_dtype "$DTYPE" \
        --packing packed4 \
        --mask_packing packedbits \
        --skip_tied_lm_head

      python eval_metrics.py \
        --checkpoint "$OUT" \
        --input_file "$CHAR_INPUT" \
        --dataset_dir "$CHAR_DATASET" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --batch_size "$BATCH_SIZE" \
        --block_size "$BLOCK_SIZE" | tee "$LOG"

      PPL=$(grep "Perplexity" "$LOG" | awk '{print $3}')
      LOSS=$(grep "Mean loss" "$LOG" | awk '{print $5}')
      BPT=$(grep "Bits per token" "$LOG" | awk '{print $5}')
      ACC=$(grep "Top-1 accuracy" "$LOG" | awk '{print $4}' | tr -d '%')

      echo "$OUT,$BITS,$SPARSITY,$GROUPSIZE,$PPL,$LOSS,$BPT,$ACC" >> "$CHAR_EVAL_CSV"

    done
  done
done

python analyze_joint_checkpoint_sizes.py \
  --original "$CHAR_ORIG" \
  --checkpoints "$CHAR_CKPTDIR"/*.pt \
  --csv "$CHAR_SIZE_CSV"

echo ""
echo "Done."
echo "GPT-2 eval CSV: $GPT2_EVAL_CSV"
echo "GPT-2 size CSV: $GPT2_SIZE_CSV"
echo "CHAR eval CSV:  $CHAR_EVAL_CSV"
echo "CHAR size CSV:  $CHAR_SIZE_CSV"