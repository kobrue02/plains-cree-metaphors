#!/usr/bin/env bash
# Fine-tune XLM-RoBERTa on the Cree-English parallel corpus.
#
# Prerequisites:
#   bash jobs/setup_awesome_align.sh
#   uv run python -m src.mt.awesome_prep   (generates data/awesome_align/)
#
# Run from project root:
#   bash jobs/train_awesome_align.sh

set -euo pipefail
source vendor/awesome-align-venv/bin/activate

TRAIN_FILE="data/awesome_align/train.txt"
DEV_FILE="data/awesome_align/dev.txt"
OUTPUT_DIR="data/awesome_align/model"

mkdir -p "$OUTPUT_DIR"

# Hyperparameters tuned for ~5k paragraph pairs on Apple Silicon:
#   - TLM: unsupervised cross-lingual masked LM (no gold alignments needed)
#   - CO:  contrastive objective boosting recall (good for polysynthetic languages
#          where one Cree word can align to many English words)
#   - 3 epochs over ~5k pairs ≈ reasonable signal without overfitting
#   - batch 2 + grad_accum 4 = effective batch 8

awesome-train \
    --output_dir="$OUTPUT_DIR" \
    --model_name_or_path=xlm-roberta-base \
    --extraction softmax \
    --do_train \
    --train_tlm \
    --train_co \
    --train_data_file="$TRAIN_FILE" \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --save_steps 1000 \
    --do_eval \
    --eval_data_file="$DEV_FILE" \
    --align_layer 8 \
    --overwrite_output_dir

echo ""
echo "Fine-tuned model saved to $OUTPUT_DIR"
echo "Next: bash jobs/infer_awesome_align.sh"
