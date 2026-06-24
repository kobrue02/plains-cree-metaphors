#!/usr/bin/env bash
# Extract word alignments from the fine-tuned model over the full corpus.
#
# Run from project root:
#   bash jobs/infer_awesome_align.sh [model_dir]
#
# Outputs:
#   data/awesome_align/alignments.txt       Pharaoh format  (i-j pairs per line)
#   data/awesome_align/alignments_words.txt Aligned word pairs (human-readable)
#   data/awesome_align/alignments_probs.txt Per-alignment confidence scores

set -euo pipefail
source vendor/awesome-align-venv/bin/activate

MODEL_DIR="${1:-data/awesome_align/model}"
DATA_DIR="data/awesome_align"

# Run over the full corpus (train + dev combined) to get all word pairs
cat "$DATA_DIR/train.txt" "$DATA_DIR/dev.txt" > "$DATA_DIR/full.txt"

awesome-align \
    --output_file="$DATA_DIR/alignments.txt" \
    --model_name_or_path="$MODEL_DIR" \
    --data_file="$DATA_DIR/full.txt" \
    --extraction softmax \
    --align_layer 8 \
    --output_prob_file="$DATA_DIR/alignments_probs.txt" \
    --output_word_file="$DATA_DIR/alignments_words.txt" \
    --batch_size 32

echo ""
echo "Alignments written to $DATA_DIR/alignments.txt"
echo "Word pairs written to $DATA_DIR/alignments_words.txt"
echo ""
echo "Next: feed into InfoAlign with:"
echo "  from src.mt.awesome_prep import pharaoh_to_word_pairs"
echo "  pairs = pharaoh_to_word_pairs('$DATA_DIR/full.txt', '$DATA_DIR/alignments.txt')"
