#!/bin/bash
#SBATCH --job-name=SilverSFT
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Runs ONE freeze_n_layers value of the hierarchical silver-SFT classifier.
# Submitted by jobs/silver_sft_freeze_sweep.sh (one job per n in 0..16) — not
# meant to be called directly. Saves locally only (no --hub-model-id): the
# sweep pushes only the overall winner at the end (scripts/train/push_best_silver_sft.py),
# to avoid cluttering the Hub with 17 intermediate checkpoints — the exact
# mistake that made recovering a specific past run's checkpoint unreliable
# earlier in this project (see silver_sft_sweep_results.parquet instead,
# written by src.figurative.silver_sft._save_sweep_result).
#
# Usage: sbatch jobs/silver_sft_worker.sh <freeze_n_layers>

set -euo pipefail
N="$1"

python scripts/train/train_silver.py \
    --checkpoint KonradBRG/xlm-mlm-plains-cree-en-tlm \
    --output-dir "data/figurative/silver_sft_sweep/n${N}" \
    --epochs 10 \
    --batch-size 16 \
    --freeze-n-layers "$N" \
    --hierarchical
