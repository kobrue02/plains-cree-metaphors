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

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"

export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $TORCH_EXTENSIONS_DIR $HF_HOME

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

python scripts/train/train_silver.py \
    --checkpoint KonradBRG/xlm-mlm-plains-cree-en-tlm \
    --output-dir "data/figurative/silver_sft_sweep/n${N}" \
    --epochs 10 \
    --batch-size 16 \
    --freeze-n-layers "$N" \
    --hierarchical \
    --wandb-project fnlp-silver-sft
