#!/bin/bash
#SBATCH --job-name=SilverSFTNoTLM
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Raw-encoder ablation for Section 6.2's evaluation paragraph: silver-SFT
# starting directly from the base (non-TLM-adapted) XLM-100 checkpoint,
# isolating whether TLM adaptation itself matters or silver labels alone
# carry the classifier. Same hyperparameters as the main TLM+Silver-SFT
# pipeline (freeze_n_layers=1, matching the local run this replaces) —
# only the starting checkpoint differs. A one-off ablation, not a sweep,
# so it pushes directly rather than going through the
# save-locally-then-push-best indirection jobs/silver_sft_worker.sh uses.
#
# Usage: sbatch jobs/silver_sft_no_tlm.sh

set -euo pipefail

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
    --checkpoint FacebookAI/xlm-mlm-100-1280 \
    --hub-model-id KonradBRG/xlm-mlm-plains-cree-en-silver-sft-no-tlm \
    --output-dir data/figurative/silver_sft_no_tlm \
    --epochs 10 \
    --batch-size 16 \
    --freeze-n-layers 1 \
    --hierarchical \
    --wandb-project fnlp-silver-sft
