#!/bin/bash
#SBATCH --job-name=FigEnglish_Cree
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# English fine-tune (VUA20+MAGPIE+FLUTE) for the "no adaptation" / "+TLM"
# rows of the classifier results table. Works with any multilingual encoder.
#
# Usage:
#   sbatch jobs/train_figurative_english.sh --experiment baseline \
#       --encoder FacebookAI/xlm-mlm-100-1280 \
#       --hub-model-id KonradBRG/xlm-mlm-100-1280-figurative
#
# All scripts/train/train_figurative_english.py flags are forwarded directly.

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

echo "Starting English figurative fine-tune with args: $@"
python3 scripts/train/train_figurative_english.py \
    --wandb-project fnlp-figurative \
    "$@"

if [ $? -eq 0 ]; then
    echo "Done."
else
    echo "Failed." && exit 1
fi
