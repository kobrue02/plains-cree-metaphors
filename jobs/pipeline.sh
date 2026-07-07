#!/bin/bash
#SBATCH --job-name=Pipeline_Cree
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# End-to-end pipeline: TLM → CLKD → Calibrate
#
# Usage:
#   sbatch jobs/pipeline.sh --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm
#   sbatch jobs/pipeline.sh --base-model facebook/xlm-v-base --model-id xlm-v --max-length 128
#   sbatch jobs/pipeline.sh --base-model cis-lmu/glot500-base --model-id glot500 --skip-tlm
#
# All pipeline.py flags are forwarded directly — see python pipeline.py --help

# 1. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"

# 2. Environment
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
export HF_HOME=$WORK/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $TORCH_EXTENSIONS_DIR $HF_HOME

# 3. Project
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

# 4. Run — all CLI args passed through from sbatch
echo "Starting pipeline with args: $@"
python3 pipeline.py \
    --wandb-project fnlp-pipeline \
    "$@"

if [ $? -eq 0 ]; then
    echo "Pipeline complete."
else
    echo "Pipeline failed." && exit 1
fi
