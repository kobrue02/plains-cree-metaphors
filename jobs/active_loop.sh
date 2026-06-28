#!/bin/bash
#SBATCH --job-name=ActiveLoop_Cree
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Active annotation loop: infer → DeepSeek annotate → retrain
#
# Usage:
#   sbatch jobs/active_loop.sh
#   sbatch jobs/active_loop.sh --skip-infer
#   sbatch jobs/active_loop.sh --no-retrain --max-annotate 500
#   sbatch jobs/active_loop.sh --push-to-hub KonradBRG/xlm-mlm-plains-cree-en-active-v1
#
# Requires DEEPSEEK_API_KEY in environment or .env file.

# 1. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"

# 2. Environment
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. DeepSeek API key — load from .env if not already set
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors
if [ -z "$DEEPSEEK_API_KEY" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -E '^DEEPSEEK_API_KEY=' "$PROJECT_ROOT/.env" | xargs)
fi
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set." && exit 1
fi

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

# 5. Run
echo "Starting active annotation loop with args: $@"
python3 scripts/active_loop.py "$@"

if [ $? -eq 0 ]; then
    echo "Active loop complete."
else
    echo "Active loop failed." && exit 1
fi
