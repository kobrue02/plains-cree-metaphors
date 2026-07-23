#!/bin/bash
#SBATCH --job-name=EvalAll
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Runs scripts/evals/eval_all.py's gold-subset task, optionally restricted to
# one model via --model (safe to use now — eval_all.py upserts by model name
# instead of overwriting the whole output file).
#
# Usage: sbatch jobs/eval_all_worker.sh ["Model Name"]
#   sbatch jobs/eval_all_worker.sh                                    # every model in _VALIDATION_MODELS
#   sbatch jobs/eval_all_worker.sh "TLM+Silver-SFT (hierarchical)"    # just this one

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

if [ -n "${1:-}" ]; then
    python scripts/evals/eval_all.py --model "$1"
else
    python scripts/evals/eval_all.py
fi
