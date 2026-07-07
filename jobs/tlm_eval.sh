#!/bin/bash
#SBATCH --job-name=TLM_Eval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Usage:
#   sbatch jobs/tlm_eval.sh
#   sbatch jobs/tlm_eval.sh --model data/tlm_xlm-mlm --n 500

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=$WORK/cache/huggingface
mkdir -p $HF_HOME

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

python scripts/evaluate/tlm_eval.py \
    --model KonradBRG/xlm-mlm-plains-cree-en-tlm \
    --baseline FacebookAI/xlm-mlm-100-1280 \
    --n 500 \
    "$@"
