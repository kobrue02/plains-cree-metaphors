#!/bin/bash
#SBATCH --job-name=Eval_ZeroShot
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Zero-shot evaluation suite (no labelled Cree data required):
#   (1) eval_consistency.py  — English-Cree label agreement vs DeBERTa teacher
#   (2) eval_figurative_rate.py — predicted label distribution on Bloomfield corpus
#   (3) eval_simile_detection.py — tâpiskôc silver-standard simile test

# 1. Project root (defined first — cache paths below anchor to it, not $WORK,
#    which is unset in the batch environment on this cluster)
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors

# 2. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 3. Environment Variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $TORCH_EXTENSIONS_DIR $HF_HOME

# 4. Project Setup
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

# 4. Run evaluations
echo "=== (1) English-Cree Consistency ==="
python3 scripts/evals/eval_all.py --task consistency

echo "=== (2) Figurative Rate on Bloomfield ==="
python3 scripts/evals/eval_all.py --task figurative-rate

echo "=== (3) Simile Detection (tâpiskôc) ==="
python3 scripts/evals/eval_all.py --task simile

echo "All evaluations complete."
echo "Results in data/figurative/eval_consistency.csv"
echo "           data/figurative/eval_figurative_rate.csv"
echo "           data/figurative/eval_simile_detection.csv"
