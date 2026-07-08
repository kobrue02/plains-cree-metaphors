#!/bin/bash
#SBATCH --job-name=Eval_CV
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Honest, held-out evaluation via k-fold cross-validation (scripts/evals/eval_cv.py).
# Requires jobs/calibrate_cv.sh to have produced data/calibrated_{model_id}-fold{0..4}/
# for whichever conditions you want scored — conditions missing fold checkpoints
# are skipped, not failed.

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

# 5. Run evaluation
echo "=== Cross-validated evaluation ==="
python3 scripts/evals/eval_cv.py "$@"

echo "Results in data/figurative/eval_cv_full.csv"
echo "           data/figurative/eval_cv_gold.csv"
