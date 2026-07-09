#!/bin/bash
#SBATCH --job-name=CalibrateCV
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Runs all 5 CV folds sequentially in ONE job allocation. Submitted by
# jobs/calibrate_cv.sh, which is the thing to actually call — not this file
# directly. Trades cross-fold parallelism (5 GPUs at once) for paying
# queue-wait + environment setup (module load, uv sync) once instead of 5
# times; worth it when queue wait dominates fold runtime on your cluster.
#
# All args are forwarded to pipeline.py for every fold, with --holdout-fold
# appended per iteration.

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

echo "Starting 5-fold CV calibration with args: $@"
for FOLD in 0 1 2 3 4; do
  echo ""
  echo "═══ Fold $FOLD ═══"
  # --skip-tlm --skip-clkd are non-negotiable here, not just defaults — this
  # script's only job is the calibration stage from an already-produced
  # checkpoint. Without them pipeline.py would retrain TLM+CLKD from scratch
  # for every one of the 5 folds.
  python3 pipeline.py "$@" --skip-tlm --skip-clkd --holdout-fold "$FOLD"
  if [ $? -ne 0 ]; then
    echo "Fold $FOLD failed." && exit 1
  fi
done

echo "All 5 folds complete."
