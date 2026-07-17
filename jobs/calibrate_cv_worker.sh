#!/bin/bash
#SBATCH --job-name=CalibrateCV
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

# Runs ONE CV fold. Submitted by jobs/calibrate_cv.sh, which submits 5 of
# these (one per fold) — not meant to be called directly. Used to run all 5
# folds sequentially in a single allocation to save on queue-wait +
# environment setup, but gpu_a100_short's walltime cap (30 min, down from
# the 2h30m this used to request) no longer leaves room for that — so each
# fold is now its own job instead.
#
# All args (including --holdout-fold, appended by the caller) are forwarded
# to pipeline.py as-is.

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

echo "Starting single-fold CV calibration with args: $@"
# --skip-tlm --skip-clkd are non-negotiable here, not just defaults — this
# script's only job is the calibration stage from an already-produced
# checkpoint. Without them pipeline.py would retrain TLM+CLKD from scratch.
python3 pipeline.py "$@" --skip-tlm --skip-clkd
if [ $? -ne 0 ]; then
  echo "Fold failed." && exit 1
fi

echo "Fold complete."
