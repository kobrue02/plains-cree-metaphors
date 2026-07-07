#!/bin/bash
#SBATCH --job-name=Sweep_Agent
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
# For TLM/CLKD/pipeline sweeps (trials take ~1.5-3h) switch to:
#   --partition=gpu_a100_il  --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Run one sweep trial.  Submit multiple times for parallel search.
#
# Usage:
#   # Create the sweep first (run locally):
#   wandb sweep sweeps/calibrate.yaml       # → kobrue02/fnlp-tune/abc1xyz2
#
#   # Submit N agents (each agent runs 1 trial then exits):
#   for i in $(seq 1 5); do
#     sbatch jobs/sweep_agent.sh kobrue02/fnlp-tune/abc1xyz2
#   done
#
# Notes:
#   - Each agent exits after completing one trial (--count 1) so SLURM job
#     time is predictable.  Resubmit agents until the sweep is complete.
#   - TLM/CLKD trials need 4h; calibrate trials need ~1h (adjust --time).

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch jobs/sweep_agent.sh <entity/project/sweep_id>"
    exit 1
fi

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

# 4. Run one trial
echo "Starting sweep agent: $SWEEP_ID"
wandb agent --count 1 "$SWEEP_ID"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Trial complete."
else
    echo "Trial failed with exit code $STATUS." && exit $STATUS
fi
