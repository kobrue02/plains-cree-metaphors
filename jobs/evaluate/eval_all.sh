#!/bin/bash
#SBATCH --job-name=Eval_All
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

# Runs scripts/evals/eval_all.py's validation task (the in-sample sanity
# check against the DeepSeek-annotated pool — see scripts/evals/eval_cv.py
# for the honest cross-validated numbers) across every CLKD/calibrated
# checkpoint. Models are loaded and freed one at a time to avoid OOM.
# No --task flag anymore — the script no longer accepts one; the idiom
# golden-set/figurative-rate/simile/consistency tasks it used to also run
# were removed along with their underlying diagnostics.
# Output: data/figurative/eval_validation_{full,gold}.parquet

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
echo "Starting full evaluation sweep..."

python3 scripts/evals/eval_all.py

if [ $? -eq 0 ]; then
    echo "Evaluation completed. Results saved under data/figurative/"
else
    echo "Evaluation failed with exit code $?."
    exit 1
fi
