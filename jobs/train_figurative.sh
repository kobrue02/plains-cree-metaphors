#!/bin/bash
#SBATCH --job-name=Train_Figurative
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# 3-class figurative language detector (literal / idiom / metaphor).
# Trains on sentence-level VUA20 (aggregated from token labels) + MAGPIE idiom
# dataset, on top of the TLM-adapted XLM-R base encoder.

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

# 4. Train
echo "Starting 3-class figurative training (TLM base)..."

python3 main.py \
    --train-figurative \
    --figurative-experiment tlm_base \
    --batch-size 32 \
    --epochs 10 \
    --hub-model-id KonradBRG/xlm-r-plains-cree-en-tlm-figurative \
    --wandb-project fnlp-figurative

if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed with exit code $?."
    exit 1
fi
