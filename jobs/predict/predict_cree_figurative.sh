#!/bin/bash
#SBATCH --job-name=Predict_Figurative
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Zero-shot Cree figurative language detection (literal/idiom/metaphor).
# Also evaluates cross-lingual idiom transfer on the 11 Plains Cree idioms
# in data/idioms.txt by comparing predictions for the Cree and English sides.

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

# 4. Predict
echo "Running 3-class figurative detection on Bloomfield Cree sentences..."

python3 main.py \
    --predict-figurative \
    --figurative-checkpoint KonradBRG/xlm-r-plains-cree-en-tlm-figurative \
    --figurative-output data/bloomfield_figurative.csv

if [ $? -eq 0 ]; then
    echo "Done."
else
    echo "Prediction failed with exit code $?."
    exit 1
fi
