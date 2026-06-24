#!/bin/bash
#SBATCH --job-name=Predict_Cree_DeBERTa
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

# Zero-shot metaphor detection on Plains Cree using an English DeBERTa model
# trained on VUA20 (tommyleo2077/deberta-v3-large-metaphor).
# No Cree-specific fine-tuning — pure cross-lingual transfer baseline.

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
echo "Running DeBERTa zero-shot on Cree sentences..."

python3 main.py \
    --predict-cree \
    --checkpoint tommyleo2077/deberta-v3-large-metaphor \
    --predict-output data/bloomfield_metaphors_deberta.csv

if [ $? -ne 0 ]; then
    echo "Prediction failed."
    exit 1
fi

# 5. Compare against annotations
echo "Comparing against LLM annotations..."

python3 main.py \
    --compare \
    --predict-output data/bloomfield_metaphors_deberta.csv

if [ $? -eq 0 ]; then
    echo "Done."
else
    echo "Comparison failed with exit code $?."
fi
