#!/bin/bash
#SBATCH --job-name=Distill_Fig_KL
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Cross-lingual adaptation via binary KL distillation.
# The model's prediction on the English translation (teacher, temperature=2)
# is collapsed to binary (literal vs. figurative) and used to supervise the
# model's prediction on the Cree text (student).  Temperature=2 softens the
# teacher to hedge against cross-linguistic type mismatches.
# Requires the figurative model to already exist on the Hub.

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

# 4. Distil
echo "Running binary-KL distillation (temperature=2.0)..."

python3 main.py \
    --distill-figurative \
    --distill-mode binary_kl \
    --distill-checkpoint KonradBRG/xlm-r-plains-cree-en-tlm-figurative \
    --distill-output data/figurative/distilled_kl \
    --distill-temperature 2.0 \
    --batch-size 16 \
    --epochs 10 \
    --hub-model-id KonradBRG/xlm-r-plains-cree-en-tlm-figurative-kl \
    --wandb-project fnlp-figurative

if [ $? -ne 0 ]; then
    echo "Distillation failed."
    exit 1
fi

# 5. Evaluate on idiom golden test set
echo "Evaluating on Cree idiom golden set..."

python3 main.py \
    --eval-idioms \
    --figurative-checkpoint data/figurative/distilled_kl

echo "Done."
