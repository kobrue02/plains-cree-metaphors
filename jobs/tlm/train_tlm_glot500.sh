#!/bin/bash
#SBATCH --job-name=TLM_Glot500_Cree_EN
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# TLM fine-tune cis-lmu/glot500-base on Cree-English pairs.
# Glot500 is XLM-R-based and already covers Plains Cree (crk) via GlotCorpus.
# TLM sharpens the Cree-English cross-lingual alignment for downstream CLKD.

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

# 4. Execute TLM Fine-tuning
echo "Starting TLM fine-tuning (glot500-base) on 1 x A100..."

python3 main.py \
    --fine-tune \
    --sentences-file data/sentences.txt \
    --model-name cis-lmu/glot500-base \
    --epochs 15 \
    --batch-size 16 \
    --tlm-output-dir data/tlm_model_glot500 \
    --hub-model-id KonradBRG/glot500-base-plains-cree-en-tlm \
    --wandb-project fnlp-tlm

if [ $? -eq 0 ]; then
    echo "TLM Glot500 fine-tuning completed successfully."
else
    echo "TLM Glot500 fine-tuning failed with exit code $?."
    exit 1
fi
