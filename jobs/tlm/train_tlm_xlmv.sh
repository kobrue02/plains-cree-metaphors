#!/bin/bash
#SBATCH --job-name=TLM_XLMV_Cree_EN
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

# TLM fine-tune facebook/xlm-v-base on Cree-English pairs.
# XLM-V uses a 1M-token vocabulary (vs 250K for XLM-R), giving better
# subword coverage for morphologically rich low-resource languages.

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
echo "Starting TLM fine-tuning (xlm-v-base) on 1 x A100..."

python3 main.py \
    --fine-tune \
    --sentences-file data/sentences.txt \
    --model-name facebook/xlm-v-base \
    --epochs 15 \
    --batch-size 4 \
    --grad-accum 8 \
    --max-length 128 \
    --tlm-output-dir data/tlm_model_xlmv \
    --hub-model-id KonradBRG/xlm-v-base-plains-cree-en-tlm \
    --wandb-project fnlp-tlm

if [ $? -eq 0 ]; then
    echo "TLM XLM-V fine-tuning completed successfully."
else
    echo "TLM XLM-V fine-tuning failed with exit code $?."
    exit 1
fi
