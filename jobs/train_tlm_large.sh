#!/bin/bash
#SBATCH --job-name=TLM_Large_Cree_EN
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

# TLM fine-tune xlm-roberta-large on Cree-English pairs.
# Large has 24 transformer layers so hidden_states[12] is a genuine mid-layer.

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
mkdir -p logs data/tlm_model_large

# 4. Execute TLM Fine-tuning (large)
echo "Starting TLM fine-tuning (xlm-roberta-large) on 1 x A100..."

python3 main.py \
    --fine-tune \
    --sentences-file data/sentences.txt \
    --model-name xlm-roberta-large \
    --epochs 10 \
    --batch-size 32 \
    --hub-model-id KonradBRG/xlm-r-large-plains-cree-en-tlm \
    --wandb-project fnlp-tlm

if [ $? -eq 0 ]; then
    echo "TLM large fine-tuning completed successfully."
else
    echo "TLM large fine-tuning failed with exit code $?."
fi
