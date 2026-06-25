#!/bin/bash
#SBATCH --job-name=CLKD_XLMV_TLM
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

# CLKD with TLM-warmed XLM-V student.
# Compared to clkd_xlmv_direct.sh, this variant first sharpens
# cross-lingual alignment on the Cree-English parallel corpus via TLM.
#
# Teacher : KonradBRG/deberta-v3-base-figurative  (frozen, English)
# Student : KonradBRG/xlm-v-base-plains-cree-en-tlm  (TLM warmup)

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

# 4. Run CLKD
echo "Starting CLKD (XLM-V + TLM warmup)..."

python3 main.py \
    --distill-figurative \
    --distill-mode clkd \
    --distill-checkpoint KonradBRG/xlm-v-base-plains-cree-en-tlm \
    --distill-teacher KonradBRG/deberta-v3-base-figurative \
    --distill-freeze-layers 0 \
    --distill-output data/figurative/clkd_xlmv_tlm \
    --epochs 10 \
    --batch-size 16 \
    --hub-model-id KonradBRG/xlm-v-base-plains-cree-en-clkd-tlm \
    --wandb-project fnlp-figurative

if [ $? -eq 0 ]; then
    echo "CLKD (XLM-V + TLM) completed successfully."
else
    echo "CLKD (XLM-V + TLM) failed with exit code $?."
    exit 1
fi
