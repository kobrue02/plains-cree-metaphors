#!/bin/bash
#SBATCH --job-name=Probe_Metaphor_Large
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

# Probing experiment: freeze the TLM-adapted XLM-R large encoder and train
# only the classification head. Runs both final-layer and layer-12 variants.
# Much faster than full fine-tuning (~1k trainable params vs ~560M).

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

# 4. Probe — final layer
echo "=== Probing: XLM-R large, final layer ==="
python3 main.py \
    --metaphor \
    --experiment tlm_last_layer \
    --encoder KonradBRG/xlm-r-large-plains-cree-en-tlm \
    --hub-model-id KonradBRG/xlm-r-large-plains-cree-en-tlm-probe-last \
    --freeze-encoder \
    --batch-size 32 \
    --epochs 10 \
    --wandb-project fnlp-metaphor

if [ $? -ne 0 ]; then
    echo "Probe (final layer) failed. Aborting."
    exit 1
fi

# 5. Probe — layer 12
echo "=== Probing: XLM-R large, layer 12 ==="
python3 main.py \
    --metaphor \
    --experiment tlm_layer_12 \
    --encoder KonradBRG/xlm-r-large-plains-cree-en-tlm \
    --hub-model-id KonradBRG/xlm-r-large-plains-cree-en-tlm-probe-layer12 \
    --freeze-encoder \
    --batch-size 32 \
    --epochs 10 \
    --wandb-project fnlp-metaphor

if [ $? -eq 0 ]; then
    echo "Both probing experiments completed successfully."
else
    echo "Probe (layer 12) failed with exit code $?."
fi
