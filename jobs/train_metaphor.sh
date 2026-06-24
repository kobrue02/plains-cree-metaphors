#!/bin/bash
#SBATCH --job-name=Metaphor_TLM
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Fine-tune TLM-adapted XLM-R on VUA20 metaphor detection.
# Runs two variants: final hidden layer and hidden_states[12].

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
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/FNLP
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

# 4. Train — final hidden layer
echo "=== Variant 1: TLM encoder, final hidden layer ==="
python3 experiments/run_metaphor.py train --experiment tlm_last_layer --batch_size 32

if [ $? -ne 0 ]; then
    echo "Variant 1 failed. Aborting."
    exit 1
fi

# 5. Train — hidden_states[12]
echo "=== Variant 2: TLM encoder, hidden_states[12] ==="
python3 experiments/run_metaphor.py train --experiment tlm_layer_12 --batch_size 32

if [ $? -ne 0 ]; then
    echo "Variant 2 failed."
    exit 1
fi

echo "Both variants completed successfully."
