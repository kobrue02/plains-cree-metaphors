#!/bin/bash
#SBATCH --job-name=AgreementEval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Computes agree_with_classifier: runs the production (non-CV) calibrated
# classifier over the full sentence pool, then compares its predictions
# against the Qwen silver labels (data/figurative/deepseek_labels_qwen_*.parquet).
#
# Usage:
#   sbatch jobs/agreement_eval.sh
#   sbatch jobs/agreement_eval.sh --checkpoint <other-ckpt>   # forwarded to predict_pool.py only

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"

export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $TORCH_EXTENSIONS_DIR $HF_HOME

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

CHECKPOINT="KonradBRG/xlm-mlm-plains-cree-en-figurative"
SILVER_FILE="data/figurative/deepseek_labels_qwen_qwen3.5-122b-a10b.parquet"
if [ "$1" = "--checkpoint" ]; then
  CHECKPOINT="$2"
  shift 2
fi

echo "=== predict_pool.py (checkpoint=$CHECKPOINT) ==="
python3 scripts/annotate/predict_pool.py --checkpoint "$CHECKPOINT" \
    --restrict-to "$SILVER_FILE" "$@"
if [ $? -ne 0 ]; then
  echo "predict_pool.py failed." && exit 1
fi

echo "=== deepseek_agreement_eval.py ==="
python3 scripts/annotate/deepseek_agreement_eval.py \
    --deepseek-file "$SILVER_FILE"
if [ $? -ne 0 ]; then
  echo "deepseek_agreement_eval.py failed." && exit 1
fi

echo "Done."
