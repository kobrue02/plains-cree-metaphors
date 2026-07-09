#!/bin/bash
# Submit ONE SLURM job that runs all 5 fold-holdout calibration runs
# (cross-validation) for ONE condition, reusing an existing CLKD/TLM/base
# checkpoint — no need to redo TLM/CLKD.
#
# Each run excludes one cv_folds.parquet fold from training and is never
# pushed to the Hub (see pipeline.py --holdout-fold); they exist purely so
# scripts/evals/eval_cv.py can aggregate genuinely held-out predictions across
# the whole annotation pool. Run scripts/data/build_cv_folds.py once first.
#
# The 5 folds run sequentially inside jobs/calibrate_cv_worker.sh's single
# allocation (see that file), rather than as 5 independent sbatch jobs, to
# avoid paying queue-wait + environment setup 5 times over. Trades
# cross-fold parallelism for that — if your cluster has 5 GPUs free at once,
# the old 5-separate-jobs approach was faster wall-clock; this wins when
# queue wait is the dominant cost.
#
# Usage:
#   bash jobs/calibrate_cv.sh --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm-abl-full --calibrate-from KonradBRG/xlm-mlm-plains-cree-en-clkd
#   bash jobs/calibrate_cv.sh ... --dependency afterok:12345          # wait for a prerequisite job (e.g. fresh CLKD training)
#   bash jobs/calibrate_cv.sh ... --dry-run
#   bash jobs/calibrate_cv.sh ... --calibrate-lr 5e-6 --calibrate-epochs 15   # forwarded as-is
#
# Prints the submitted job ID to stdout (via sbatch --parsable) so callers
# (e.g. jobs/ablation.sh) can capture it — everything else goes to stderr.

DRY_RUN=0
ARGS=()
BASE_MODEL=""
MODEL_ID=""
CALIBRATE_FROM=""
DEPENDENCY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --calibrate-from) CALIBRATE_FROM="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$BASE_MODEL" ] || [ -z "$MODEL_ID" ] || [ -z "$CALIBRATE_FROM" ]; then
  echo "Usage: bash jobs/calibrate_cv.sh --base-model <hf-id> --model-id <id> --calibrate-from <ckpt> [--dependency afterok:<jobid>] [extra pipeline.py flags] [--dry-run]" >&2
  exit 1
fi

SBATCH_ARGS=(--job-name="cv_${MODEL_ID}" --parsable)
if [ -n "$DEPENDENCY" ]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

echo "" >&2
echo "── ${MODEL_ID} (5 folds, one job)${DEPENDENCY:+ [depends on $DEPENDENCY]} ──" >&2
echo "  sbatch ${SBATCH_ARGS[*]} jobs/calibrate_cv_worker.sh --base-model $BASE_MODEL --model-id $MODEL_ID --calibrate-from $CALIBRATE_FROM ${ARGS[*]}" >&2

if [ "$DRY_RUN" -eq 0 ]; then
  sbatch "${SBATCH_ARGS[@]}" jobs/calibrate_cv_worker.sh \
    --base-model "$BASE_MODEL" \
    --model-id "$MODEL_ID" \
    --calibrate-from "$CALIBRATE_FROM" \
    "${ARGS[@]}"
fi
