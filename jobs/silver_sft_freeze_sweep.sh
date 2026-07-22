#!/bin/bash
# Sweep freeze_n_layers = 0..16 (XLM-100's full transformer depth) for the
# hierarchical silver-SFT classifier (jobs/silver_sft_worker.sh, one sbatch
# job per n), then submit jobs/silver_sft_push_best.sh as a dependent job
# that reads data/figurative/silver_sft_sweep_results.parquet once every
# n-run has finished and pushes only the best-macro-F1 checkpoint to the Hub.
#
# Uses --dependency=afterany (not afterok): if one or two n-values fail for
# an unrelated reason (transient cluster issue, OOM, etc.), the push-best
# step still runs and just picks the best among whichever runs actually wrote
# a result — afterok would leave it queued forever if even one job failed.
#
# Usage (run from project root on the cluster):
#   bash jobs/silver_sft_freeze_sweep.sh
#   bash jobs/silver_sft_freeze_sweep.sh --dry-run     # print commands without submitting

DRY_RUN=0
for arg in "$@"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

run_or_print() {
  local label="$1"; shift
  echo "── $label ──" >&2
  echo "  $*" >&2
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

JOB_IDS=()
for N in $(seq 0 16); do
  JOB_ID=$(run_or_print "freeze_n_layers=${N}" \
    sbatch --parsable \
      --job-name="silver_n${N}" \
      jobs/silver_sft_worker.sh "$N")
  [ -n "$JOB_ID" ] && JOB_IDS+=("$JOB_ID")
done

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run complete — no jobs submitted."
  exit 0
fi

DEPENDENCY=$(IFS=:; echo "afterany:${JOB_IDS[*]}")
sbatch --dependency="$DEPENDENCY" jobs/silver_sft_push_best.sh

echo "All 17 sweep jobs submitted, push-best job queued after them."
echo "Monitor with: squeue -u \$USER"
