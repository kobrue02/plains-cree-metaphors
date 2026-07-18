#!/bin/bash
# Submit all six ablation conditions for a given base model, going straight to
# 5-fold CV calibration (jobs/calibrate_cv.sh) rather than also producing a
# full-data "production" calibrated checkpoint — the paper only ever reports
# the CV numbers (scripts/evals/eval_cv.py), so a production pass for a
# non-final ablation arm would just burn compute and push an unused Hub
# checkpoint. Conditions that need fresh TLM/CLKD training submit that job
# first, then chain the CV calibration job onto it via --dependency so it
# doesn't start until training succeeds — no manual waiting required.
#
# The six conditions differ only in which checkpoint feeds calibration:
#
#   full            TLM → CLKD → 5-fold Calibrate  (uses existing checkpoints)
#   no_tlm          base → CLKD → 5-fold Calibrate (CLKD retrained from base model)
#   no_clkd         TLM → 5-fold Calibrate         (calibrate from TLM output)
#   neither         base → 5-fold Calibrate        (calibrate from base model directly)
#   mono_mlm        mono MLM → TLM → CLKD → 5-fold Calibrate
#   tlm_contrastive TLM + InfoNCE → CLKD → 5-fold Calibrate
#
# Usage (run from project root on the cluster):
#   bash jobs/ablation.sh
#   bash jobs/ablation.sh --base-model facebook/xlm-v-base --model-id xlm-v
#   bash jobs/ablation.sh --dry-run     # print commands without submitting
#
# --base-model/--model-id default to the FacebookAI/xlm-mlm-100-1280 lineage
# (xlm-mlm) that all prior ablation/alpha-sweep work has used. Point them at
# a different encoder once the encoder comparison (see
# scripts/evals/eval_cv.py's "... (base pipeline)" conditions) picks a winner
# other than xlm-mlm — note eval_cv.py's CV_CONDITIONS will also need a new
# set of "Ablation: ..." entries for that model_id at that point, since it
# currently only has the xlm-mlm-abl-* ones wired up.

DRY_RUN=0
BASE_MODEL="FacebookAI/xlm-mlm-100-1280"
MODEL_ID="xlm-mlm"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

TLM_CKPT="data/tlm_${MODEL_ID}"
TLM_HUB="KonradBRG/${MODEL_ID}-plains-cree-en-tlm"
CLKD_CKPT="data/clkd_${MODEL_ID}"
CLKD_HUB="KonradBRG/${MODEL_ID}-plains-cree-en-clkd"

# Sweep-best CLKD hyperparameters (data/sweep_best/konradbrg-uni_FNLP_vjqf6ngx.json,
# eval/kl_epoch=0.0575) — applied to every condition that trains a fresh CLKD
# stage (no_tlm/mono_mlm/tlm_contrastive below). "full" doesn't retrain CLKD
# here (it reuses whatever's already on the Hub for $MODEL_ID) — make sure
# that checkpoint was itself produced with these same flags, or "full" and
# the other three conditions aren't actually comparable.
CLKD_FLAGS=(--clkd-epochs 5 --clkd-lr 2.8635565931749224e-05 --clkd-temperature 4 --freeze-layers 6)

# Resolve checkpoint paths: prefer local dir, fall back to Hub. Only valid
# for checkpoints that already exist at script-submission time (full/no_clkd/
# neither below) — conditions that train a fresh checkpoint (no_tlm/mono_mlm/
# tlm_contrastive) can't use this, since the local dir won't exist until
# their training job finishes; those always target the Hub id directly,
# guaranteed present by the time the dependent CV job starts because
# --push-intermediates makes the training job push before it exits.
#
# This is a genuine leave-one-out ablation over the PRODUCTION recipe — i.e.
# no_clkd/neither deliberately reuse the same alpha=0.2-InfoNCE TLM checkpoint
# "full" uses, isolating "what does removing CLKD do to our actual released
# TLM," not a separately-trained alpha=0.0 vanilla baseline.
if [ -d "$TLM_CKPT" ]; then
  TLM_RESOLVED="$TLM_CKPT"
else
  TLM_RESOLVED="$TLM_HUB"
fi

if [ -d "$CLKD_CKPT" ]; then
  CLKD_RESOLVED="$CLKD_CKPT"
else
  CLKD_RESOLVED="$CLKD_HUB"
fi

echo "Base model : $BASE_MODEL"
echo "Model ID   : $MODEL_ID"

run_or_print() {
  local label="$1"; shift
  echo ""
  echo "── $label ──"
  echo "  $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

# Submits a pipeline.py training job with --skip-calibrate (always — see
# header) and echoes its SLURM job ID to stdout via --parsable, so the caller
# can chain the CV calibration job onto it. Everything else goes to stderr.
submit_train() {
  local label="$1"; shift
  echo "" >&2
  echo "── $label (train) ──" >&2
  echo "  sbatch --parsable $*" >&2
  if [ "$DRY_RUN" -eq 0 ]; then
    sbatch --parsable "$@"
  fi
}

# ── Condition A: full — reuses the existing CLKD checkpoint directly, no
# training job needed; 5-fold calibration is the only step. ────────────────
run_or_print "full (5-fold calibrate from existing CLKD checkpoint)" \
  bash jobs/calibrate_cv.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-full" \
    --calibrate-from "$CLKD_RESOLVED" \
    --calibrate-lr 5e-6 \
    --calibrate-epochs 15

# ── Condition B: no_tlm (base → CLKD → 5-fold calibrate) ────────────────────
# CLKD must be retrained from the base model, so train first (~2h), then the
# CV job waits on it via --dependency.
NO_TLM_JOB=$(submit_train "no_tlm (CLKD from base model)" \
  --job-name=abl_no_tlm --time=04:00:00 \
  jobs/pipeline.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-no-tlm" \
    --skip-tlm \
    --clkd-from "$BASE_MODEL" \
    --skip-calibrate \
    --push-intermediates \
    "${CLKD_FLAGS[@]}")
DEP_ARGS=()
[ -n "$NO_TLM_JOB" ] && DEP_ARGS=(--dependency "afterok:$NO_TLM_JOB")
run_or_print "no_tlm (5-fold calibrate, waits on training job ${NO_TLM_JOB:-N/A})" \
  bash jobs/calibrate_cv.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-no-tlm" \
    --calibrate-from "KonradBRG/${MODEL_ID}-abl-no-tlm-plains-cree-en-clkd" \
    --calibrate-lr 5e-6 \
    --calibrate-epochs 15 \
    "${DEP_ARGS[@]}"

# ── Condition C: no_clkd — reuses the existing (production, alpha=0.2) TLM
# checkpoint directly, same one "full" uses. Leave-one-out: this isolates
# what CLKD adds on top of the actual released TLM, not a separately-trained
# vanilla baseline. ──────────────────────────────────────────────────────────
run_or_print "no_clkd (5-fold calibrate from TLM checkpoint)" \
  bash jobs/calibrate_cv.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-no-clkd" \
    --calibrate-from "$TLM_RESOLVED" \
    --calibrate-lr 5e-6 \
    --calibrate-epochs 15

# ── Condition D: neither — reuses the base model directly. ──────────────────
run_or_print "neither (5-fold calibrate from base model directly)" \
  bash jobs/calibrate_cv.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-neither" \
    --calibrate-from "$BASE_MODEL" \
    --calibrate-lr 5e-6 \
    --calibrate-epochs 15

# ── Condition E: mono_mlm (mono MLM → TLM → CLKD → 5-fold calibrate) ────────
# Most expensive training job (~2h mono + ~2h TLM + ~2h CLKD); the CV job
# waits on it via --dependency.
MONO_JOB=$(submit_train "mono_mlm (Cree MLM warmup → TLM → CLKD)" \
  --job-name=abl_mono_mlm --time=08:00:00 \
  jobs/pipeline.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-mono-mlm" \
    --mono-mlm \
    --skip-calibrate \
    --push-intermediates \
    "${CLKD_FLAGS[@]}")
DEP_ARGS=()
[ -n "$MONO_JOB" ] && DEP_ARGS=(--dependency "afterok:$MONO_JOB")
run_or_print "mono_mlm (5-fold calibrate, waits on training job ${MONO_JOB:-N/A})" \
  bash jobs/calibrate_cv.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-mono-mlm" \
    --calibrate-from "KonradBRG/${MODEL_ID}-abl-mono-mlm-plains-cree-en-clkd" \
    --calibrate-lr 5e-6 \
    --calibrate-epochs 15 \
    "${DEP_ARGS[@]}"

# ── Condition F: tlm_contrastive (TLM+InfoNCE → CLKD → 5-fold calibrate) ────
CONTRASTIVE_JOB=$(submit_train "tlm_contrastive (TLM with InfoNCE alignment → CLKD)" \
  --job-name=abl_tlm_contrastive --time=08:00:00 \
  jobs/pipeline.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-tlm-contrastive" \
    --contrastive-alpha 0.1 \
    --skip-calibrate \
    --push-intermediates \
    "${CLKD_FLAGS[@]}")
DEP_ARGS=()
[ -n "$CONTRASTIVE_JOB" ] && DEP_ARGS=(--dependency "afterok:$CONTRASTIVE_JOB")
run_or_print "tlm_contrastive (5-fold calibrate, waits on training job ${CONTRASTIVE_JOB:-N/A})" \
  bash jobs/calibrate_cv.sh \
    --base-model "$BASE_MODEL" \
    --model-id "${MODEL_ID}-abl-tlm-contrastive" \
    --calibrate-from "KonradBRG/${MODEL_ID}-abl-tlm-contrastive-plains-cree-en-clkd" \
    --calibrate-lr 5e-6 \
    --calibrate-epochs 15 \
    "${DEP_ARGS[@]}"

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run complete — no jobs submitted."
else
  echo "All ablation jobs submitted (CV-only — no production calibration checkpoints). Monitor with: squeue -u \$USER"
fi
