"""
End-to-end figurative language pipeline for Plains Cree.

Stages (all optional, all chainable):
  1. TLM        — Translation Language Modelling warmup on Cree-English pairs
  2. CLKD       — Cross-Lingual Knowledge Distillation from English DeBERTa teacher
  3. Calibrate  — Low-LR adjustment on DeepSeek-annotated Bloomfield validation set

Hub naming (derived from --model-id):
  TLM        → {hub-prefix}/{model-id}-plains-cree-en-tlm
  CLKD       → {hub-prefix}/{model-id}-plains-cree-en-clkd
  Calibrated → {hub-prefix}/{model-id}-plains-cree-en-calibrated

Examples:
  # Full pipeline, xlm-mlm base
  python pipeline.py --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm

  # XLM-V needs shorter max-length to avoid OOM
  python pipeline.py --base-model facebook/xlm-v-base --model-id xlm-v --max-length 128

  # Glot500, skip TLM — reuse the legacy TLM checkpoint (pushed under a
  # different --model-id than this run, so point CLKD at it explicitly;
  # --skip-tlm alone would look for the nonexistent KonradBRG/glot500-plains-cree-en-tlm)
  python pipeline.py --base-model cis-lmu/glot500-base --model-id glot500 \
      --skip-tlm --clkd-from KonradBRG/glot500-base-plains-cree-en-tlm

  # Same caveat applies to XLM-V if you ever skip its TLM stage:
  # --clkd-from KonradBRG/xlm-v-base-plains-cree-en-tlm

  # Resume from existing CLKD checkpoint, only calibrate
  python pipeline.py --model-id xlm-mlm --skip-tlm --skip-clkd
"""

from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funcs import fine_tune, figurative_distill, calibrate
from scripts.hub.push_model_cards import push_card_for
from scripts.hub.create_collections import add_model_to_collection

TEACHER        = "KonradBRG/deberta-v3-base-figurative"
SENTENCES_FILE = "data/sentences.parquet"
ANNOT_FILE     = "data/figurative/annotations.parquet"


def hub_id(prefix: str, model_id: str, stage: str) -> str:
    return f"{prefix}/{model_id}-plains-cree-en-{stage}"


def publish(repo_id: str, stage: str, base_model: str, **stage_kwargs) -> None:
    """Push a model card and add the checkpoint to its Hub collection.
    Called right after a stage pushes weights, so cards/collections stay current
    without a manual step. Best-effort — must not fail an otherwise-successful job."""
    push_card_for(repo_id, stage, base_model, **stage_kwargs)
    add_model_to_collection(repo_id, stage)


def local_dir(model_id: str, stage: str) -> str:
    return f"data/{stage}_{model_id}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="TLM → CLKD → Calibrate pipeline for Plains Cree figurative detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # identity
    p.add_argument("--base-model",  required=True,
                   help="Base HF model ID (e.g. FacebookAI/xlm-mlm-100-1280)")
    p.add_argument("--model-id",    required=True,
                   help="Short name used in Hub IDs and local dirs (e.g. xlm-mlm)")
    p.add_argument("--hub-prefix",  default="KonradBRG",
                   help="HF Hub user/org prefix (default: KonradBRG)")

    # stage skipping / additions
    p.add_argument("--mono-mlm",       action="store_true",
                   help="Stage 0: run monolingual Cree MLM before TLM (ablation E)")
    p.add_argument("--skip-tlm",       action="store_true", help="Skip TLM stage")
    p.add_argument("--skip-clkd",      action="store_true", help="Skip CLKD stage")
    p.add_argument("--skip-calibrate", action="store_true", help="Skip calibration stage")

    # ablation overrides — inject a checkpoint for clkd/calibration without touching main checkpoint dirs
    p.add_argument("--clkd-from",      default=None, metavar="CKPT",
                   help="Override the input checkpoint for CLKD (e.g. the base model)")
    p.add_argument("--calibrate-from", default=None, metavar="CKPT",
                   help="Override the input checkpoint for calibration (e.g. TLM or base model)")

    # push behaviour
    p.add_argument("--push-intermediates", action="store_true",
                   help="Also push TLM and CLKD checkpoints to Hub (calibrated is always pushed)")

    # shared hyperparams
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--max-length",  type=int,   default=256,
                   help="Max token length — set 128 for XLM-V to avoid OOM")
    p.add_argument("--wandb-project", default=None)

    # TLM
    p.add_argument("--tlm-epochs",  type=int,   default=15)
    p.add_argument("--tlm-lr",      type=float, default=2e-5)
    p.add_argument("--grad-accum",  type=int,   default=2)
    p.add_argument("--sentences-file", default=SENTENCES_FILE)
    p.add_argument("--contrastive-alpha",       type=float, default=0.0,
                   help="Weight on InfoNCE alignment loss during TLM (0 = pure TLM)")
    p.add_argument("--contrastive-temperature", type=float, default=0.05,
                   help="Softmax temperature for InfoNCE (default: 0.05)")

    # CLKD
    p.add_argument("--clkd-epochs",      type=int,   default=10)
    p.add_argument("--clkd-lr",          type=float, default=5e-6)
    p.add_argument("--clkd-temperature", type=float, default=2.0)
    p.add_argument("--freeze-layers",    type=int,   default=0)
    p.add_argument("--teacher",          default=TEACHER)
    p.add_argument("--corpus-file",      default="data/bloomfield_texts_sentences.parquet")

    # calibration
    p.add_argument("--calibrate-epochs", type=int,   default=10)
    p.add_argument("--calibrate-lr",     type=float, default=5e-6)
    p.add_argument("--literal-ratio",    type=int,   default=3,
                   help="Literals per figurative sentence in calibration data (default: 3)")
    p.add_argument("--annot-file",       default=ANNOT_FILE)
    p.add_argument("--gold-only",        action="store_true",
                   help="Calibrate on footnote_applies=True sentences only")
    p.add_argument("--holdout-fold",     type=int, default=None, metavar="K",
                   help="Exclude cross-validation fold K from calibration training "
                        "(see scripts/data/build_cv_folds.py). Used for honest CV "
                        "evaluation, not for the published model — output is kept "
                        "local and never pushed to the Hub.")

    args = p.parse_args()

    prefix   = args.hub_prefix
    mid      = args.model_id

    # ── derived paths ──────────────────────────────────────────────────────────
    mono_local  = local_dir(mid, "mono")
    tlm_local   = local_dir(mid, "tlm")
    clkd_local  = local_dir(mid, "clkd")
    cal_local   = local_dir(mid, "calibrated")

    tlm_hub  = hub_id(prefix, mid, "tlm")
    clkd_hub = hub_id(prefix, mid, "clkd")
    cal_hub  = hub_id(prefix, mid, "calibrated")

    # ── Stage 0: Monolingual Cree MLM (ablation only) ─────────────────────────
    # if run, stage 1 starts from this checkpoint instead of the base model
    tlm_base = args.base_model
    if args.mono_mlm:
        print(f"\n{'='*60}\n  Stage 0 · Mono MLM  ({args.base_model})\n{'='*60}")
        fine_tune(
            sentences_file=args.sentences_file,
            model_name=args.base_model,
            output_dir=mono_local,
            epochs=args.tlm_epochs,
            learning_rate=args.tlm_lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
            wandb_project=args.wandb_project,
            src_col="text_cree",
            tgt_col="text_cree",  # Cree-only: no cross-lingual signal
        )
        tlm_base = mono_local

    # ── Stage 1: TLM ──────────────────────────────────────────────────────────
    if not args.skip_tlm:
        print(f"\n{'='*60}\n  Stage 1 · TLM  ({tlm_base})\n{'='*60}")
        fine_tune(
            sentences_file=args.sentences_file,
            model_name=tlm_base,
            output_dir=tlm_local,
            epochs=args.tlm_epochs,
            learning_rate=args.tlm_lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
            hub_model_id=tlm_hub if args.push_intermediates else None,
            wandb_project=args.wandb_project,
            contrastive_alpha=args.contrastive_alpha,
            contrastive_temperature=args.contrastive_temperature,
        )
        tlm_ckpt = tlm_local
        if args.push_intermediates:
            publish(
                tlm_hub, "tlm", args.base_model,
                epochs=args.tlm_epochs, batch_size=args.batch_size, max_length=args.max_length,
                warm_start=mono_local if args.mono_mlm else None,
                contrastive_alpha=args.contrastive_alpha,
                contrastive_temperature=args.contrastive_temperature,
            )
    else:
        tlm_ckpt = tlm_local if os.path.isdir(tlm_local) else tlm_hub
        print(f"\nSkipping TLM — using checkpoint: {tlm_ckpt}")

    # ── Stage 2: CLKD ─────────────────────────────────────────────────────────
    if not args.skip_clkd:
        clkd_input = args.clkd_from if args.clkd_from else tlm_ckpt
        print(f"\n{'='*60}\n  Stage 2 · CLKD  ({clkd_input})\n{'='*60}")
        figurative_distill(
            checkpoint=clkd_input,
            teacher_checkpoint=args.teacher,
            freeze_n_layers=args.freeze_layers,
            mode="clkd",
            corpus_file=args.corpus_file,
            epochs=args.clkd_epochs,
            batch_size=args.batch_size,
            learning_rate=args.clkd_lr,
            temperature=args.clkd_temperature,
            max_length=args.max_length,
            output_dir=clkd_local,
            hub_model_id=clkd_hub if args.push_intermediates else None,
            wandb_project=args.wandb_project,
        )
        clkd_ckpt = clkd_local
        if args.push_intermediates:
            publish(
                clkd_hub, "clkd", args.base_model,
                student_base=clkd_input, teacher=args.teacher,
                freeze_n_layers=args.freeze_layers, temperature=args.clkd_temperature,
                epochs=args.clkd_epochs, tlm_warmup=(clkd_input != args.base_model),
            )
    else:
        clkd_ckpt = clkd_local if os.path.isdir(clkd_local) else clkd_hub
        print(f"\nSkipping CLKD — using checkpoint: {clkd_ckpt}")

    # ── Stage 3: Calibrate ────────────────────────────────────────────────────
    if not args.skip_calibrate:
        is_cv_run = args.holdout_fold is not None
        if is_cv_run:
            cal_local = f"{cal_local}-fold{args.holdout_fold}"

        cal_input = args.calibrate_from if args.calibrate_from else clkd_ckpt
        print(f"\n{'='*60}\n  Stage 3 · Calibrate  ({cal_input})\n{'='*60}")
        calibrate(
            checkpoint=cal_input,
            output_dir=cal_local,
            hub_model_id=None if is_cv_run else cal_hub,
            annot_file=args.annot_file,
            epochs=args.calibrate_epochs,
            batch_size=min(args.batch_size, 8),
            learning_rate=args.calibrate_lr,
            literal_ratio=args.literal_ratio,
            max_length=min(args.max_length, 128),
            gold_only=args.gold_only,
            wandb_project=args.wandb_project,
            holdout_fold=args.holdout_fold,
        )
        if not is_cv_run:
            publish(
                cal_hub, "calibrated", args.base_model,
                student_base=cal_input, literal_ratio=args.literal_ratio,
                gold_only=args.gold_only, epochs=args.calibrate_epochs,
                learning_rate=args.calibrate_lr,
            )

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"  Calibrated model → {cal_hub}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
