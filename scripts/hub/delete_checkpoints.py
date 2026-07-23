"""Delete stale Hub checkpoints (alpha-sweep runs, superseded ablation/CLKD/TLM variants), while explicitly keeping the specific checkpoints that still back published paper tables."""

from __future__ import annotations
import sys
from huggingface_hub import HfApi

AUTHOR = "KonradBRG"

TO_DELETE = [
    # alpha sweep
    "xlm-mlm-alpha-0p05-plains-cree-en-calibrated",
    "xlm-mlm-alpha-0p15-plains-cree-en-calibrated",
    "xlm-mlm-alpha-0p2-plains-cree-en-calibrated",
    "xlm-mlm-alpha-0p3-plains-cree-en-calibrated",
    "xlm-mlm-alpha-0p4-plains-cree-en-calibrated",
    "xlm-mlm-alpha-0p5-plains-cree-en-calibrated",
    "xlm-mlm-alpha-0p75-plains-cree-en-calibrated",
    "xlm-mlm-alpha-1p0-plains-cree-en-calibrated",
    # ablation-calibrated (legacy SFT-per-condition checkpoints; superseded by
    # CV-only ablation which never pushes a calibrated checkpoint)
    "xlm-mlm-abl-full-plains-cree-en-calibrated",
    "xlm-mlm-abl-mono-mlm-plains-cree-en-calibrated",
    "xlm-mlm-abl-neither-plains-cree-en-calibrated",
    "xlm-mlm-abl-no-clkd-plains-cree-en-calibrated",
    "xlm-mlm-abl-no-tlm-plains-cree-en-calibrated",
    "xlm-mlm-abl-tlm-contrastive-plains-cree-en-calibrated",
    # stale/superseded CLKD naming variants
    "glot500-base-plains-cree-en-clkd-direct",
    "glot500-base-plains-cree-en-clkd-tlm",
    "xlm-mlm-100-1280-plains-cree-en-clkd-frozen12",
    "xlm-mlm-100-1280-plains-cree-en-clkd-full",
    "xlm-v-base-plains-cree-en-clkd-direct",
    # TLM checkpoints superseded or already consumed by a kept CLKD checkpoint
    "glot500-base-plains-cree-en-tlm",
    "glot500-plains-cree-en-tlm",
    "xlm-mlm-100-1280-plains-cree-en-tlm-figurative",
    "xlm-mlm-abl-mono-mlm-plains-cree-en-tlm",
    "xlm-mlm-abl-tlm-contrastive-plains-cree-en-tlm",
    "xlm-r-large-plains-cree-en-tlm-figurative",
    "xlm-r-plains-cree-en-tlm",
    "xlm-r-plains-cree-en-tlm-figurative",
    "xlm-v-base-plains-cree-en-tlm",
    "xlm-v-plains-cree-en-tlm",
]


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    skip_confirm = "--yes" in sys.argv

    print(f"{len(TO_DELETE)} repos to delete:")
    for name in TO_DELETE:
        print(f"  {AUTHOR}/{name}")

    if dry_run:
        print("\nDry run — nothing deleted.")
        return

    if not skip_confirm:
        answer = input(f"\nDelete these {len(TO_DELETE)} repos from the Hub? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    api = HfApi()
    deleted, failed = 0, []
    for name in TO_DELETE:
        repo_id = f"{AUTHOR}/{name}"
        try:
            api.delete_repo(repo_id=repo_id, repo_type="model")
            print(f"deleted  {repo_id}")
            deleted += 1
        except Exception as e:
            print(f"FAILED   {repo_id} — {e}")
            failed.append(repo_id)

    print(f"\n{deleted}/{len(TO_DELETE)} deleted.")
    if failed:
        print("Failed:")
        for repo_id in failed:
            print(f"  {repo_id}")


if __name__ == "__main__":
    main()
