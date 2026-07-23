"""Create HuggingFace Collections and add all trained models to them; run directly to (re)sync every known model into its collection."""

from __future__ import annotations
import argparse, sys

from huggingface_hub import create_collection, add_collection_item, HfApi

AUTHOR    = "KonradBRG"
NAMESPACE = AUTHOR

api = HfApi()

COLLECTIONS = [
    dict(
        stage       = "tlm",
        title       = "Plains Cree TLM Encoders",
        description = "Multilingual encoders TLM-fine-tuned on Plains Cree–English pairs. Student foundations for CLKD figurative language transfer.",
        models = [
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-tlm",
            f"{AUTHOR}/glot500-base-plains-cree-en-tlm",
            f"{AUTHOR}/xlm-v-base-plains-cree-en-tlm",
            f"{AUTHOR}/xlm-mlm-plains-cree-en-tlm",
            f"{AUTHOR}/xlm-v-plains-cree-en-tlm",
        ],
    ),
    dict(
        stage       = "figurative",
        title       = "Plains Cree Figurative Language Classifiers",
        description = "4-class figurative language detectors (literal/idiom/metaphor/simile) trained on VUA20+MAGPIE+FLUTE with TLM-adapted encoders.",
        models = [
            f"{AUTHOR}/deberta-v3-base-figurative",
            f"{AUTHOR}/xlm-r-plains-cree-en-tlm-figurative",
            f"{AUTHOR}/xlm-r-large-plains-cree-en-tlm-figurative",
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-tlm-figurative",
        ],
    ),
    dict(
        stage       = "clkd",
        title       = "Plains Cree CLKD Models",
        description = "Figurative language classifiers distilled into Plains Cree via CLKD. DeBERTa-v3 teacher, three student architectures, two warmup strategies.",
        models = [
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12",
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-clkd-full",
            f"{AUTHOR}/glot500-base-plains-cree-en-clkd-direct",
            f"{AUTHOR}/glot500-base-plains-cree-en-clkd-tlm",
            f"{AUTHOR}/xlm-v-base-plains-cree-en-clkd-direct",
            f"{AUTHOR}/xlm-v-base-plains-cree-en-clkd-tlm",
            f"{AUTHOR}/xlm-mlm-plains-cree-en-clkd",
            f"{AUTHOR}/xlm-v-plains-cree-en-clkd",
        ],
    ),
    dict(
        stage       = "calibrated",
        title       = "Plains Cree Calibrated Figurative Classifiers",
        description = "Final CLKD checkpoints calibrated on DeepSeek-annotated Bloomfield validation data — the base pipeline runs plus every jobs/ablation.sh condition.",
        models = [
            # base pipeline (pipeline.py --model-id xlm-mlm/glot500/xlm-v)
            f"{AUTHOR}/xlm-mlm-plains-cree-en-calibrated",
            f"{AUTHOR}/glot500-plains-cree-en-calibrated",
            f"{AUTHOR}/xlm-v-plains-cree-en-calibrated",
            # ablation study (jobs/ablation.sh)
            f"{AUTHOR}/xlm-mlm-abl-full-plains-cree-en-calibrated",
            f"{AUTHOR}/xlm-mlm-abl-no-tlm-plains-cree-en-calibrated",
            f"{AUTHOR}/xlm-mlm-abl-no-clkd-plains-cree-en-calibrated",
            f"{AUTHOR}/xlm-mlm-abl-neither-plains-cree-en-calibrated",
            f"{AUTHOR}/xlm-mlm-abl-mono-mlm-plains-cree-en-calibrated",
            f"{AUTHOR}/xlm-mlm-abl-tlm-contrastive-plains-cree-en-calibrated",
        ],
    ),
]

STAGE_TO_COLLECTION = {c["stage"]: c for c in COLLECTIONS}
_slug_cache: dict[str, str] = {}


def _collection_slug(stage: str) -> str:
    if stage not in _slug_cache:
        meta = STAGE_TO_COLLECTION[stage]
        collection = create_collection(
            title       = meta["title"],
            description = meta["description"],
            namespace   = NAMESPACE,
            private     = False,
            exists_ok   = True,
        )
        _slug_cache[stage] = collection.slug
    return _slug_cache[stage]


def add_model_to_collection(repo_id: str, stage: str) -> None:
    """Create (if needed) the collection for `stage` and add `repo_id` to it.
    Never raises — a collection add failing should not fail an otherwise-successful job."""
    if stage not in STAGE_TO_COLLECTION:
        print(f"[hub] no collection mapping for stage '{stage}', skipping {repo_id}")
        return
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        slug = _collection_slug(stage)
        add_collection_item(collection_slug=slug, item_id=repo_id, item_type="model", exists_ok=True)
        print(f"[hub] {repo_id} → collection '{STAGE_TO_COLLECTION[stage]['title']}'")
    except Exception as exc:
        print(f"[hub] collection add failed for {repo_id} — {exc}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=None, help="Only add this single repo_id to its collection")
    p.add_argument("--stage", default=None, choices=list(STAGE_TO_COLLECTION), help="Stage for --repo (required if --repo is set)")
    args = p.parse_args()

    if args.repo:
        if not args.stage:
            sys.exit("--stage is required when --repo is given")
        add_model_to_collection(args.repo, args.stage)
        sys.exit(0)

    for coll in COLLECTIONS:
        print(f"\nCreating collection: {coll['title']}")
        try:
            collection = create_collection(
                title       = coll["title"],
                description = coll["description"],
                namespace   = NAMESPACE,
                private     = False,
                exists_ok   = True,
            )
            slug = collection.slug
            _slug_cache[coll["stage"]] = slug
            print(f"  slug: {slug}")
        except Exception as exc:
            print(f"  ERROR creating collection: {exc}")
            continue

        for repo_id in coll["models"]:
            try:
                api.repo_info(repo_id=repo_id, repo_type="model")
                add_collection_item(
                    collection_slug = slug,
                    item_id         = repo_id,
                    item_type       = "model",
                    exists_ok       = True,
                )
                print(f"  ✓  {repo_id}")
            except Exception as exc:
                print(f"  ✗  {repo_id} — {exc}")
