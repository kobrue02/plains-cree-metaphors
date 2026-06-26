"""
Create HuggingFace Collections and add all trained models to them.
Run locally: python scripts/create_collections.py
Requires: huggingface-cli login (or HF_TOKEN env var set)

Collections created:
  1. Plains Cree TLM Encoders
  2. Plains Cree Figurative Language Classifiers
  3. Plains Cree CLKD Models
"""

from huggingface_hub import create_collection, add_collection_item, HfApi

AUTHOR = "KonradBRG"
NAMESPACE = AUTHOR

api = HfApi()

COLLECTIONS = [
    dict(
        title       = "Plains Cree TLM Encoders",
        description = "Multilingual encoders TLM-fine-tuned on Plains Cree–English pairs. Student foundations for CLKD figurative language transfer.",
        models = [
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-tlm",
            f"{AUTHOR}/glot500-base-plains-cree-en-tlm",
            f"{AUTHOR}/xlm-v-base-plains-cree-en-tlm",
        ],
    ),
    dict(
        title       = "Plains Cree Figurative Language Classifiers",
        description = "4-class figurative language detectors (literal/idiom/metaphor/simile) trained on VUA20+MAGPIE+FLUTE with TLM-adapted encoders.",
        models = [
            f"{AUTHOR}/deberta-v3-base-figurative",
            f"{AUTHOR}/xlm-r-plains-cree-en-tlm-figurative",
            f"{AUTHOR}/xlm-r-large-plains-cree-en-tlm-figurative",
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-figurative",
        ],
    ),
    dict(
        title       = "Plains Cree CLKD Models",
        description = "Figurative language classifiers distilled into Plains Cree via CLKD. DeBERTa-v3 teacher, three student architectures, two warmup strategies.",
        models = [
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12",
            f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-clkd-full",
            f"{AUTHOR}/glot500-base-plains-cree-en-clkd-direct",
            f"{AUTHOR}/glot500-base-plains-cree-en-clkd-tlm",
            f"{AUTHOR}/xlm-v-base-plains-cree-en-clkd-direct",
            f"{AUTHOR}/xlm-v-base-plains-cree-en-clkd-tlm",
        ],
    ),
]


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
