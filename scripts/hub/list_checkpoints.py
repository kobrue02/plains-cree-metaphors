"""List every KonradBRG Hub model repo relevant to this project, categorized so it's safe to decide what to delete before actually deleting anything."""

from __future__ import annotations
from huggingface_hub import HfApi

AUTHOR = "KonradBRG"


def categorize(repo_id: str) -> str:
    name = repo_id.split("/", 1)[-1]
    if "-alpha-" in name:
        return "alpha_sweep"
    if name.endswith("-tlm") or name.endswith("-clkd") or "-tlm-" in name or "-clkd-" in name:
        return "intermediate (tlm/clkd)"
    if "-abl-" in name and ("-calibrated" in name or "-figurative" in name):
        return "ablation final (legacy — CV-only ablation jobs shouldn't push these anymore)"
    if name in ("deberta-v3-base-figurative",):
        return "teacher"
    return "final/production"


def main() -> None:
    api = HfApi()
    models = [m.id for m in api.list_models(author=AUTHOR) if "plains-cree" in m.id or "figurative" in m.id]

    buckets: dict[str, list[str]] = {}
    for repo_id in sorted(models):
        buckets.setdefault(categorize(repo_id), []).append(repo_id)

    for cat, repos in buckets.items():
        print(f"\n=== {cat} ({len(repos)}) ===")
        for r in repos:
            print(f"  {r}")

    print(f"\nTotal: {len(models)} repos")


if __name__ == "__main__":
    main()
