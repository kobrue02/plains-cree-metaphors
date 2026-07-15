"""
Interactive CLI to manually spot-check a sample of the silver-labeled sentences
(data/figurative/deepseek_labels.parquet, the dictionary-grounded silver pool).

Unlike the gold set (all 228 reviewed, see verify_gold.py), the silver pool is
too large (9,613 sentences) to review exhaustively, so this draws a fixed
sample once and lets you work through it over multiple sittings.

Sampling is stratified by label by default (--n-per-class per label, so
idiom/metaphor/simile — 233/563/90 of 9,613 — aren't drowned out by literal).
This means the sample is NOT population-representative: --report gives both
the raw per-class agreement (how good DeepSeek is *within* each class) and a
population-weighted overall estimate (using the full silver set's actual
label proportions), so the two don't get conflated.

The sample is drawn once and saved to SAMPLE_FILE (refuses to redraw if it
already exists — pass --redraw to force a fresh sample, which discards any
review progress on the old one). Review progress is checkpointed to a
resume-safe JSONL cache, same pattern as verify_gold.py.

Usage:
  python scripts/annotate/verify_silver.py                  # review unverified rows
  python scripts/annotate/verify_silver.py --redo            # also re-review already-verified rows
  python scripts/annotate/verify_silver.py --only-label idiom
  python scripts/annotate/verify_silver.py --report          # print agreement stats only
"""

from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

SILVER_FILE = "data/figurative/deepseek_labels.parquet"
SAMPLE_FILE = "data/figurative/silver_verification_sample.parquet"
CACHE_JSONL = "data/figurative/silver_verification_cache.jsonl"
OUTPUT_FILE = "data/figurative/silver_verified.parquet"

LABELS = ["literal", "idiom", "metaphor", "simile"]
LABEL_KEYS = {"l": "literal", "i": "idiom", "m": "metaphor", "s": "simile"}


def build_sample(silver: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for label in LABELS:
        pool = silver[silver["deepseek_label"] == label]
        n = min(n_per_class, len(pool))
        parts.append(pool.sample(n=n, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle label order


def load_cache(path: str) -> dict[tuple[int, int], dict]:
    cache: dict[tuple[int, int], dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[(rec["paragraph_id"], rec["sentence_id"])] = rec
    return cache


def append_cache(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_report(sample: pd.DataFrame, cache: dict, silver: pd.DataFrame) -> None:
    verified = [cache[(pid, sid)] for pid, sid in zip(sample["paragraph_id"], sample["sentence_id"])
                if (pid, sid) in cache]
    if not verified:
        print("No sentences verified yet.")
        return

    df = pd.DataFrame(verified)
    n = len(df)
    agree = (df["human_label"] == df["deepseek_label"]).sum()
    print(f"\n{'='*64}")
    print(f"Verified {n}/{len(sample)} sampled sentences")
    print(f"Overall raw agreement (unweighted): {agree}/{n} ({agree/n:.1%})")
    print(f"{'='*64}")

    print("\nPer-class agreement (within the stratified sample):")
    per_class_rate = {}
    for label in LABELS:
        sub = df[df["deepseek_label"] == label]
        if sub.empty:
            continue
        rate = (sub["human_label"] == sub["deepseek_label"]).mean()
        per_class_rate[label] = rate
        print(f"  {label:10s}: {(sub['human_label'] == sub['deepseek_label']).sum():3d}/{len(sub):3d}  ({rate:.1%})")

    pop_counts = silver["deepseek_label"].value_counts()
    pop_total = pop_counts.sum()
    if per_class_rate and pop_total:
        weighted = sum(per_class_rate[l] * pop_counts.get(l, 0) for l in per_class_rate) / \
                   sum(pop_counts.get(l, 0) for l in per_class_rate)
        print(f"\nPopulation-weighted overall estimate (using full silver set's label "
              f"proportions instead of the stratified sample's): {weighted:.1%}")

    disagreements = df[df["human_label"] != df["deepseek_label"]]
    if not disagreements.empty:
        print(f"\nDisagreements ({len(disagreements)}):")
        for _, v in disagreements.iterrows():
            print(f"  [{v['paragraph_id']}:{v['sentence_id']}] "
                  f"deepseek={v['deepseek_label']!r} -> human={v['human_label']!r}"
                  + (f"   note: {v['note']}" if v.get("note") else ""))


def review(sample: pd.DataFrame, cache: dict, redo: bool) -> None:
    todo = sample if redo else sample[~sample.apply(
        lambda r: (r["paragraph_id"], r["sentence_id"]) in cache, axis=1)]
    if todo.empty:
        print("Nothing left to review. Pass --redo to re-review, or --report for stats.")
        return

    print(f"{len(todo)} sentence(s) to review "
          f"({len(sample) - len(todo)} already verified). Ctrl-C or 'q' to stop.\n")

    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        print(f"\n{'-'*70}")
        print(f"[{i}/{len(todo)}]  paragraph {row['paragraph_id']}, sentence {row['sentence_id']}")
        print(f"  Cree:     {row['text_cree']}")
        print(f"  English:  {row['text_en']}")
        print(f"  DeepSeek label: {row['deepseek_label']}")
        reasoning = str(row.get("reasoning", "")).strip()
        if reasoning:
            tail = reasoning[-500:]
            print(f"  DeepSeek reasoning (tail): ...{tail}")

        prompt = (f"  Accept '{row['deepseek_label']}'? "
                  f"[enter=yes / l/i/m/s=literal,idiom,metaphor,simile / q=quit]: ")
        try:
            resp = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return

        if resp == "q":
            print("Stopped.")
            return

        if resp == "":
            human_label = row["deepseek_label"]
        elif resp in LABEL_KEYS:
            human_label = LABEL_KEYS[resp]
        elif resp in LABELS:
            human_label = resp
        else:
            print(f"  Unrecognized input {resp!r}, treating as accept.")
            human_label = row["deepseek_label"]

        note = ""
        if human_label != row["deepseek_label"]:
            try:
                confirm = input(f"  -> change to '{human_label}' (was '{row['deepseek_label']}')? "
                                 f"[enter=confirm / n=keep '{row['deepseek_label']}']: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped.")
                return
            if confirm == "n":
                human_label = row["deepseek_label"]
            else:
                try:
                    note = input("  Note (optional, why you changed it): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nStopped.")
                    return

        rec = {
            "paragraph_id":    int(row["paragraph_id"]),
            "sentence_id":     int(row["sentence_id"]),
            "deepseek_label":  row["deepseek_label"],
            "human_label":     human_label,
            "note":            note,
        }
        append_cache(CACHE_JSONL, rec)
        cache[(rec["paragraph_id"], rec["sentence_id"])] = rec

    print("\nAll done for this pass.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--silver-file", default=SILVER_FILE)
    p.add_argument("--sample-file", default=SAMPLE_FILE)
    p.add_argument("--cache",       default=CACHE_JSONL)
    p.add_argument("--out",         default=OUTPUT_FILE)
    p.add_argument("--n-per-class", type=int, default=25,
                   help="Sentences to sample per label (default 25 -> 100 total)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--redraw",      action="store_true",
                   help="Discard the existing sample and draw a fresh one (loses review progress on the old sample)")
    p.add_argument("--redo",        action="store_true", help="Re-review already-verified rows too")
    p.add_argument("--only-label",  choices=LABELS, default=None,
                   help="Only review sentences DeepSeek assigned this label")
    p.add_argument("--report",      action="store_true", help="Print agreement stats only, no review")
    args = p.parse_args()

    silver = pd.read_parquet(args.silver_file)

    if args.redraw and os.path.exists(args.sample_file):
        os.remove(args.sample_file)
    if os.path.exists(args.sample_file):
        sample = pd.read_parquet(args.sample_file)
        print(f"Using existing sample -> {args.sample_file} ({len(sample)} sentences). "
              f"Pass --redraw for a fresh one.")
    else:
        sample = build_sample(silver, args.n_per_class, args.seed)
        sample.to_parquet(args.sample_file, index=False)
        print(f"Drew a fresh stratified sample ({len(sample)} sentences) -> {args.sample_file}")

    if args.only_label:
        sample = sample[sample["deepseek_label"] == args.only_label].reset_index(drop=True)

    cache = load_cache(args.cache)

    if args.report:
        print_report(sample, cache, silver)
        return

    review(sample, cache, redo=args.redo)
    print_report(sample, cache, silver)

    sample = sample.copy()
    sample["human_label"] = sample.apply(
        lambda r: cache.get((r["paragraph_id"], r["sentence_id"]), {}).get("human_label"), axis=1)
    sample["verified"] = sample["human_label"].notna()
    sample.to_parquet(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
