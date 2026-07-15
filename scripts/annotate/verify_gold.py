"""
Interactive CLI to manually verify DeepSeek's gold-set annotations
(data/figurative/bloomfield_annotated.parquet, footnote_applies == True).

For each sentence, shows the Cree text, English translation, Bloomfield's
footnote (the actual evidence DeepSeek was given), and DeepSeek's assigned
label + rationale. You confirm or correct the label.

Resume-safe: verdicts are checkpointed to a jsonl cache keyed by
(paragraph_id, sentence_id), so you can quit (Ctrl-C or 'q') and pick back up
later without re-reviewing anything.

Usage:
  python scripts/annotate/verify_gold.py                  # review unverified rows
  python scripts/annotate/verify_gold.py --redo           # also re-review already-verified rows
  python scripts/annotate/verify_gold.py --only-idiom      # filter to one label first
  python scripts/annotate/verify_gold.py --report          # just print agreement stats, no review
"""

from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

GOLD_FILE   = "data/figurative/bloomfield_annotated.parquet"
CACHE_JSONL = "data/figurative/gold_verification_cache.jsonl"
OUTPUT_FILE = "data/figurative/gold_verified.parquet"

LABELS = ["literal", "idiom", "metaphor", "simile"]
LABEL_KEYS = {"l": "literal", "i": "idiom", "m": "metaphor", "s": "simile"}


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


def print_report(gold: pd.DataFrame, cache: dict) -> None:
    verified = [cache[(pid, sid)] for pid, sid in zip(gold["paragraph_id"], gold["sentence_id"])
                if (pid, sid) in cache]
    if not verified:
        print("No sentences verified yet.")
        return
    n = len(verified)
    agree = sum(1 for v in verified if v["human_label"] == v["deepseek_label"])
    print(f"\n{'='*60}")
    print(f"Verified {n}/{len(gold)} sentences")
    print(f"Human agrees with DeepSeek: {agree}/{n} ({agree/n:.1%})")
    print(f"{'='*60}")
    disagreements = [v for v in verified if v["human_label"] != v["deepseek_label"]]
    if disagreements:
        print("\nDisagreements:")
        for v in disagreements:
            print(f"  [{v['paragraph_id']}:{v['sentence_id']}] "
                  f"deepseek={v['deepseek_label']!r} -> human={v['human_label']!r}"
                  + (f"   note: {v['note']}" if v.get("note") else ""))


def review(gold: pd.DataFrame, cache: dict, redo: bool) -> None:
    todo = gold if redo else gold[~gold.apply(
        lambda r: (r["paragraph_id"], r["sentence_id"]) in cache, axis=1)]
    if todo.empty:
        print("Nothing left to review. Pass --redo to re-review, or --report for stats.")
        return

    print(f"{len(todo)} sentence(s) to review "
          f"({len(gold) - len(todo)} already verified). Ctrl-C or 'q' to stop.\n")

    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        print(f"\n{'-'*70}")
        print(f"[{i}/{len(todo)}]  {row['source_file']}  "
              f"(paragraph {row['paragraph_id']}, sentence {row['sentence_id']})")
        print(f"  Cree:     {row['text_cree']}")
        print(f"  English:  {row['text_en']}")
        if row.get("footnote_en"):
            print(f"  Footnote: {row['footnote_en']}")
        print(f"  DeepSeek label:     {row['label']}")
        print(f"  DeepSeek rationale: {row['rationale']}")

        prompt = (f"  Accept '{row['label']}'? "
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
            human_label = row["label"]
        elif resp in LABEL_KEYS:
            human_label = LABEL_KEYS[resp]
        elif resp in LABELS:
            human_label = resp
        else:
            print(f"  Unrecognized input {resp!r}, treating as accept.")
            human_label = row["label"]

        note = ""
        if human_label != row["label"]:
            try:
                confirm = input(f"  -> change to '{human_label}' (was '{row['label']}')? "
                                 f"[enter=confirm / n=keep '{row['label']}']: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped.")
                return
            if confirm == "n":
                human_label = row["label"]
            else:
                try:
                    note = input("  Note (optional, why you changed it): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nStopped.")
                    return

        rec = {
            "paragraph_id":    int(row["paragraph_id"]),
            "sentence_id":     int(row["sentence_id"]),
            "deepseek_label":  row["label"],
            "human_label":     human_label,
            "note":            note,
        }
        append_cache(CACHE_JSONL, rec)
        cache[(rec["paragraph_id"], rec["sentence_id"])] = rec

    print("\nAll done for this pass.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-file", default=GOLD_FILE)
    p.add_argument("--cache",     default=CACHE_JSONL)
    p.add_argument("--out",       default=OUTPUT_FILE)
    p.add_argument("--redo",       action="store_true", help="Re-review already-verified rows too")
    p.add_argument("--only-label", choices=LABELS, default=None,
                   help="Only review sentences DeepSeek assigned this label")
    p.add_argument("--report",    action="store_true", help="Print agreement stats only, no review")
    args = p.parse_args()

    gold = pd.read_parquet(args.gold_file)
    gold = gold[gold["footnote_applies"] == True].reset_index(drop=True)
    if args.only_label:
        gold = gold[gold["label"] == args.only_label].reset_index(drop=True)

    cache = load_cache(args.cache)

    if args.report:
        print_report(gold, cache)
        return

    review(gold, cache, redo=args.redo)
    print_report(gold, cache)

    gold["human_label"] = gold.apply(
        lambda r: cache.get((r["paragraph_id"], r["sentence_id"]), {}).get("human_label"), axis=1)
    gold["verified"] = gold["human_label"].notna()
    gold.to_parquet(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
