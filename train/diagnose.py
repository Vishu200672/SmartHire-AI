"""
diagnose.py
-----------
Prints actual raw cosine scores for every pair so you can see the
true empirical distribution and set optimal calibration thresholds.

Run this ONCE after any model change (base or fine-tuned) to re-derive
the correct CALIBRATION_LOW and CALIBRATION_HIGH values.

Usage (run from SmartHireAI folder):
    python train/diagnose.py
    python train/diagnose.py --model models/smarthire-finetuned

Author: SmartHire AI
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE  = "train/training_data.json"


def main():
    parser = argparse.ArgumentParser(description="Diagnose raw cosine score distribution")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--data",  default=DATA_FILE)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("ERROR: Run  pip install sentence-transformers  first.")
        sys.exit(1)

    with open(args.data, "r") as f:
        pairs = json.load(f)
    pairs = [p for p in pairs if all(k in p for k in ("resume", "jd", "score"))]

    print(f"\nLoading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"\nComputing raw cosine scores for {len(pairs)} pairs...\n")
    print(f"{'Gold':>6}  {'Raw Cosine':>11}  Resume (snippet)")
    print("-" * 75)

    raw_scores  = []
    gold_scores = []

    for pair in pairs:
        embs = model.encode([pair["resume"], pair["jd"]], normalize_embeddings=True)
        raw  = float(np.dot(embs[0], embs[1]))
        raw_scores.append(raw)
        gold_scores.append(pair["score"])
        print(f"  {pair['score']:>4.2f}  {raw:>11.4f}  {pair['resume'][:50]}...")

    print("-" * 75)

    print(f"\n=== RAW COSINE DISTRIBUTION ===")
    sorted_r = sorted(raw_scores)
    n        = len(sorted_r)
    print(f"  Count  : {n}")
    print(f"  Min    : {min(raw_scores):.4f}")
    print(f"  Max    : {max(raw_scores):.4f}")
    print(f"  Mean   : {sum(raw_scores)/n:.4f}")
    print(f"  P10    : {sorted_r[max(0,int(n*0.10))]:.4f}")
    print(f"  P25    : {sorted_r[max(0,int(n*0.25))]:.4f}")
    print(f"  Median : {sorted_r[n//2]:.4f}")
    print(f"  P75    : {sorted_r[min(n-1,int(n*0.75))]:.4f}")
    print(f"  P90    : {sorted_r[min(n-1,int(n*0.90))]:.4f}")

    print(f"\n=== RAW SCORES BY GOLD TIER ===")
    buckets = [
        ("Highly Recommended (gold>=0.85)", lambda g: g >= 0.85),
        ("Recommended        (0.65-0.84)",  lambda g: 0.65 <= g < 0.85),
        ("Consider           (0.35-0.64)",  lambda g: 0.35 <= g < 0.65),
        ("Not Recommended    (gold<0.35)",  lambda g: g < 0.35),
    ]
    tier_buckets = {}
    for label, fn in buckets:
        bucket = [r for r, g in zip(raw_scores, gold_scores) if fn(g)]
        tier_buckets[label] = bucket
        if bucket:
            print(f"  {label}:")
            print(f"    n={len(bucket)}  min={min(bucket):.4f}  max={max(bucket):.4f}  "
                  f"mean={sum(bucket)/len(bucket):.4f}")

    print(f"\n=== RECOMMENDED CALIBRATION ===")
    not_rec  = [r for r, g in zip(raw_scores, gold_scores) if g < 0.35]
    high_rec = [r for r, g in zip(raw_scores, gold_scores) if g >= 0.85]
    if not_rec and high_rec:
        suggested_low  = round(max(not_rec), 4)
        suggested_high = round(min(high_rec), 4)
        overlap        = suggested_low >= suggested_high
        print(f"  CALIBRATION_LOW  = {suggested_low}   (max raw cosine of Not-Recommended pairs)")
        print(f"  CALIBRATION_HIGH = {suggested_high}   (min raw cosine of Highly-Recommended pairs)")
        print(f"  Window width     = {suggested_high - suggested_low:.4f}")
        if overlap:
            print(f"\n  WARNING: Tiers overlap! The model cannot separate them with calibration alone.")
            print(f"  SOLUTION: Run fine-tuning:  python train/finetune.py")
        else:
            print(f"\n  Copy these values into src/similarity.py and train/evaluate.py")
    print()


if __name__ == "__main__":
    main()
