"""EX per subset for a BIRD-dev + Spider-dev run. Failure details are shown for the tuning split only; locked test
subsets (BIRD mini-dev questions, Spider dev) get aggregate EX and outcome counts, never per-item reading.
    python scripts/xiyan_breakdown.py results/p1x_xiyan/<run> data/bird_dev_tune_ids.json"""
import json
import math
import sys
from collections import Counter

rows = [json.loads(l) for l in open(f"{sys.argv[1]}/per_item.jsonl", encoding="utf-8")]
print("gold_error items excluded:", [r["id"] for r in rows if r["outcome"] == "gold_error"])
rows = [r for r in rows if r["outcome"] != "gold_error"]
tune = set(json.load(open(sys.argv[2])))


def ex(sub):
    n = len(sub)
    p = sum(r["outcome"] == "correct" for r in sub) / n
    return f"EX={p:.2%} ±{1.96 * math.sqrt(p * (1 - p) / n):.1%} (n={n})"


bird = [r for r in rows if r["source"] == "bird"]
subsets = {"bird_dev (all, published set)": bird,
           "  bird_dev tuning split": [r for r in bird if r["id"] in tune],
           "  bird_dev rest (mini-dev questions) [LOCKED]": [r for r in bird if r["id"] not in tune],
           "spider_dev [LOCKED]": [r for r in rows if r["source"] == "spider"]}
for name, sub in subsets.items():
    print(f"{name:48} {ex(sub)}  outcomes={dict(Counter(r['outcome'] for r in sub))}")
t = [r for r in bird if r["id"] in tune]
print("tuning-split failure taxonomy:", dict(Counter(r["label"] for r in t if r["outcome"] != "correct").most_common()))
print("tuning-split by difficulty:", {d: f"{sum(r['outcome'] == 'correct' for r in t if r['difficulty'] == d) / max(1, sum(r['difficulty'] == d for r in t)):.1%}"
                                       for d in ("simple", "moderate", "challenging")})
print("tuning-split mean tokens:", round(sum(r["tokens"] for r in t) / len(t), 1),
      "cut:", sum(r["cut"] for r in t), "multiple statements:", sum(r.get("extract_status") == "multiple" for r in t))
