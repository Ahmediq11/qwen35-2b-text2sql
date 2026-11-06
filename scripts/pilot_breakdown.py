"""EX per source with a 95% interval (items as the unit; samples averaged per item) and failure labels, for pilot runs.
    python scripts/pilot_breakdown.py results/p1_pilot/qwen35-2b_nothink results/p1_pilot/qwen35-2b_think"""
import json
import math
import sys
from collections import Counter, defaultdict

for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{run}/per_item.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["outcome"] != "gold_error"]
    print(f"== {run.split('/')[-1]}")
    for src in ("spider", "bird", None):
        sub = [r for r in rows if src is None or r["source"] == src]
        per = defaultdict(list)
        for r in sub:
            per[r["id"]].append(r["outcome"] == "correct")
        acc = [sum(v) / len(v) for v in per.values()]
        m = sum(acc) / len(acc)
        sd = math.sqrt(sum((a - m) ** 2 for a in acc) / (len(acc) - 1))
        ci = 1.96 * sd / math.sqrt(len(acc))
        labels = Counter(r["label"] for r in sub if r["outcome"] != "correct")
        print(f"  {src or 'all':6} items={len(acc):3}  EX={m:.0%} ±{ci:.0%}  failures={dict(labels.most_common())}")
