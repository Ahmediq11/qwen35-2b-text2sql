"""Paired comparison of generation runs on the same ids (sample 0): EX, cut share, rescues/losses, paired SE.
    python scripts/paired.py <run_dir>/gen <base_tag> <tag>..."""
import json
import math
import sys
from collections import Counter

root, base, tags = sys.argv[1], sys.argv[2], sys.argv[3:]


def rows(tag):
    out = {}
    for l in open(f"{root}/{tag}/per_item.jsonl"):
        r = json.loads(l)
        if r["outcome"] != "gold_error" and r["sample"] == 0:
            out[r["id"]] = r
    return out


b = rows(base)
print(f"{'run':14} {'n':>4} {'EX':>7} {'cut':>6} {'mean tok':>9} {'gain':>8} {'±SE':>6} {'rescue':>6} {'loss':>5}  "
      f"rescued-from / new failures")
for t in [base] + tags:
    e = rows(t)
    ids = sorted(set(b) & set(e))
    x = [(b[i]["outcome"] == "correct", e[i]["outcome"] == "correct") for i in ids]
    d = [int(y) - int(z) for z, y in x]
    n = len(ids)
    mean = sum(d) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1) / n) if n > 1 else 0
    resc = [i for i in ids if e[i]["outcome"] == "correct" and b[i]["outcome"] != "correct"]
    loss = [i for i in ids if b[i]["outcome"] == "correct" and e[i]["outcome"] != "correct"]
    ex = sum(z for _, z in x) / n
    cut = sum(e[i]["cut"] for i in ids) / n
    tok = sum(e[i]["tokens"] for i in ids) / n
    print(f"{t:14} {n:4} {ex:7.1%} {cut:6.1%} {tok:9.0f} {mean:+8.1%} {se:6.1%} {len(resc):6} {len(loss):5}  "
          f"{dict(Counter(b[i]['outcome'] for i in resc))} / {dict(Counter(e[i]['outcome'] for i in loss))}")
