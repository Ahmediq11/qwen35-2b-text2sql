"""Print a few failed answers of a run (practice data only) to sanity-check the scorer and the taxonomy.
    python scripts/show_errors.py results/<run> [label] [n]"""
import json
import sys

run, lab, n = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None), int(sys.argv[3]) if len(sys.argv) > 3 else 6
rows = [json.loads(l) for l in open(f"{run}/per_item.jsonl")]
assert all(r["id"].split("_")[1] == "train" for r in rows), "this script is for practice runs only"
shown = 0
for r in rows:
    if r["outcome"] == "correct" or (lab and r["label"] != lab):
        continue
    print(f"--- {r['id']} {r['outcome']} [{r['label']}] err={r.get('error')}")
    print("PRED:", (r.get("pred_sql") or "")[:300])
    print("GOLD:", r["gold_sql"][:300])
    shown += 1
    if shown >= n:
        break
