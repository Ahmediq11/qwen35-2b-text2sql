"""List answers scored as gold_error in result runs.   python scripts/list_gold_errors.py results/p1_pilot/*"""
import glob
import json
import sys

for run in sys.argv[1:]:
    for f in glob.glob(f"{run}/per_item.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["outcome"] == "gold_error":
                print(run.split("/")[-1], r["id"], r["sample"], (r["error"] or "")[:200])
