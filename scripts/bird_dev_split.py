"""Overlap of BIRD mini-dev with the full BIRD dev set, matched by (db_id, normalized question); writes the tuning
split = BIRD dev items NOT in mini-dev (for choosing inference/edit settings), while mini-dev stays the locked test.
Uses questions and databases only, never model outputs.
    python scripts/bird_dev_split.py data data/bird_dev_tune_ids.json"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import data as D  # noqa: E402


def key(it):
    return it["db_id"], re.sub(r"\s+", " ", it["question"].strip().lower())


def sql_key(it):
    return it["db_id"], re.sub(r"\s+", " ", it["gold_sql"].strip().lower().rstrip(";"))


dev, mini = D.load(sys.argv[1], "bird_dev"), D.load(sys.argv[1], "bird_minidev")
mini_keys, mini_sql = {key(it) for it in mini}, {sql_key(it) for it in mini}
in_mini = [it for it in dev if key(it) in mini_keys]
near = [it for it in dev if key(it) not in mini_keys and sql_key(it) in mini_sql]   # same gold SQL, reworded question
print(f"extra dev items excluded because their gold SQL equals a mini-dev gold SQL: {len(near)}")
tune = [it["id"] for it in dev if key(it) not in mini_keys and sql_key(it) not in mini_sql]
print(f"bird_dev {len(dev)} items / {len({it['db_id'] for it in dev})} dbs; mini-dev {len(mini)} / "
      f"{len({it['db_id'] for it in mini})} dbs; mini-dev questions found in dev: {len(in_mini)}; "
      f"same db set: {set(it['db_id'] for it in dev) == set(it['db_id'] for it in mini)}")
print(f"tuning split (dev minus mini-dev): {len(tune)} items")
json.dump(tune, open(sys.argv[2], "w"))
