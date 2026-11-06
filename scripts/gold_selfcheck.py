"""Sanity check of the scorer and the schema prompts: every gold query, scored as its own prediction, must be correct.
Also reports schema prompt sizes (characters) per dataset. Uses gold SQL only, never model outputs.

    python scripts/gold_selfcheck.py data spider_dev bird_minidev ...
"""
import os
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import data as D  # noqa: E402
import prompts as P  # noqa: E402
import sql_eval as E  # noqa: E402


def one(it):
    try:
        r = E.score("```sql\n" + it["gold_sql"] + "\n```", it["gold_sql"], it["db_path"])
        return r["outcome"], r.get("error"), it["id"]
    except Exception as e:
        return "gold_error", repr(e)[:120], it["id"]


if __name__ == "__main__":
    root = sys.argv[1]
    for name in sys.argv[2:]:
        items = D.load(root, name)
        with Pool(16) as p:
            res = p.map(one, items, chunksize=8)
        c = Counter(r[0] for r in res)
        bad = [r for r in res if r[0] != "correct"][:5]
        sizes = sorted(len(P.qwen_user(it)) for it in {it["db_id"]: it for it in items}.values())
        print(f"{name}: {dict(c)}  examples of non-correct: {bad}")
        print(f"  schema prompt chars per db: median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
