"""Score a generation run: per_item.jsonl (outcome + failure label per answer) and summary.json (PLAN §4.4).

    python src/score.py results/<run_id> [--data-root data] [--workers 16]
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import data as D
import failures as F
import prompts as P
import sql_eval as E

_ITEMS = {}


def _init(data_root, dataset):
    _ITEMS.update({it["id"]: it for name in dataset.split(",") for it in D.load(data_root, name)})


def _one(args):
    g, thinking, family = args
    it = _ITEMS[g["id"]]
    text = P.answer_text(g["text"], family)
    try:
        r = E.score(text, it["gold_sql"], it["db_path"], thinking=bool(thinking), finish_reason=g["finish_reason"])
    except Exception as e:   # gold query failed: keep the item, mark it, never count it as a model error
        r = {"outcome": "gold_error", "pred_sql": None, "extract_status": None, "error": repr(e)}
    r["label"] = F.label(r, it["gold_sql"]) if r["outcome"] != "gold_error" else None
    return dict(g, **r, cut=g["finish_reason"] == "length", source=it["source"], db_id=it["db_id"],
                difficulty=it.get("difficulty"), gold_sql=it["gold_sql"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    gens = [json.loads(l) for l in open(os.path.join(a.run, "gen.jsonl"))]
    with Pool(a.workers, initializer=_init, initargs=(a.data_root, cfg["dataset"])) as pool:
        rows = pool.map(_one, [(g, cfg["thinking"], cfg["family"]) for g in gens], chunksize=4)
    rows = [r for r in rows if r["outcome"] != "gold_error"] + [r for r in rows if r["outcome"] == "gold_error"]
    with open(os.path.join(a.run, "per_item.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r["outcome"] != "gold_error"]
    summary = {"config": {k: cfg[k] for k in ("model", "profile", "dataset", "samples", "sampling", "dtype")},
               "gold_errors": len(rows) - len(ok), **E.summarize(ok)}
    fails = [r for r in ok if r["outcome"] != "correct"]
    summary["failure_taxonomy"] = dict(Counter(r["label"] for r in fails).most_common())
    summary["failure_taxonomy_share"] = {k: v / len(fails) for k, v in summary["failure_taxonomy"].items()} if fails else {}
    by = defaultdict(list)
    for r in ok:
        by[r.get("difficulty") or "all"].append(r)
    summary["EX_by_difficulty"] = {k: sum(r["outcome"] == "correct" for r in v) / len(v) for k, v in by.items()}
    summary["gpu_hours"] = cfg.get("wall_s", 0) / 3600
    summary["tok_per_s"] = cfg.get("tok_per_s")
    json.dump(summary, open(os.path.join(a.run, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("n", "EX", "cut_share", "no_sql", "syntax_error", "exec_error",
                                              "wrong_result", "mean_tokens", "failure_taxonomy", "gpu_hours")}))


if __name__ == "__main__":
    main()
