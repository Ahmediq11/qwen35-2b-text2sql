"""SFT data for the LoRA distillation run (PLAN §7, Phase 4): SynSQL-2.5M chain-of-thought answers rendered in the
exact prompt format we evaluate with, the CoT placed inside <think> so the model learns short, structured reasoning.

Filters: database file present; prompt (schema DDL + 3 rows, question, evidence, instruction) <= --max-prompt
tokens; CoT <= --max-cot tokens; the gold SQL runs on its database within 5 s and returns >= 1 row.
Decontamination: drop any item whose normalized question appears in Spider dev / BIRD dev / BIRD mini-dev, or whose
db_id equals one of their database names. Stratified by sql_complexity.

    python src/sft_data.py --synsql data/synsql --out data/sft/synsql_20k.jsonl --n 20000
"""
import argparse
import collections
import json
import os
import random
import re
import sqlite3
import sys
import time
import zipfile

import ijson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import prompts as P  # noqa: E402

SHARE = {"Simple": 0.2, "Moderate": 0.35, "Complex": 0.3, "Highly Complex": 0.15}


def norm(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def runs_ok(db_path, sql, timeout_s=5.0):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    t0 = time.time()
    con.set_progress_handler(lambda: int(time.time() - t0 > timeout_s), 10000)
    try:
        return len(con.execute(sql).fetchmany(1)) == 1
    except Exception:
        return False
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synsql", default="data/synsql")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model-dir", default="models/Qwen3.5-2B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--keep-prob", type=float, default=0.03, help="streaming pre-sample rate over the 2.5M items")
    ap.add_argument("--max-prompt", type=int, default=6000)
    ap.add_argument("--max-cot", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    rng = random.Random(a.seed)

    db_root = os.path.join(a.synsql, "databases")
    if not os.path.isdir(db_root):                      # extract into its own folder; read-only use afterwards
        with zipfile.ZipFile(os.path.join(a.synsql, "databases.zip")) as z:
            z.extractall(db_root)
    paths = {}
    for dp, _, fs in os.walk(db_root):
        for f in fs:
            if f.endswith(".sqlite"):
                paths[f[:-7]] = os.path.join(dp, f)

    test_q, test_db = set(), set()
    for name in ("spider_dev", "bird_dev", "bird_minidev"):
        for it in D.load(a.data_root, name):
            test_q.add(norm(it["question"]))
            test_db.add(it["db_id"])

    quota = {k: int(v * a.n) for k, v in SHARE.items()}
    got = collections.defaultdict(list)
    drop = collections.Counter()
    seen = 0
    with open(os.path.join(a.synsql, "data.json"), "rb") as f:
        for o in ijson.items(f, "item"):
            seen += 1
            if rng.random() > a.keep_prob:
                continue
            c = o.get("sql_complexity")
            if c not in quota or len(got[c]) >= quota[c]:
                drop["quota"] += 1
                continue
            if o["db_id"] in test_db or norm(o["question"]) in test_q:
                drop["contamination"] += 1
                continue
            db = paths.get(o["db_id"])
            if db is None:
                drop["no_db"] += 1
                continue
            item = {"question": o["question"], "evidence": o.get("external_knowledge") or "", "db_path": db,
                    "db_id": o["db_id"]}
            prompt = tok.apply_chat_template(P.messages(item, "qwen"), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
            if len(tok(prompt).input_ids) > a.max_prompt:
                drop["long_prompt"] += 1
                continue
            cot = o["cot"].strip()
            if len(tok(cot).input_ids) > a.max_cot:
                drop["long_cot"] += 1
                continue
            if not runs_ok(db, o["sql"]):
                drop["sql_fails_or_empty"] += 1
                continue
            target = cot + "\n</think>\n\n```sql\n" + o["sql"].strip() + "\n```"
            got[c].append({"prompt": prompt, "target": target, "db_id": o["db_id"], "complexity": c,
                           "style": o.get("question_style")})
            if all(len(got[k]) >= quota[k] for k in quota):
                break
            if seen % 200000 == 0:
                print(seen, {k: len(v) for k, v in got.items()}, dict(drop), flush=True)
    rows = [r for v in got.values() for r in v]
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {"seen": seen, "kept": len(rows), "by_complexity": {k: len(v) for k, v in got.items()},
             "dropped": dict(drop), "dbs": len({r["db_id"] for r in rows}),
             "test_dbs_overlap": len({r["db_id"] for r in rows} & test_db)}
    json.dump(stats, open(a.out.replace(".jsonl", ".stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
