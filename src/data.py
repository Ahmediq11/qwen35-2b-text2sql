"""Dataset loaders, schema serialization and practice splits for Spider 1.0 and BIRD.

Every loader returns items as dicts:
    {"id", "source", "split", "db_id", "db_path", "question", "evidence", "gold_sql"}
Locked test sets: spider/dev and bird_minidev (PLAN §0 rule 4). Practice pools: spider/train and bird/train.

    python src/data.py summary <data_root>                     # counts per split, missing DBs
    python src/data.py splits <data_root> <out.json> [seed]    # discover/confirm split of practice data by database
"""
import glob
import hashlib
import json
import os
import sqlite3
import sys

LOCKED = {("spider", "dev"), ("bird", "minidev")}


def _find(root, name):
    """first file called `name` under root (the official zips nest their folders differently)."""
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f"{name} under {root}")
    return hits[0]


def _db_dir(root, marker_db):
    """the folder whose children are <db_id>/<db_id>.sqlite, located via one known database."""
    return os.path.dirname(os.path.dirname(_find(root, f"{marker_db}.sqlite")))


def load_spider(data_root, split):
    root = os.path.join(data_root, "spider")
    files = {"dev": ["dev.json"], "train": ["train_spider.json", "train_others.json"]}[split]
    raw = [r for f in files for r in json.load(open(_find(root, f), encoding="utf-8"))]
    # dev databases live in database/ (newer zips also ship test_database/ with the same dbs)
    dbs = _db_dir(root, raw[0]["db_id"])
    return [{"id": f"spider_{split}_{i}", "source": "spider", "split": split, "db_id": r["db_id"],
             "db_path": os.path.join(dbs, r["db_id"], r["db_id"] + ".sqlite"), "question": r["question"],
             "evidence": "", "gold_sql": r["query"]} for i, r in enumerate(raw)]


def load_bird(data_root, split):
    if split == "minidev":
        root = os.path.join(data_root, "bird_minidev")
        raw = json.load(open(_find(root, "mini_dev_sqlite.json"), encoding="utf-8"))
    elif split == "dev":     # the full BIRD dev set (1,534), on which XiYan's numbers are published
        root = os.path.join(data_root, "bird_dev")
        raw = json.load(open(_find(root, "dev.json"), encoding="utf-8"))
    else:
        root = os.path.join(data_root, "bird_train")
        raw = json.load(open(_find(root, "train.json"), encoding="utf-8"))
    dbs = _db_dir(root, raw[0]["db_id"])
    return [{"id": f"bird_{split}_{r.get('question_id', i)}", "source": "bird", "split": split, "db_id": r["db_id"],
             "db_path": os.path.join(dbs, r["db_id"], r["db_id"] + ".sqlite"), "question": r["question"],
             "evidence": r.get("evidence") or "", "gold_sql": r["SQL"], "difficulty": r.get("difficulty")}
            for i, r in enumerate(raw)]


def load(data_root, name):
    """name: spider_dev | spider_train | bird_minidev | bird_train"""
    src, split = name.split("_", 1)
    return load_spider(data_root, split) if src == "spider" else load_bird(data_root, split)


# ---------------- schema serialization ----------------
_SCHEMA_CACHE = {}


def _cell(v, max_len=60):
    if isinstance(v, bytes):
        return "<blob>"
    s = repr(v) if isinstance(v, str) else str(v)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def schema_text(db_path, sample_rows=3):
    """CREATE TABLE DDL as stored in sqlite_master plus `sample_rows` example rows per table."""
    key = (db_path, sample_rows)
    if key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    parts = []
    tables = con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                         "ORDER BY rowid").fetchall()
    for name, ddl in tables:
        parts.append(ddl.strip() + ";")
        if sample_rows:
            cur = con.execute(f'SELECT * FROM "{name}" LIMIT {int(sample_rows)}')
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            if rows:
                parts.append(f"/* {len(rows)} example rows from {name}:\n" + " | ".join(cols) + "\n" +
                             "\n".join(" | ".join(_cell(v) for v in r) for r in rows) + "\n*/")
    con.close()
    _SCHEMA_CACHE[key] = "\n\n".join(parts)
    return _SCHEMA_CACHE[key]


# ---------------- practice splits ----------------
def practice_splits(data_root, seed=0, confirm_share=0.5):
    """discover/confirm split of the practice pools, by database: no db_id appears in both halves."""
    out = {"seed": seed, "confirm_share": confirm_share, "discover": [], "confirm": [], "dbs": {}}
    for name in ("spider_train", "bird_train"):
        items = load(data_root, name)
        for db in sorted({it["db_id"] for it in items}):
            h = int(hashlib.sha256(f"{seed}:{name}:{db}".encode()).hexdigest(), 16) / 2 ** 256
            out["dbs"][f"{name}:{db}"] = "confirm" if h < confirm_share else "discover"
        for it in items:
            out[out["dbs"][f"{name}:{it['db_id']}"]].append(it["id"])
    out["counts"] = {k: len(out[k]) for k in ("discover", "confirm")}
    return out


def _summary(data_root):
    for name in ("spider_dev", "spider_train", "bird_minidev", "bird_train"):
        try:
            items = load(data_root, name)
        except FileNotFoundError as e:
            print(f"{name}: missing ({e})")
            continue
        dbs = {it["db_id"]: it["db_path"] for it in items}
        missing = [d for d, p in dbs.items() if not os.path.exists(p)]
        lock = " [LOCKED TEST]" if tuple(name.split("_", 1)) in LOCKED else ""
        print(f"{name}{lock}: {len(items)} items, {len(dbs)} dbs, {len(missing)} missing dbs {missing[:5]}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        _summary(sys.argv[2])
    elif sys.argv[1] == "splits":
        s = practice_splits(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
        json.dump(s, open(sys.argv[3], "w"), indent=1)
        print(s["counts"], sum(v == "confirm" for v in s["dbs"].values()), "of", len(s["dbs"]), "dbs in confirm")
