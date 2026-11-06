"""SQL extraction from model output and execution accuracy (EX) against SQLite (PLAN §4.3).

Outcome per answer (exactly one): correct | cut | no_sql | syntax_error | exec_error | timeout | wrong_result.
EX compares result multisets; row order matters only when the gold query has a top-level ORDER BY.
"""
import re
import sqlite3
import time
from collections import Counter

import sqlglot
from sqlglot import exp

FENCE_SQL = re.compile(r"```[ \t]*sql[^\n]*\n(.*?)```", re.S | re.I)
FENCE_ANY = re.compile(r"```[^\n]*\n(.*?)```", re.S)
BARE_SELECT = re.compile(r"\b((?:WITH\b|SELECT\b)[\s\S]*?)(?:;|\n\s*\n|$)", re.I)
SYNTAX_MARKERS = ("syntax error", "incomplete input", "unrecognized token", "near \"")


def answer_part(text, thinking=False):
    """(text after the last </think>, closed?). Thinking that never closes means the answer was cut. With thinking on,
    Qwen's chat template puts the opening <think> in the prompt, so the generated text may contain only </think>."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], True
    return ("", False) if (thinking or "<think>" in text) else (text, True)


def extract_sql(text, thinking=False):
    """returns (sql or None, status) with status in ok | cut | no_sql | multiple."""
    ans, closed = answer_part(text, thinking)
    if thinking and not closed:
        return None, "cut"
    blocks = FENCE_SQL.findall(ans)
    if not blocks:
        blocks = [b for b in FENCE_ANY.findall(ans) if re.search(r"\bSELECT\b", b, re.I)]
    if blocks:
        sql = blocks[-1].strip()
    else:
        sel = BARE_SELECT.findall(ans)
        if not sel:
            return None, "no_sql"
        sql = sel[-1].strip()
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return None, "no_sql"
    # more than one statement in the final block: keep the last one, flag it
    stmts = [s.strip() for s in _split_statements(sql) if s.strip()]
    if len(stmts) > 1:
        return stmts[-1], "multiple"
    return sql, "ok"


def _split_statements(sql):
    out, cur, q = [], [], None
    for ch in sql:
        if q:
            q = None if ch == q else q
        elif ch in "'\"`":
            q = ch
        elif ch == ";":
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return out


class QueryTimeout(Exception):
    pass


def execute(db_path, sql, timeout_s=30.0, row_cap=100_000):
    """run `sql` on a read-only connection; returns rows (list of tuples). Raises sqlite3.Error / QueryTimeout."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(row_cap + 1)
        return rows[:row_cap]
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e) and time.monotonic() > deadline:
            raise QueryTimeout(f"over {timeout_s}s") from None
        raise
    finally:
        con.close()


def has_top_level_order(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return bool(re.search(r"\bORDER\s+BY\b", sql.rsplit(")", 1)[-1], re.I))
    if isinstance(tree, exp.Union):   # ORDER BY of a compound select hangs off the union
        return tree.args.get("order") is not None
    return tree.args.get("order") is not None


def _norm(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, float):
        return round(v, 6)
    return v


def results_match(pred_rows, gold_rows, ordered):
    p = [tuple(_norm(v) for v in r) for r in pred_rows]
    g = [tuple(_norm(v) for v in r) for r in gold_rows]
    return p == g if ordered else Counter(p) == Counter(g)


_GOLD = {}


def gold_result(db_path, gold_sql, timeout_s=300.0):
    # 300 s: two BIRD mini-dev gold queries (518, 701) need more than 60 s; predictions keep the 30 s limit
    key = (db_path, gold_sql)
    if key not in _GOLD:
        _GOLD[key] = (execute(db_path, gold_sql, timeout_s), has_top_level_order(gold_sql))
    return _GOLD[key]


def score(text, gold_sql, db_path, thinking=False, finish_reason=None, timeout_s=30.0):
    """full scoring of one model answer -> dict with outcome, extracted sql, error message."""
    sql, st = extract_sql(text, thinking)
    out = {"pred_sql": sql, "extract_status": st, "error": None}
    if st == "cut" or (sql is None and finish_reason == "length"):
        out["outcome"] = "cut"
        return out
    if sql is None:
        out["outcome"] = "no_sql"
        return out
    gold_rows, ordered = gold_result(db_path, gold_sql)
    try:
        pred_rows = execute(db_path, sql, timeout_s)
    except QueryTimeout as e:
        out.update(outcome="timeout", error=str(e))
        return out
    except (sqlite3.Error, sqlite3.Warning, ValueError) as e:
        msg = str(e)
        syntax = any(m in msg.lower() for m in SYNTAX_MARKERS)
        out.update(outcome="syntax_error" if syntax else "exec_error", error=msg)
        return out
    out["outcome"] = "correct" if results_match(pred_rows, gold_rows, ordered) else "wrong_result"
    out["n_pred_rows"], out["n_gold_rows"] = len(pred_rows), len(gold_rows)
    return out


def tail_dup8(text, chars=2000):
    w = text[-chars:].split()
    grams = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def summarize(rows):
    """rows: per-answer dicts with outcome, tokens, cut (finish_reason == length), text."""
    n = len(rows)
    oc = Counter(r["outcome"] for r in rows)
    toks = sorted(r["tokens"] for r in rows)
    cut_rows = [r for r in rows if r["cut"]]
    return {"n": n, "EX": oc["correct"] / n, "cut_share": len(cut_rows) / n,
            "no_sql": oc["no_sql"] / n, "syntax_error": oc["syntax_error"] / n,
            "exec_error": (oc["exec_error"] + oc["timeout"]) / n, "timeout": oc["timeout"] / n,
            "wrong_result": oc["wrong_result"] / n, "outcomes": dict(oc),
            "mean_tokens": sum(toks) / n, "median_tokens": toks[n // 2],
            "mean_cut_tail_duplicate8": (sum(tail_dup8(r["text"]) for r in cut_rows) / len(cut_rows)
                                         if cut_rows else None)}
