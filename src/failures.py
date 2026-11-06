"""Failure taxonomy (PLAN §4.5): one primary cause per failed answer, rules first.

loop/cut | no_sql/format | dialect | schema_hallucination | join_logic | aggregation/grouping | filter_value | other
"""
import re

import sqlglot
from sqlglot import exp

# functions / syntax that exist in MySQL or PostgreSQL but not in SQLite
NON_SQLITE_FUNCS = {"date_format", "datediff", "now", "curdate", "year", "month", "day", "concat_ws", "if",
                    "isnull", "to_char", "to_date", "extract", "date_part", "string_agg", "array_agg", "regexp_like",
                    "ilike", "left", "right", "timestampdiff", "str_to_date", "datepart", "getdate", "len", "top"}
DIALECT_PATTERNS = [r"::\s*\w+", r"\bILIKE\b", r"\bTOP\s+\d+\b", r"\bFETCH\s+FIRST\b", r"\bINTERVAL\s+'?\d",
                    r"\bSELECT\s+TOP\b"]


def _tables(tree):
    return {t.name.lower() for t in tree.find_all(exp.Table)}


def _literals(tree):
    return {str(l.this).lower() for l in tree.find_all(exp.Literal)}


def _has(tree, kind):
    return any(True for _ in tree.find_all(kind))


def label(rec, gold_sql):
    """rec: a scored answer (outcome, pred_sql, error, extract_status). Returns a taxonomy label or None if correct."""
    oc = rec["outcome"]
    if oc == "correct":
        return None
    if oc == "cut":
        return "loop/cut"
    if oc == "no_sql" or rec.get("extract_status") == "multiple":
        return "no_sql/format"
    sql = rec.get("pred_sql") or ""
    err = (rec.get("error") or "").lower()
    if oc in ("syntax_error", "exec_error", "timeout"):
        m = re.search(r"no such function: (\w+)", err)
        if m or any(re.search(p, sql, re.I) for p in DIALECT_PATTERNS):
            return "dialect"
        if "no such table" in err or "no such column" in err or "ambiguous column" in err:
            return "schema_hallucination"
        if oc == "syntax_error":
            try:
                sqlglot.parse_one(sql, read="sqlite")
            except Exception:
                for d in ("mysql", "postgres", "tsql"):
                    try:
                        sqlglot.parse_one(sql, read=d)
                        return "dialect"
                    except Exception:
                        pass
            return "other"
        return "other"
    # wrong_result: compare the structure of prediction and gold
    try:
        p, g = sqlglot.parse_one(sql, read="sqlite"), sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return "other"
    if _tables(p) != _tables(g) or _has(p, exp.Join) != _has(g, exp.Join):
        return "join_logic"
    aggs = (exp.AggFunc, exp.Group, exp.Having)
    if any(_has(p, k) != _has(g, k) for k in aggs) or (
            {type(f).__name__ for f in p.find_all(exp.AggFunc)} != {type(f).__name__ for f in g.find_all(exp.AggFunc)}):
        return "aggregation/grouping"
    if _literals(p) != _literals(g) or _has(p, exp.Where) != _has(g, exp.Where):
        return "filter_value"
    return "other"
