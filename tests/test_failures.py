import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import failures as F  # noqa: E402

GOLD = "SELECT T1.name FROM customer T1 JOIN orders T2 ON T1.id = T2.customer_id WHERE T2.amount > 15"


@pytest.mark.parametrize("rec,gold,want", [
    ({"outcome": "correct"}, GOLD, None),
    ({"outcome": "cut"}, GOLD, "loop/cut"),
    ({"outcome": "no_sql"}, GOLD, "no_sql/format"),
    ({"outcome": "wrong_result", "extract_status": "multiple", "pred_sql": "SELECT 1"}, GOLD, "no_sql/format"),
    ({"outcome": "exec_error", "pred_sql": "SELECT DATE_FORMAT(day,'%Y') FROM orders",
      "error": "no such function: DATE_FORMAT"}, GOLD, "dialect"),
    ({"outcome": "syntax_error", "pred_sql": "SELECT amount::int FROM orders", "error": 'near ":": syntax error'},
     GOLD, "dialect"),
    ({"outcome": "exec_error", "pred_sql": "SELECT nickname FROM customer", "error": "no such column: nickname"},
     GOLD, "schema_hallucination"),
    ({"outcome": "wrong_result", "pred_sql": "SELECT name FROM customer"}, GOLD, "join_logic"),
    ({"outcome": "wrong_result", "pred_sql": "SELECT COUNT(T1.name) FROM customer T1 JOIN orders T2 "
                                             "ON T1.id = T2.customer_id WHERE T2.amount > 15"}, GOLD,
     "aggregation/grouping"),
    ({"outcome": "wrong_result", "pred_sql": "SELECT T1.name FROM customer T1 JOIN orders T2 "
                                             "ON T1.id = T2.customer_id WHERE T2.amount > 10"}, GOLD, "filter_value"),
])
def test_label(rec, gold, want):
    assert F.label(rec, gold) == want
