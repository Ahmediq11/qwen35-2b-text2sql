"""30 hand-made scorer cases (PLAN §4.3): equivalent queries, wrong columns, syntax errors, timeouts, extraction."""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import sql_eval as E  # noqa: E402


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("db") / "shop.sqlite")
    con = sqlite3.connect(p)
    con.executescript("""
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT, age INTEGER);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customer(id), amount REAL, day TEXT);
        INSERT INTO customer VALUES (1,'Ann','Paris',30),(2,'Bob','Rome',41),(3,'Cy','Paris',25),(4,'Di','Oslo',41);
        INSERT INTO orders VALUES (1,1,10.5,'2024-01-02'),(2,1,20.0,'2024-02-03'),(3,2,5.0,'2024-02-04'),
                                  (4,3,7.25,'2024-03-01'),(5,3,7.25,'2024-03-02');
    """)
    con.commit()
    con.close()
    return p


def ans(sql):
    return f"Here is the query:\n```sql\n{sql}\n```"


# ---- execution accuracy: (prediction, gold, expected outcome) ----
CASES = [
    # equivalent queries
    ("SELECT name FROM customer WHERE city = 'Paris'", "SELECT name FROM customer WHERE city='Paris'", "correct"),
    ("SELECT c.name FROM customer AS c WHERE c.city = 'Paris'", "SELECT name FROM customer WHERE city='Paris'", "correct"),
    ("SELECT name FROM customer WHERE city IN ('Paris')", "SELECT name FROM customer WHERE city='Paris'", "correct"),
    ("SELECT COUNT(*) FROM orders", "SELECT count(id) FROM orders", "correct"),
    ("SELECT name FROM customer WHERE age > 40 ORDER BY name", "SELECT name FROM customer WHERE age > 40", "correct"),
    ("SELECT city, COUNT(*) FROM customer GROUP BY city", "SELECT city, count(*) FROM customer GROUP BY city ORDER BY city DESC", "wrong_result"),
    ("SELECT name FROM customer ORDER BY age DESC, name", "SELECT name FROM customer ORDER BY age DESC, name", "correct"),
    ("SELECT name FROM customer ORDER BY age, name", "SELECT name FROM customer ORDER BY age DESC, name", "wrong_result"),
    ("SELECT SUM(amount) FROM orders", "SELECT sum(amount) FROM orders", "correct"),
    ("SELECT AVG(age) * 1.0 FROM customer", "SELECT avg(age) FROM customer", "correct"),
    ("SELECT CAST(SUM(amount) AS INTEGER) FROM orders WHERE customer_id = 2", "SELECT sum(amount) FROM orders WHERE customer_id = 2", "correct"),
    ("SELECT T1.name FROM customer T1 JOIN orders T2 ON T1.id = T2.customer_id WHERE T2.amount > 15",
     "SELECT name FROM customer WHERE id IN (SELECT customer_id FROM orders WHERE amount > 15)", "correct"),
    # duplicates matter (multiset, not set)
    ("SELECT DISTINCT amount FROM orders WHERE customer_id = 3", "SELECT amount FROM orders WHERE customer_id = 3", "wrong_result"),
    ("SELECT amount FROM orders WHERE customer_id = 3", "SELECT amount FROM orders WHERE customer_id = 3", "correct"),
    # wrong columns / values / logic
    ("SELECT city FROM customer WHERE city = 'Paris'", "SELECT name FROM customer WHERE city='Paris'", "wrong_result"),
    ("SELECT name, city FROM customer WHERE city = 'Paris'", "SELECT name FROM customer WHERE city='Paris'", "wrong_result"),
    ("SELECT name FROM customer WHERE city = 'paris'", "SELECT name FROM customer WHERE city='Paris'", "wrong_result"),
    ("SELECT COUNT(*) FROM customer", "SELECT count(*) FROM orders", "wrong_result"),
    # errors
    ("SELECT name FROM customers", "SELECT name FROM customer", "exec_error"),
    ("SELECT nickname FROM customer", "SELECT name FROM customer", "exec_error"),
    ("SELECT DATE_FORMAT(day, '%Y') FROM orders", "SELECT strftime('%Y', day) FROM orders", "exec_error"),
    ("SELECT name FROM customer WHERE", "SELECT name FROM customer", "syntax_error"),
    ("SELEC name FROM customer", "SELECT name FROM customer", "syntax_error"),
    ("SELECT name FROM customer WHERE city = 'Paris", "SELECT name FROM customer", "syntax_error"),
    # timeout: a huge cross join
    ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT count(*) FROM n",
     "SELECT 1", "timeout"),
]


@pytest.mark.parametrize("pred,gold,want", CASES)
def test_execution(db, pred, gold, want):
    r = E.score(ans(pred), gold, db, timeout_s=1.0 if want == "timeout" else 30.0)
    assert r["outcome"] == want, r


# ---- extraction ----
def test_last_sql_block_wins(db):
    t = "```sql\nSELECT 1\n```\nActually:\n```sql\nSELECT name FROM customer\n```"
    assert E.extract_sql(t) == ("SELECT name FROM customer", "ok")


def test_thinking_unclosed_is_cut(db):
    t = "<think>Let me see... wait, wait, wait"
    assert E.score(t, "SELECT 1", db, thinking=True)["outcome"] == "cut"


def test_thinking_without_tags_is_cut(db):
    # the chat template opens <think> in the prompt: a cut answer has no tag at all, but may quote SQL while reasoning
    t = "Let me write SELECT name FROM customer WHERE age = 25 ... wait, wait, wait"
    assert E.score(t, "SELECT name FROM customer WHERE age=25", db, thinking=True)["outcome"] == "cut"


def test_text_after_think_only(db):
    t = "<think>maybe ```sql\nSELECT 1\n``` no</think>\n```sql\nSELECT name FROM customer WHERE age = 25\n```"
    assert E.score(t, "SELECT name FROM customer WHERE age=25", db, thinking=True)["outcome"] == "correct"


def test_bare_select_fallback(db):
    t = "The answer is SELECT name FROM customer WHERE city = 'Oslo';"
    assert E.score(t, "SELECT name FROM customer WHERE city='Oslo'", db)["outcome"] == "correct"


def test_prose_only_is_no_sql(db):
    assert E.score("There are four customers.", "SELECT count(*) FROM customer", db)["outcome"] == "no_sql"


def test_multiple_statements_keep_last(db):
    t = ans("SELECT 1; SELECT name FROM customer WHERE age = 30")
    sql, st = E.extract_sql(t)
    assert st == "multiple" and sql == "SELECT name FROM customer WHERE age = 30"


def test_read_only(db):
    r = E.score(ans("DELETE FROM customer"), "SELECT 1", db)
    assert r["outcome"] in ("exec_error", "syntax_error") and "readonly" in r["error"].lower()
    assert E.execute(db, "SELECT count(*) FROM customer") == [(4,)]


def test_case_count():
    assert len(CASES) + 7 >= 30
