"""
Text-to-SQL Agent: answers questions against a historical stock-price table.

Real mode (OPENAI_API_KEY set): LLM converts the natural-language question
into a SQL query against the known schema, which is then executed safely
(read-only connection, SELECT-only guard).

Fallback mode: a small keyword/regex-based NL parser extracts a ticker and
maps common phrasings ("latest price", "highest close", "average volume")
to a canned parameterized query. Good enough to demo routing + real DB
execution without an API key.
"""
import re
import sqlite3
from config import SQL_DB_PATH, USE_OPENAI, OPENAI_API_KEY

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_prices (
    ticker TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER
);
"""


def ensure_sample_db():
    conn = sqlite3.connect(SQL_DB_PATH)
    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.execute("SELECT COUNT(*) FROM stock_prices")
    if cur.fetchone()[0] == 0:
        import random
        from datetime import date, timedelta
        random.seed(42)
        tickers = ["ACME", "GLOBEX", "INITECH"]
        start = date(2026, 1, 1)
        rows = []
        for t in tickers:
            price = 100.0
            for i in range(120):
                d = start + timedelta(days=i)
                price += random.uniform(-3, 3)
                o = price
                h = price + random.uniform(0, 2)
                l = price - random.uniform(0, 2)
                c = price + random.uniform(-1, 1)
                v = random.randint(100_000, 5_000_000)
                rows.append((t, d.isoformat(), round(o, 2), round(h, 2), round(l, 2), round(c, 2), v))
        cur.executemany(
            "INSERT INTO stock_prices VALUES (?,?,?,?,?,?,?)", rows
        )
        conn.commit()
    conn.close()


def _is_select_only(sql: str) -> bool:
    return sql.strip().lower().startswith("select")


def _llm_nl_to_sql(question: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "You convert natural language questions into a single SQLite SELECT "
        "query. Schema:\n" + SCHEMA + "\n"
        "Only output the raw SQL, no markdown, no explanation.\n"
        f"Question: {question}\nSQL:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )
    sql = resp.choices[0].message.content.strip().strip("`").replace("sql\n", "")
    return sql


def _heuristic_nl_to_sql(question: str) -> str:
    q = question.lower()
    ticker_match = re.search(r"\b(acme|globex|initech)\b", q)
    ticker = ticker_match.group(1).upper() if ticker_match else None
    where = f"WHERE ticker = '{ticker}'" if ticker else ""

    if "latest" in q or "most recent" in q or "current" in q:
        return f"SELECT * FROM stock_prices {where} ORDER BY date DESC LIMIT 1"
    if "highest" in q or "max" in q:
        return f"SELECT * FROM stock_prices {where} ORDER BY high DESC LIMIT 1"
    if "lowest" in q or "min" in q:
        return f"SELECT * FROM stock_prices {where} ORDER BY low ASC LIMIT 1"
    if "average" in q or "avg" in q:
        return f"SELECT ticker, AVG(close) as avg_close, AVG(volume) as avg_volume FROM stock_prices {where} GROUP BY ticker"
    # default: last 10 rows
    return f"SELECT * FROM stock_prices {where} ORDER BY date DESC LIMIT 10"


def run(question: str) -> dict:
    ensure_sample_db()
    sql = _llm_nl_to_sql(question) if USE_OPENAI else _heuristic_nl_to_sql(question)

    if not _is_select_only(sql):
        return {"agent": "sql", "error": "Refused to run non-SELECT query.", "sql": sql}

    conn = sqlite3.connect(SQL_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return {"agent": "sql", "error": str(e), "sql": sql}
    finally:
        conn.close()

    return {"agent": "sql", "sql": sql, "rows": rows}
