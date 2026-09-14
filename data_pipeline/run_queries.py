"""
run_queries.py
===============
Module 1 of the Zepto Data & AI Platform capstone project.

Runs a set of SQL queries against data/zepto_books.db (built by
scrape_and_load.py) and demonstrates:
    - SELECT + WHERE
    - ORDER BY
    - LIMIT
    - DISTINCT
    - IN and BETWEEN
    - a JOIN between books and categories

It also loads two of the query results into pandas DataFrames via
pd.read_sql(...), and for the JOIN query it independently reproduces the
identical result using pd.merge(...) on in-memory DataFrames (no SQL for
that second computation), then programmatically compares the two results.

Run:
    source "<repo>/.venv/Scripts/activate"
    python run_queries.py
"""

import os
import sqlite3

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "zepto_books.db")

TOP_N_PER_CATEGORY = 3


def divider(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_query(conn, label, sql, params=()):
    """Print the SQL and its actual result rows, return the rows."""
    divider(label)
    print("SQL:")
    print(sql.strip())
    cur = conn.cursor()
    cur.execute(sql, params)
    col_names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"\nOutput ({len(rows)} row(s)):")
    print(col_names)
    for r in rows:
        print(r)
    return col_names, rows


def main():
    conn = sqlite3.connect(DB_PATH)

    # ------------------------------------------------------------------
    # Query 1: SELECT + WHERE + ORDER BY + LIMIT
    #   Top 5 most expensive books that have a 5-star rating.
    # ------------------------------------------------------------------
    q1_sql = """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        WHERE rating = 5
        ORDER BY price_gbp DESC
        LIMIT 5;
    """
    q1_cols, q1_rows = run_query(conn, "Query 1: SELECT + WHERE + ORDER BY + LIMIT "
                                        "(top 5 most expensive 5-star books)", q1_sql)

    # ------------------------------------------------------------------
    # Query 2: DISTINCT
    #   Distinct category names present in the books table (via JOIN would
    #   also work, but categories table alone already demonstrates DISTINCT
    #   meaningfully on the books-derived rating values instead).
    # ------------------------------------------------------------------
    q2_sql = """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """
    q2_cols, q2_rows = run_query(conn, "Query 2: DISTINCT (distinct star ratings present)", q2_sql)

    # ------------------------------------------------------------------
    # Query 3: BETWEEN
    #   Books priced between £20 and £30 (inclusive).
    # ------------------------------------------------------------------
    q3_sql = """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp ASC;
    """
    q3_cols, q3_rows = run_query(conn, "Query 3: BETWEEN (books priced £20-£30)", q3_sql)

    # ------------------------------------------------------------------
    # Query 4: IN
    #   Books belonging to a specific subset of categories.
    # ------------------------------------------------------------------
    q4_sql = """
        SELECT b.title, b.price_gbp, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Mystery', 'Classics')
        ORDER BY c.category_name, b.price_gbp DESC;
    """
    q4_cols, q4_rows = run_query(conn, "Query 4: IN + JOIN (books in 'Mystery' or 'Classics')", q4_sql)

    # ------------------------------------------------------------------
    # Query 5: JOIN -- top-N highest rated (then cheapest as tiebreak)
    #   books per category.
    # ------------------------------------------------------------------
    q5_sql = f"""
        SELECT category_name, title, price_gbp, rating
        FROM (
            SELECT
                c.category_name AS category_name,
                b.title AS title,
                b.price_gbp AS price_gbp,
                b.rating AS rating,
                ROW_NUMBER() OVER (
                    PARTITION BY c.category_id
                    ORDER BY b.rating DESC, b.price_gbp ASC
                ) AS rn
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        WHERE rn <= {TOP_N_PER_CATEGORY}
        ORDER BY category_name, rating DESC, price_gbp ASC;
    """
    q5_cols, q5_rows = run_query(
        conn,
        f"Query 5: JOIN (top {TOP_N_PER_CATEGORY} highest-rated books per category)",
        q5_sql,
    )

    # ------------------------------------------------------------------
    # Query 6: Aggregation across the JOIN (bonus, not required but useful)
    #   Average price per category.
    # ------------------------------------------------------------------
    q6_sql = """
        SELECT c.category_name, COUNT(*) AS num_books, ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY avg_price_gbp DESC;
    """
    q6_cols, q6_rows = run_query(conn, "Query 6: JOIN + GROUP BY (avg price per category)", q6_sql)

    # ------------------------------------------------------------------
    # pandas: load at least two query results via pd.read_sql
    # ------------------------------------------------------------------
    divider("pandas: pd.read_sql for Query 1 and Query 3")
    df_q1 = pd.read_sql(q1_sql, conn)
    df_q3 = pd.read_sql(q3_sql, conn)
    print("\ndf_q1 (top 5 most expensive 5-star books):")
    print(df_q1)
    print("\ndf_q3 (books priced £20-£30):")
    print(df_q3)

    # ------------------------------------------------------------------
    # JOIN equivalence check: pd.read_sql(JOIN SQL) vs pd.merge on
    # in-memory DataFrames loaded from the two tables independently.
    # ------------------------------------------------------------------
    divider("JOIN equivalence check: pd.read_sql(SQL JOIN) vs pd.merge()")

    # (a) The SQL-JOIN result (Query 4, the IN + JOIN query) via pd.read_sql.
    df_sql_join = pd.read_sql(q4_sql, conn)
    print("\ndf_sql_join (from pd.read_sql of the SQL JOIN query):")
    print(df_sql_join)

    # (b) Independently reproduce the same result using pd.merge on
    # in-memory DataFrames loaded from the two raw tables (no SQL JOIN).
    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)

    merged_df = pd.merge(books_df, categories_df, on="category_id", how="inner")
    merged_df = merged_df[merged_df["category_name"].isin(["Mystery", "Classics"])]
    df_pandas_merge = merged_df[["title", "price_gbp", "category_name"]].sort_values(
        by=["category_name", "price_gbp"], ascending=[True, False]
    ).reset_index(drop=True)
    print("\ndf_pandas_merge (from pd.merge on in-memory DataFrames, no SQL JOIN):")
    print(df_pandas_merge)

    # Normalize both for comparison: same column order, sorted the same way,
    # index reset.
    df_sql_join_cmp = df_sql_join[["title", "price_gbp", "category_name"]].sort_values(
        by=["category_name", "price_gbp"], ascending=[True, False]
    ).reset_index(drop=True)
    df_pandas_merge_cmp = df_pandas_merge.reset_index(drop=True)

    match = df_sql_join_cmp.equals(df_pandas_merge_cmp)
    print(f"\nDo the SQL-JOIN result and the pandas pd.merge result match? {match}")

    if not match:
        # Fall back to a value-by-value comparison for diagnostics.
        diff = df_sql_join_cmp.compare(df_pandas_merge_cmp)
        print("Differences:")
        print(diff)

    conn.close()

    return {
        "q1": (q1_cols, q1_rows),
        "q2": (q2_cols, q2_rows),
        "q3": (q3_cols, q3_rows),
        "q4": (q4_cols, q4_rows),
        "q5": (q5_cols, q5_rows),
        "q6": (q6_cols, q6_rows),
        "join_match": match,
    }


if __name__ == "__main__":
    main()
