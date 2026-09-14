# SQL Queries and Real Output

This file contains the exact SQL query strings and the exact real output
captured from actually running `run_queries.py` against
`data/zepto_books.db` (built by `scrape_and_load.py`). No numbers below are
invented -- they are copy-pasted from the real console output of that run.

Database at time of this run: **163 books** across **5 categories**
(Travel, Mystery, Historical Fiction, Sequential Art, Classics).

---

## Query 1 -- SELECT + WHERE + ORDER BY + LIMIT

Top 5 most expensive books that have a 5-star rating.

```sql
SELECT title, price_gbp, price_inr, rating
FROM books
WHERE rating = 5
ORDER BY price_gbp DESC
LIMIT 5;
```

**Output (5 rows):**

| title | price_gbp | price_inr | rating |
|---|---|---|---|
| El Deafo | 57.62 | 6078.91 | 5 |
| The Sandman, Vol. 3: Dream Country (The Sandman (volumes) #3) | 55.55 | 5860.525 | 5 |
| A Flight of Arrows (The Pathfinders #2) | 55.53 | 5858.415 | 5 |
| The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) | 52.3 | 5517.65 | 5 |
| Scott Pilgrim's Precious Little Life (Scott Pilgrim #1) | 52.29 | 5516.595 | 5 |

---

## Query 2 -- DISTINCT

Distinct star ratings present in the books table.

```sql
SELECT DISTINCT rating
FROM books
ORDER BY rating;
```

**Output (5 rows):**

```
(1,)
(2,)
(3,)
(4,)
(5,)
```

---

## Query 3 -- BETWEEN

Books priced between £20 and £30 (inclusive).

```sql
SELECT title, price_gbp, price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 30
ORDER BY price_gbp ASC;
```

**Output (34 rows -- first 10 shown, full 34 captured in `run_queries_output.txt`):**

| title | price_gbp | price_inr |
|---|---|---|
| Blood Defense (Samantha Brinkman #1) | 20.3 | 2141.65 |
| Love, Lies and Spies | 20.55 | 2168.025 |
| Between Shades of Gray | 20.79 | 2193.345 |
| Delivering the Truth (Quaker Midwife Mystery #1) | 20.89 | 2203.895 |
| Fruits Basket, Vol. 6 (Fruits Basket #6) | 20.96 | 2211.28 |
| Voyager (Outlander #3) | 21.07 | 2222.885 |
| Saga, Volume 3 (Saga (Collected Editions) #3) | 21.57 | 2275.635 |
| Paper Girls, Vol. 1 (Paper Girls #1-5) | 21.71 | 2290.405 |
| Giant Days, Vol. 2 (Giant Days #5-8) | 22.11 | 2332.605 |
| The Silkworm (Cormoran Strike #2) | 23.05 | 2431.775 |
| ... (24 more rows, up to 29.87 GBP) | | |

(Total row count actually returned: **34**.)

---

## Query 4 -- IN + JOIN

Books belonging to a specific subset of categories, joined against the
`categories` table.

```sql
SELECT b.title, b.price_gbp, c.category_name
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE c.category_name IN ('Mystery', 'Classics')
ORDER BY c.category_name, b.price_gbp DESC;
```

**Output (51 rows -- 19 Classics + 32 Mystery). Top of each group:**

| title | price_gbp | category_name |
|---|---|---|
| Candide | 58.63 | Classics |
| Animal Farm | 57.22 | Classics |
| Alice in Wonderland (Alice's Adventures in Wonderland #1) | 55.53 | Classics |
| The Pilgrim's Progress | 50.26 | Classics |
| ... (15 more Classics rows down to 14.82) | | |
| Boar Island (Anna Pigeon #19) | 59.48 | Mystery |
| The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1) | 57.7 | Mystery |
| The Past Never Ends | 56.5 | Mystery |
| ... (29 more Mystery rows down to 10.69) | | |

(Total row count actually returned: **51** -- full list captured in
`run_queries_output.txt`.)

---

## Query 5 -- JOIN (top-N per category, window function)

Top 3 highest-rated books per category (ties broken by cheapest price),
computed with a `ROW_NUMBER() OVER (PARTITION BY ...)` window function
joined against `categories`.

```sql
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
WHERE rn <= 3
ORDER BY category_name, rating DESC, price_gbp ASC;
```

**Output (15 rows -- exactly 3 per category):**

| category_name | title | price_gbp | rating |
|---|---|---|---|
| Classics | The Secret Garden | 15.08 | 4 |
| Classics | The Complete Stories and Poems (The Works of Edgar Allan Poe [Cameo Edition]) | 26.78 | 4 |
| Classics | Little Women (Little Women #1) | 28.07 | 4 |
| Historical Fiction | A Spy's Devotion (The Regency Spies of London #1) | 16.97 | 5 |
| Historical Fiction | Between Shades of Gray | 20.79 | 5 |
| Historical Fiction | Voyager (Outlander #3) | 21.07 | 5 |
| Mystery | The Girl You Lost | 12.29 | 5 |
| Mystery | The Silkworm (Cormoran Strike #2) | 23.05 | 5 |
| Mystery | What Happened on Beale Street (Secrets of the South Mysteries #2) | 25.37 | 5 |
| Sequential Art | Fruits Basket, Vol. 2 (Fruits Basket #2) | 11.64 | 5 |
| Sequential Art | Superman Vol. 1: Before Truth (Superman by Gene Luen Yang #1) | 11.89 | 5 |
| Sequential Art | Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Princess Jellyfish 2-in-1 Omnibus #1) | 13.61 | 5 |
| Travel | 1,000 Places to See Before You Die | 26.08 | 5 |
| Travel | Full Moon over Noah's Ark: An Odyssey to Mount Ararat and Beyond | 49.43 | 4 |
| Travel | A Year in Provence (Provence #1) | 56.88 | 4 |

(Note: the apostrophe in "Noah's Ark" displays as a mojibake `?` in the raw
Git-Bash console capture due to terminal codepage limitations -- the value
stored in the database and returned by Python is the correct Unicode
right-single-quote character U+2019, verified separately with
`json.dumps(title, ensure_ascii=False)`.)

---

## Query 6 -- JOIN + GROUP BY (bonus aggregation)

Average price and book count per category.

```sql
SELECT c.category_name, COUNT(*) AS num_books, ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp
FROM books b
JOIN categories c ON b.category_id = c.category_id
GROUP BY c.category_name
ORDER BY avg_price_gbp DESC;
```

**Output (5 rows):**

| category_name | num_books | avg_price_gbp |
|---|---|---|
| Travel | 11 | 39.79 |
| Classics | 19 | 36.55 |
| Sequential Art | 75 | 34.57 |
| Historical Fiction | 26 | 33.64 |
| Mystery | 32 | 31.72 |

---

## pandas: pd.read_sql

Query 1 and Query 3's results were loaded into pandas DataFrames via
`pd.read_sql(sql, conn)`:

- `df_q1` -- 5 rows x 4 columns (title, price_gbp, price_inr, rating),
  matching Query 1's output exactly.
- `df_q3` -- 34 rows x 3 columns (title, price_gbp, price_inr), matching
  Query 3's output exactly.

---

## pd.read_sql vs pd.merge equivalence check (JOIN query)

For Query 4 (the `IN` + `JOIN` query), the result was computed two
independent ways and compared:

1. **`df_sql_join`** -- `pd.read_sql(q4_sql, conn)`, i.e. the SQL engine
   performs the JOIN.
2. **`df_pandas_merge`** -- `books` and `categories` were each loaded
   independently into DataFrames with plain `SELECT * FROM <table>`, then
   joined in-memory using `pd.merge(books_df, categories_df,
   on="category_id", how="inner")`, filtered to `category_name IN
   ('Mystery', 'Classics')`, and sorted identically -- no SQL JOIN used for
   this second computation.

Both DataFrames were normalized (same columns, same sort order, reset
index) and compared with `DataFrame.equals(...)`.

**Actual real output printed by `run_queries.py`:**

```
Do the SQL-JOIN result and the pandas pd.merge result match? True
```

Both DataFrames contained the identical 51 rows (19 Classics + 32 Mystery)
in the identical order, byte-for-byte equal per `pandas.DataFrame.equals`.

---

The complete, unedited console output of the actual `run_queries.py` run
that produced every number above is saved alongside this file as
`run_queries_output.txt`.
