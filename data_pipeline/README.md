# Module 1 -- Data Pipeline (Zepto Data & AI Platform capstone)

A scrape -> clean -> convert -> store -> query pipeline built against
the public scraping-practice site
[books.toscrape.com](https://books.toscrape.com).

## What this module does

1. **Scrape** (`scrape_and_load.py`): discovers category links from the
   site's left-hand navigation on the index page, then fully paginates
   each category's listing pages, reading title, price, star rating
   (CSS class word), and availability text directly off each
   `article.product_pod` block (no need to visit individual book detail
   pages).
   fails to parse cleanly instead of crashing.
3. **Convert**: converts GBP prices to INR using a fixed, hardcoded
   project-defined constant (not a live exchange rate).
4. **Store**: loads the cleaned rows into a normalized two-table SQLite
   database, rebuilt from scratch on every run.
5. **Query** (`run_queries.py`): runs 6 SQL queries demonstrating
   `SELECT`/`WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`, `BETWEEN`, and a
   `JOIN`, loads results into pandas via `pd.read_sql`, and reproduces
   the JOIN result with `pd.merge` on in-memory DataFrames to
   verify the two approaches agree.

## Setup & run commands

```bash
# From the zepto-data-ai-platform repo root:
source "./.venv/Scripts/activate"

cd data_pipeline
python scrape_and_load.py
python run_queries.py
```

Both scripts are fully automatic -- no manual steps, no credentials, no
API keys. `scrape_and_load.py` deletes and recreates
`data/zepto_books.db` from scratch every time it is run, so the pipeline
is idempotent and reproducible.

Here is the output from running `python scrape_and_load.py`:

```
Discovering categories and scraping books.toscrape.com ...
  scraped category 'Travel': 11 books (running total: 11 books, 1 categories)
  scraped category 'Mystery': 32 books (running total: 43 books, 2 categories)
  scraped category 'Historical Fiction': 26 books (running total: 69 books, 3 categories)
  scraped category 'Sequential Art': 75 books (running total: 144 books, 4 categories)
  scraped category 'Classics': 19 books (running total: 163 books, 5 categories)

Raw scrape complete: 163 raw rows across 5 categories: ['Travel', 'Mystery', 'Historical Fiction', 'Sequential Art', 'Classics']

Cleaning complete: 163 clean rows, 0 row(s) dropped as malformed.

Database (re)built at: <repo>\data_pipeline\data\zepto_books.db
Categories written (5): ['Travel', 'Mystery', 'Historical Fiction', 'Sequential Art', 'Classics']
Books written: 163

Fixed conversion rate used: 1 GBP = 105.5 INR (project-defined constant, not a live market rate)
```

- **Total books loaded:** 163
- **Categories (5):** Travel, Mystery, Historical Fiction, Sequential Art, Classics
- **Rows dropped as malformed:** 0

The complete real query output from `run_queries.py` (all 6 queries plus
the pandas `pd.read_sql` / `pd.merge` equivalence check) is captured
verbatim in [`sql_queries.md`](sql_queries.md) and
produced by `pd.read_sql` and the result reproduced via
`pd.merge` on in-memory DataFrames were compared with
`DataFrame.equals(...)` and printed:

```
Do the SQL-JOIN result and the pandas pd.merge result match? True
```

## Cleaning / parsing decisions

- **`price_gbp`**: parsed from the raw price string (e.g. `"£51.77"`) by
  stripping every character that isn't a digit or a decimal point, then
  calling `float(...)`. If the result is empty or not a valid float, the
  row is dropped.
- **`rating`**: the star-rating CSS class word (`One`..`Five`) is mapped to
  an integer 1-5 via a fixed lookup dict. An unrecognized word means the
  row is dropped.
- **`in_stock`**: the availability text is checked with
  `.strip().lower().startswith("in stock")` -> `True`/`False`. An empty/
  missing availability string (a structural parse failure) means the row
  is dropped.
- **Strategy: drop malformed rows.** Instead of guessing a value for
  a row that fails to parse, I drop it entirely and report the count.
  books.toscrape.com is a well-formed site, so malformed rows are rare
  (0 rows dropped in my run). Dropping them is the cleanest approach.

## Currency conversion

**1 GBP = 105.50 INR** -- a fixed, project-defined constant
(`GBP_TO_INR_RATE = 105.50` in `scrape_and_load.py`), explicitly *not* a
live/market exchange rate. It is hardcoded so results are reproducible
run-to-run and clearly documented in code with a comment to that effect.

## Database schema

SQLite database at `data_pipeline/data/zepto_books.db`, two related
tables:

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```

- `categories.category_id` is the primary key; `category_name` is unique.
- `books.book_id` is the primary key; `books.category_id` is a foreign key
  referencing `categories.category_id`, giving a normalized one-to-many
  relationship (one category -> many books).

## Files in this module

- `scrape_and_load.py` -- scrape, clean, convert, and load into SQLite.
- `run_queries.py` -- run the 6 SQL queries + pandas read_sql/merge check.
- `sql_queries.md` -- exact queries and exact real captured output.
- `run_queries_output.txt` -- raw, unedited console output of the actual
  `run_queries.py` run (evidence backing `sql_queries.md`).
- `README.md` -- this file.
- `data/zepto_books.db` -- the built SQLite database (regenerated by
  `scrape_and_load.py`; not hand-edited).
