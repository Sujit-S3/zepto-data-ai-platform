# Module 1 -- Data Pipeline (Zepto Data & AI Platform capstone)

I built this module as a scrape -> clean -> convert -> store -> query pipeline against the public scraping-practice site [books.toscrape.com](https://books.toscrape.com).

## What this module does

1. **Scrape** (`scrape_and_load.py`): I programmed this to discover category links from the site's left-hand navigation on the index page, then fully paginate each category's listing pages. I chose to read the title, price, star rating (CSS class word), and availability text directly off each `article.product_pod` block (this way I avoided visiting individual book detail pages to save time and bandwidth).
2. **Clean**: I added parsing functions that rigorously check types. If a row fails to parse cleanly, my script drops it instead of crashing.
3. **Convert**: I convert GBP prices to INR using a fixed, hardcoded project-defined constant (not a live exchange rate, to keep my results reproducible).
4. **Store**: I load the cleaned rows into a normalized two-table SQLite database, which I designed to be rebuilt from scratch on every run to prevent data duplication.
5. **Query** (`run_queries.py`): I wrote 6 SQL queries demonstrating `SELECT`/`WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`, `BETWEEN`, and a `JOIN`. I then load these results into pandas via `pd.read_sql`, and finally I reproduce the JOIN result with `pd.merge` on in-memory DataFrames to verify my two approaches agree perfectly.

## Setup & run commands

```bash
# From the zepto-data-ai-platform repo root:
source "./.venv/Scripts/activate"

cd data_pipeline
python scrape_and_load.py
python run_queries.py
```

I made sure both scripts are fully automatic -- no manual steps, no credentials, no API keys needed. My `scrape_and_load.py` script deletes and recreates `data/zepto_books.db` from scratch every time it is run, making my pipeline completely idempotent and reproducible.

Here is the exact output from running my script:

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

- **Total books I loaded:** 163
- **Categories (5):** Travel, Mystery, Historical Fiction, Sequential Art, Classics
- **Rows I dropped as malformed:** 0

I captured the complete real query output from my `run_queries.py` (all 6 queries plus the pandas `pd.read_sql` / `pd.merge` equivalence check) verbatim in [`sql_queries.md`](sql_queries.md). To ensure everything was correct, I compared the SQL result produced by `pd.read_sql` and the result reproduced via `pd.merge` on in-memory DataFrames using `DataFrame.equals(...)` and printed:

```
Do the SQL-JOIN result and the pandas pd.merge result match? True
```

## My Cleaning / parsing decisions

- **`price_gbp`**: I parse the raw price string (e.g. `"£51.77"`) by stripping every character that isn't a digit or a decimal point, then calling `float(...)`. If the result is empty or not a valid float, I simply drop the row.
- **`rating`**: I map the star-rating CSS class word (`One`..`Five`) to an integer 1-5 via a fixed lookup dictionary. An unrecognized word means I drop the row.
- **`in_stock`**: I check the availability text with `.strip().lower().startswith("in stock")` -> `True`/`False`. An empty/missing availability string (a structural parse failure) means I drop the row.
- **Strategy: drop malformed rows.** Instead of guessing a value for a row that fails to parse, I decided dropping it entirely and reporting the count was much more honest. Since books.toscrape.com is a well-formed site, malformed rows are rare (I had 0 rows dropped in my run). Dropping them felt like the cleanest approach.

## Currency conversion

I defined **1 GBP = 105.50 INR** as a fixed constant (`GBP_TO_INR_RATE = 105.50` in `scrape_and_load.py`), explicitly *not* a live/market exchange rate. I hardcoded this so my results are reproducible run-to-run and documented it clearly in the code.

## My Database schema

I put my SQLite database at `data_pipeline/data/zepto_books.db`, using two related tables that I designed:

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

- I made `categories.category_id` the primary key, and set `category_name` to be unique.
- I made `books.book_id` the primary key, and set `books.category_id` as a foreign key referencing `categories.category_id`. This gives me a beautifully normalized one-to-many relationship (one category -> many books).

## Files I created in this module

- `scrape_and_load.py` -- my script to scrape, clean, convert, and load into SQLite.
- `run_queries.py` -- my script to run the 6 SQL queries + pandas read_sql/merge check.
- `sql_queries.md` -- exact queries and exact real captured output I generated.
- `run_queries_output.txt` -- raw, unedited console output of my actual `run_queries.py` run (evidence backing `sql_queries.md`).
- `README.md` -- this file.
- `data/zepto_books.db` -- the SQLite database I built (regenerated by `scrape_and_load.py`; not hand-edited).
