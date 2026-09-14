"""
scrape_and_load.py
===================
Module 1 of the Zepto Data & AI Platform capstone project.

Scrapes books from the public scraping-practice site https://books.toscrape.com,
cleans/parses the raw fields, converts GBP prices to INR using a fixed
project-defined conversion constant, and loads the cleaned data into a
normalized SQLite database (two related tables: categories, books).

Runnable end-to-end with zero manual steps:
    source "<repo>/.venv/Scripts/activate"
    python scrape_and_load.py

Every run deletes and rebuilds data/zepto_books.db from scratch.
"""

import os
import re
import sqlite3
import time

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Constants / configuration
# --------------------------------------------------------------------------

BASE_URL = "https://books.toscrape.com/"
INDEX_URL = BASE_URL + "index.html"

# Fixed project-defined currency conversion constant.
# This is NOT a live/market exchange rate -- it is hardcoded per the
# assignment brief so that results are reproducible.
GBP_TO_INR_RATE = 105.50

# Minimum thresholds the scrape must reach before we stop pulling more
# categories.
MIN_TOTAL_BOOKS = 60
MIN_CATEGORIES = 5
# Soft cap on how many categories we will walk through (in nav order) even
# if the thresholds above are already satisfied earlier -- keeps the scrape
# comfortably over both thresholds per the assignment brief.
MAX_CATEGORIES = 6

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "zepto_books.db")

RATING_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

HEADERS = {"User-Agent": "Mozilla/5.0 (capstone-project-scraper)"}

REQUEST_DELAY_SECONDS = 0.2  # small politeness delay between requests


# --------------------------------------------------------------------------
# Scraping helpers
# --------------------------------------------------------------------------

def fetch(url: str) -> BeautifulSoup:
    """GET a URL and return a parsed BeautifulSoup document.

    NOTE: books.toscrape.com serves UTF-8 encoded HTML but does not send a
    charset in its Content-Type header, so `requests` falls back to the
    HTTP default of ISO-8859-1 for resp.text/resp.encoding. Decoding the
    UTF-8 bytes as ISO-8859-1 mangles multi-byte punctuation (e.g. the
    right single quote in "Noah's Ark" becomes "â€™" mojibake). We pass the
    raw bytes (resp.content) to BeautifulSoup instead so it can sniff the
    actual encoding from the bytes/meta tag, which correctly resolves to
    UTF-8.
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(resp.content, "html.parser")


def discover_categories(index_url: str = INDEX_URL):
    """Return a list of (category_name, category_url) tuples in the order
    they appear in the site's left-hand navigation, skipping the top-level
    'Books' (all books) entry."""
    soup = fetch(index_url)
    nav = soup.find("div", class_="side_categories")
    links = nav.find_all("a")
    categories = []
    for a in links[1:]:  # links[0] is the "Books" (all books) top entry
        name = a.get_text(strip=True)
        href = a["href"]  # e.g. "catalogue/category/books/travel_2/index.html"
        url = BASE_URL + href
        categories.append((name, url))
    return categories


def scrape_category(category_name: str, category_url: str):
    """Fully paginate one category listing and return a list of raw book
    dicts (title, price_text, star_rating_word, availability_text,
    category)."""
    books = []
    page_url = category_url
    while page_url:
        soup = fetch(page_url)
        pods = soup.find_all("article", class_="product_pod")
        for pod in pods:
            title = pod.h3.a["title"]
            price_text = pod.find("p", class_="price_color").get_text(strip=True)
            rating_p = pod.find("p", class_="star-rating")
            star_rating_word = rating_p["class"][1] if rating_p and len(rating_p["class"]) > 1 else ""
            availability_text = pod.find("p", class_="instock availability").get_text(strip=True)
            books.append(
                {
                    "title": title,
                    "price_text": price_text,
                    "star_rating_word": star_rating_word,
                    "availability_text": availability_text,
                    "category": category_name,
                }
            )

        # Pagination: relative "next" link, e.g. "page-2.html"
        pager_next = soup.find("li", class_="next")
        if pager_next and pager_next.a:
            next_href = pager_next.a["href"]
            page_url = page_url.rsplit("/", 1)[0] + "/" + next_href
        else:
            page_url = None
    return books


def scrape_books(min_total_books=MIN_TOTAL_BOOKS, min_categories=MIN_CATEGORIES,
                  max_categories=MAX_CATEGORIES):
    """Walk categories in nav order, fully scraping each, until we have
    comfortably passed both the total-book and category-count thresholds
    (or we run out of categories / hit the soft cap)."""
    all_categories = discover_categories()
    raw_books = []
    categories_used = []

    for name, url in all_categories:
        if len(categories_used) >= max_categories:
            break
        cat_books = scrape_category(name, url)
        raw_books.extend(cat_books)
        categories_used.append(name)
        print(f"  scraped category '{name}': {len(cat_books)} books "
              f"(running total: {len(raw_books)} books, {len(categories_used)} categories)")
        if len(raw_books) >= min_total_books and len(categories_used) >= min_categories:
            break

    return raw_books, categories_used


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------

def parse_price_gbp(price_text: str):
    """Parse a raw price string like '£51.77' (or with mangled encoding of
    the £ symbol) into a float. Returns None if it cannot be parsed."""
    # Strip everything that isn't a digit or a decimal point.
    cleaned = re.sub(r"[^0-9.]", "", price_text)
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_rating(star_rating_word: str):
    """Map a rating word (One..Five) to an integer 1-5. Returns None if the
    word is not recognized."""
    return RATING_WORD_TO_INT.get(star_rating_word)


def parse_in_stock(availability_text: str):
    """Parse the availability text into a boolean. 'In stock...' => True,
    anything else => False. Returns None only if the text is empty/missing
    (structural parse failure), which is treated as malformed."""
    if not availability_text:
        return None
    return availability_text.strip().lower().startswith("in stock")


def clean_books(raw_books):
    """Clean/parse each raw scraped book dict. Rows that fail to parse
    cleanly (price, rating, or availability) are dropped rather than
    crashing the pipeline or being filled with fabricated values.

    Returns (cleaned_rows, dropped_count).
    """
    cleaned = []
    dropped = 0

    for row in raw_books:
        price_gbp = parse_price_gbp(row["price_text"])
        rating = parse_rating(row["star_rating_word"])
        in_stock = parse_in_stock(row["availability_text"])
        title = row["title"].strip() if row["title"] else ""

        if price_gbp is None or rating is None or in_stock is None or not title:
            dropped += 1
            continue

        price_inr = price_gbp * GBP_TO_INR_RATE

        cleaned.append(
            {
                "title": title,
                "price_gbp": price_gbp,
                "price_inr": price_inr,
                "rating": rating,
                "in_stock": in_stock,
                "category": row["category"],
            }
        )

    return cleaned, dropped


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def build_database(cleaned_books, db_path=DB_PATH):
    """(Re)create data/zepto_books.db from scratch and load the cleaned
    books into a normalized two-table schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Recreate the DB file from scratch every run.
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL,
            in_stock    INTEGER NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(category_id)
        );
        """
    )

    category_ids = {}
    for book in cleaned_books:
        cat_name = book["category"]
        if cat_name not in category_ids:
            cur.execute(
                "INSERT INTO categories (category_name) VALUES (?);", (cat_name,)
            )
            category_ids[cat_name] = cur.lastrowid

    for book in cleaned_books:
        cur.execute(
            """
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                book["title"],
                book["price_gbp"],
                book["price_inr"],
                book["rating"],
                1 if book["in_stock"] else 0,
                category_ids[book["category"]],
            ),
        )

    conn.commit()
    conn.close()
    return category_ids


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("Discovering categories and scraping books.toscrape.com ...")
    raw_books, categories_used = scrape_books()
    print(f"\nRaw scrape complete: {len(raw_books)} raw rows across "
          f"{len(categories_used)} categories: {categories_used}")

    cleaned_books, dropped = clean_books(raw_books)
    print(f"\nCleaning complete: {len(cleaned_books)} clean rows, "
          f"{dropped} row(s) dropped as malformed.")

    category_ids = build_database(cleaned_books)
    print(f"\nDatabase (re)built at: {DB_PATH}")
    print(f"Categories written ({len(category_ids)}): {list(category_ids.keys())}")
    print(f"Books written: {len(cleaned_books)}")
    print(f"\nFixed conversion rate used: 1 GBP = {GBP_TO_INR_RATE} INR "
          f"(project-defined constant, not a live market rate)")


if __name__ == "__main__":
    main()
