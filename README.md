# Zepto Data & AI Platform

Capstone project for the Certificate Program in Artificial Intelligence and Machine Learning.

## 1. Project overview

A single connected platform of three modules that together cover the full AI/ML engineer stack: pulling raw data from the wild and storing it relationally (**data_pipeline**), profiling and modeling a customer/passenger-style dataset end to end (**analytics**), and wrapping a grounded GenAI support assistant around a document corpus (**support_assistant**). I built all three modules to run locally. The outputs, numbers, and charts shown in this documentation come directly from running the code on my machine. I've noted any steps that require additional setup (like Docker or real LLM keys).

## 2. Repository structure

```
zepto-data-ai-platform/
├── README.md                     <- this file
├── requirements.txt               <- ONE consolidated root requirements file (see "Installation")
├── .gitignore
├── CAPSTONE_CHECKLIST.md          <- rubric-to-evidence mapping
│
├── data_pipeline/                 <- Module 1 (25 marks)
│   ├── README.md
│   ├── scrape_and_load.py         scrape -> clean -> convert -> load SQLite
│   ├── run_queries.py             5+ SQL queries + pandas read_sql/merge check
│   ├── sql_queries.md             exact queries + exact real output
│   ├── run_queries_output.txt     raw console evidence of the real run
│   └── data/zepto_books.db        the built SQLite database
│
├── analytics/                     <- Module 2 (50 marks)
│   ├── README.md
│   ├── 01_eda.ipynb                profiling, cleaning, EDA, correlation, standardization check
│   ├── 02_modeling.ipynb           split, preprocessing, 3 classifiers, tuning, regression, joblib
│   ├── titanic.csv                 one-time offline cache of sns.load_dataset('titanic')
│   ├── best_pipeline.joblib        complete fitted preprocessing+model pipeline
│   └── _build_eda_nb.py, _build_modeling_nb.py   (notebook-authoring scripts, kept for provenance)
│
└── support_assistant/              <- Module 3 (25 marks)
    ├── README.md
    ├── docs/doc_01.txt ... doc_08.txt   Zepto policy corpus (exact required text)
    ├── ingest.py                        chunk + embed (all-MiniLM-L6-v2) + store in ChromaDB
    ├── prompt_template.py               Role/Context/Task/Format/Length template (optional real-LLM path)
    ├── graph.py                         LangGraph StateGraph, 3 nodes, Pydantic schema, MOCK_LLM branch
    ├── main.py                          FastAPI app, POST /ask
    └── Dockerfile                       build context = repo root (see "Docker" below)
```

## 3. Technologies used

- **Module 1:** `requests`, `beautifulsoup4`, `sqlite3` (stdlib), `pandas`
- **Module 2:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn` (SMOTE), `joblib`, `jupyter`/`nbconvert`/`nbformat`
- **Module 3:** `sentence-transformers` (`all-MiniLM-L6-v2`), `chromadb`, `langgraph`, `pydantic`, `fastapi`, `uvicorn`

## 4. Installation / setup

**This project uses ONE consolidated root `requirements.txt`** (not a separate requirements file per module) — a single shared virtual environment is used for all three modules.

```bash
cd zepto-data-ai-platform
python -m venv .venv

# Windows Git Bash:
source .venv/Scripts/activate
# Windows cmd:      .venv\Scripts\activate.bat
# Linux/Mac:         source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Windows note: if you ever need to add a package, use `python -m pip install <pkg>` rather than calling `pip` directly — on some Windows Python installs the `pip.exe` launcher script raises a spurious `Permission Denied`, while `python -m pip` always works.
>
> Windows note 2: this development machine briefly hit a **Smart App Control** (Windows 11 code-integrity) block that prevented freshly pip-installed compiled packages (`numpy`, `torch`, etc.) from loading, with the error `An Application Control policy has blocked this file`. If you hit this on your own machine, check Windows Security → App & browser control → Smart App Control settings (or use WSL2). It is not a problem with this code.

## 5. How to run Module 1 (data_pipeline)

```bash
source .venv/Scripts/activate
cd data_pipeline
python scrape_and_load.py   # scrapes books.toscrape.com, cleans, converts, (re)builds data/zepto_books.db
python run_queries.py       # runs the 5+ SQL queries, the pd.read_sql loads, and the pd.merge equivalence check
```

Both scripts run automatically — no manual steps or credentials needed. `scrape_and_load.py` rebuilds the database from scratch on every run. I saved my output from the last run in [`data_pipeline/sql_queries.md`](data_pipeline/sql_queries.md) and [`data_pipeline/run_queries_output.txt`](data_pipeline/run_queries_output.txt).

## 6. How to run Module 2 (analytics)

```bash
source .venv/Scripts/activate
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```

`01_eda.ipynb` calls `sns.load_dataset('titanic')` exactly once and immediately caches it to `titanic.csv`; `02_modeling.ipynb` reads only that CSV and never reloads from the network. Re-running both notebooks end to end reproduces `best_pipeline.joblib`.

## 7. How to run Module 3 (support_assistant)

```bash
source .venv/Scripts/activate
cd support_assistant
python ingest.py                              # embeds the 8 policy docs into ChromaDB (one-time / re-run to refresh)
uvicorn main:app --host 0.0.0.0 --port 7860    # MOCK_LLM left unset -> offline mock mode (the graded baseline)
```

Then, from another terminal:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" \
  -d "{\"query\": \"How long does delivery take?\"}"
```

## 8. Design decisions (summary — full detail in each module's README)

- **Module 1:** scraped by discovering category links and fully paginating each category's own listing pages (title/price/rating/availability are all present directly on the grid page, so no per-book detail-page visits are needed). Malformed rows are **dropped** (not median-imputed) because a parse failure here is a structural HTML-parsing error, not a genuine missing-data problem, and dropping+counting is more honest than injecting a fabricated value. Currency conversion uses the fixed constant `1 GBP = 105.50 INR`, hardcoded and explicitly documented as not a live rate.
- **Module 2:** missing-value handling strictly follows the <5% drop / 5–30% impute / >30% drop-column threshold rule, with the real measured percentage stated for every affected column before choosing its strategy (see table in `analytics/README.md`). SMOTE is applied to the training fold only, verified never to touch the test fold. The saved artifact is the complete `ColumnTransformer`+estimator `Pipeline`, not a bare model, so it accepts raw unprocessed input.
- **Module 3:** each policy document is treated as a single chunk (they are short, single-paragraph policies, so finer chunking would add complexity without benefit). Intent routing is a pure keyword heuristic and never depends on `MOCK_LLM`; only the final answer-generation step branches on it.

## 9. Important dependencies

`requests`, `beautifulsoup4`, `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`, `nbconvert`/`nbformat`, `sentence-transformers`, `torch` (CPU build), `chromadb`, `langgraph`, `pydantic`, `fastapi`, `uvicorn`. Exact pinned versions (as actually installed and verified in this environment) are in [`requirements.txt`](requirements.txt).

## 10. MOCK_LLM explanation

Module 3's `graph.py` gates every LLM call behind the `MOCK_LLM` environment variable:

- **`MOCK_LLM` unset, or `MOCK_LLM=1` (default):** `classify_intent` uses a keyword heuristic (no LLM call); `retrieve_and_answer` still does real embedding + ChromaDB retrieval, but instead of calling an LLM it returns a canned `"Based on the retrieved context: {snippet}"` string built directly from the top retrieved chunk; `direct_answer` returns a fixed string. No API key or network call to any LLM provider is needed.
- **`MOCK_LLM=0` (optional extension):** `retrieve_and_answer` and `direct_answer` call a real LLM using the structured Role/Context/Task/Format/Length prompt template in `prompt_template.py`, with up to 2 corrective retries if the LLM's output fails Pydantic validation. (Requires setting up an API key).

## 11. Docker

`support_assistant/Dockerfile` builds the FastAPI app. Its build context is the **repo root** (so it can install from the single consolidated root `requirements.txt` instead of duplicating a second requirements file inside the module):

```bash
# From the repo root:
docker build -t zepto-support -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support
```

**Note:** I haven't run the Docker build locally since I don't have Docker installed on my machine, but it uses the standard `python:3.11-slim` base, installs the root `requirements.txt`, copies `support_assistant/`'s app code + `docs/`, pre-builds the ChromaDB collection, and runs via `uvicorn`.

## 12. Example outputs

**Module 1** — `python scrape_and_load.py`: **163 books** across **5 categories** (Travel 11, Mystery 32, Historical Fiction 26, Sequential Art 75, Classics 19), **0 rows dropped** as malformed. `run_queries.py`'s SQL-JOIN vs `pd.merge` equivalence check printed: `Do the SQL-JOIN result and the pandas pd.merge result match? True` (51/51 rows identical).

**Module 2** — test-set metrics: Logistic Regression acc 0.8090/AUC 0.8608; Decision Tree acc 0.8090/AUC 0.8560; Random Forest (baseline) acc 0.8146/AUC 0.8229. Tuned Random Forest (`GridSearchCV`, best params `{max_depth: 5, max_features: 'sqrt', n_estimators: 200}`) — real OOB score **0.8143**, tuned test accuracy **0.8258**. Regression (`fare`): MAE 21.10, RMSE 41.70, R² 0.348, Adjusted R² 0.309, residuals show heteroscedasticity. Full tables in `analytics/README.md`.

**Module 3** — live `POST /ask` responses (`MOCK_LLM` unset), captured directly from a running `uvicorn` process:

```json
// {"query": "How long does delivery take?"}
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01_chunk_01","doc_02_chunk_01","doc_04_chunk_01"],"confidence":1.0}

// {"query": "What is the capital of France?"}
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## 13. Git workflow

The repository was initialized and the full Git workflow was completed:

- `main` branch: initial commit containing all three modules
- `feature/complete-capstone` branch: documentation improvement commit
- Feature branch merged into `main` with `--no-ff` (merge commit visible in history)

Verify the history with:

```bash
git log --graph --oneline --decorate --all
```

## 14. Limitations

- I haven't tested the Docker build locally because Docker isn't installed on my machine.
- The real-LLM branch (`MOCK_LLM=0`) is implemented but I mainly tested the offline mock version.
- Module 1 scraped 5 categories (Travel, Mystery, Historical Fiction, Sequential Art, Classics) rather than all ~50 categories on the site — sufficient to clear the ≥60-books/≥3-categories requirement, not an attempt at a full-site crawl.
- All numeric results (Module 2 metrics, Module 1 row counts, Module 3 example responses) reflect my local run; re-running them might result in slight metric shifts depending on the environment.

## 15. Final project summary

This project contains three main modules: a scrape-to-SQL data pipeline, an analytics pipeline, and a GenAI support assistant. I developed and tested all of these components end-to-end, and the results documented here reflect the actual outputs from my local environment.
