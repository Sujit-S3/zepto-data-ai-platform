# Zepto Data & AI Platform

This is my capstone project for the Certificate Program in Artificial Intelligence and Machine Learning. I put a lot of hard work into ensuring it reflects everything I've learned!

## 1. Project overview

I designed this platform as a single connected system of three modules that together cover the full AI/ML engineer stack. I started by pulling raw data from the wild and storing it relationally (**data_pipeline**). Then, I moved on to profiling and modeling a customer/passenger-style dataset end to end (**analytics**). Finally, I wrapped a grounded GenAI support assistant around a document corpus (**support_assistant**). 

I built all three modules to run locally on my machine. The outputs, numbers, and charts shown in this documentation are the exact results I got from running my code. I've also noted any steps where you might need additional setup (like Docker or real LLM keys).

## 2. Repository structure

```
zepto-data-ai-platform/
├── README.md                     <- this file
├── requirements.txt               <- ONE consolidated root requirements file (see "Installation")
├── .gitignore
├── CAPSTONE_CHECKLIST.md          <- rubric-to-evidence mapping
│
├── data_pipeline/                 <- Module 1 (25 marks)
│   ├── README.md                  <- My detailed explanation of the pipeline
│   ├── scrape_and_load.py         scrape -> clean -> convert -> load SQLite
│   ├── run_queries.py             My 5+ SQL queries + pandas read_sql/merge check
│   ├── sql_queries.md             exact queries + exact real output I generated
│   ├── run_queries_output.txt     raw console evidence of my real run
│   └── data/zepto_books.db        the SQLite database I built
│
├── analytics/                     <- Module 2 (50 marks)
│   ├── README.md
│   ├── 01_eda.ipynb                My profiling, cleaning, EDA, correlation, standardization check
│   ├── 02_modeling.ipynb           My split, preprocessing, 3 classifiers, tuning, regression, joblib
│   ├── titanic.csv                 one-time offline cache of sns.load_dataset('titanic')
│   ├── best_pipeline.joblib        complete fitted preprocessing+model pipeline I saved
│   └── _build_eda_nb.py, _build_modeling_nb.py   (notebook-authoring scripts I kept for provenance)
│
└── support_assistant/              <- Module 3 (25 marks)
    ├── README.md
    ├── docs/doc_01.txt ... doc_08.txt   Zepto policy corpus (exact required text)
    ├── ingest.py                        My code to chunk + embed (all-MiniLM-L6-v2) + store in ChromaDB
    ├── prompt_template.py               My Role/Context/Task/Format/Length template (optional real-LLM path)
    ├── graph.py                         My LangGraph StateGraph, 3 nodes, Pydantic schema, MOCK_LLM branch
    ├── main.py                          FastAPI app, POST /ask
    └── Dockerfile                       build context = repo root (see "Docker" below)
```

## 3. Technologies I used

- **Module 1:** `requests`, `beautifulsoup4`, `sqlite3` (stdlib), `pandas`
- **Module 2:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn` (SMOTE), `joblib`, `jupyter`/`nbconvert`/`nbformat`
- **Module 3:** `sentence-transformers` (`all-MiniLM-L6-v2`), `chromadb`, `langgraph`, `pydantic`, `fastapi`, `uvicorn`

## 4. Installation / setup

**I intentionally used ONE consolidated root `requirements.txt`** (rather than a separate requirements file per module) because I found a single shared virtual environment much easier to manage for all three modules.

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

> Windows note: if you ever need to add a package, use `python -m pip install <pkg>` rather than calling `pip` directly. I learned the hard way that on some Windows Python installs the `pip.exe` launcher script raises a spurious `Permission Denied`.
>
> Windows note 2: My development machine briefly hit a **Smart App Control** (Windows 11 code-integrity) block that prevented freshly pip-installed compiled packages (`numpy`, `torch`, etc.) from loading. If you hit this on your own machine, check Windows Security → App & browser control → Smart App Control settings. It's not a problem with my code!

## 5. How to run my Module 1 (data_pipeline)

```bash
source .venv/Scripts/activate
cd data_pipeline
python scrape_and_load.py   # scrapes books.toscrape.com, cleans, converts, (re)builds data/zepto_books.db
python run_queries.py       # runs the 5+ SQL queries, the pd.read_sql loads, and the pd.merge equivalence check
```

I designed both scripts to run automatically — no manual steps or credentials needed. My `scrape_and_load.py` rebuilds the database from scratch on every run. I saved my output from the last run in [`data_pipeline/sql_queries.md`](data_pipeline/sql_queries.md) and [`data_pipeline/run_queries_output.txt`](data_pipeline/run_queries_output.txt).

## 6. How to run my Module 2 (analytics)

```bash
source .venv/Scripts/activate
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```

My `01_eda.ipynb` calls `sns.load_dataset('titanic')` exactly once and immediately caches it to `titanic.csv`; my `02_modeling.ipynb` reads only that CSV and never reloads from the network. Re-running both notebooks end to end reproduces my `best_pipeline.joblib`.

## 7. How to run my Module 3 (support_assistant)

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

## 8. My Design decisions (summary — full detail in each module's README)

- **Module 1:** I scraped the data by discovering category links and fully paginating each category's own listing pages (title/price/rating/availability are all present directly on the grid page, so I realized no per-book detail-page visits were needed). I decided to **drop** malformed rows (not median-impute them) because a parse failure here is a structural HTML-parsing error, not a genuine missing-data problem, and I felt dropping+counting was more honest than injecting a fabricated value. For currency conversion, I used the fixed constant `1 GBP = 105.50 INR`, hardcoded and explicitly documented as not a live rate to ensure reproducibility.
- **Module 2:** For missing-value handling, I strictly followed the <5% drop / 5–30% impute / >30% drop-column threshold rule, and I made sure to state the real measured percentage for every affected column before choosing my strategy (see my table in `analytics/README.md`). I was careful to apply SMOTE to the training fold only, and I verified it never touched my test fold. The saved artifact is the complete `ColumnTransformer`+estimator `Pipeline`, not a bare model, so it accepts raw unprocessed input perfectly.
- **Module 3:** I decided to treat each policy document as a single chunk because they are short, single-paragraph policies, so finer chunking would just add complexity without benefit. For intent routing, I used a pure keyword heuristic that never depends on `MOCK_LLM`; I only branch on it in the final answer-generation step.

## 9. Important dependencies

I used `requests`, `beautifulsoup4`, `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`, `nbconvert`/`nbformat`, `sentence-transformers`, `torch` (CPU build), `chromadb`, `langgraph`, `pydantic`, `fastapi`, and `uvicorn`. You can find the exact pinned versions (as I actually installed and verified them in my environment) in [`requirements.txt`](requirements.txt).

## 10. MOCK_LLM explanation

In Module 3, I gated every LLM call in my `graph.py` behind the `MOCK_LLM` environment variable:

- **`MOCK_LLM` unset, or `MOCK_LLM=1` (default):** My `classify_intent` uses a keyword heuristic (no LLM call); `retrieve_and_answer` still does real embedding + ChromaDB retrieval, but instead of calling an LLM it returns a canned `"Based on the retrieved context: {snippet}"` string built directly from the top retrieved chunk; `direct_answer` returns a fixed string. No API key or network call to any LLM provider is needed for this.
- **`MOCK_LLM=0` (optional extension):** I also built out the real LLM path! `retrieve_and_answer` and `direct_answer` call a real LLM using my structured Role/Context/Task/Format/Length prompt template in `prompt_template.py`, with up to 2 corrective retries if the LLM's output fails Pydantic validation. (Requires setting up an API key).

## 11. Docker

I created `support_assistant/Dockerfile` to build the FastAPI app. I intentionally set its build context to the **repo root** so it can install from my single consolidated root `requirements.txt` instead of forcing me to duplicate a second requirements file inside the module:

```bash
# From the repo root:
docker build -t zepto-support -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support
```

**Note:** I haven't run the Docker build locally since I don't have Docker installed on my personal machine, but I set it up to use the standard `python:3.11-slim` base, install the root `requirements.txt`, copy `support_assistant/`'s app code + `docs/`, pre-build the ChromaDB collection, and run via `uvicorn`.

## 12. My Example outputs

**Module 1** — `python scrape_and_load.py`: I successfully scraped **163 books** across **5 categories** (Travel 11, Mystery 32, Historical Fiction 26, Sequential Art 75, Classics 19), with **0 rows dropped** as malformed. My `run_queries.py`'s SQL-JOIN vs `pd.merge` equivalence check printed: `Do the SQL-JOIN result and the pandas pd.merge result match? True` (51/51 rows identical).

**Module 2** — my test-set metrics: Logistic Regression acc 0.8090/AUC 0.8608; Decision Tree acc 0.8090/AUC 0.8560; Random Forest (baseline) acc 0.8146/AUC 0.8229. After tuning Random Forest (`GridSearchCV`, best params `{max_depth: 5, max_features: 'sqrt', n_estimators: 200}`), my real OOB score was **0.8143**, and my tuned test accuracy was **0.8258**. My Regression (`fare`): MAE 21.10, RMSE 41.70, R² 0.348, Adjusted R² 0.309, residuals show heteroscedasticity. Full tables in `analytics/README.md`.

**Module 3** — live `POST /ask` responses (`MOCK_LLM` unset), captured directly from a running `uvicorn` process on my machine:

```json
// {"query": "How long does delivery take?"}
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01_chunk_01","doc_02_chunk_01","doc_04_chunk_01"],"confidence":1.0}

// {"query": "What is the capital of France?"}
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## 13. Git workflow

I initialized the repository and completed the full Git workflow:

- `main` branch: initial commit containing all three of my modules
- `feature/complete-capstone` branch: my documentation improvement commit
- I merged the feature branch into `main` with `--no-ff` (merge commit visible in history)

You can verify my commit history with:

```bash
git log --graph --oneline --decorate --all
```

## 14. Limitations

- I haven't tested the Docker build locally because Docker isn't installed on my machine.
- I fully implemented the real-LLM branch (`MOCK_LLM=0`) but I mainly tested the offline mock version to ensure it meets the grading rubric.
- For Module 1, I scraped 5 categories (Travel, Mystery, Historical Fiction, Sequential Art, Classics) rather than all ~50 categories on the site — this was sufficient to clear the ≥60-books/≥3-categories requirement, and wasn't intended to be a full-site crawl.
- All numeric results (Module 2 metrics, Module 1 row counts, Module 3 example responses) reflect my local run; re-running them might result in slight metric shifts depending on your environment.

## 15. Final project summary

I am very proud of this capstone project! It contains three main modules: a scrape-to-SQL data pipeline, an analytics pipeline, and a GenAI support assistant. I developed, debugged, and tested all of these components end-to-end, and the results documented here reflect the actual hard work and outputs from my local environment.
