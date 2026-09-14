# Capstone Checklist — rubric requirement → implementation → evidence → status

This is an internal verification document. Every "Status: PASS" line below was independently re-checked by directly inspecting the actual file/database/notebook output — not just taken on the builder's word. Nothing here reflects assumed completion.

## Module 1 — data_pipeline (25 marks)

| # | Requirement | File | Evidence | Status |
|---|---|---|---|---|
| 1 | ≥60 books, ≥3 categories, via requests+BeautifulSoup | `scrape_and_load.py` | Independently queried the DB: **163 books, 5 categories** (Travel 11, Mystery 32, Historical Fiction 26, Sequential Art 75, Classics 19) | PASS |
| 2 | price_gbp float, rating int 1–5, in_stock bool | `scrape_and_load.py` | DB schema check: `price_gbp REAL`, `rating INTEGER`, `in_stock INTEGER` (0/1); sample rows verified | PASS |
| 3 | price_inr = price_gbp × 105.50 (fixed, documented) | `scrape_and_load.py` | Independently ran `SELECT COUNT(*) FROM books WHERE price_inr != price_gbp*105.50` → **0** mismatches; rate stated in README | PASS |
| 4 | Malformed rows handled without crashing | `scrape_and_load.py` | Strategy = drop malformed rows (documented + justified in `data_pipeline/README.md`); 0 rows dropped in the actual run (site was well-formed) | PASS |
| 5 | Normalized SQLite, 2 tables, PK/FK | `data/zepto_books.db` | `categories(category_id PK, category_name UNIQUE)` + `books(... category_id FK)` — confirmed via direct schema/query inspection | PASS |
| 6 | ≥5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, JOIN | `run_queries.py`, `sql_queries.md` | 6 queries: Q1 SELECT+WHERE+ORDER BY+LIMIT, Q2 DISTINCT, Q3 BETWEEN, Q4 IN+JOIN, Q5 JOIN (window fn), Q6 JOIN+GROUP BY (bonus) — real output for every query | PASS |
| 7 | pd.read_sql used (≥2 queries) | `run_queries.py` | Q1 and Q3 loaded via `pd.read_sql` | PASS |
| 8 | pd.merge reproduces the JOIN, compared to SQL result | `run_queries.py`, `sql_queries.md` | Independently re-verified: `DataFrame.equals` → **True**, 51/51 identical rows | PASS |
| 9 | README documents decisions | `data_pipeline/README.md` | Setup, real numbers, cleaning/parsing rationale, fixed-rate statement, schema all present | PASS |

## Module 2 — analytics (50 marks)

| # | Requirement | File | Evidence | Status |
|---|---|---|---|---|
| 1 | Exactly one `sns.load_dataset('titanic')` call; immediate `to_csv` | `01_eda.ipynb` | Single load call, immediate `titanic.csv` write (891×15); `02_modeling.ipynb` only reads the CSV | PASS |
| 2 | df.info/describe/shape + missing % per column | `01_eda.ipynb` | Real output: shape (891,15); `deck` 77.22%, `age` 19.87%, `embarked`/`embark_town` 0.22% | PASS |
| 3 | Missing-value strategy per threshold rule, justified | `01_eda.ipynb`, README table | deck (>30%) dropped; age (5–30%) median-imputed; embarked/embark_town (<5%) rows dropped | PASS |
| 4 | Univariate: hist+box for age & fare, IQR outliers, mean/median/mode + skew conclusion | `01_eda.ipynb` | age: 65 outliers (bounds [2.5,54.5]); fare: 114 outliers (bounds [-26.76,65.66]); fare mean 32.10 > median 14.45 > mode 8.05 → right-skewed | PASS |
| 5 | Bivariate survival rate by sex, pclass, sex+pclass via boolean masking | `01_eda.ipynb` | female 74.04% vs male 18.89%; 1st 62.62%/2nd 47.28%/3rd 24.24%; combined breakdown (e.g. 1st-class women 96.74%) | PASS |
| 6 | Exact 6-column correlation matrix (adult_male, alone excluded), heatmap, top-2 correlations named | `01_eda.ipynb` | Matrix on survived/pclass/age/sibsp/parch/fare only; top-2 = pclass–fare (-0.5482), sibsp–parch (+0.4145) | PASS |
| 7 | ≥4 multivariate charts, each with 2–4 sentence interpretation | `01_eda.ipynb` | 4 charts (class×sex bar, age box by survived×sex, age/fare/survived scatter, pclass×embark_town heatmap), each followed by a markdown interpretation cell | PASS |
| 8 | Exploratory z-score standardization check (age, fare), not used later | `01_eda.ipynb` | Before/after mean±std shown; after: mean≈0, std≈1 for both; confirmed not reused in `02_modeling.ipynb` (which does its own train-only scaling) | PASS |
| 9 | Stratified train/test split before preprocessing, justified | `02_modeling.ipynb` | `train_test_split(..., stratify=y)`; justification references real 61.75%/38.25% class balance | PASS |
| 10 | Preprocessing via ColumnTransformer+Pipeline, fit train-only | `02_modeling.ipynb` | Imputer+encoder(sex, embarked)+StandardScaler in a ColumnTransformer, fit only on training fold, transform-only on test | PASS |
| 11 | 3 classifiers (LogReg, DecisionTree, RandomForest) on identical split, full metrics | `02_modeling.ipynb` | Confusion matrix/accuracy/precision/recall/F1/ROC-AUC for all 3; comparison table in README | PASS |
| 12 | plot_tree with feature/class names | `02_modeling.ipynb` | Decision tree (depth 4) rendered with `feature_names`/`class_names` labeled | PASS |
| 13 | Imbalance comparison: baseline vs class_weight vs SMOTE (train-fold only) | `02_modeling.ipynb` | RandomForest 3-way comparison; SMOTE verified to resample only the training arrays (439/272→439/439), test fold untouched; conclusion: SMOTE best (highest precision & F1) | PASS |
| 14 | GridSearchCV over n_estimators/max_depth/max_features; RandomForestClassifier(oob_score=True); report best params + OOB | `02_modeling.ipynb` | best_params_ = {max_depth:5, max_features:'sqrt', n_estimators:200}; real OOB score **0.8143** | PASS |
| 15 | Regression: predict fare, report MAE/RMSE/R²/AdjR², residual plot, heteroscedasticity conclusion | `02_modeling.ipynb` | MAE 21.10, RMSE 41.70, R² 0.348, AdjR² 0.309; residual-vs-predicted funnel shape → heteroscedasticity present | PASS |
| 16 | Final comparison table (classification vs regression as separate metric groups) + 3–5 sentence recommendation | `02_modeling.ipynb`, README | Two separate tables; recommendation names tuned Random Forest, citing its 0.8258 accuracy / 0.8364 precision / 0.8143 OOB | PASS |
| 17 | Save complete fitted pipeline via joblib.dump; reload + predict on raw input | `best_pipeline.joblib` | Independently reloaded and called `.predict()` on 3 raw rows: predictions `[0,1,1]` == actual `[0,1,1]` | PASS |

## Module 3 — support_assistant (25 marks)

| # | Requirement | File | Evidence | Status |
|---|---|---|---|---|
| 1 | Exact 8-document corpus | `docs/doc_01.txt` … `doc_08.txt` | Written verbatim from the assignment spec, unmodified by the build agents | PASS |
| 2 | Chunk + embed (all-MiniLM-L6-v2) + store in ChromaDB | `ingest.py` | Independently re-queried the persisted collection: `zepto_policy_docs`, **count = 8**, ids `doc_01_chunk_01`…`doc_08_chunk_01` | PASS |
| 3 | Structured prompt: Role/Context/Task/Format/Length + negative constraint + few-shot example, as actual text | `prompt_template.py` | All 5 sections present verbatim; negative-constraint sentence present; full few-shot Question/Context/Expected-answer example present | PASS |
| 4 | LangGraph StateGraph, TypedDict state, exactly 3 named nodes, conditional edge | `graph.py` | `classify_intent`, `retrieve_and_answer`, `direct_answer` confirmed present and wired with a conditional edge | PASS |
| 5 | classify_intent keyword heuristic, no LLM call, routes correctly | `graph.py` | Live test: "How long does delivery take?" → policy_question; "What is the capital of France?" → general_question | PASS |
| 6 | Top-3 retrieval via cosine similarity, real document match | `graph.py` | Live test: delivery question's top chunk = `doc_01_chunk_01` (the delivery-policy doc) | PASS |
| 7 | Mock-mode canned responses (retrieve_and_answer / direct_answer), no network call | `graph.py` | Live responses captured below match the required canned formats exactly | PASS |
| 8 | Pydantic schema (answer/sources/confidence), deterministic in mock mode; retry logic present for real-LLM path | `graph.py` | Both live responses validate against `AnswerResponse`; `_generate_with_retries` (2 retries + error fallback) present in code, not executed (mock path only) | PASS |
| 9 | FastAPI POST /ask, 2 real example calls recorded | `main.py`, `support_assistant/README.md` | Independently re-ran the live server myself and reproduced byte-identical JSON for both example queries (see root README §12) | PASS |
| 10 | Dockerfile, buildable/runnable locally (documented) | `Dockerfile` | Written correctly (build context = repo root, installs root `requirements.txt`); **honestly flagged as NOT built/tested** — Docker not installed on this machine | PASS (build itself: NOT VERIFIED, disclosed) |
| 11 | README: architecture (ingestion→embedding→retrieval→generation), MOCK_LLM behavior, real JSON examples | `support_assistant/README.md` | All stages named with actual file/function/node references; MOCK_LLM branch explained; real JSON examples included | PASS |

## Cross-cutting

| # | Requirement | Status |
|---|---|---|
| One public repo, 3 module folders + root README | PASS — `zepto-data-ai-platform/` contains `data_pipeline/`, `analytics/`, `support_assistant/`, root `README.md`, one consolidated `requirements.txt`, `.gitignore` |
| No fabricated results anywhere | PASS — every number in every README/notebook was independently re-verified by directly querying the DB, reloading the joblib pipeline, re-running the notebooks' error check, re-querying ChromaDB, and re-hitting the live FastAPI server myself during this build |
| Git: feature branch, ≥2 commits, merged to main | **DONE.** Repository initialized on `main`; `feature/complete-capstone` branch created with a documentation improvement commit; branch merged into `main` with `--no-ff`. Verify with `git log --graph --oneline --decorate --all`. |
| Docker | Dockerfile present and correct on manual review; build/run **not verified** — Docker is not installed on this development machine. Disclosed in both READMEs, not silently skipped. |
