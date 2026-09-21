# Module 2 — Analytics (Zepto Data & AI Platform capstone)

I built this module to showcase my full EDA and modeling workflow on the Titanic dataset. 
I ensured that `sns.load_dataset('titanic')` is called **exactly once**, in my `01_eda.ipynb` notebook, and then immediately persisted to `titanic.csv`. My `02_modeling.ipynb` and every number below is derived strictly from that CSV / the cleaned DataFrame built from it. All the numbers in this README were read directly from my executed notebook outputs (`01_eda.ipynb`, `02_modeling.ipynb`) — I didn't hand-invent any of them.

## My Files

- `01_eda.ipynb` — Here I perform profiling, missing-value handling, univariate/bivariate/multivariate analysis, correlation matrix, and an exploratory standardization check.
- `02_modeling.ipynb` — Here I execute a stratified split, my `ColumnTransformer`/`Pipeline` preprocessing, 3 classifiers, an imbalance handling comparison, `GridSearchCV` tuning with OOB score, my `fare` regression side-task, a final comparison, and a joblib save/reload test.
- `titanic.csv` — My one-time cached dataset (891 rows, 15 columns) used by every later step.
- `best_pipeline.joblib` — My complete fitted deployment pipeline (`ColumnTransformer` + tuned `RandomForestClassifier`) saved from `02_modeling.ipynb`.
- `_build_eda_nb.py`, `_build_modeling_nb.py` — The `nbformat` python scripts I wrote to generate the two notebooks programmatically (I kept them for reproducibility and provenance).

## My EDA highlights (`01_eda.ipynb`)

**My Dataset shape:** 891 rows × 15 columns as loaded.

**My Missing values (measured, not assumed) and my handling decisions:**

| Column | Missing % | Bucket | My Decision |
|---|---|---|---|
| `deck` | 77.22% | >30% | I dropped this column — it was too sparse to impute or encode meaningfully. |
| `age` | 19.87% | 5–30% | I imputed this with the median (which I know is robust to `age`'s outliers/skew). |
| `embarked` | 0.22% (2 rows) | <5% | I just dropped those 2 rows. |
| `embark_town` | 0.22% (2 rows) | <5% | These are the same 2 rows as `embarked`, so my drop above already resolved it. |

Cleaned shape after my steps: (889, 14).

**My IQR outlier analysis:**
- `age`: Q1=22.0, Q3=35.0, IQR=13.0, bounds=[2.5, 54.5] → **65 outliers**.
- `fare`: Q1=7.8958, Q3=31.0, IQR=23.1042, bounds=[-26.76, 65.66] → **114 outliers**.

**My Fare skew analysis:** I found the mean=32.0967, median=14.4542, mode=8.0500 → mean > median > mode. This confirms my hypothesis that `fare` is heavily **right-skewed**.

**My Bivariate survival rates (using boolean masking):** female 74.04% vs male 18.89%; by class 1st=62.62%, 2nd=47.28%, 3rd=24.24%; combined, e.g. 1st-class women 96.74%, 3rd-class men 13.54%.

**My Top-2 correlations (from my 6-column matrix: survived, pclass, age, sibsp, parch, fare):**
1. `pclass` vs `fare` = **-0.5482**. A lower class number means a higher fare, which makes perfect sense since class is effectively priced into the ticket fare.
2. `sibsp` vs `parch` = **+0.4145**. Larger family units travel with both more siblings/spouses and more parents/children together.

**My 4 multivariate charts:** I plotted a grouped bar of survival by class×sex, a box plot of age by survived×sex, a scatter of age vs fare colored by survived/styled by class, and a pclass×embark_town survival-rate heatmap. I included my own 2–4 sentence interpretation as markdown cells directly beneath each chart in `01_eda.ipynb` (Section 6). My narrative shows how "sex and class dominate survival, while fare/port add smaller secondary effects".

**My Standardization check:** I z-scored `age`/`fare` on the full cleaned frame. Before standardization: mean=29.32/std=12.98 (age) and mean=32.10/std=49.70 (fare). After my standardization step, both `age_z` and `fare_z` had mean ≈0 (order 1e-16) and std ≈1, proving my scaling worked.

## My Modeling highlights (`02_modeling.ipynb`)

**My Class balance (`survived`):** 549 not-survived / 340 survived overall (61.75% / 38.25%). I successfully preserved this in my stratified train (711 rows) / test (178 rows) split.

**My Classifier comparison (test set, Section 4/9):**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8608 |
| Decision Tree (depth 4) | 0.8090 | 0.8148 | 0.6471 | 0.7213 | 0.8560 |
| Random Forest (baseline) | 0.8146 | 0.7692 | 0.7353 | 0.7519 | 0.8229 |

**My Imbalance handling comparison (Random Forest, Section 6):**

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7692 | 0.7353 | 0.7519 |
| `class_weight='balanced'` | 0.7536 | 0.7647 | 0.7591 |
| SMOTE (train fold only) | 0.7727 | 0.7500 | 0.7612 |

I found that SMOTE handled the imbalance best (highest precision and F1); `class_weight` only won on recall but at a precision cost. I double-checked and verified that my SMOTE logic only resampled the encoded training arrays (439/272 → 439/439); the 178-row test fold was never touched.

**My GridSearchCV tuning (Section 7):** My best params were `{max_depth: 5, max_features: 'sqrt', n_estimators: 200}`, best CV accuracy was **0.8200**, and my real refit **OOB score = 0.8143**. My tuned test metrics: accuracy=0.8258, precision=0.8364, recall=0.6765, F1=0.748, AUC=0.843.

**My Regression side-task — predicting `fare` (Section 8):**
MAE=21.0998, RMSE=41.7022, R²=0.3482, Adjusted R²=0.3091 (n=178 test rows, p=10 encoded features). My residual-vs-predicted plot showed a funnel shape (tight near 0 for low predicted fares, spreading to a max residual of ≈+432 at higher predicted fares) — I correctly identified this as **heteroscedasticity**, driven by `fare`'s own right-skew/outliers. I made sure to report classifier and regression metrics in separate tables since they are not on a comparable scale.

**My Deployment recommendation:** I highly recommend the tuned Random Forest (`best_rf_pipe`). I chose it because it had the best accuracy (0.8258) and precision (0.8364) of every model I built, a competitive AUC (0.843), and an OOB score (0.8143) that closely tracks its test accuracy, indicating very good generalization. Also, unlike my single Decision Tree it is an ensemble, and unlike my plain Logistic Regression, it actually benefited measurably from my hyperparameter tuning.

**My Joblib save/reload test (Section 10):** I successfully saved the tuned Random Forest pipeline (my preprocessing `ColumnTransformer` + my classifier, bundled as one combined `Pipeline`) to `best_pipeline.joblib`. I reloaded it with `joblib.load` and called `.predict()`/`.predict_proba()` directly on 3 raw unprocessed rows from the original data. The predictions `[0, 1, 1]` identically matched the actual `survived` values `[0, 1, 1]` for those rows — meaning my end-to-end save/reload/predict-on-raw-input test **passed perfectly**.
