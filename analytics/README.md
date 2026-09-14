# Module 2 — Analytics (Zepto Data & AI Platform capstone)

This module contains the full EDA + modeling workflow on the Titanic dataset.
`sns.load_dataset('titanic')` is called **exactly once**, in `01_eda.ipynb`,
and immediately persisted to `titanic.csv`; `02_modeling.ipynb` and every
number below is derived only from that CSV / the cleaned DataFrame built
from it. All numbers in this README were read directly from the executed
notebook outputs (`01_eda.ipynb`, `02_modeling.ipynb`) — none were
hand-invented.

## Files

- `01_eda.ipynb` — profiling, missing-value handling, univariate/bivariate/
  multivariate analysis, correlation matrix, exploratory standardization check.
- `02_modeling.ipynb` — stratified split, `ColumnTransformer`/`Pipeline`
  preprocessing, 3 classifiers, imbalance handling comparison, `GridSearchCV`
  tuning with OOB score, `fare` regression, final comparison, joblib save/reload.
- `titanic.csv` — the one-time cached dataset (891 rows, 15 columns) used by
  every later step.
- `best_pipeline.joblib` — the complete fitted deployment pipeline
  (`ColumnTransformer` + tuned `RandomForestClassifier`) from `02_modeling.ipynb`.
- `_build_eda_nb.py`, `_build_modeling_nb.py` — the `nbformat` scripts used to
  author the two notebooks (kept for reproducibility/provenance).

## EDA highlights (`01_eda.ipynb`)

**Dataset shape:** 891 rows × 15 columns as loaded.

**Missing values (measured, not assumed) and handling decisions:**

| Column | Missing % | Bucket | Decision |
|---|---|---|---|
| `deck` | 77.22% | >30% | Dropped the column — too sparse to impute or encode meaningfully. |
| `age` | 19.87% | 5–30% | Imputed with the median (robust to `age`'s outliers/skew). |
| `embarked` | 0.22% (2 rows) | <5% | Dropped those 2 rows. |
| `embark_town` | 0.22% (2 rows) | <5% | Same 2 rows as `embarked`, resolved by the same drop. |

Cleaned shape after these steps: (889, 14).

**IQR outlier analysis:**
- `age`: Q1=22.0, Q3=35.0, IQR=13.0, bounds=[2.5, 54.5] → **65 outliers**.
- `fare`: Q1=7.8958, Q3=31.0, IQR=23.1042, bounds=[-26.76, 65.66] → **114 outliers**.

**Fare skew:** mean=32.0967, median=14.4542, mode=8.0500 → mean > median > mode,
confirming `fare` is **right-skewed**.

**Bivariate survival rates (boolean masking):** female 74.04% vs male 18.89%;
by class 1st=62.62%, 2nd=47.28%, 3rd=24.24%; combined, e.g. 1st-class women
96.74%, 3rd-class men 13.54%.

**Top-2 correlations (6-column matrix: survived, pclass, age, sibsp, parch, fare):**
1. `pclass` vs `fare` = **-0.5482** (lower class number → higher fare; class is
   effectively priced into the ticket fare).
2. `sibsp` vs `parch` = **+0.4145** (larger family units travel with both more
   siblings/spouses and more parents/children together).

**4 multivariate charts** (grouped bar of survival by class×sex, box plot of
age by survived×sex, scatter of age vs fare colored by survived/styled by
class, and a pclass×embark_town survival-rate heatmap) each have their own
2–4 sentence interpretation as markdown cells directly beneath them in
`01_eda.ipynb` (Section 6), building a "sex and class dominate survival,
fare/port add smaller secondary effects" narrative.

**Standardization check:** `age`/`fare` z-scored on the full cleaned frame;
before mean=29.32/std=12.98 (age) and mean=32.10/std=49.70 (fare); after,
both `age_z` and `fare_z` have mean ≈0 (order 1e-16) and std ≈1.

## Modeling highlights (`02_modeling.ipynb`)

**Class balance (`survived`):** 549 not-survived / 340 survived overall
(61.75% / 38.25%), preserved in the stratified train (711 rows) / test
(178 rows) split.

**Classifier comparison (test set, Section 4/9):**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8608 |
| Decision Tree (depth 4) | 0.8090 | 0.8148 | 0.6471 | 0.7213 | 0.8560 |
| Random Forest (baseline) | 0.8146 | 0.7692 | 0.7353 | 0.7519 | 0.8229 |

**Imbalance handling comparison (Random Forest, Section 6):**

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7692 | 0.7353 | 0.7519 |
| `class_weight='balanced'` | 0.7536 | 0.7647 | 0.7591 |
| SMOTE (train fold only) | 0.7727 | 0.7500 | 0.7612 |

SMOTE handled the imbalance best (highest precision and F1); `class_weight`
only wins on recall at a precision cost. SMOTE was verified to resample only
the encoded training arrays (439/272 → 439/439); the 178-row test fold was
never touched.

**GridSearchCV tuning (Section 7):** best params
`{max_depth: 5, max_features: 'sqrt', n_estimators: 200}`, best CV accuracy
**0.8200**, real refit **OOB score = 0.8143**. Tuned test metrics:
accuracy=0.8258, precision=0.8364, recall=0.6765, F1=0.748, AUC=0.843.

**Regression side-task — predicting `fare` (Section 8):**
MAE=21.0998, RMSE=41.7022, R²=0.3482, Adjusted R²=0.3091 (n=178 test rows,
p=10 encoded features). The residual-vs-predicted plot shows a funnel shape
(tight near 0 for low predicted fares, spreading to a max residual of
≈+432 at higher predicted fares) — this is **heteroscedasticity**, driven by
`fare`'s own right-skew/outliers. Classifier and regression metrics are
reported in separate tables and are not on a comparable scale.

**Deployment recommendation:** the tuned Random Forest (`best_rf_pipe`) is
recommended — it has the best accuracy (0.8258) and precision (0.8364) of
every model built, a competitive AUC (0.843), and an OOB score (0.8143) that
closely tracks its test accuracy, indicating good generalization; unlike the
single Decision Tree it is an ensemble, and unlike plain Logistic Regression
it benefited measurably from tuning.

**Joblib save/reload test (Section 10):** the tuned Random Forest pipeline
(preprocessing `ColumnTransformer` + classifier, as one combined `Pipeline`)
was saved to `best_pipeline.joblib`, reloaded with `joblib.load`, and called
`.predict()`/`.predict_proba()` directly on 3 raw unprocessed rows from the
original data. Predictions `[0, 1, 1]` matched the actual `survived` values
`[0, 1, 1]` for those rows — the end-to-end save/reload/predict-on-raw-input
test **passed**.
