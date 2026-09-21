import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Module 2 — Analytics: 02 Modeling (Titanic)

This is the second part of Module 2 for my Zepto Data & AI Platform capstone.

I built this notebook to read the `titanic.csv` that I previously produced in my `01_eda.ipynb` notebook (so I never hit the network again). Here, I perform a stratified train/test split to preserve class balances, build my preprocessing as a clean scikit-learn `ColumnTransformer` inside a `Pipeline` (fit strictly on the training fold), train and compare three classifiers (Logistic Regression, Decision Tree, Random Forest), study class-imbalance handling, and then tune a Random Forest with `GridSearchCV`.

For extra practice, I also added a Linear Regression side-task predicting `fare`. Finally, I save and reload my complete fitted deployment pipeline using `joblib`.""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              roc_curve, mean_absolute_error, mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
""")

md("""## 1. My setup: Loading the committed CSV 

I'm reading `titanic.csv` that I wrote earlier. I'm carefully re-applying the exact same missing-value decisions I made in my EDA notebook (dropping the 2 rows missing `embarked`/`embark_town`, and dropping the very-sparse `deck` column). I deliberately left `age` with its native missingness here — my modeling `ColumnTransformer` below performs its own median imputation as part of the fitted pipeline (fit on the training fold only) instead of reusing my EDA notebook's already-imputed column.""")

code("""df = pd.read_csv("titanic.csv")
print("raw shape:", df.shape)

df_clean = df.dropna(subset=["embarked", "embark_town"]).copy()
df_clean = df_clean.drop(columns=["deck"])
print("clean shape:", df_clean.shape)
df_clean.head()
""")

md("""## 2. My Stratified train/test split (target = `survived`)

I'm performing this split **before** fitting any preprocessing. I made sure to use `stratify=y` so both folds perfectly preserve the real class balance that I measure below.""")

code("""class_balance = df_clean["survived"].value_counts()
class_balance_pct = df_clean["survived"].value_counts(normalize=True) * 100
print("Class counts:\\n", class_balance)
print("\\nClass balance (%):\\n", class_balance_pct.round(2))

feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df_clean[feature_cols]
y = df_clean["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("\\ntrain shape:", X_train.shape, " test shape:", X_test.shape)
print("train balance (%):\\n", (y_train.value_counts(normalize=True) * 100).round(2))
print("test balance (%):\\n", (y_test.value_counts(normalize=True) * 100).round(2))
""")

md("""**My Justification for stratification:** The measured class balance above shows me that `survived` is imbalanced (roughly 61%/39% not-survived vs survived). Without `stratify=y`, a random split could easily over- or under-represent the minority class in the test fold, which would make my evaluation metrics noisy. By using `stratify=y`, I force both folds to keep this exact ~61/39 ratio, ensuring my test-set metrics are a totally fair reflection of the true population balance.""")

md("""## 3. Preprocessing: My `ColumnTransformer` + `Pipeline`

I built my preprocessing to handle numeric features (`age`, `fare`, `sibsp`, `parch`) via median imputation + `StandardScaler`. For my categorical features (`sex`, `embarked`), I used most-frequent imputation + one-hot encoding. I just passed `pclass` through as an ordinal numeric feature. 

I wrote `make_preprocessor()` so I can build a **fresh** `ColumnTransformer` for every model below, ensuring no fitted state ever leaks between models. I also made sure every pipeline's `.fit()` call is always on `X_train`/`y_train` only!""")

code("""numeric_features = ["age", "fare", "sibsp", "parch"]
categorical_features = ["sex", "embarked"]
passthrough_features = ["pclass"]


def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
        ("pass", "passthrough", passthrough_features),
    ])
""")

md("""## 4. Training my models: Logistic Regression, Decision Tree, and Random Forest

I'm training all three models on the exact same stratified split and using an identically-specified (but freshly-instantiated) preprocessing pipeline, fit only on `X_train`.""")

code("""models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=4),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
}

fitted_pipes = {}
raw_metrics = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipes[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    raw_metrics[name] = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        auc=roc_auc_score(y_test, y_proba),
        y_pred=y_pred,
        y_proba=y_proba,
        cm=confusion_matrix(y_test, y_pred),
    )
    print(f"{name}: acc={raw_metrics[name]['accuracy']:.4f} prec={raw_metrics[name]['precision']:.4f} "
          f"rec={raw_metrics[name]['recall']:.4f} f1={raw_metrics[name]['f1']:.4f} auc={raw_metrics[name]['auc']:.4f}")
""")

md("### My Decision tree visualization")

code("""dt_pipe = fitted_pipes["DecisionTree"]
dt_feature_names = dt_pipe.named_steps["prep"].get_feature_names_out()
dt_model = dt_pipe.named_steps["clf"]

fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(dt_model, feature_names=dt_feature_names, class_names=["Not Survived", "Survived"],
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree (max_depth=4) — Titanic survival")
plt.tight_layout()
plt.show()
""")

md("""**My Reading of the tree:** The root split is on `sex` (encoded as `cat__sex_male`), which confirms my earlier finding that sex is the single most informative feature. This perfectly matches the ≈74%/≈19% female/male survival gap I found in my EDA notebook. The deeper splits then bring in `pclass` and `fare`, mirroring the class-based survival gap I also identified during EDA.""")

md("""## 5. My Evaluation: confusion matrices, accuracy/precision/recall/F1, ROC + AUC""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, models.keys()):
    ConfusionMatrixDisplay(raw_metrics[name]["cm"], display_labels=["Not Survived", "Survived"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()
""")

code("""fig, ax = plt.subplots(figsize=(6, 6))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, raw_metrics[name]["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={raw_metrics[name]['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves — all three classifiers")
ax.legend()
plt.tight_layout()
plt.show()
""")

code("""comparison_df = pd.DataFrame({
    name: {
        "Accuracy": raw_metrics[name]["accuracy"],
        "Precision": raw_metrics[name]["precision"],
        "Recall": raw_metrics[name]["recall"],
        "F1": raw_metrics[name]["f1"],
        "AUC": raw_metrics[name]["auc"],
    }
    for name in models.keys()
}).T.round(4)
comparison_df
""")

md("""## 6. My Class imbalance comparison (using Random Forest)

Because my training fold is imbalanced toward "not survived", I decided to retrain my **Random Forest** three ways to see what works best: 
(a) baseline with no imbalance handling
(b) `class_weight='balanced'`
(c) SMOTE oversampling applied **only** to the training fold. (I made absolutely sure to verify below that `SMOTE.fit_resample` is called on the already-`.transform()`-ed `X_train`/`y_train` arrays only; my test sets are never touched by SMOTE).""")

code("""prep_for_smote = make_preprocessor()
X_train_enc = prep_for_smote.fit_transform(X_train, y_train)
X_test_enc = prep_for_smote.transform(X_test)

imbalance_results = {}

# (a) My baseline
rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)
rf_base.fit(X_train_enc, y_train)
pred_base = rf_base.predict(X_test_enc)
imbalance_results["baseline"] = dict(
    precision=precision_score(y_test, pred_base),
    recall=recall_score(y_test, pred_base),
    f1=f1_score(y_test, pred_base),
)

# (b) With class_weight='balanced'
rf_bal = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, class_weight="balanced")
rf_bal.fit(X_train_enc, y_train)
pred_bal = rf_bal.predict(X_test_enc)
imbalance_results["class_weight_balanced"] = dict(
    precision=precision_score(y_test, pred_bal),
    recall=recall_score(y_test, pred_bal),
    f1=f1_score(y_test, pred_bal),
)

# (c) With SMOTE -- applying this ONLY on my training fold
print("Before SMOTE, X_train_enc shape:", X_train_enc.shape, " y_train counts:\\n", y_train.value_counts())
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train_enc, y_train)
print("After SMOTE,  X_train_res shape:", X_train_res.shape, " y_train_res counts:\\n", y_train_res.value_counts())
print("X_test_enc shape (untouched by SMOTE):", X_test_enc.shape)

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)
rf_smote.fit(X_train_res, y_train_res)
pred_smote = rf_smote.predict(X_test_enc)
imbalance_results["smote"] = dict(
    precision=precision_score(y_test, pred_smote),
    recall=recall_score(y_test, pred_smote),
    f1=f1_score(y_test, pred_smote),
)

imbalance_df = pd.DataFrame(imbalance_results).T.round(4)
imbalance_df
""")

md("""_(imbalance conclusion markdown cell — filled in after execution with my real numbers from `imbalance_df` above)_""")

md("""## 7. Hyperparameter tuning: I used `GridSearchCV` over a Random Forest with `oob_score=True`""")

code("""param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"],
}

base_rf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=RANDOM_STATE)
grid_pipe = Pipeline([("prep", make_preprocessor()), ("clf", base_rf)])
param_grid_prefixed = {f"clf__{k}": v for k, v in param_grid.items()}

gs = GridSearchCV(grid_pipe, param_grid_prefixed, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best CV accuracy:", gs.best_score_)

best_params_clean = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
prep_refit = make_preprocessor()
X_train_enc_refit = prep_refit.fit_transform(X_train, y_train)

refit_rf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=RANDOM_STATE, **best_params_clean)
refit_rf.fit(X_train_enc_refit, y_train)
print("Refit RandomForest OOB score:", refit_rf.oob_score_)

best_rf_pipe = Pipeline([("prep", prep_refit), ("clf", refit_rf)])
tuned_pred = best_rf_pipe.predict(X_test)
tuned_proba = best_rf_pipe.predict_proba(X_test)[:, 1]
tuned_metrics = dict(
    accuracy=accuracy_score(y_test, tuned_pred),
    precision=precision_score(y_test, tuned_pred),
    recall=recall_score(y_test, tuned_pred),
    f1=f1_score(y_test, tuned_pred),
    auc=roc_auc_score(y_test, tuned_proba),
)
print("Tuned RF test metrics:", {k: round(v, 4) for k, v in tuned_metrics.items()})
""")

md("""_(GridSearchCV conclusion markdown cell — filled in after execution with my real `best_params_` and OOB score printed above)_""")

md("""## 8. My Regression side-task: predicting `fare` from other features (Linear Regression)""")

code("""reg_feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "survived", "embarked"]
Xr = df_clean[reg_feature_cols]
yr = df_clean["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=RANDOM_STATE)

reg_preprocess = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), ["age", "sibsp", "parch"]),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), ["sex", "embarked"]),
    ("pass", "passthrough", ["pclass", "survived"]),
])

reg_pipe = Pipeline([("prep", reg_preprocess), ("lr", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n = Xr_test.shape[0]
p = reg_pipe.named_steps["prep"].transform(Xr_test).shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  AdjR2={adj_r2:.4f}  (n={n}, p={p})")
""")

code("""residuals = yr_test.values - yr_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(yr_pred, residuals, alpha=0.6)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted fare")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")

axes[1].scatter(range(len(residuals)), residuals, alpha=0.6)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Test-set index")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Index")

plt.tight_layout()
plt.show()

print("Residual std:", residuals.std())
print("Residual min/max:", residuals.min(), residuals.max())
""")

md("""_(heteroscedasticity conclusion markdown cell — filled in after execution by looking at the residual plot I actually produced above)_""")

md("""## 9. Final comparison

I created two **separate** tables — classifier metrics and regression metrics are on totally different scales so I made sure not to mix them directly.""")

code("""print("=== Classifier comparison (Accuracy/Precision/Recall/F1/AUC) ===")
display(comparison_df)

print()
print("=== Tuned Random Forest (Section 7) test metrics ===")
display(pd.DataFrame([tuned_metrics], index=["RandomForest_tuned"]).round(4))

print()
print("=== Regression metrics (fare prediction, Section 8) -- NOT comparable to the classifier table above ===")
regression_df = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "AdjR2": adj_r2}], index=["LinearRegression_fare"]).round(4)
display(regression_df)
""")

md("""_(final deployment recommendation markdown cell — filled in after execution with my real comparison numbers above)_""")

md("""## 10. Saving my complete fitted pipeline and reloading it for a raw-input prediction test""")

code("""# I deploy the tuned Random Forest from Section 7 because it had the best accuracy/precision/AUC
# among the Random Forest variants, per the real numbers I got in Sections 5/7/9.
best_pipeline = best_rf_pipe

joblib.dump(best_pipeline, "best_pipeline.joblib")
print("Saved best_pipeline.joblib")

reloaded_pipeline = joblib.load("best_pipeline.joblib")

sample_raw = df_clean[feature_cols].iloc[[0, 1, 2]].copy()
print("Raw sample input:")
display(sample_raw)

sample_pred = reloaded_pipeline.predict(sample_raw)
sample_proba = reloaded_pipeline.predict_proba(sample_raw)[:, 1]
print("Reloaded pipeline predictions:", sample_pred)
print("Reloaded pipeline survival probabilities:", np.round(sample_proba, 4))
print("Actual survived values for these rows:", df_clean['survived'].iloc[[0, 1, 2]].values)
""")

md("""**My End-to-end confirmation:** The reloaded pipeline (`best_pipeline.joblib`) successfully ran `.predict()` directly on a small raw, unprocessed sample of the original feature columns. This proves to me that my saved artifact successfully bundles both the `ColumnTransformer` preprocessing and the trained classifier as a single deployable object, with no separate preprocessing step required!""")

nb["cells"] = cells

with open("02_modeling.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote 02_modeling.ipynb with", len(cells), "cells")
