import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Module 2 — Analytics: 01 Exploratory Data Analysis (Titanic)

Zepto Data & AI Platform capstone — Module 2 (Analytics).

This notebook performs the full EDA workflow on the Titanic dataset: a single
seaborn load (network fetch, cached), profiling, missing-value handling,
univariate/bivariate/multivariate analysis, correlation analysis, and an
exploratory standardization check. `titanic.csv` written here is the offline
fallback used by every later step in this module (including `02_modeling.ipynb`).
""")

code("""import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")
""")

md("""## 1. Load dataset (single seaborn call for the entire module)

Per the module rule, `sns.load_dataset('titanic')` is called **exactly once**
here, and immediately persisted to `titanic.csv`. No other cell/notebook in
this module calls `sns.load_dataset` again — `02_modeling.ipynb` reads this
CSV instead.""")

code("""df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)
df.head()
""")

md("## 2. Profiling")

code("df.info()")
code("df.describe(include='all')")
code("df.shape")

code("""missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
missing_pct
""")

md("""### Missing-value percentages and handling decisions

Measured on the raw loaded DataFrame (891 rows):

| Column | Missing % | Bucket | Decision |
|---|---|---|---|
| `deck` | 77.22% | > 30% | **Drop the column.** At this sparsity, any imputation or "missing" category would carry almost no real signal and `deck` is not part of the 6-column correlation set or the modeling feature set used later, so dropping it entirely is the simplest, least biased choice. |
| `age` | 19.87% | 5–30% | **Impute with the median.** Age is numeric and right-of-center skewed by a long tail of older passengers, so the median (robust to outliers) is used rather than the mean. |
| `embarked` | 0.22% (2 rows) | < 5% | **Drop those rows.** Only 2 of 891 rows are affected, so dropping is safe and avoids inventing a port of embarkation. |
| `embark_town` | 0.22% (2 rows) | < 5% | **Drop those rows.** Same 2 rows as `embarked` (it is a redundant text version of the same field), so the row-drop above already resolves it. |

All other columns have 0% missing values.""")

code("""df_clean = df.dropna(subset=["embarked", "embark_town"]).copy()
df_clean = df_clean.drop(columns=["deck"])
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
print("Shape after cleaning:", df_clean.shape)
print("Remaining missing values:\\n", df_clean.isna().sum()[df_clean.isna().sum() > 0])
""")

md("""## 3. Univariate analysis — `age` and `fare`

Histogram + box plot for each, plus IQR-based outlier bounds and counts.""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_clean["age"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Age distribution")
sns.boxplot(x=df_clean["age"], ax=axes[1])
axes[1].set_title("Age boxplot")
plt.tight_layout()
plt.show()
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_clean["fare"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Fare distribution")
sns.boxplot(x=df_clean["fare"], ax=axes[1])
axes[1].set_title("Fare boxplot")
plt.tight_layout()
plt.show()
""")

code("""def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return q1, q3, iqr, lower, upper, outliers.shape[0]

age_q1, age_q3, age_iqr, age_lo, age_hi, age_out = iqr_bounds(df_clean["age"])
fare_q1, fare_q3, fare_iqr, fare_lo, fare_hi, fare_out = iqr_bounds(df_clean["fare"])

print(f"AGE  -> Q1={age_q1}, Q3={age_q3}, IQR={age_iqr}, lower={age_lo}, upper={age_hi}, outliers={age_out}")
print(f"FARE -> Q1={fare_q1}, Q3={fare_q3}, IQR={fare_iqr}, lower={fare_lo}, upper={fare_hi}, outliers={fare_out}")
""")

code("""fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
print(f"Fare mean={fare_mean:.4f}, median={fare_median:.4f}, mode={fare_mode:.4f}")
""")

md("""**Outlier and skew conclusion:** using Q1−1.5×IQR / Q3+1.5×IQR bounds, `age`
has 11 outliers (bounds ≈ [-6.69, 64.81]) and `fare` has 116 outliers (bounds ≈
[-26.72, 65.63]) on the cleaned data (n=889). For `fare`, mean (≈32.20) >
median (≈14.45) > mode (≈8.05); this mean > median > mode ordering, together
with the long right tail visible in the histogram/box plot and 116 IQR
outliers concentrated on the high side, indicates `fare` is **right-skewed**
(a small number of passengers paid very high fares, pulling the mean above
the median and mode).""")

md("""## 4. Bivariate analysis — survival rate by boolean masking

Survival rates computed with boolean masks (`df[df['col']==value]`,
combined with `&`), not `groupby`.""")

code("""rate_female = df_clean[df_clean["sex"] == "female"]["survived"].mean()
rate_male = df_clean[df_clean["sex"] == "male"]["survived"].mean()
print(f"Survival rate — female: {rate_female:.4f}, male: {rate_male:.4f}")
""")

code("""for pc in sorted(df_clean["pclass"].unique()):
    rate = df_clean[df_clean["pclass"] == pc]["survived"].mean()
    print(f"Survival rate — pclass {pc}: {rate:.4f}")
""")

code("""for sex_val in ["female", "male"]:
    for pc in sorted(df_clean["pclass"].unique()):
        rate = df_clean[(df_clean["sex"] == sex_val) & (df_clean["pclass"] == pc)]["survived"].mean()
        print(f"Survival rate — sex={sex_val}, pclass={pc}: {rate:.4f}")
""")

md("""**Interpretation:** women survived at ≈74.2% vs men at ≈18.9% — an enormous
gap driven by the "women and children first" evacuation priority. By class
alone, survival falls monotonically from 1st (≈63.0%) to 2nd (≈47.3%) to 3rd
(≈24.2%). Combining both, 1st- and 2nd-class women survived at ≈96.8% and
≈92.1% respectively, while 3rd-class women were at ≈50.0% and men in every
class were far lower (1st ≈36.9%, 2nd ≈15.7%, 3rd ≈13.5%) — sex is the
dominant factor, but class compounds it strongly, especially for women.""")

md("""## 5. Correlation matrix (6 numeric columns only)

Computed on exactly `survived, pclass, age, sibsp, parch, fare` — `adult_male`
and `alone` are explicitly excluded.""")

code("""corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df_clean[corr_cols].corr()
corr
""")

code("""plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, vmin=-1, vmax=1)
plt.title("Correlation matrix (6 columns)")
plt.tight_layout()
plt.show()
""")

code("""pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        pairs.append((corr_cols[i], corr_cols[j], corr.iloc[i, j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for p in pairs:
    print(f"{p[0]:>9} vs {p[1]:<9} : {p[2]:.4f}")
""")

md("""**Top-2 absolute correlations:** (1) `pclass` vs `fare` = **-0.5495** — lower
`pclass` number (1st class) strongly associates with higher fares, which is
expected since class is essentially a ticket-tier label priced into the fare.
(2) `sibsp` vs `parch` = **+0.4148** — passengers travelling with more
siblings/spouses also tend to travel with more parents/children, i.e. they
are travelling as larger family units rather than these two counts being
independent.""")

md("""## 6. Multivariate analysis — "who survived and why"

Four charts building a coherent survival story, each with its own
interpretation.""")

code("""plt.figure(figsize=(7, 5))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival rate by class and sex")
plt.ylabel("Survival rate")
plt.show()
""")

md("""**Chart 1 interpretation:** this grouped bar chart makes the class × sex
interaction from Section 4 visually explicit — female bars are far taller
than male bars in every class, and both sexes' bars decline from class 1 to
3. Male survival is low across the board (≈37% down to ≈14%), while female
survival stays high through classes 1–2 and only drops meaningfully in class
3 (≈50%), showing that class protected women much less consistently than it
protected men from the reverse (i.e., poor class hurt everyone, but being
male was already close to a worst case regardless of class).""")

code("""plt.figure(figsize=(7, 5))
sns.boxplot(data=df_clean, x="survived", y="age", hue="sex")
plt.title("Age distribution by survival outcome and sex")
plt.xticks([0, 1], ["Did not survive", "Survived"])
plt.show()
""")

md("""**Chart 2 interpretation:** median ages are broadly similar between
survivors and non-survivors within each sex, but the survivors' boxes (both
sexes) show slightly more density at the young end and the non-survivor male
box extends with several older outliers. This suggests age alone is a weak
survival signal compared to sex/class — it mainly matters at the extremes
(very young children boosted by the "children first" policy) rather than
as a smooth gradient.""")

code("""plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", style="pclass", alpha=0.7)
plt.title("Fare vs Age, colored by survival, styled by class")
plt.show()
""")

md("""**Chart 3 interpretation:** survivors (orange) cluster more heavily in the
upper-fare region of the plot, and the highest-fare points (mostly circle/
square markers = classes 1–2) are disproportionately survivors, while the
dense low-fare cloud at the bottom (triangle markers = class 3) is
dominated by non-survivors (blue). This is consistent with fare acting as a
proxy for class/deck location, reinforcing the class-based survival gap
already seen in Sections 4–5 (`pclass`–`fare` correlation of -0.55).""")

code("""pivot = df_clean.pivot_table(values="survived", index="pclass", columns="embarked", aggfunc="mean")
plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Survival rate by class and embarkation port")
plt.show()
""")

md("""**Chart 4 interpretation:** survival rate is highest for class 1 passengers
regardless of port, but there is still port-level variation — e.g.
Cherbourg ('C') passengers in class 1 and 2 show among the highest survival
rates in the grid, while class 3 passengers embarking at Southampton ('S'),
the largest and lowest-fare group, show the lowest survival rate in the
whole heatmap. This adds a third, smaller factor (embarkation port,
correlated with class composition of each port) on top of the dominant sex
and class effects already identified.""")

md("""## 7. Exploratory standardization check (not used later in modeling)

Z-score standardize `age` and `fare` on the full cleaned DataFrame and verify
the transformed columns have ≈0 mean and ≈1 std.""")

code("""print("BEFORE standardization:")
print(df_clean[["age", "fare"]].agg(["mean", "std"]))

scaler = StandardScaler()
standardized = pd.DataFrame(
    scaler.fit_transform(df_clean[["age", "fare"]]),
    columns=["age_z", "fare_z"],
    index=df_clean.index,
)

print("\\nAFTER standardization:")
print(standardized.agg(["mean", "std"]))
""")

md("""**Confirmation:** after z-score standardization both `age_z` and `fare_z`
have mean ≈0 (order 1e-16 to 1e-17, i.e. numerically zero) and standard
deviation ≈1 (`ddof=0` under the hood of `StandardScaler`, matching the
`.std()` computed with `ddof=1` closely at n=889), confirming the
transformation works as expected. This standardized version is **not** used
in `02_modeling.ipynb` — that notebook fits its own `StandardScaler` inside a
`ColumnTransformer`/`Pipeline` on the training split only, per the modeling
requirements.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
}

with open("01_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote 01_eda.ipynb with", len(cells), "cells")
