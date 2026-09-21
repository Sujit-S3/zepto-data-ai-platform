import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Module 2 — Analytics: 01 Exploratory Data Analysis (Titanic)

This is the first part of Module 2 for my Zepto Data & AI Platform capstone.

I created this notebook to perform a full Exploratory Data Analysis (EDA) workflow on the Titanic dataset. My process involves a single seaborn load (fetching from the network and caching it), detailed profiling, strict missing-value handling based on threshold rules, and extensive univariate, bivariate, and multivariate analyses. I also added a correlation analysis and an exploratory standardization check just to make sure I understand the preprocessing steps. 

I set it up so that `titanic.csv` is written here as an offline fallback. This CSV is then used by my `02_modeling.ipynb` notebook to ensure I don't hit the network again.
""")

code("""import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")
""")

md("""## 1. My Dataset Loading Strategy (single seaborn call)

Per the module requirements, I am calling `sns.load_dataset('titanic')` **exactly once** in this cell. I immediately persist the result to `titanic.csv`. I made sure no other cell or notebook in this module calls `sns.load_dataset` again — my `02_modeling.ipynb` script is strictly programmed to read this CSV instead.""")

code("""df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)
df.head()
""")

md("## 2. My Data Profiling Approach")

code("df.info()")
code("df.describe(include='all')")
code("df.shape")

code("""missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
missing_pct
""")

md("""### My missing-value percentages and handling decisions

When I measured the missing values on the raw loaded DataFrame (891 rows), here is what I found and how I decided to handle them:

| Column | Missing % | Bucket | My Decision |
|---|---|---|---|
| `deck` | 77.22% | > 30% | **Drop the column.** Since over 77% of the data is missing, I realized any imputation or "missing" category would carry almost no real signal. Plus, `deck` isn't part of my 6-column correlation set or my modeling feature set, so dropping it entirely was the simplest, least biased choice I could make. |
| `age` | 19.87% | 5–30% | **Impute with the median.** Age is numeric but right-of-center skewed by a long tail of older passengers. Because of this skew, I decided to use the median (which is robust to outliers) rather than the mean. |
| `embarked` | 0.22% (2 rows) | < 5% | **Drop those rows.** Since only 2 out of 891 rows are affected, dropping them is completely safe and saves me from inventing a port of embarkation out of thin air. |
| `embark_town` | 0.22% (2 rows) | < 5% | **Drop those rows.** These are the exact same 2 rows as `embarked` (it is just a redundant text version of the same field), so the row-drop I decided on above already resolves this perfectly. |

All other columns have 0% missing values.""")

code("""df_clean = df.dropna(subset=["embarked", "embark_town"]).copy()
df_clean = df_clean.drop(columns=["deck"])
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
print("Shape after cleaning:", df_clean.shape)
print("Remaining missing values:\\n", df_clean.isna().sum()[df_clean.isna().sum() > 0])
""")

md("""## 3. Univariate analysis — `age` and `fare`

For my univariate analysis, I decided to generate a histogram and a box plot for both age and fare. I also calculated the IQR-based outlier bounds and exact counts to get a true feel for the spread of the data.""")

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

md("""**My outlier and skew conclusion:** using Q1−1.5×IQR and Q3+1.5×IQR bounds, I found that `age` has 11 outliers (bounds ≈ [-6.69, 64.81]) and `fare` has 116 outliers (bounds ≈ [-26.72, 65.63]) on the cleaned data (n=889). 

For `fare`, the mean (≈32.20) > median (≈14.45) > mode (≈8.05). This mean > median > mode ordering, together with the long right tail I saw in the histogram/box plot and the 116 IQR outliers concentrated on the high side, indicates to me that `fare` is **right-skewed** (a small number of passengers paid very high fares, pulling the mean far above the median and mode).""")

md("""## 4. Bivariate analysis — survival rate by boolean masking

I computed the survival rates using explicit boolean masks (e.g., `df[df['col']==value]` combined with `&`), rather than relying on a simple `groupby`, just to demonstrate manual filtering.""")

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

md("""**My Interpretation:** Women survived at ≈74.2% while men survived at ≈18.9%. This is an enormous gap driven by the "women and children first" evacuation priority. 

When I looked at class alone, survival falls monotonically from 1st (≈63.0%) to 2nd (≈47.3%) to 3rd (≈24.2%). 

Combining both, I noticed 1st- and 2nd-class women survived at ≈96.8% and ≈92.1% respectively, while 3rd-class women were at ≈50.0%. Men in every class were far lower (1st ≈36.9%, 2nd ≈15.7%, 3rd ≈13.5%). My conclusion is that sex is the dominant factor, but class compounds it strongly, especially for women.""")

md("""## 5. Correlation matrix (6 numeric columns only)

I computed this on exactly `survived, pclass, age, sibsp, parch, fare`. I explicitly excluded `adult_male` and `alone` to keep the matrix focused.""")

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

md("""**My Top-2 absolute correlations:** 
1) `pclass` vs `fare` = **-0.5495**. A lower `pclass` number (1st class) strongly associates with higher fares. I expected this since class is essentially a ticket-tier label priced into the fare.
2) `sibsp` vs `parch` = **+0.4148**. Passengers travelling with more siblings/spouses also tend to travel with more parents/children. This makes sense to me because they are travelling as larger family units rather than these two counts being independent.""")

md("""## 6. Multivariate analysis — "who survived and why"

I created four charts to build a coherent survival story, and I provided my own interpretation for each.""")

code("""plt.figure(figsize=(7, 5))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival rate by class and sex")
plt.ylabel("Survival rate")
plt.show()
""")

md("""**My interpretation for Chart 1:** This grouped bar chart makes the class × sex interaction from Section 4 visually explicit. The female bars are far taller than the male bars in every class, and both sexes' bars decline from class 1 to 3. Male survival is low across the board (≈37% down to ≈14%), while female survival stays high through classes 1–2 and only drops meaningfully in class 3 (≈50%). This shows me that class protected women much less consistently than it protected men from the reverse (i.e., poor class hurt everyone, but being male was already close to a worst-case scenario regardless of class).""")

code("""plt.figure(figsize=(7, 5))
sns.boxplot(data=df_clean, x="survived", y="age", hue="sex")
plt.title("Age distribution by survival outcome and sex")
plt.xticks([0, 1], ["Did not survive", "Survived"])
plt.show()
""")

md("""**My interpretation for Chart 2:** Median ages are broadly similar between survivors and non-survivors within each sex, but I noticed the survivors' boxes (for both sexes) show slightly more density at the young end. Also, the non-survivor male box extends with several older outliers. This suggests to me that age alone is a weak survival signal compared to sex/class — it mainly matters at the extremes (very young children boosted by the "children first" policy) rather than acting as a smooth gradient.""")

code("""plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", style="pclass", alpha=0.7)
plt.title("Fare vs Age, colored by survival, styled by class")
plt.show()
""")

md("""**My interpretation for Chart 3:** The survivors (orange) cluster more heavily in the upper-fare region of the plot, and the highest-fare points (mostly circle/square markers = classes 1–2) are disproportionately survivors. Meanwhile, the dense low-fare cloud at the bottom (triangle markers = class 3) is heavily dominated by non-survivors (blue). This perfectly aligns with my earlier finding that fare acts as a proxy for class/deck location, reinforcing the class-based survival gap I saw in Sections 4–5 (where `pclass`–`fare` correlation was -0.55).""")

code("""pivot = df_clean.pivot_table(values="survived", index="pclass", columns="embarked", aggfunc="mean")
plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Survival rate by class and embarkation port")
plt.show()
""")

md("""**My interpretation for Chart 4:** Survival rate is highest for class 1 passengers regardless of port, but I can still see port-level variation. For example, Cherbourg ('C') passengers in class 1 and 2 show among the highest survival rates in the grid, while class 3 passengers embarking at Southampton ('S') — the largest and lowest-fare group — show the lowest survival rate in the whole heatmap. This adds a third, smaller factor (embarkation port, which is correlated with the class composition of each port) on top of the dominant sex and class effects I already identified.""")

md("""## 7. My exploratory standardization check (not used later in modeling)

I wanted to quickly Z-score standardize `age` and `fare` on the full cleaned DataFrame just to verify that the transformed columns have ≈0 mean and ≈1 std.""")

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

md("""**My Confirmation:** After z-score standardization, I confirmed that both `age_z` and `fare_z` have mean ≈0 (order 1e-16 to 1e-17, i.e. numerically zero) and standard deviation ≈1 (`ddof=0` under the hood of `StandardScaler`, matching the `.std()` computed with `ddof=1` closely at n=889). This confirms my transformation works exactly as expected. I am **not** using this standardized version in `02_modeling.ipynb` — instead, I programmed that notebook to fit its own `StandardScaler` inside a `ColumnTransformer`/`Pipeline` on the training split only, strictly adhering to the modeling requirements.""")

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
