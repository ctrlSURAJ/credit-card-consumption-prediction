"""
==============================================================================
CAPSTONE PROJECT: Predicting Credit Card Consumption for a Leading Bank
==============================================================================
Author: Suraj
Script 1 of 3: Data Loading, Cleaning & Exploratory Data Analysis (EDA)

This script:
  1. Loads the three raw data sources (Demographics, Behavior, Target)
  2. Performs data quality checks and documents every issue found
  3. Cleans the data (fixes bad rows, outliers, missing values)
  4. Merges everything into a single analytical dataset
  5. Runs EDA (univariate, bivariate, correlation) and saves charts
  6. Saves a clean, merged dataset for the feature engineering / modeling
     script to pick up (02_feature_engineering_modeling.py)
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

import os

# Paths are relative to this script, so the project runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DEMO   = os.path.join(BASE_DIR, "data", "CustomerDemographics.xlsx")
RAW_BEHAV  = os.path.join(BASE_DIR, "data", "CustomerBehaviorData.xlsx")
RAW_TARGET = os.path.join(BASE_DIR, "data", "CreditConsumptionData.xlsx")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. LOAD RAW DATA
# ------------------------------------------------------------------
demo   = pd.read_excel(RAW_DEMO)
behav  = pd.read_excel(RAW_BEHAV)
target = pd.read_excel(RAW_TARGET)

print("Raw shapes -> demo:", demo.shape, "| behav:", behav.shape, "| target:", target.shape)
assert set(demo.ID) == set(behav.ID) == set(target.ID), "ID sets differ across files!"

# ------------------------------------------------------------------
# 2. DATA QUALITY CHECKS (documented findings)
# ------------------------------------------------------------------
dq_log = []

# 2a. A single corrupted record in Demographics: account_type / gender /
#     Income all show the literal value 0 instead of a real category.
bad_ids = demo.loc[demo["account_type"] == 0, "ID"].tolist()
dq_log.append(f"Found {len(bad_ids)} demographic record(s) with placeholder '0' "
              f"instead of a real category (IDs: {bad_ids}). These fields will be "
              f"imputed with the mode.")
demo.loc[demo["account_type"] == 0, "account_type"] = np.nan
demo.loc[demo["gender"] == 0, "gender"] = np.nan
demo.loc[demo["Income"] == 0, "Income"] = np.nan

# 2b. 'NetBanking_Flag' and 'Avg_days_between_transaction' appear to be
#     mislabeled relative to the data dictionary: the column named
#     NetBanking_Flag holds large integer values (0-974, looks like a day
#     count), while the column named Avg_days_between_transaction only ever
#     holds 0/1 (looks like a flag). We rename them to match their actual
#     content so downstream features are semantically correct.
assert set(demo["Avg_days_between_transaction"].unique()) <= {0, 1}
dq_log.append("Columns 'NetBanking_Flag' and 'Avg_days_between_transaction' were "
              "swapped relative to their content (the 'flag' column only ever held "
              "large day-count-like integers, and vice versa). Renamed to match "
              "actual content: netbanking_flag (0/1) <- old Avg_days_between_transaction, "
              "avg_days_between_transaction (integer) <- old NetBanking_Flag.")
demo = demo.rename(columns={
    "NetBanking_Flag": "avg_days_between_transaction",
    "Avg_days_between_transaction": "netbanking_flag",
})

# 2c. region_code is identical to Tenure_with_Bank for every single record
#     (100% match) -> it carries zero incremental information and is in
#     fact a duplicated/mis-populated column. Dropped to avoid
#     multicollinearity / redundant noise.
pct_match = (demo["region_code"] == demo["Tenure_with_Bank"]).mean()
dq_log.append(f"'region_code' is identical to 'Tenure_with_Bank' for {pct_match:.0%} "
              f"of records -> treated as a duplicated/corrupted column and dropped.")
demo = demo.drop(columns=["region_code"])

# 2d. Implausible ages (>100, up to 223) and employment tenures (>100 yrs,
#     up to 203) are clearly data-entry errors -- there's a clean gap
#     between the plausible range (25-69 / 0.5-100) and the corrupted
#     values (118+ / 100+), so these are treated as missing and imputed
#     with the column median.
n_age_bad = (demo["age"] > 100).sum()
n_ten_bad = (demo["Emp_Tenure_Years"] > 100).sum()
dq_log.append(f"{n_age_bad} records have implausible age (>100, up to 223) and "
              f"{n_ten_bad} have implausible Emp_Tenure_Years (>100, up to 203). "
              f"Both show a clean gap between plausible and implausible values, "
              f"so these are set to NaN and median-imputed.")
demo.loc[demo["age"] > 100, "age"] = np.nan
demo.loc[demo["Emp_Tenure_Years"] > 100, "Emp_Tenure_Years"] = np.nan

# 2e. Behavioral file: a handful of scattered nulls (1-2 records per column)
#     across spend/count/investment fields, plus 2 missing loan_enq values.
n_behav_nulls = int(behav.isna().sum().sum())
dq_log.append(f"CustomerBehaviorData has {n_behav_nulls} scattered null cells "
              f"across various columns (<=2 per column, out of 20,000 rows) -> "
              f"median-imputed for numeric columns; loan_enq nulls filled as 'N' "
              f"(no enquiry, the majority class / the only alternative to 'Y').")

numeric_behav_cols = behav.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_behav_cols:
    if col != "ID" and behav[col].isna().any():
        behav[col] = behav[col].fillna(behav[col].median())
behav["loan_enq"] = behav["loan_enq"].fillna("N")

# 2f. Impute remaining demo categoricals / numerics after nulling step 2a/2d
for col in ["account_type", "gender", "Income"]:
    demo[col] = demo[col].fillna(demo[col].mode()[0])
for col in ["age", "Emp_Tenure_Years"]:
    demo[col] = demo[col].fillna(demo[col].median())

print("\nData quality log:")
for i, msg in enumerate(dq_log, 1):
    print(f"  {i}. {msg}")

with open(f"{OUT_DIR}/data_quality_log.txt", "w") as f:
    f.write("\n\n".join(f"{i}. {m}" for i, m in enumerate(dq_log, 1)))

# ------------------------------------------------------------------
# 3. MERGE
# ------------------------------------------------------------------
df = demo.merge(behav, on="ID", how="inner").merge(target, on="ID", how="left")
print("\nMerged shape:", df.shape)
df.to_pickle(f"{OUT_DIR}/merged_clean.pkl")

# ------------------------------------------------------------------
# 4. EDA
# ------------------------------------------------------------------
train_mask = df["cc_cons"].notna()
print(f"\nTrain (target known): {train_mask.sum()}  |  Score (target missing): {(~train_mask).sum()}")

# 4a. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df.loc[train_mask, "cc_cons"], bins=60, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribution of cc_cons (target)")
axes[0].set_xlabel("Average CC spend, next 3 months")

sns.histplot(np.log1p(df.loc[train_mask, "cc_cons"]), bins=60, ax=axes[1], color="#DD8452")
axes[1].set_title("Distribution of log1p(cc_cons)")
axes[1].set_xlabel("log1p(cc_cons)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_target_distribution.png")
plt.close()

# 4b. Target vs key categoricals
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.boxplot(data=df[train_mask], x="Income", y="cc_cons", ax=axes[0],
            order=["LOW", "MEDIUM", "HIGH"], palette="Blues")
axes[0].set_title("CC Consumption by Income Level")
axes[0].set_yscale("log")

sns.boxplot(data=df[train_mask], x="account_type", y="cc_cons", ax=axes[1], palette="Greens")
axes[1].set_title("CC Consumption by Account Type")
axes[1].set_yscale("log")

sns.boxplot(data=df[train_mask], x="gender", y="cc_cons", ax=axes[2], palette="Oranges")
axes[2].set_title("CC Consumption by Gender")
axes[2].set_yscale("log")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_target_by_categoricals.png")
plt.close()

# 4c. Correlation heatmap of key numeric drivers with the target
num_cols = ["age", "Emp_Tenure_Years", "Tenure_with_Bank", "netbanking_flag",
            "avg_days_between_transaction", "cc_cons_apr", "cc_cons_may", "cc_cons_jun",
            "dc_cons_apr", "dc_cons_may", "dc_cons_jun", "card_lim",
            "credit_amount_jun", "debit_amount_jun", "investment_1", "investment_2",
            "investment_3", "investment_4", "emi_active", "cc_cons"]
corr = df.loc[train_mask, num_cols].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
plt.title("Correlation Heatmap - Key Numeric Features vs Target")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_correlation_heatmap.png")
plt.close()

# 4d. Top correlations with target, printed for the report
top_corr = corr["cc_cons"].drop("cc_cons").sort_values(key=np.abs, ascending=False)
print("\nTop correlations with target (cc_cons):")
print(top_corr.head(10))
top_corr.to_csv(f"{OUT_DIR}/top_correlations.csv")

# 4e. Monthly spend trend (recent months matter more?)
fig, ax = plt.subplots(figsize=(7, 4.5))
monthly_corr = df.loc[train_mask, ["cc_cons_apr", "cc_cons_may", "cc_cons_jun", "cc_cons"]].corr()["cc_cons"].drop("cc_cons")
monthly_corr.plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_title("Correlation of Monthly CC Spend with Future 3-Month Avg Spend")
ax.set_ylabel("Pearson correlation")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_monthly_trend_correlation.png")
plt.close()

print("\nEDA charts saved to", OUT_DIR)
print("Script 1 complete.")
