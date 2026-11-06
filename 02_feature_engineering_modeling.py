"""
==============================================================================
CAPSTONE PROJECT: Predicting Credit Card Consumption for a Leading Bank
==============================================================================
Script 2 of 3: Feature Engineering, Model Training & Validation

This script:
  1. Loads the cleaned/merged dataset from script 1
  2. Engineers features from the raw monthly spend/behavior columns
  3. Encodes categoricals
  4. Trains multiple candidate models (Linear Regression, Ridge, Random
     Forest, Gradient Boosting, XGBoost) on a train/validation split
  5. Evaluates every model with RMSPE (the metric specified in the brief)
  6. Picks the best model, refits on all labeled data, and saves it +
     the feature list for the scoring script
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, json, joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import os

# Paths are relative to this script, so the project runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")

np.random.seed(42)

df = pd.read_pickle(f"{OUT_DIR}/merged_clean.pkl")

# ------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ------------------------------------------------------------------
d = df.copy()

# Spend aggregates (credit + debit card combined activity)
d["cc_cons_total_q2"] = d[["cc_cons_apr", "cc_cons_may", "cc_cons_jun"]].sum(axis=1)
d["dc_cons_total_q2"] = d[["dc_cons_apr", "dc_cons_may", "dc_cons_jun"]].sum(axis=1)
d["cc_cons_avg_q2"]   = d[["cc_cons_apr", "cc_cons_may", "cc_cons_jun"]].mean(axis=1)
d["cc_cons_std_q2"]   = d[["cc_cons_apr", "cc_cons_may", "cc_cons_jun"]].std(axis=1)

# Momentum / trend: is spend accelerating into June?
d["cc_cons_trend"]     = d["cc_cons_jun"] - d["cc_cons_apr"]
d["cc_cons_trend_pct"] = d["cc_cons_trend"] / (d["cc_cons_apr"] + 1)
d["cc_share_of_total_spend"] = d["cc_cons_total_q2"] / (d["cc_cons_total_q2"] + d["dc_cons_total_q2"] + 1)

# Transaction activity
d["cc_count_total_q2"] = d[["cc_count_apr", "cc_count_may", "cc_count_jun"]].sum(axis=1)
d["dc_count_total_q2"] = d[["dc_count_apr", "dc_count_may", "dc_count_jun"]].sum(axis=1)
d["cc_avg_ticket_size"] = d["cc_cons_total_q2"] / (d["cc_count_total_q2"] + 1)

# Card utilization
d["card_utilization"] = d["cc_cons_avg_q2"] / (d["card_lim"] + 1)

# Banking relationship / cash flow features
d["credit_amount_total_q2"] = d[["credit_amount_apr", "credit_amount_may", "credit_amount_jun"]].sum(axis=1)
d["debit_amount_total_q2"]  = d[["debit_amount_apr", "debit_amount_may", "debit_amount_jun"]].sum(axis=1)
d["net_cashflow_q2"]        = d["credit_amount_total_q2"] - d["debit_amount_total_q2"]
d["credit_count_total_q2"]  = d[["credit_count_apr", "credit_count_may", "credit_count_jun"]].sum(axis=1)
d["debit_count_total_q2"]   = d[["debit_count_apr", "debit_count_may", "debit_count_jun"]].sum(axis=1)

# Investments & loans
d["total_investments"] = d[["investment_1", "investment_2", "investment_3", "investment_4"]].sum(axis=1)
d["has_active_loan"]   = ((d["personal_loan_active"] > 0) | (d["vehicle_loan_active"] > 0)).astype(int)
d["loan_enq_flag"]     = (d["loan_enq"] == "Y").astype(int)

# Categorical encodings
d["income_encoded"] = d["Income"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
d["account_type_current"] = (d["account_type"] == "current").astype(int)
d["gender_M"] = (d["gender"] == "M").astype(int)

FEATURES = [
    # demographics
    "age", "Emp_Tenure_Years", "Tenure_with_Bank", "netbanking_flag",
    "avg_days_between_transaction", "income_encoded", "account_type_current", "gender_M",
    # monthly spend
    "cc_cons_apr", "cc_cons_may", "cc_cons_jun", "dc_cons_apr", "dc_cons_may", "dc_cons_jun",
    # engineered spend aggregates
    "cc_cons_total_q2", "dc_cons_total_q2", "cc_cons_avg_q2", "cc_cons_std_q2",
    "cc_cons_trend", "cc_cons_trend_pct", "cc_share_of_total_spend",
    # transaction activity
    "cc_count_apr", "cc_count_may", "cc_count_jun", "dc_count_apr", "dc_count_may", "dc_count_jun",
    "cc_count_total_q2", "dc_count_total_q2", "cc_avg_ticket_size",
    # card / limits
    "card_lim", "card_utilization",
    # loans
    "personal_loan_active", "vehicle_loan_active", "personal_loan_closed",
    "vehicle_loan_closed", "has_active_loan", "loan_enq_flag", "emi_active",
    # investments
    "investment_1", "investment_2", "investment_3", "investment_4", "total_investments",
    # cash flow
    "credit_amount_total_q2", "debit_amount_total_q2", "net_cashflow_q2",
    "credit_count_total_q2", "debit_count_total_q2",
    "max_credit_amount_apr", "max_credit_amount_may", "max_credit_amount_jun",
]

print(f"Total engineered features: {len(FEATURES)}")

train_df = d[d["cc_cons"].notna()].reset_index(drop=True)
score_df = d[d["cc_cons"].isna()].reset_index(drop=True)
print(f"Train rows: {len(train_df)}  |  Score (to-predict) rows: {len(score_df)}")

X = train_df[FEATURES].fillna(0)
y = train_df["cc_cons"]
y_log = np.log1p(y)

X_score = score_df[FEATURES].fillna(0)

# ------------------------------------------------------------------
# 2. RMSPE METRIC
# ------------------------------------------------------------------
def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-6, None)  # predictions must stay positive
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

# ------------------------------------------------------------------
# 3. TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------------
X_train, X_val, y_train, y_val, ylog_train, ylog_val = train_test_split(
    X, y, y_log, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)

results = []

# --- Model 1: Linear Regression (log target) ---
lr = LinearRegression()
lr.fit(X_train_sc, ylog_train)
pred = np.expm1(lr.predict(X_val_sc))
results.append(("Linear Regression (log target)", rmspe(y_val, pred)))

# --- Model 2: Ridge Regression (log target) ---
ridge = Ridge(alpha=5.0)
ridge.fit(X_train_sc, ylog_train)
pred = np.expm1(ridge.predict(X_val_sc))
results.append(("Ridge Regression (log target)", rmspe(y_val, pred)))

# --- Model 3: Random Forest (log target) ---
rf = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, ylog_train)
pred = np.expm1(rf.predict(X_val))
results.append(("Random Forest (log target)", rmspe(y_val, pred)))

# --- Model 4: Gradient Boosting (log target) ---
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, random_state=42)
gbr.fit(X_train, ylog_train)
pred = np.expm1(gbr.predict(X_val))
results.append(("Gradient Boosting (log target)", rmspe(y_val, pred)))

# --- Model 5: XGBoost (log target) ---
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, ylog_train, eval_set=[(X_val, ylog_val)], verbose=False)
pred = np.expm1(xgb_model.predict(X_val))
results.append(("XGBoost (log target)", rmspe(y_val, pred)))

# --- Model 6: XGBoost on raw (non-log) target, for comparison ---
xgb_raw = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1
)
xgb_raw.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
pred = np.expm1(np.log1p(xgb_raw.predict(X_val)))  # no-op transform for symmetry; raw preds used directly below
pred = xgb_raw.predict(X_val)
results.append(("XGBoost (raw target)", rmspe(y_val, pred)))

results_df = pd.DataFrame(results, columns=["Model", "Validation RMSPE"]).sort_values("Validation RMSPE")
print("\n=== Model comparison (holdout validation set) ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# ------------------------------------------------------------------
# 4. 5-FOLD CROSS-VALIDATION ON THE BEST MODEL (XGBoost, log target)
# ------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    m = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=42, n_jobs=-1)
    m.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
    p = np.expm1(m.predict(X.iloc[va_idx]))
    score = rmspe(y.iloc[va_idx], p)
    cv_scores.append(score)
    print(f"  Fold {fold}: RMSPE = {score:.4f}")

print(f"\n5-fold CV RMSPE: mean={np.mean(cv_scores):.4f}  std={np.std(cv_scores):.4f}")

with open(f"{OUT_DIR}/cv_scores.json", "w") as f:
    json.dump({"fold_scores": cv_scores, "mean": float(np.mean(cv_scores)),
               "std": float(np.std(cv_scores))}, f, indent=2)

# ------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (final XGBoost model, log target, all labeled data)
# ------------------------------------------------------------------
final_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1
)
final_model.fit(X, y_log)

imp = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
imp.to_csv(f"{OUT_DIR}/feature_importance.csv")

plt.figure(figsize=(8, 8))
imp.head(20).sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Top 20 Feature Importances (Final XGBoost Model)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png")
plt.close()

# ------------------------------------------------------------------
# 6. VALIDATION DIAGNOSTIC PLOT: actual vs predicted (holdout)
# ------------------------------------------------------------------
val_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42, n_jobs=-1)
val_model.fit(X_train, ylog_train)
val_pred = np.expm1(val_model.predict(X_val))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_val, val_pred, alpha=0.25, s=12, color="#4C72B0")
lims = [0, max(y_val.max(), val_pred.max())]
axes[0].plot(lims, lims, "r--", linewidth=1)
axes[0].set_xlabel("Actual cc_cons")
axes[0].set_ylabel("Predicted cc_cons")
axes[0].set_title("Actual vs Predicted (Holdout Set)")

resid_pct = (y_val - val_pred) / y_val
sns.histplot(resid_pct, bins=60, ax=axes[1], color="#DD8452")
axes[1].set_title("Percentage Residuals ((Actual-Pred)/Actual)")
axes[1].set_xlabel("% error")
axes[1].axvline(0, color="black", linewidth=1)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/validation_diagnostics.png")
plt.close()

# ------------------------------------------------------------------
# 7. SAVE FINAL MODEL + ARTIFACTS FOR SCORING SCRIPT
# ------------------------------------------------------------------
joblib.dump(final_model, f"{OUT_DIR}/final_xgb_model.pkl")
with open(f"{OUT_DIR}/feature_list.json", "w") as f:
    json.dump(FEATURES, f, indent=2)

print("\nModel, feature list, and diagnostics saved to", OUT_DIR)
print("Script 2 complete.")
