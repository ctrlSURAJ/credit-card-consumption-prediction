"""
==============================================================================
CAPSTONE PROJECT: Predicting Credit Card Consumption for a Leading Bank
==============================================================================
Script 3 of 3: Score the 5,000 customers with missing target values and
export the final predictions workbook.
==============================================================================
"""

import pandas as pd
import numpy as np
import json, joblib, os

# Paths are relative to this script, so the project runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")

df = pd.read_pickle(f"{OUT_DIR}/merged_clean.pkl")
model = joblib.load(f"{OUT_DIR}/final_xgb_model.pkl")
with open(f"{OUT_DIR}/feature_list.json") as f:
    FEATURES = json.load(f)

# Recreate the same feature engineering as script 2
d = df.copy()
d["cc_cons_total_q2"] = d[["cc_cons_apr", "cc_cons_may", "cc_cons_jun"]].sum(axis=1)
d["dc_cons_total_q2"] = d[["dc_cons_apr", "dc_cons_may", "dc_cons_jun"]].sum(axis=1)
d["cc_cons_avg_q2"]   = d[["cc_cons_apr", "cc_cons_may", "cc_cons_jun"]].mean(axis=1)
d["cc_cons_std_q2"]   = d[["cc_cons_apr", "cc_cons_may", "cc_cons_jun"]].std(axis=1)
d["cc_cons_trend"]     = d["cc_cons_jun"] - d["cc_cons_apr"]
d["cc_cons_trend_pct"] = d["cc_cons_trend"] / (d["cc_cons_apr"] + 1)
d["cc_share_of_total_spend"] = d["cc_cons_total_q2"] / (d["cc_cons_total_q2"] + d["dc_cons_total_q2"] + 1)
d["cc_count_total_q2"] = d[["cc_count_apr", "cc_count_may", "cc_count_jun"]].sum(axis=1)
d["dc_count_total_q2"] = d[["dc_count_apr", "dc_count_may", "dc_count_jun"]].sum(axis=1)
d["cc_avg_ticket_size"] = d["cc_cons_total_q2"] / (d["cc_count_total_q2"] + 1)
d["card_utilization"] = d["cc_cons_avg_q2"] / (d["card_lim"] + 1)
d["credit_amount_total_q2"] = d[["credit_amount_apr", "credit_amount_may", "credit_amount_jun"]].sum(axis=1)
d["debit_amount_total_q2"]  = d[["debit_amount_apr", "debit_amount_may", "debit_amount_jun"]].sum(axis=1)
d["net_cashflow_q2"]        = d["credit_amount_total_q2"] - d["debit_amount_total_q2"]
d["credit_count_total_q2"]  = d[["credit_count_apr", "credit_count_may", "credit_count_jun"]].sum(axis=1)
d["debit_count_total_q2"]   = d[["debit_count_apr", "debit_count_may", "debit_count_jun"]].sum(axis=1)
d["total_investments"] = d[["investment_1", "investment_2", "investment_3", "investment_4"]].sum(axis=1)
d["has_active_loan"]   = ((d["personal_loan_active"] > 0) | (d["vehicle_loan_active"] > 0)).astype(int)
d["loan_enq_flag"]     = (d["loan_enq"] == "Y").astype(int)
d["income_encoded"] = d["Income"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
d["account_type_current"] = (d["account_type"] == "current").astype(int)
d["gender_M"] = (d["gender"] == "M").astype(int)

score_df = d[d["cc_cons"].isna()].reset_index(drop=True)
X_score = score_df[FEATURES].fillna(0)

pred_log = model.predict(X_score)
pred = np.expm1(pred_log)
pred = np.clip(pred, 0, None)

output = score_df[["ID"]].copy()
output["predicted_cc_cons"] = pred.round(2)

print("Predictions summary:")
print(output["predicted_cc_cons"].describe())

# Save as Excel
output.to_excel(f"{OUT_DIR}/Predicted_Credit_Consumption.xlsx", index=False, sheet_name="Predictions")

# Also save a combined file: full 20,000 customers, actual where known, predicted where missing
full = df[["ID", "cc_cons"]].copy()
full = full.merge(output, on="ID", how="left")
full["cc_cons_final"] = full["cc_cons"].fillna(full["predicted_cc_cons"])
full["source"] = np.where(full["cc_cons"].notna(), "Actual", "Predicted")
full.to_excel(f"{OUT_DIR}/Full_Customer_Base_Credit_Consumption.xlsx", index=False, sheet_name="AllCustomers")

print("\nSaved:")
print(" -", f"{OUT_DIR}/Predicted_Credit_Consumption.xlsx  ({len(output)} rows, missing-target customers only)")
print(" -", f"{OUT_DIR}/Full_Customer_Base_Credit_Consumption.xlsx  ({len(full)} rows, all customers)")
print("\nScript 3 complete.")
