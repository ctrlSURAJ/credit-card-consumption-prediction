import numpy as np
import pandas as pd

# Load the three files (keep this script in the same folder as the Excel files)
demo = pd.read_excel("data/CustomerDemographics.xlsx")
behav = pd.read_excel("data/CustomerBehaviorData.xlsx")
target = pd.read_excel("data/CreditConsumptionData.xlsx")

# Join all three on customer ID
new_tab = demo.merge(behav, on="ID", how="inner").merge(target, on="ID", how="left")

print(new_tab.shape)
print(new_tab.head())

# Summary by account type: avg credit card consumption, number of customers, max credit limit
summary = new_tab.groupby("account_type").agg(
    avg_cc_consumption=("cc_cons", "mean"),
    num_customers=("ID", "count"),
    max_credit_limit=("card_lim", "max")
).reset_index()

# Percentage of male customers per account type
male_pct = new_tab.groupby("account_type")["gender"].apply(
    lambda x: (x == "M").mean() * 100
).reset_index(name="pct_male")

# Combine both into one table
summary = summary.merge(male_pct, on="account_type")

print(summary)
