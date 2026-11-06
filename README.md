# Predicting Credit Card Consumption (Bank Capstone)

End-to-end machine learning project that predicts each customer's average credit card spend for the next three months (July to September), using demographics and April to June behaviour data from a leading bank.

## Problem
- 20,000 customers, 3 source files: demographics, behaviour (transactions, loans, investments), and target `cc_cons`
- Target is missing for 5,000 customers. Train on the 15,000 labelled customers, predict the missing 5,000
- Evaluation metric: RMSPE (Root Mean Square Percentage Error)

## Approach
1. **Data cleaning**: fixed swapped column labels (`NetBanking_Flag` and `Avg_days_between_transaction`), dropped `region_code` (identical to `Tenure_with_Bank`), treated impossible ages and tenures (above 100) as missing, median/mode imputation for the rest
2. **EDA**: target distribution (right skewed, so log transform), spend by income/account type/gender, correlation heatmap
3. **Feature engineering**: about 50 features such as quarterly spend totals and averages, spend trend, card utilisation, average ticket size, net cash flow, loan and investment flags
4. **Modelling**: six models compared on an 80/20 split, best model validated with 5-fold cross-validation

## Results

| Model | Validation RMSPE |
|---|---|
| Gradient Boosting (log target) | 0.2572 |
| XGBoost (log target) | 0.2577 |
| Random Forest (log target) | 0.2607 |
| XGBoost (raw target) | 0.2754 |
| Linear Regression (log target) | 0.4721 |
| Ridge Regression (log target) | 0.4721 |

Final model: XGBoost on log-transformed target. 5-fold CV RMSPE = 0.2603 (std 0.0026). Recent credit and debit card spend are the strongest predictors.

## Project structure
```
data/        raw Excel files (not tracked, see below)
outputs/     charts, model comparison, feature importance, predictions
docs/        Model_Documentation.docx (full write-up)
01_data_prep_eda.py
02_feature_engineering_modeling.py
03_score_predictions.py
import_data.py   quick merge and account-type summary
```

## How to run
```bash
pip install -r requirements.txt
# place CustomerDemographics.xlsx, CustomerBehaviorData.xlsx and
# CreditConsumptionData.xlsx inside the data/ folder
python 01_data_prep_eda.py
python 02_feature_engineering_modeling.py
python 03_score_predictions.py
```
The raw data belongs to AnalytixLabs and is not included in this repository.

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn

## Author
Suraj
