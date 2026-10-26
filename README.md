# Customer Churn Prediction

Predicts which telecom customers are likely to leave and ranks them by risk so a retention team can contact the right people first.

**Dataset:** Telco Customer Churn (7,043 customers, 21 columns, 26.5% churn). Download it from Kaggle and save it as `data/telco_churn.csv`.

## Results

All numbers below come from `python src/train.py` (random seed 42, 80/20 stratified split, 5,634 train / 1,409 test rows).

### Model comparison

| Model | CV ROC-AUC (5-fold, train) | Test ROC-AUC |
|---|---|---|
| Logistic Regression | 0.845 | 0.842 |
| Random Forest (selected) | 0.847 | 0.843 |
| Hist Gradient Boosting | 0.845 | 0.845 |

The three models perform almost the same. Random Forest was selected on cross-validation score only; the test set was not used to choose it.

### Random Forest on the test set

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.50 | 77.3% | 53.8% | 0.634 |
| 0.33 (tuned for recall) | 86.9% | 45.3% | 0.595 |

The 0.33 threshold maximises F2 on out-of-fold training predictions, favouring recall because missing a churner costs more than contacting a loyal customer.

### Business view (test set)

| Metric | Value |
|---|---|
| Churn rate in highest-risk 10% of customers | 76.4% (2.9x the 26.5% average) |
| Churners reached by contacting the top 20% | 50.3% |
| Churning monthly revenue reached by the top 20% | 53.1% |

### Honest comparison with a simple rule

Flagging every month-to-month customer reaches 88.0% of churners at 42.6% precision and flags 54.9% of customers. The tuned model reaches 86.9% of churners at 45.3% precision and flags about 51%. The model is only slightly better as a yes/no flag. Its main value is **ranking**: it lets the team start with the riskiest customers and stop when the budget runs out (see the gains and lift numbers above).

## What drives churn

Churn rate by segment (full tables in `reports/eda_summary.md`):

| Segment | Churn rate |
|---|---|
| Month-to-month contract | 42.7% |
| Two-year contract | 2.8% |
| First 12 months of tenure | 47.4% |
| Tenure over 48 months | 9.5% |
| Fiber optic internet | 41.9% |
| Pays by electronic check | 45.3% |

Permutation importance (drop in test ROC-AUC when a feature is shuffled) ranks the top drivers as Contract (0.050), tenure (0.015), InternetService (0.014), TotalCharges (0.008) and tenure_group (0.008). These show association, not cause.

## Project structure

```
data/            put telco_churn.csv here (not committed)
src/data.py      loading, cleaning, feature engineering
src/eda.py       segment analysis and charts
src/train.py     model comparison, threshold tuning, evaluation, saved model
src/predict.py   score a CSV of customers from the command line
app.py           Streamlit app (single customer or CSV upload)
reports/         metrics.json, eda_summary.md, feature_importance.csv
```

## How to run

```bash
pip install -r requirements.txt
python src/eda.py        # writes charts to reports/figures
python src/train.py      # trains, evaluates, saves models/churn_model.joblib
python src/predict.py data/telco_churn.csv --top 10
streamlit run app.py
```

## Method

1. **Cleaning:** `TotalCharges` is stored as text; 11 blank values all belong to customers with tenure 0, so they are set to 0.
2. **Features:** number of add-on services, auto-pay flag, has partner or dependents, average monthly spend, tenure group.
3. **Modelling:** one-hot encoding plus scaling inside a scikit-learn pipeline, class weighting for the 26.5% positive rate, 5-fold stratified cross-validation.
4. **Threshold:** chosen on out-of-fold training predictions so the test set stays untouched.
5. **Evaluation:** ROC-AUC, recall, precision, F1, cumulative gains and lift on the held-out test set.

## Limitations

- One public dataset, one random split. No external or time-based validation.
- Class weighting is used instead of SMOTE, and scikit-learn models are used instead of XGBoost, because only scikit-learn was available when this was built. Feature importance uses permutation importance rather than SHAP for the same reason.
- The Streamlit app was syntax-checked but not launched in the build environment. The scoring code it calls (`src/predict.py`) was run and tested.
- The model scores customers; it does not say which retention offer will work. That needs an A/B test.
