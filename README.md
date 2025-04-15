# 💼 Loan Default Prediction using LightGBM & SHAP

This project leverages machine learning to predict loan defaults using historical Lending Club data. It applies SMOTE to handle class imbalance and uses a LightGBM model for classification. SHAP values provide explainability to model decisions.

---

## 📊 Project Overview

The goal of this project is to:

- Predict whether a loan will default (`Charged Off`) or be repaid (`Fully Paid`).
- Use only numeric features and clean data to train a robust classifier.
- Handle class imbalance with SMOTE.
- Interpret results with SHAP values for transparency.
- Identify high-risk applicants for further financial scrutiny.

---

## 🗃️ Dataset

- **Source:** [Lending Club Loan Data (2007–2018Q4)](https://www.lendingclub.com/)
- **Used File:** `accepted_2007_to_2018Q4.csv`
- **Rows Used:** 200,000 (approximately 5% of the full dataset)

---

## 📌 Features & Target

- **Target Variable:** `loan_status`
  - `Fully Paid` → 0
  - `Charged Off` → 1
- **Input Features:** All numeric features after data cleaning.

---

## 🧰 Tech Stack

- **Language:** Python
- **ML Library:** LightGBM
- **Preprocessing:** scikit-learn, pandas, numpy
- **Resampling:** SMOTE (`imbalanced-learn`)
- **Interpretability:** SHAP
- **Visualization:** Matplotlib

---

## ⚙️ Pipeline Summary

1. **Load Data**
   - Use 200k rows to keep memory manageable.
2. **Filter Classes**
   - Keep only loans with clear outcomes (`Fully Paid`, `Charged Off`).
3. **Drop Missing Values**
   - Remove columns with >10% NaNs and rows with any remaining NaNs.
4. **Feature Selection**
   - Use only numeric columns.
5. **Scaling**
   - Standardize features using `StandardScaler`.
6. **Class Balancing**
   - Apply SMOTE to balance the binary classes.
7. **Model Training**
   - Train a `LightGBMClassifier`.
8. **Evaluation**
   - Report precision, recall, F1 score, and a classification report.
9. **Explainability**
   - Use SHAP to understand feature contributions.
10. **Export**
    - Save predictions and SHAP summary.

---

## 🧪 Sample Results

--- LightGBM --- Precision: 0.8043 Recall: 0.8176 F1 Score: 0.8109

yaml
Copy
Edit

---

## 🔍 Explainability

SHAP (SHapley Additive exPlanations) was used to explain the top 100 test samples.

📁 Output: `shap_summary.png`

```python
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], feature_names=X.columns)
📂 Output Files
predicted_defaulters.csv — Predicted high-risk loans.

shap_summary.png — SHAP summary plot.

✅ Recommendations
Manual Review: Flag applicants predicted as defaulters for deeper risk assessment.

Risk-based Pricing: Adjust interest rates based on model scores.

Model Transparency: Use SHAP plots to justify approval/denial decisions.

Model Maintenance: Periodically retrain the model on new data.

▶️ How to Run

Clone this repository or copy the script.

Make sure your Lending Club CSV file is accessible.

Install required packages:

pip install pandas numpy scikit-learn imbalanced-learn lightgbm shap matplotlib

Run the script in your IDE or terminal:

python loan_default_prediction.py


👨‍💻 Author
Talha Saeed
Data Scientist | Python ML Developer
📍 Sharjah
🔗 GitHub
