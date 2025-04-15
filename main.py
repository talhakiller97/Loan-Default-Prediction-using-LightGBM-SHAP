import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import shap
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print("Loading and preprocessing data...")
# Load 5% of the dataset
df = pd.read_csv(
    r"C:\Users\Talha Saeed\PycharmProjects\week2task4\accepted_2007_to_2018Q4.csv",
    low_memory=False,
    nrows=200000  # Approx. 5% of the full dataset
)

# Keep only 'Fully Paid' and 'Charged Off'
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# Drop columns with too many missing values
df = df.dropna(axis=1, thresh=int(0.9 * len(df)))

# Drop rows with missing values (after dropping high-NaN columns)
df = df.dropna()

# Separate features and target AFTER all cleaning
y = df['loan_status']
X = df.select_dtypes(include=[np.number]).drop('loan_status', axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset with SMOTE
print("Balancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.01, random_state=42
)

# LightGBM Model
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

# Evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model("LightGBM", y_test, y_pred_lgb)

# Identify predicted defaulters
print("\nIdentifying predicted defaulters...")
results_df = pd.DataFrame(X_test, columns=X.columns)
results_df['actual'] = y_test
results_df['predicted'] = y_pred_lgb

predicted_defaulters = results_df[results_df['predicted'] == 1]
print("\nTop 50 predicted defaulters:")
print(predicted_defaulters.head(50))

# Optionally export defaulters to CSV
predicted_defaulters.to_csv("predicted_defaulters.csv", index=False)

# SHAP Explainability
print("\nGenerating SHAP explanations...")
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test[:100])

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test[:100], feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

# Recommendations
print("\nRECOMMENDATIONS:")
print("1. Flag high-risk applicants using model predictions for further manual review.")
print("2. Adjust lending criteria or offer higher interest rates to offset risk.")
print("3. Use model explainability tools like SHAP to understand which features impact default risk.")
print("4. Retrain model regularly using updated loan data to adapt to changing borrower behaviors.")
