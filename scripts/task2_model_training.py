import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score
from xgboost import XGBClassifier

<<<<<<< HEAD
# Create figures directory if it doesn't exist
=======
# Make sure figures directory exists
>>>>>>> 3de606c (WIP: local changes before pulling)
os.makedirs("C:/Users/HP/fraud_detection/figures", exist_ok=True)

# Load datasets
credit_df = pd.read_csv(r'C:\Users\HP\fraud_detection\data\creditcard.csv')
fraud_df = pd.read_csv(r'C:\Users\HP\fraud_detection\data\Fraud_Data.csv')

<<<<<<< HEAD
def prepare_data(df, target_column):
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=[np.number])  # Drop non-numeric columns like datetime
=======
# Split and preprocess
def prepare_data(df, target_column):
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=[np.number])  # Drop non-numeric (e.g., datetime)
>>>>>>> 3de606c (WIP: local changes before pulling)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

<<<<<<< HEAD
def calculate_scale_pos_weight(y):
    # Calculate scale_pos_weight for XGBoost dynamically
    return np.sum(y == 0) / np.sum(y == 1)

def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"\n=== {name} ===")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fraud', 'Fraud'],
                yticklabels=['No Fraud', 'Fraud'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("True")
=======
def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"\n=== {name} ===")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel("Predicted"); plt.ylabel("True")
>>>>>>> 3de606c (WIP: local changes before pulling)
    plt.savefig(f'C:/Users/HP/fraud_detection/figures/{name}_conf_matrix.png')
    plt.clf()

    print(classification_report(y_true, y_pred))
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC-PR: {average_precision_score(y_true, y_proba):.4f}")

<<<<<<< HEAD
# Prepare creditcard dataset
Xc_train, Xc_test, yc_train, yc_test = prepare_data(credit_df, 'Class')
Xc_train, Xc_test = scale_data(Xc_train, Xc_test)
scale_pos_weight_credit = calculate_scale_pos_weight(yc_train)

# Prepare Fraud_Data dataset
Xf_train, Xf_test, yf_train, yf_test = prepare_data(fraud_df, 'class')
Xf_train, Xf_test = scale_data(Xf_train, Xf_test)
scale_pos_weight_fraud = calculate_scale_pos_weight(yf_train)

# Initialize models
lr = LogisticRegression(max_iter=5000, class_weight='balanced')

xgb_credit = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_credit)
xgb_fraud = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_fraud)
=======
# Prepare creditcard.csv
Xc_train, Xc_test, yc_train, yc_test = prepare_data(credit_df, 'Class')
Xc_train, Xc_test = scale_data(Xc_train, Xc_test)

# Prepare Fraud_Data.csv
Xf_train, Xf_test, yf_train, yf_test = prepare_data(fraud_df, 'class')
Xf_train, Xf_test = scale_data(Xf_train, Xf_test)

# Initialize models
lr = LogisticRegression(max_iter=5000, class_weight='balanced')
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=10)
>>>>>>> 3de606c (WIP: local changes before pulling)

# Train & evaluate Logistic Regression on creditcard.csv
lr.fit(Xc_train, yc_train)
yc_pred = lr.predict(Xc_test)
yc_proba = lr.predict_proba(Xc_test)[:, 1]
evaluate_model("LogReg_CreditCard", yc_test, yc_pred, yc_proba)

# Train & evaluate Logistic Regression on Fraud_Data.csv
lr.fit(Xf_train, yf_train)
yf_pred = lr.predict(Xf_test)
yf_proba = lr.predict_proba(Xf_test)[:, 1]
evaluate_model("LogReg_FraudData", yf_test, yf_pred, yf_proba)

# Train & evaluate XGBoost on creditcard.csv
<<<<<<< HEAD
xgb_credit.fit(Xc_train, yc_train)
yc_pred_xgb = xgb_credit.predict(Xc_test)
yc_proba_xgb = xgb_credit.predict_proba(Xc_test)[:, 1]
evaluate_model("XGB_CreditCard", yc_test, yc_pred_xgb, yc_proba_xgb)

# Train & evaluate XGBoost on Fraud_Data.csv
xgb_fraud.fit(Xf_train, yf_train)
yf_pred_xgb = xgb_fraud.predict(Xf_test)
yf_proba_xgb = xgb_fraud.predict_proba(Xf_test)[:, 1]
evaluate_model("XGB_FraudData", yf_test, yf_pred_xgb, yf_proba_xgb)

# Summary printout to clarify best models
print("\n=== Summary ===")
print(f"CreditCard Dataset - XGBoost F1 Score: {f1_score(yc_test, yc_pred_xgb):.4f}")
print(f"CreditCard Dataset - Logistic Regression F1 Score: {f1_score(yc_test, yc_pred):.4f}")
print(f"FraudData Dataset - XGBoost F1 Score: {f1_score(yf_test, yf_pred_xgb):.4f}")
print(f"FraudData Dataset - Logistic Regression F1 Score: {f1_score(yf_test, yf_pred):.4f}")
print("\nBased on F1 Scores and AUC-PR, XGBoost models outperform Logistic Regression on both datasets, especially creditcard.csv.")
=======
xgb.fit(Xc_train, yc_train)
yc_pred_xgb = xgb.predict(Xc_test)
yc_proba_xgb = xgb.predict_proba(Xc_test)[:, 1]
evaluate_model("XGB_CreditCard", yc_test, yc_pred_xgb, yc_proba_xgb)

# Train & evaluate XGBoost on Fraud_Data.csv
xgb.fit(Xf_train, yf_train)
yf_pred_xgb = xgb.predict(Xf_test)
yf_proba_xgb = xgb.predict_proba(Xf_test)[:, 1]
evaluate_model("XGB_FraudData", yf_test, yf_pred_xgb, yf_proba_xgb)
>>>>>>> 3de606c (WIP: local changes before pulling)
