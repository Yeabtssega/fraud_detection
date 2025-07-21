import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create folder for SHAP plots
os.makedirs("C:/Users/HP/fraud_detection/figures/shap", exist_ok=True)

# Load data
credit_df = pd.read_csv(r'C:\Users\HP\fraud_detection\data\creditcard.csv')
fraud_df = pd.read_csv(r'C:\Users\HP\fraud_detection\data\Fraud_Data.csv')

def prepare_data(df, target_col):
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])  # Remove non-numeric
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

# Prepare creditcard data
Xc_train, Xc_test, yc_train, yc_test = prepare_data(credit_df, 'Class')
Xc_train_scaled, Xc_test_scaled = scale_data(Xc_train, Xc_test)

# Prepare fraud data
Xf_train, Xf_test, yf_train, yf_test = prepare_data(fraud_df, 'class')
Xf_train_scaled, Xf_test_scaled = scale_data(Xf_train, Xf_test)

# Train XGBoost models (you can reuse your best hyperparameters)
xgb_credit = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=np.sum(yc_train == 0)/np.sum(yc_train == 1))
xgb_credit.fit(Xc_train_scaled, yc_train)

xgb_fraud = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=np.sum(yf_train == 0)/np.sum(yf_train == 1))
xgb_fraud.fit(Xf_train_scaled, yf_train)

# Initialize SHAP explainer for creditcard
explainer_credit = shap.TreeExplainer(xgb_credit)
shap_values_credit = explainer_credit.shap_values(Xc_test_scaled)

# Summary plot (global feature importance)
shap.summary_plot(shap_values_credit, Xc_test, plot_type="bar", show=False)
plt.savefig("C:/Users/HP/fraud_detection/figures/shap/creditcard_summary_bar.png")
plt.clf()

shap.summary_plot(shap_values_credit, Xc_test, show=False)
plt.savefig("C:/Users/HP/fraud_detection/figures/shap/creditcard_summary_dot.png")
plt.clf()

# Force plot for a specific test instance (local explanation)
idx = 0  # Change this to any index of interest
force_plot = shap.force_plot(explainer_credit.expected_value, shap_values_credit[idx], Xc_test.iloc[idx], matplotlib=True, show=False)
plt.savefig("C:/Users/HP/fraud_detection/figures/shap/creditcard_force_plot.png")
plt.clf()

# Repeat for fraud dataset
explainer_fraud = shap.TreeExplainer(xgb_fraud)
shap_values_fraud = explainer_fraud.shap_values(Xf_test_scaled)

shap.summary_plot(shap_values_fraud, Xf_test, plot_type="bar", show=False)
plt.savefig("C:/Users/HP/fraud_detection/figures/shap/frauddata_summary_bar.png")
plt.clf()

shap.summary_plot(shap_values_fraud, Xf_test, show=False)
plt.savefig("C:/Users/HP/fraud_detection/figures/shap/frauddata_summary_dot.png")
plt.clf()

force_plot_fraud = shap.force_plot(explainer_fraud.expected_value, shap_values_fraud[idx], Xf_test.iloc[idx], matplotlib=True, show=False)
plt.savefig("C:/Users/HP/fraud_detection/figures/shap/frauddata_force_plot.png")
plt.clf()
