# 🕵️‍♂️ Fraud Detection Using Machine Learning

## 📌 Overview
This project explores fraud detection using machine learning techniques across two datasets (`creditcard.csv` and `Fraud_Data.csv`). It covers:
- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis (EDA)
- Model training and evaluation (Logistic Regression & XGBoost)

---

## 📁 Project Structure

fraud_detection/
│
├── data/ # Raw and cleaned datasets
├── scripts/ # Python scripts for EDA, preprocessing, modeling
├── figures/ # Plots and visualizations
├── models/ # Saved models (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

markdown
Copy
Edit

---

## 🧼 Task 1: Data Cleaning & Preprocessing

- Removed duplicates and handled missing values
- Engineered features:
  - `hour_of_day`, `day_of_week`, `time_since_signup`
  - Transaction frequency
  - Country mapping from IP address
- Handled class imbalance using SMOTE
- Saved EDA plots to `figures/` folder

**To run:**
```bash
python scripts/preprocess.py
🤖 Task 2: Model Training & Evaluation
Models:
Logistic Regression: Baseline for interpretability

XGBoost: Handles imbalance and captures nonlinearity

Metrics:
F1 Score: Balance between precision and recall

AUC-PR: Preferred over ROC for imbalanced data

Results:
Dataset	Model	F1-Score	AUC-PR
creditcard.csv	Logistic Regression	0.1247	0.7050
creditcard.csv	XGBoost	0.8456	0.8385
Fraud_Data.csv	Logistic Regression	0.1616	0.0956
Fraud_Data.csv	XGBoost	0.2826	0.3208

🔍 Conclusion:
XGBoost significantly outperforms Logistic Regression, especially on creditcard.csv. It achieved an F1 score of 0.85, showing strong fraud detection performance. Results on Fraud_Data.csv suggest weaker features or more noise, indicating a need for better feature engineering.

⚙️ Requirements
Install dependencies:
pandas==1.5.3
numpy==1.24.4
scikit-learn==1.2.2
imbalanced-learn==0.10.1
matplotlib==3.7.1
seaborn==0.12.2
xgboost

👤 Author
Yeabtsega Tilahun
📧 yeabtsegatilahun77@gmail.com
🔗 GitHub Profile

