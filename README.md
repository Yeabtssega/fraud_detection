# Fraud Detection Project — Task 1: Data Analysis and Preprocessing

## Overview

This repository contains the data cleaning, exploratory data analysis (EDA), and preprocessing scripts for fraud detection on e-commerce and banking datasets.

## Data

- Raw datasets are in the `data/` folder.
- EDA figures generated using Python scripts and saved in `figures/` folder.

## Task 1 Summary

- Cleaned datasets by removing duplicates, handling missing values, and fixing data types.
- Created new features including time-based variables (`hour_of_day`, `day_of_week`, `time_since_signup`), transaction counts, and geolocation (country) from IP address.
- Addressed class imbalance by applying SMOTE on the training set.
- Saved EDA visualizations as PNG files in the `figures/` directory.

## Running the Code

Run the preprocessing and EDA script:

HEAD

4273f2d (Added Task 2 evaluation summary and results)
python scripts/preprocess.py
📊 Task 2: Model Training & Evaluation
Two models were trained and compared on both creditcard.csv and Fraud_Data.csv:

Logistic Regression: Used as a baseline due to its simplicity and interpretability

XGBoost: Chosen as a powerful gradient boosting model, ideal for handling class imbalance

Evaluation Metrics:

F1 Score: Balances precision and recall, critical for fraud detection

AUC-PR: Better than ROC-AUC for imbalanced data

Dataset	Model	F1-Score	AUC-PR
creditcard.csv	Logistic Regression	0.1247	0.7050
creditcard.csv	XGBoost	0.8456	0.8385
Fraud_Data.csv	Logistic Regression	0.1616	0.0956
Fraud_Data.csv	XGBoost	0.2826	0.3208

Conclusion:

XGBoost significantly outperforms Logistic Regression on both datasets. For creditcard.csv, it achieved an F1 score of 0.85 with an AUC-PR of 0.84, making it highly effective at detecting fraud. For Fraud_Data.csv, results were more modest, suggesting either weaker features or higher noise. Overall, XGBoost is the best-performing model.
