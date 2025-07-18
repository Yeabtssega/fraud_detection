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

```bash
python scripts/preprocess.py
