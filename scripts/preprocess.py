import os
import pandas as pd
import socket
import struct
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Create figures folder if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except Exception:
        return None

def find_country(ip_int, ip_df):
    matched = ip_df[(ip_df['lower_bound_ip_address'] <= ip_int) & (ip_df['upper_bound_ip_address'] >= ip_int)]
    if not matched.empty:
        return matched.iloc[0]['country']
    else:
        return 'Unknown'

def save_eda_figures(df):
    # Fraud class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='class', data=df)
    plt.title('Fraud Class Distribution')
    plt.savefig('figures/fraud_class_distribution.png')
    plt.close()

    # Purchase value distribution by class
    plt.figure(figsize=(8,5))
    sns.boxplot(x='class', y='purchase_value', data=df)
    plt.title('Purchase Value Distribution by Fraud Class')
    plt.yscale('log')
    plt.savefig('figures/purchase_value_distribution.png')
    plt.close()

    # Fraud by sex
    plt.figure(figsize=(6,4))
    sns.countplot(x='sex', hue='class', data=df)
    plt.title('Fraud Distribution by Sex')
    plt.savefig('figures/fraud_by_sex.png')
    plt.close()

    # Fraud by source
    plt.figure(figsize=(8,5))
    sns.countplot(x='source', hue='class', data=df)
    plt.title('Fraud Distribution by Source')
    plt.xticks(rotation=45)
    plt.savefig('figures/fraud_by_source.png')
    plt.close()

    # Fraud by country (top 15 countries by fraud proportion)
    fraud_country = df.groupby(['country', 'class']).size().unstack(fill_value=0)
    fraud_country_prop = fraud_country.div(fraud_country.sum(axis=1), axis=0)
    fraud_country_prop = fraud_country_prop[1].sort_values(ascending=False).head(15)
    plt.figure(figsize=(12,6))
    sns.barplot(x=fraud_country_prop.index, y=fraud_country_prop.values)
    plt.title('Top 15 Countries by Fraud Proportion')
    plt.ylabel('Proportion of Fraudulent Transactions')
    plt.xticks(rotation=45)
    plt.savefig('figures/fraud_by_country.png')
    plt.close()

    # Age distribution by fraud class
    plt.figure(figsize=(8,5))
    sns.boxplot(x='class', y='age', data=df)
    plt.title('Age Distribution by Fraud Class')
    plt.savefig('figures/age_distribution_by_class.png')
    plt.close()

def main():
    print("Loading datasets...")
    fraud_df = pd.read_csv('data/Fraud_Data.csv', parse_dates=['signup_time', 'purchase_time'])
    ip_df = pd.read_csv('data/IpAddress_to_Country.csv')
    credit_df = pd.read_csv('data/creditcard.csv')  # Loaded but not processed here

    print("Converting IP addresses...")
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)

    print("Mapping IP addresses to countries...")
    fraud_df['country'] = fraud_df['ip_int'].apply(lambda x: find_country(x, ip_df))

    print("Feature engineering...")
    fraud_df['transaction_count'] = fraud_df.groupby('user_id')['user_id'].transform('count')
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()

    print("Saving EDA figures...")
    save_eda_figures(fraud_df)

    print("Preparing data for modeling...")
    # Drop unused columns
    fraud_model_df = fraud_df.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'ip_int'])

    # One-hot encode categorical features
    X = pd.get_dummies(fraud_model_df.drop(columns=['class']), drop_first=True)
    y = fraud_model_df['class']

    print("Splitting dataset into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Scaling numeric features...")
    numeric_cols = ['purchase_value', 'age', 'transaction_count', 'hour_of_day', 'day_of_week', 'time_since_signup']

    scaler = StandardScaler()
    X_train_resampled[numeric_cols] = scaler.fit_transform(X_train_resampled[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("Preprocessing complete.")
    print(f"Training set size: {X_train_resampled.shape}")
    print(f"Fraud counts in training set:\n{y_train_resampled.value_counts()}")
    print(f"Test set size: {X_test.shape}")
    print(f"Fraud counts in test set:\n{y_test.value_counts()}")

if __name__ == "__main__":
    main()
