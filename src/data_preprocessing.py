 # src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean_data():
 data = pd.read_csv("my_project/data/Fraud.csv") # شامل Time, V1..V28, Amount, Class


 X = data.drop('Class', axis=1)
 y = data['Class']


 # مقیاس‌بندی Time و Amount
 numerical_cols = ['Time', 'Amount']
 scaler = StandardScaler()
 X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

 return X_train, X_test, y_train, y_test, scaler, numerical_cols

