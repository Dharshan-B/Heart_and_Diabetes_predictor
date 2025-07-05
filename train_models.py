# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

os.makedirs('models', exist_ok=True)

# === DIABETES MODEL ===
diabetes_df = pd.read_csv('datasets/diabetes.csv')

X_dia = diabetes_df.drop(columns='Outcome', axis=1)
y_dia = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X_dia, y_dia, test_size=0.2, random_state=42)
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_train, y_train)

y_pred_dia = diabetes_model.predict(X_test)
print("Diabetes Model Accuracy:", accuracy_score(y_test, y_pred_dia))

# Save model
with open('models/diabetes_model.sav', 'wb') as f:
    pickle.dump(diabetes_model, f)

# === HEART DISEASE MODEL ===
heart_df = pd.read_csv('datasets/heart.csv')

# Identify the target column
if 'HeartDisease' in heart_df.columns:
    target_col = 'HeartDisease'
elif 'target' in heart_df.columns:
    target_col = 'target'
else:
    raise Exception("Cannot find heart disease target column.")

# Encode categorical columns (e.g., Sex, ChestPainType, etc.)
from sklearn.preprocessing import LabelEncoder

df = heart_df.copy()
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X_heart = df.drop(columns=target_col, axis=1)
y_heart = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

heart_model = RandomForestClassifier()
heart_model.fit(X_train, y_train)

y_pred_heart = heart_model.predict(X_test)
print("Heart Disease Model Accuracy:", accuracy_score(y_test, y_pred_heart))

# Save model
with open('models/heart_model.sav', 'wb') as f:
    pickle.dump(heart_model, f)

