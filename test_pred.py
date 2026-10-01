import pandas as pd
import joblib
import json

with open('credit-card-fraud-detection/models/final_config.json', 'r') as f:
    config = json.load(f)

model = joblib.load(f"credit-card-fraud-detection/models/{config['model_file']}")
scaler = joblib.load("credit-card-fraud-detection/models/scaler.pkl")

# Test all zeros
input_data = {f"V{i}": 0.0 for i in range(1, 29)}
input_data["Amount"] = 100.0

df = pd.DataFrame([input_data])
df[['Amount']] = scaler.transform(df[['Amount']])

proba = model.predict_proba(df)[0, 1]
print(f"All zeros proba: {proba * 100:.4f}%")

# Find a real fraud case
raw_data = pd.read_csv('credit-card-fraud-detection/data/raw/creditcard.csv')
fraud_cases = raw_data[raw_data['Class'] == 1].head(1)
if len(fraud_cases) > 0:
    fraud_row = fraud_cases.drop(columns=['Time', 'Class'])
    fraud_row_scaled = fraud_row.copy()
    fraud_row_scaled[['Amount']] = scaler.transform(fraud_row_scaled[['Amount']])
    fraud_proba = model.predict_proba(fraud_row_scaled)[0, 1]
    print(f"Real fraud proba: {fraud_proba * 100:.4f}%")
    print("\nReal fraud values to use:")
    for col in fraud_row.columns:
        print(f"{col}: {fraud_row[col].values[0]}")
