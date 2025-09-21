# src/prediction_app.py
import pandas as pd

def predict_transaction(model, scaler, numerical_cols):
    print("\n--- Predict Fraud for a New Transaction ---")
    
    # گرفتن اطلاعات ورودی از کاربر
    time = float(input("Enter transaction time (e.g., 10000): "))
    amount = float(input("Enter transaction amount: "))

    # پیش‌فرض برای V1 تا V28 صفر است
    v_features = [0.0] * 28

    # ساخت دیتافریم ورودی
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_data = [ [time] + v_features + [amount] ]
    input_df = pd.DataFrame(input_data, columns=columns)

    # مقیاس‌بندی Time و Amount
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # پیش‌بینی
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    print("\nPrediction result:")
    if prediction[0] == 1:
        print("🚨 This transaction is FRAUDULENT!")
    else:
        print("✅ This transaction is NOT fraudulent.")
    print(f"Fraud Probability: {probability[0]*100:.2f}%")
