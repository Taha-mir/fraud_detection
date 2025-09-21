# main.py
from src.data_preprocessing import load_and_clean_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.prediction_app import predict_transaction

if __name__ == "__main__":
 print("Loading and preparing data...")
 X_train_scaled, X_test_scaled, y_train, y_test, scaler, numerical_cols = load_and_clean_data()


 print("\nTraining model...")
 model = train_model(X_train_scaled, y_train)

 print("\nEvaluating model...")
 evaluate_model(model, X_test_scaled, y_test)

 print("\nMaking prediction for a new transaction...")
 predict_transaction(model, scaler, numerical_cols)