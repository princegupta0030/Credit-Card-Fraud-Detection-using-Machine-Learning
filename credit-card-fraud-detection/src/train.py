import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import preprocess_pipeline

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression model with class weighting.
    """
    print("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train a Random Forest Classifier with class weighting.
    """
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1  # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Save the model using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def run_training():
    # Adjusted paths to work regardless of where the script is executed from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'creditcard.csv')
    models_dir = os.path.join(base_dir, 'models')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    if not os.path.exists(raw_data_path):
        print(f"Error: Raw dataset not found at {raw_data_path}")
        print("Please run the EDA notebook or download script first.")
        return

    print("Preparing data...")
    # Using the existing preprocessing pipeline
    # Note: We rely on the algorithm's class_weight='balanced' here instead of SMOTE,
    # as requested, to evaluate this approach.
    X_train, X_test, y_train, y_test = preprocess_pipeline(raw_data_path, scaler_path)

    # 1. Train and save Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_path = os.path.join(models_dir, 'logistic_regression.pkl')
    save_model(lr_model, lr_path)

    # 2. Train and save Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_path = os.path.join(models_dir, 'random_forest.pkl')
    save_model(rf_model, rf_path)

    print("\nTraining completed successfully!")
    print("\nFiles created in models directory:")
    for f in os.listdir(models_dir):
        print(f" - {f}")

if __name__ == "__main__":
    run_training()
