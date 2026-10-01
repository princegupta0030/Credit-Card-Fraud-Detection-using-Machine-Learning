import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.data_preprocessing import load_data, clean_data, get_features_and_target, split_data, preprocess_pipeline
from src.feature_engineering import engineer_features, apply_smote

RAW_DATA_PATH = os.path.join(base_dir, 'data', 'raw', 'creditcard.csv')
MODELS_DIR = os.path.join(base_dir, 'models')

def test_dataset_loading():
    assert os.path.exists(RAW_DATA_PATH), "Dataset file not found!"
    df = load_data(RAW_DATA_PATH)
    assert not df.empty, "Dataset is empty!"
    assert 'Class' in df.columns, "Target column 'Class' is missing."

def test_preprocessing():
    df = load_data(RAW_DATA_PATH)

    # Check cleaning
    df_clean = clean_data(df)
    assert 'Time' not in df_clean.columns, "Time column was not dropped during cleaning."
    assert df_clean.isnull().sum().sum() == 0, "Missing values were not handled."
    assert df_clean.duplicated().sum() == 0, "Duplicates were not dropped."

    # Check split sizes and stratification
    X, y = get_features_and_target(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train) > len(X_test), "Train set should be larger than test set."

    # Ratio difference should be minuscule
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.001, "Stratification failed."

def test_data_leakage_in_pipeline():
    X_train, X_test, y_train, y_test = preprocess_pipeline(RAW_DATA_PATH, apply_engineering=False)

    # Test that Amount is scaled to ~0 mean and ~1 std strictly on train set
    assert abs(X_train['Amount'].mean()) < 0.01, "Scaler was not properly fitted to train set mean."
    assert abs(X_train['Amount'].std() - 1.0) < 0.01, "Scaler was not properly fitted to train set std."

    # Test set should be slightly off because the scaler didn't see it (No Leakage)
    assert X_test['Amount'].mean() != X_train['Amount'].mean(), "Potential data leakage detected!"

def test_smote():
    X_train, _, y_train, _ = preprocess_pipeline(RAW_DATA_PATH, apply_engineering=False)
    X_smote, y_smote = apply_smote(X_train, y_train, random_state=42)

    # SMOTE should perfectly balance the classes
    assert y_smote.mean() == 0.5, "SMOTE did not perfectly balance the classes."
    assert len(X_smote) > len(X_train), "SMOTE did not synthesize new samples."

def test_reproducibility():
    X_train1, X_test1, _, _ = preprocess_pipeline(RAW_DATA_PATH, apply_engineering=False)
    X_train2, X_test2, _, _ = preprocess_pipeline(RAW_DATA_PATH, apply_engineering=False)

    assert X_train1.equals(X_train2), "Train split is not reproducible with random_state=42!"
    assert X_test1.equals(X_test2), "Test split is not reproducible with random_state=42!"

def test_model_loading_and_feature_order():
    final_model_path = os.path.join(MODELS_DIR, 'final_model.pkl')
    assert os.path.exists(final_model_path), "Final model not found."

    model = joblib.load(final_model_path)
    X_train, _, _, _ = preprocess_pipeline(RAW_DATA_PATH, apply_engineering=False)

    # The columns the model was trained on must exactly match the columns in X_train
    model_features = list(model.feature_names_in_)
    pipeline_features = list(X_train.columns)

    assert model_features == pipeline_features, f"Feature mismatch! Model expects {model_features}, but pipeline provides {pipeline_features}."

def test_invalid_input_to_model():
    final_model_path = os.path.join(MODELS_DIR, 'final_model.pkl')
    model = joblib.load(final_model_path)

    # Create bad dataframe (missing columns)
    bad_df = pd.DataFrame({"V1": [1.0], "Amount": [100.0]})

    with pytest.raises(ValueError):
         model.predict(bad_df)
