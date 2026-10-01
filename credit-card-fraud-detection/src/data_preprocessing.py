import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath):
    """Load the dataset from the given filepath."""
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Clean the dataset:
    - Handle missing values
    - Remove duplicates
    - Drop non-predictive columns
    """
    # 1. Handle missing values (though we know there are none in this dataset)
    df = df.dropna()

    # 2. Handle duplicates
    # Removing exact duplicate rows as they don't provide new information
    # and can artificially inflate model performance if split between train/test
    df = df.drop_duplicates()

    # 3. Drop non-predictive columns
    # 'Time' is the seconds elapsed between each transaction and the first transaction.
    # Without cyclical feature engineering, it's not directly useful as a continuous predictor.
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    return df

def get_features_and_target(df, target_col='Class'):
    """Separate features X and target y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets with stratification."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def scale_data(X_train, X_test, cols_to_scale=None):
    """
    Scale numerical features to prevent data leakage.
    The scaler is fitted ONLY on the training data.
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if cols_to_scale is None:
        # Scale all columns
        X_train_scaled[:] = scaler.fit_transform(X_train)
        X_test_scaled[:] = scaler.transform(X_test)
    else:
        # Scale only specific columns
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler

def save_scaler(scaler, output_path):
    """Save the fitted scaler for future use in inference."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(scaler, output_path)

def preprocess_pipeline(filepath, scaler_output_path=None):
    """
    Run the full preprocessing pipeline.
    """
    print("Loading data...")
    df = load_data(filepath)

    print("Cleaning data...")
    df = clean_data(df)

    print("Separating features and target...")
    X, y = get_features_and_target(df)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Scaling features...")
    # Typically V1-V28 are already scaled by PCA, but 'Amount' definitely needs scaling.
    # To be safe and uniform, we scale the 'Amount' column.
    # (Applying StandardScaler to PCA components won't hurt, but just 'Amount' is more targeted).
    cols_to_scale = ['Amount'] if 'Amount' in X_train.columns else None

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test, cols_to_scale=cols_to_scale)

    if scaler_output_path:
        print(f"Saving scaler to {scaler_output_path}...")
        save_scaler(scaler, scaler_output_path)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Test the pipeline
    raw_data_path = "../data/raw/creditcard.csv"
    if os.path.exists(raw_data_path):
        X_train, X_test, y_train, y_test = preprocess_pipeline(
            filepath=raw_data_path,
            scaler_output_path="../models/scaler.pkl"
        )
        print("Pipeline executed successfully.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train fraud ratio: {y_train.mean():.4f}")
        print(f"y_test fraud ratio: {y_test.mean():.4f}")
    else:
        print(f"Dataset not found at {raw_data_path}")
