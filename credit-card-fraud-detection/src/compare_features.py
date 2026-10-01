import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.data_preprocessing import preprocess_pipeline
from src.evaluate import evaluate_model

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name="Logistic Regression"):
    if model_name == "Logistic Regression":
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    metrics, _, _, _ = evaluate_model(model, X_test, y_test)
    return metrics

def compare_features():
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'creditcard.csv')

    print("\n" + "="*50)
    print("1. EVALUATING BASELINE (NO FEATURE ENGINEERING)")
    print("="*50)
    X_train_base, X_test_base, y_train, y_test = preprocess_pipeline(raw_data_path, apply_engineering=False)

    metrics_lr_base = train_and_evaluate(X_train_base, y_train, X_test_base, y_test, "Logistic Regression")
    metrics_rf_base = train_and_evaluate(X_train_base, y_train, X_test_base, y_test, "Random Forest")

    print("\n" + "="*50)
    print("2. EVALUATING ENGINEERED FEATURES")
    print("="*50)
    X_train_eng, X_test_eng, _, _ = preprocess_pipeline(raw_data_path, apply_engineering=True)

    metrics_lr_eng = train_and_evaluate(X_train_eng, y_train, X_test_eng, y_test, "Logistic Regression")
    metrics_rf_eng = train_and_evaluate(X_train_eng, y_train, X_test_eng, y_test, "Random Forest")

    print("\n" + "="*80)
    print("                      COMPARISON RESULTS")
    print("="*80)

    results = [
        {"Model": "Logistic Regression", "Features": "Baseline", **metrics_lr_base},
        {"Model": "Logistic Regression", "Features": "Engineered", **metrics_lr_eng},
        {"Model": "Random Forest", "Features": "Baseline", **metrics_rf_base},
        {"Model": "Random Forest", "Features": "Engineered", **metrics_rf_eng}
    ]

    df_results = pd.DataFrame(results)

    # Sort for easier reading
    df_results = df_results.sort_values(by=["Model", "Features"], ascending=[True, True])
    print(df_results.to_markdown(index=False))

    print("\nConclusion:")
    print("Observe if the Engineered features (Log_Amount, Hour, Hour_Sin, Hour_Cos) improved the PR-AUC or F1-Score.")
    print("Often, logistic regression benefits highly from the Log_Amount transformation because it relies on normally distributed features.")

if __name__ == "__main__":
    compare_features()
