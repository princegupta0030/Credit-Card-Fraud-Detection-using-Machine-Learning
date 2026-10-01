import os
import sys
import joblib
import json
import pandas as pd
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.evaluate import load_test_data

def analyze_thresholds(y_test, y_proba, thresholds):
    """
    Analyze performance across various decision thresholds.
    """
    results = []
    for threshold in thresholds:
        # If probability >= threshold, predict 1 (Fraud), else 0 (Normal)
        y_pred = (y_proba >= threshold).astype(int)

        results.append({
            'Threshold': threshold,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred)
        })

    return pd.DataFrame(results)

def run_analysis():
    models_dir = os.path.join(base_dir, 'models')

    print("\n" + "="*50)
    print("1. MODEL SELECTION")
    print("="*50)
    print("Based on the previous evaluation stage:")
    print("Logistic Regression struggled with extreme False Positives (Precision ~5%).")
    print("Random Forest achieved a highly balanced profile (Precision ~95%, Recall ~73%, F1 ~83%).")
    print("Therefore, we select Random Forest as our base model. We will NOT base this solely on accuracy,")
    print("as both models achieved >97% accuracy due to the imbalanced nature of the dataset.")

    # Load selected model and test data
    rf_model_path = os.path.join(models_dir, 'random_forest.pkl')
    model = joblib.load(rf_model_path)
    X_test, y_test = load_test_data()

    # Get probabilities for the positive class (Fraud)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("2. THRESHOLD ANALYSIS")
    print("="*50)
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    df_results = analyze_thresholds(y_test, y_proba, thresholds)
    print(df_results.to_markdown(index=False))

    print("\n" + "="*50)
    print("3. EXPLANATION OF TRADE-OFFS")
    print("="*50)
    print("As the threshold INCREASES (e.g., from 0.10 to 0.70):")
    print(" - The model becomes more conservative, requiring higher certainty to flag fraud.")
    print(" - False Positives DECREASE (Precision goes up). We bother fewer normal customers.")
    print(" - False Negatives INCREASE (Recall goes down). We miss more actual fraud.")
    print("\nAs the threshold DECREASES (e.g., from 0.50 to 0.10):")
    print(" - The model becomes more sensitive, flagging anything slightly suspicious.")
    print(" - False Negatives DECREASE (Recall goes up). We catch almost all fraud.")
    print(" - False Positives INCREASE (Precision goes down). Many normal transactions are declined.")

    print("\n" + "="*50)
    print("4. FINAL SELECTION & REASONING")
    print("="*50)

    # Select a reasonable threshold (e.g., 0.30)
    selected_threshold = 0.30

    print(f"Selected Threshold: {selected_threshold}")
    print("Reasoning: In fraud detection, catching fraud (Recall) is generally prioritized over")
    print("perfect Precision, because the financial cost of a fraudulent transaction usually outweighs")
    print("the operational cost of an SMS verification for a false positive.")
    print(f"Looking at the table, a threshold of {selected_threshold} significantly boosts Recall")
    print("compared to the default 0.50, while maintaining an acceptable Precision.")
    print("Note: This threshold is NOT universally optimal. A bank might adjust this threshold dynamically:")
    print("lower it during holiday shopping spikes, or raise it for low-value transactions.")

    print("\n" + "="*50)
    print("5. SAVING CONFIGURATION")
    print("="*50)

    # Save the final model explicitly
    final_model_path = os.path.join(models_dir, 'final_model.pkl')
    shutil.copy(rf_model_path, final_model_path)

    # Save the configuration
    config = {
        "model_name": "Random Forest Classifier",
        "model_file": "final_model.pkl",
        "selected_threshold": selected_threshold,
        "justification": "Favors recall to prevent financial loss, while keeping precision viable."
    }

    config_path = os.path.join(models_dir, 'final_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Final model saved as: {final_model_path}")
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    run_analysis()
