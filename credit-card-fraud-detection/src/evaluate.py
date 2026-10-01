import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# Ensure src is in the python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.data_preprocessing import preprocess_pipeline

def load_test_data():
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'creditcard.csv')
    # Use the pipeline to get the strictly split X_test and y_test
    _, X_test, _, y_test = preprocess_pipeline(raw_data_path)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'PR-AUC': average_precision_score(y_test, y_proba)
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred, y_proba

def plot_confusion_matrices(cm_lr, cm_rf, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Random Forest')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_test, proba_lr, proba_rf, save_path):
    fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, proba_lr):.4f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, proba_rf):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_pr_curves(y_test, proba_lr, proba_rf, save_path):
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, proba_lr)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, proba_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_lr, precision_lr, label=f'Logistic Regression (PR-AUC = {average_precision_score(y_test, proba_lr):.4f})')
    plt.plot(recall_rf, precision_rf, label=f'Random Forest (PR-AUC = {average_precision_score(y_test, proba_rf):.4f})')

    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def run_evaluation():
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports', 'figures')
    os.makedirs(reports_dir, exist_ok=True)

    print("Loading models and data...")
    lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))

    X_test, y_test = load_test_data()

    print("\nEvaluating Logistic Regression...")
    metrics_lr, cm_lr, pred_lr, proba_lr = evaluate_model(lr_model, X_test, y_test)

    print("Evaluating Random Forest...")
    metrics_rf, cm_rf, pred_rf, proba_rf = evaluate_model(rf_model, X_test, y_test)

    print("\nGenerating and saving plots...")
    plot_confusion_matrices(cm_lr, cm_rf, os.path.join(reports_dir, 'confusion_matrix.png'))
    plot_roc_curves(y_test, proba_lr, proba_rf, os.path.join(reports_dir, 'roc_curve.png'))
    plot_pr_curves(y_test, proba_lr, proba_rf, os.path.join(reports_dir, 'precision_recall_curve.png'))
    print(f"Figures saved successfully in {reports_dir}")

    print("\n" + "="*50)
    print("           MODEL EVALUATION COMPARISON")
    print("="*50)
    comparison_df = pd.DataFrame([metrics_lr, metrics_rf], index=['Logistic Regression', 'Random Forest'])
    print(comparison_df.to_markdown())

    print("\n" + "="*50)
    print("        EVALUATION CONTEXT & TRADE-OFFS")
    print("="*50)
    print("\n1. Accuracy vs. Reality:")
    print("In highly imbalanced datasets like credit card fraud (where ~99.8% of transactions are normal),")
    print("a naive model that predicts 'Normal' for everything would achieve 99.8% accuracy.")
    print("Therefore, we DO NOT select a model simply because it has the highest accuracy.")

    print("\n2. False Positives (FP) vs. False Negatives (FN):")
    print("- False Positive (FP): A normal transaction is incorrectly flagged as fraudulent.")
    print("  Cost: Customer inconvenience, blocked cards, support call overhead.")
    print("- False Negative (FN): A fraudulent transaction goes undetected.")
    print("  Cost: Direct financial loss to the customer or the bank, loss of trust.")

    print("\n3. Precision vs. Recall Trade-off:")
    print("- Precision: Out of all transactions flagged as fraud, how many were actually fraud?")
    print("             (Focuses on minimizing FP)")
    print("- Recall: Out of all actual fraudulent transactions, how many did we successfully catch?")
    print("          (Focuses on minimizing FN)")

    print("\n4. Why Recall is Crucial for Fraud Detection:")
    print("In the context of fraud, the cost of missing a fraudulent transaction (FN) is typically much higher ")
    print("than the cost of annoying a customer with a verification text (FP). Therefore, maximizing Recall ")
    print("(catching as much fraud as possible) is generally prioritized, even if it means sacrificing some Precision.")
    print("We aim for a balance, often evaluated via the F1-score or PR-AUC.")

if __name__ == "__main__":
    run_evaluation()
