# 💳 Credit Card Fraud Detection

**An end-to-end Machine Learning project to identify fraudulent credit card transactions.**

---

## 1. Project Overview
This repository contains a complete Machine Learning pipeline built to detect fraudulent credit card transactions. It covers every stage of the data science lifecycle: data downloading, exploratory data analysis (EDA), data cleaning, feature engineering, severe class imbalance handling, model training, threshold tuning, and a Streamlit frontend application.

*Note: This is an academic/portfolio project. It demonstrates MLOps and Data Science best practices, but it is not intended to replace a production-grade banking system.*

## 2. Problem Statement
Credit card fraud costs consumers and financial institutions billions of dollars annually. The objective of this project is to build a machine learning model capable of accurately recognizing fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## 3. Why Fraud Detection is Challenging
- **Extreme Class Imbalance:** Fraudulent transactions represent a tiny fraction of total transactions (often less than 0.2%). Models naturally bias toward the majority "Normal" class.
- **Data Privacy:** Financial data is highly confidential. Open-source datasets are typically anonymized using techniques like Principal Component Analysis (PCA), making intuitive feature engineering difficult.
- **The Precision/Recall Trade-off:** We must balance catching as much fraud as possible (Recall) against declining too many legitimate transactions (Precision).

## 4. Dataset Description
The project uses the widely recognized [Credit Card Fraud Detection dataset from Kaggle (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (0.172%)
- **Missing Values:** 0

## 5. Features Used
Due to confidentiality issues, the original features and background information are not provided.
- **V1 - V28:** 28 numerical features resulting from a PCA transformation.
- **Amount:** The transaction amount.
- **Class:** The response variable (1 for fraud, 0 for normal).
- *(Note: The original `Time` column was dropped as it simply represented seconds elapsed from the first transaction and caused data leakage when identifying duplicates).*

## 6. ML Pipeline
The pipeline is designed with reproducibility and anti-leakage best practices in mind:
1. Load dataset (via `kagglehub`).
2. Engineer features (if configured).
3. Clean data (remove duplicates).
4. **Train-Test Split** (80/20, stratified on the target class).
5. Scale features (**fitted strictly on the training set**).
6. Train models.
7. Evaluate on the untouched test set.

## 7. Data Preprocessing
To prevent data leakage, the `StandardScaler` used to normalize the `Amount` feature is strictly fitted on `X_train`. The test set (`X_test`) is only transformed. This ensures the model does not "peek" at the distribution of the test set during training.

## 8. Class Imbalance Problem
Because fraud represents only ~0.17% of the data, a naive model that predicts "Normal" every time would achieve 99.83% accuracy. Accuracy is a highly misleading metric here.

## 9. SMOTE vs. Class Weighting
We explored two methods to handle this imbalance:
- **SMOTE (Synthetic Minority Over-sampling Technique):** Synthesizes new fraud examples. *Important: Applied only to the training set to prevent leakage.*
- **Class Weighting:** Instructs the algorithm to penalize mistakes on the minority class more heavily.
We ultimately utilized **Class Weighting** (`class_weight='balanced'`) during model training as it yielded excellent results without altering the underlying data distribution.

## 10. Models Used
Two base models were tested:
1. **Logistic Regression:** A highly interpretable linear baseline.
2. **Random Forest Classifier:** A powerful ensemble tree method capable of capturing non-linear relationships.

## 11. Evaluation Metrics
We evaluated models using:
- **Precision:** Out of all transactions flagged as fraud, how many were actually fraud?
- **Recall:** Out of all actual fraudulent transactions, how many did we catch?
- **F1-Score:** The harmonic mean of Precision and Recall.
- **PR-AUC:** Area Under the Precision-Recall Curve (the gold standard for highly imbalanced datasets).
- **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve.

## 12. Actual Model Results
*Results obtained from the exact final execution on the 20% untouched test set.*

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.977 | 0.065 | **0.905** | 0.121 | 0.982 | 0.691 |
| **Random Forest** | **0.999** | **0.878** | 0.757 | **0.813** | 0.949 | **0.800** |

## 13. Model Selection Reasoning
While Logistic Regression caught more fraud (Recall ~90%), its Precision was abysmal (~6%). This means 94% of the customers it flagged as fraudulent were actually innocent, which would cause unacceptable operational overhead and customer dissatisfaction.
**Random Forest** was selected as the final model because it achieved a highly balanced and robust profile, heavily outperforming Logistic Regression in PR-AUC (0.80 vs 0.69).

## 14. Threshold Selection
By default, models classify fraud if the predicted probability is >= 0.50. We performed a threshold analysis to prioritize catching fraud (Recall) while maintaining acceptable Precision.

| Threshold | Precision | Recall | F1-score |
| :---: | :---: | :---: | :---: |
| 0.10 | 0.639 | 0.821 | 0.718 |
| **0.30** | **0.826** | **0.800** | **0.812** |
| 0.50 | 0.879 | 0.768 | 0.820 |

**Selected Threshold:** `0.30`.
*Reasoning:* Lowering the threshold from 0.50 to 0.30 successfully boosts our Recall to 80% (catching more fraud) while maintaining a strong Precision of ~82% (minimal false alarms).

## 15. Streamlit Application
A beginner-friendly frontend was built using Streamlit. It loads the saved model and scaler without retraining them, applies the exact same preprocessing logic to user inputs, and flags transactions as "Potential Fraud" or "Likely Legitimate" using the selected 0.30 threshold.

## 16. Project Structure
```text
credit-card-fraud-detection/
├── data/
│   ├── raw/               <- Downloaded Kaggle dataset
│   └── processed/
├── models/                <- Saved models (Random Forest), Scaler, and config.json
├── notebooks/             <- Jupyter notebooks (EDA, Preprocessing, Imbalance Analysis)
├── reports/
│   └── figures/           <- Confusion Matrices, ROC Curves, PR Curves
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_features.py
│   └── threshold_analysis.py
├── tests/                 <- Pytest suite ensuring no data leakage and perfect reproducibility
├── app.py                 <- Streamlit Web Application
├── requirements.txt
└── README.md
```

## 17. Installation Instructions
1. Clone the repository.
2. Ensure you have Python 3.9+ installed.
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 18. How to Run Training
To execute the pipeline from start to finish, run the scripts sequentially from the root directory:
```bash
# 1. Download and explore the data
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# 2. Train the models
python src/train.py

# 3. Evaluate the models
python src/evaluate.py

# 4. Perform threshold analysis and select final model
python src/threshold_analysis.py
```

## 19. How to Run Streamlit
Once the models are trained and saved in the `models/` directory, start the UI:
```bash
streamlit run app.py
```

## 20. Limitations
- The features (`V1`-`V28`) are purely anonymized PCA components. In a real bank, features would include geographic locations, merchant categories, IP addresses, and customer transaction histories.
- The dataset is static. Real fraud patterns suffer from concept drift, requiring continuous retraining.

## 21. Future Improvements
- Implement XGBoost or LightGBM which often outperform Random Forests on tabular data.
- Build a real-time data streaming simulation (e.g., using Kafka) to mimic live transaction processing.
- Deploy the model via Docker and FastAPI to serve predictions over an API.
