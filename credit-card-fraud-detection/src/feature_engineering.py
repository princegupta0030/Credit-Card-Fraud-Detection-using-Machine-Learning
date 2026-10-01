from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

def engineer_features(df):
    """
    Apply mathematically justified feature engineering to the raw dataset.

    Transformations:
    1. Log Transformation of Amount (Log_Amount):
       - Why: The 'Amount' feature is heavily right-skewed (most transactions are small,
         a few are very large). Algorithms like Logistic Regression assume normality or
         perform poorly with extreme outliers. np.log1p (log(1+x)) squashes large values,
         making the distribution more bell-shaped and manageable for linear models.

    2. Time-of-Day Features (Hour, Hour_Sin, Hour_Cos):
       - Why: The 'Time' feature represents seconds elapsed since the first transaction.
         As a continuous, ever-increasing integer, it's not predictive. However, fraud
         often follows diurnal patterns (e.g., more fraud at night). By extracting the
         hour of the day (Time % 86400 / 3600), we get a cyclical feature. We then apply
         Sine and Cosine transformations to represent the cyclical nature of time (i.e.,
         hour 23 is close to hour 0, which a linear model wouldn't understand otherwise).

    Args:
        df: Pandas DataFrame containing 'Time' and 'Amount'.

    Returns:
        df: DataFrame with new engineered features.
    """
    df_engineered = df.copy()

    if 'Amount' in df_engineered.columns:
        # np.log1p safely handles 0.0 amounts (log(0) is undefined)
        df_engineered['Log_Amount'] = np.log1p(df_engineered['Amount'])

    if 'Time' in df_engineered.columns:
        # Extract hour of the day (0-23)
        # 86400 seconds in a day, 3600 seconds in an hour
        hour = (df_engineered['Time'] % 86400) / 3600
        df_engineered['Hour'] = hour

        # Cyclical encoding
        df_engineered['Hour_Sin'] = np.sin(hour * (2. * np.pi / 24))
        df_engineered['Hour_Cos'] = np.cos(hour * (2. * np.pi / 24))

    return df_engineered

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply Synthetic Minority Over-sampling Technique (SMOTE) to the training data.

    IMPORTANT:
    SMOTE must ONLY be applied to the training data to prevent data leakage.
    If applied before train-test split, synthetic samples derived from test data
    will bleed into the training set, artificially inflating model performance
    and leading to poor generalization in the real world.

    Args:
        X_train: Training features.
        y_train: Training target.
        random_state: Seed for reproducibility.

    Returns:
        X_train_smote: Resampled training features.
        y_train_smote: Resampled training target.
    """
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, y_train_smote

def get_class_weights(y_train):
    """
    Calculate class weights for algorithmic imbalance handling (e.g., Logistic Regression).
    This computes a dictionary suitable for passing to sklearn classifiers' `class_weight` parameter.

    Weight calculation is inversely proportional to class frequencies:
    n_samples / (n_classes * np.bincount(y))
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))

    return weight_dict
