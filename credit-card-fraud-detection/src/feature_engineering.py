from imblearn.over_sampling import SMOTE
from collections import Counter

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
