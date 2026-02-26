"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Split into train and test (80-20).
            Use random_state=42.
    STEP 3: Standardize features using StandardScaler.
            IMPORTANT:
            - Fit scaler only on X_train
            - Transform both X_train and X_test
    STEP 4: Train LinearRegression model.
    STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2
    STEP 6: Identify indices of top 3 features
            with largest absolute coefficients.

    RETURN:
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices (list length 3)
    """

    # Load diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Split into train and test (80-20), random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)        

    # Train LinearRegression model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Compute train and test metrics
    y_train_pred = lr.predict(X_train_scaled)
    y_test_pred = lr.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Identifying top 3 features with largest absolute coefficients
    coef_abs = np.abs(lr.coef_)
    top_3_feature_indices = list(coef_abs.argsort()[-3:][::-1])  # indices of top 3
    
    # Overfitting: If Train R² is significantly higher than Test R², the model is overfitting. 
    # In this dataset, the gap is usually small, suggesting a stable fit.
    # Scaling: Important because Linear Regression is sensitive to the scale of input features; 
    # It ensures all features contribute equally to the distance-based calculations.

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
            but use pipeline logic manually).
    STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.

    STEP 4: Compute:
            - mean_r2
            - std_r2

    RETURN:
        mean_r2,
        std_r2
    """

    # Load diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Standardize entire dataset
    # For CV, we scale features across the whole dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Performing 5-fold cross-validation using LinearRegression
    lr = LinearRegression()
    # Scoring='r2' computes R² for each fold
    r2_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')

    # Computing mean and standard deviation of R²
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    # Standard deviation: indicates how much R² varies across folds (model stability)
    # CV reduces variance risk: by averaging results across multiple folds,
    # it gives a more reliable estimate of model performance than a single train-test split.

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
            Use random_state=42.
    STEP 3: Standardize features.
    STEP 4: Train LogisticRegression(max_iter=5000).
    STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)

    In comments:
        Explain what a False Negative represents medically.

    RETURN:
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    """

    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # Spliting into train-test (80-20), random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LogisticRegression(max_iter=5000)
    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train_scaled, y_train)

    # Compute metrics
    y_train_pred = lr.predict(X_train_scaled)
    y_test_pred = lr.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Confusion matrix 
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    # Confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]

    # False Negative (FN) medically: A patient who actually has cancer
    # is predicted as healthy by the model. This is dangerous because
    # the patient may not receive necessary treatment.

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
    STEP 3: Standardize features.
    STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            - Train LogisticRegression(max_iter=5000, C=value)
            - Compute train accuracy
            - Compute test accuracy

    STEP 5: Store results in dictionary:
            {
                C_value: (train_accuracy, test_accuracy)
            }

    In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?

    RETURN:
        results_dictionary
    """

    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # Split into train-test (80-20), random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LogisticRegression for different C values
    C_values = [0.01, 0.1, 1, 10, 100]
    results = {}

    for C in C_values:
        lr = LogisticRegression(max_iter=5000, C=C)
        lr.fit(X_train_scaled, y_train)

        train_acc = lr.score(X_train_scaled, y_train)
        test_acc = lr.score(X_test_scaled, y_test)

        results[C] = (train_acc, test_acc)

    # Very small C (e.g., 0.01): Strong regularization → coefficients shrink → underfitting
    # Very large C (e.g., 100): Weak regularization → model fits training data closely → may overfit
    # Overfitting occurs with very large C because the model captures noise in training data. 

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Standardize entire dataset.
    STEP 3: Perform 5-fold cross-validation
            using LogisticRegression(C=1, max_iter=5000).
            Use scoring='accuracy'.

    STEP 4: Compute:
            - mean_accuracy
            - std_accuracy

    In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.

    RETURN:
        mean_accuracy,
        std_accuracy
    """

    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform 5-fold cross-validation using LogisticRegression(C=1, max_iter=5000)
    lr = LogisticRegression(C=1, max_iter=5000)
    accuracy_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy')

    # Compute mean and standard deviation of accuracy
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    # Cross-validation is critical in medical diagnosis because it ensures
    # the model generalizes well to unseen patients. It reduces the risk
    # of overestimating performance due to a single train-test split, which
    # is important when decisions affect patient health.

    return mean_accuracy, std_accuracy
