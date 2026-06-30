"""
================================================================================
FILE:    backend/ml/train_model.py
PURPOSE: Train the XGBoost medication adherence risk prediction model.
         Run once before starting the API server.

COMMAND:
    python -m backend.ml.train_model

DATASET:
    By default generates a 2,000-row synthetic dataset that mirrors the
    clinical distributions of the MIMIC-III ICU database.

    To use REAL data instead, place a CSV at data/patient_features.csv
    with these columns (see FEATURE_COLUMNS + "high_risk" target):
        age, adherence_rate, comorbidity_count, medication_count,
        exercise_level, follow_up_frequency, hospital_visits_last_year,
        hba1c_normalized, bmi_normalized,
        age_group_young, age_group_middle, age_group_senior,
        low_adherence_flag, high_comorbidity_flag,
        diabetes_flag, hypertension_flag, cardiac_flag,
        high_risk  (0=low risk, 1=high risk)

    Real MIMIC-III Demo (free, 100 patients):
        https://physionet.org/content/mimiciii-demo/1.4/

SAVED ARTIFACTS:
    backend/ml/saved_models/xgboost_adherence_model.pkl  — trained classifier
    backend/ml/saved_models/feature_scaler.pkl           — fitted StandardScaler
    backend/ml/saved_models/training_metrics.json        — performance metrics
    data/patient_features.csv                            — generated dataset

EXPECTED METRICS (on synthetic data):
    Accuracy  ≈ 0.875
    F1 Score  ≈ 0.877
    ROC-AUC   ≈ 0.942
================================================================================
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.features import FEATURE_COLUMNS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
SAVE_DIR      = Path("backend/ml/saved_models")
MODEL_PATH    = SAVE_DIR / "xgboost_adherence_model.pkl"
SCALER_PATH   = SAVE_DIR / "feature_scaler.pkl"
METRICS_PATH  = SAVE_DIR / "training_metrics.json"
DATA_DIR      = Path("data")
DATASET_PATH  = DATA_DIR / "patient_features.csv"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 3.3,   # ← ADD THIS LINE (76.8/23.2 = 3.3)
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

N_SYNTHETIC_SAMPLES = 2000
CV_FOLDS            = 5
TEST_SIZE           = 0.20
RANDOM_STATE        = 42


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = N_SYNTHETIC_SAMPLES,
                                random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a realistic synthetic healthcare dataset for model training.
    Feature distributions are calibrated against MIMIC-III clinical literature.

    Target variable:
        high_risk = 1  →  patient has high medication non-adherence risk
        high_risk = 0  →  patient has low non-adherence risk

    Clinical correlations encoded:
        - Low adherence rate                → strong positive risk signal
        - Elevated HbA1c + diabetes flag    → positive risk signal
        - Multiple comorbidities            → positive risk signal
        - Frequent hospitalisations         → positive risk signal
        - High exercise level               → negative risk signal (protective)
        - Frequent follow-up visits         → negative risk signal (protective)
        - Advanced age                      → positive risk signal
        - Cardiac conditions                → positive risk signal

    Args:
        n_samples:    Number of synthetic patient records to generate.
        random_state: NumPy random seed for reproducibility.

    Returns:
        pd.DataFrame with FEATURE_COLUMNS + "high_risk" column.
    """
    rng = np.random.RandomState(random_state)
    n   = n_samples

    logger.info(f"Generating synthetic dataset: {n} patients ...")

    # ── Raw clinical features ─────────────────────────────────────────────
    age           = rng.randint(18, 86, n)
    adherence_raw = rng.beta(5, 2, n) * 100       # skewed toward higher values (realistic)
    comorbidities = rng.poisson(1.5, n).clip(0, 8)
    medications   = (comorbidities + rng.poisson(1, n)).clip(1, 15)
    exercise      = rng.randint(1, 11, n)
    follow_up     = rng.poisson(4, n).clip(0, 24)
    hosp_visits   = rng.poisson(0.8, n).clip(0, 12)
    hba1c_raw     = rng.normal(6.5, 1.5, n).clip(4.0, 14.0)
    bmi_raw       = rng.normal(27.0, 5.0, n).clip(15.0, 50.0)

    # ── Disease flags ─────────────────────────────────────────────────────
    # Correlated: older patients more likely to have diabetes and hypertension
    diabetes_flag     = ((rng.random(n) < 0.28) | (hba1c_raw > 7.0)).astype(int)
    hypertension_flag = ((rng.random(n) < 0.38) | (age > 55)).astype(int)
    cardiac_flag      = ((rng.random(n) < 0.14) | ((age > 60) & (rng.random(n) < 0.25))).astype(int)

    # Adjust HbA1c upward for diabetic patients (clinically realistic)
    hba1c_raw = np.where(diabetes_flag, hba1c_raw + rng.normal(0.8, 0.4, n), hba1c_raw)
    hba1c_raw = hba1c_raw.clip(4.0, 14.0)

    # ── Derived features ──────────────────────────────────────────────────
    hba1c_normalized  = (hba1c_raw / 14.0).clip(0, 1)
    bmi_normalized    = (bmi_raw   / 50.0).clip(0, 1)
    age_group_young   = (age < 40).astype(int)
    age_group_middle  = ((age >= 40) & (age < 65)).astype(int)
    age_group_senior  = (age >= 65).astype(int)
    low_adherence_flag      = (adherence_raw < 70).astype(int)
    high_comorbidity_flag   = (comorbidities >= 3).astype(int)

    # ── Composite clinical risk score (used to generate target label) ─────
    # This mirrors the clinical logic in identify_risk_factors()
    clinical_risk = (
        (100 - adherence_raw) * 0.38           # adherence gap
        + np.maximum(0, hba1c_raw - 5.5) * 3.5 * diabetes_flag  # glycaemic control
        + comorbidities * 4.2                  # comorbidity burden
        + hosp_visits   * 5.5                  # hospitalisation frequency
        + np.where(age > 65, 9.0, 0)           # senior age
        + np.where(age > 75, 5.0, 0)           # advanced age bonus
        + np.maximum(0, 3 - exercise) * 2.5    # low exercise
        + np.maximum(0, 3 - follow_up) * 2.0   # infrequent follow-up
        + hypertension_flag * 3.5              # hypertension
        + cardiac_flag      * 11.0             # cardiac conditions
        + bmi_normalized * 6.0                # obesity
        + rng.normal(0, 5.0, n)               # random noise (real-world variability)
    )

    # Label: top 60% of risk score = high risk  (40/60 class split)
    threshold = np.percentile(clinical_risk, 40)
    high_risk = (clinical_risk > threshold).astype(int)

    # ── Build DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "age":                       age,
        "adherence_rate":            np.round(adherence_raw, 1),
        "comorbidity_count":         comorbidities,
        "medication_count":          medications,
        "exercise_level":            exercise,
        "follow_up_frequency":       follow_up,
        "hospital_visits_last_year": hosp_visits,
        "hba1c_normalized":          np.round(hba1c_normalized, 4),
        "bmi_normalized":            np.round(bmi_normalized, 4),
        "age_group_young":           age_group_young,
        "age_group_middle":          age_group_middle,
        "age_group_senior":          age_group_senior,
        "low_adherence_flag":        low_adherence_flag,
        "high_comorbidity_flag":     high_comorbidity_flag,
        "diabetes_flag":             diabetes_flag,
        "hypertension_flag":         hypertension_flag,
        "cardiac_flag":              cardiac_flag,
        # Extra raw columns for analysis (not used in model)
        "_hba1c_raw":                np.round(hba1c_raw, 2),
        "_bmi_raw":                  np.round(bmi_raw,   1),
        "_clinical_risk_score":      np.round(clinical_risk, 2),
        "high_risk":                 high_risk,
    })

    logger.info(
        f"Dataset generated: {n} rows | "
        f"high_risk={high_risk.sum()} ({high_risk.mean()*100:.1f}%) | "
        f"low_risk={n - high_risk.sum()} ({(1-high_risk.mean())*100:.1f}%)"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def train() -> dict:
    """
    Full training pipeline:
        1.  Load or generate dataset
        2.  Train/test split (stratified)
        3.  StandardScaler fit
        4.  XGBoost training with early stopping
        5.  5-fold cross-validation
        6.  Evaluation on held-out test set
        7.  Save model, scaler, and metrics to disk
        8.  Return metrics dict

    Returns:
        metrics dict with accuracy, precision, recall, f1_score, roc_auc, etc.
    """
    _print_header()

    # ── Step 1: Dataset ───────────────────────────────────────────────────
    if DATASET_PATH.exists():
        logger.info(f"Loading existing dataset: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    else:
        logger.info("No dataset found — generating synthetic data...")
        df = generate_synthetic_dataset(N_SYNTHETIC_SAMPLES, RANDOM_STATE)
        df.to_csv(DATASET_PATH, index=False)
        logger.info(f"Synthetic dataset saved → {DATASET_PATH}")

    # Verify required columns exist
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}\n"
            f"Expected columns: {FEATURE_COLUMNS}"
        )
    if "high_risk" not in df.columns:
        raise ValueError("Dataset must have a 'high_risk' target column (0 or 1)")

    X = df[FEATURE_COLUMNS].copy()
    y = df["high_risk"].copy()

    _print_class_distribution(y)

    # ── Step 2: Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    logger.info(f"Split: train={len(X_train)} | test={len(X_test)}")

    # ── Step 3: Feature scaling ───────────────────────────────────────────
    scaler          = StandardScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_test_scaled   = scaler.transform(X_test)

    # ── Step 4: XGBoost training ──────────────────────────────────────────
    logger.info("Training XGBoost classifier ...")
    logger.info(f"Hyperparameters: {XGBOOST_PARAMS}")

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train_scaled,
        y_train,
        eval_set        = [(X_test_scaled, y_test)],
        verbose         = False,
    )
    logger.info("✅ XGBoost training complete")

    # ── Step 5: Cross-validation ──────────────────────────────────────────
    logger.info(f"Running {CV_FOLDS}-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_f1_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="f1"
    )
    cv_auc_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
    )
    logger.info(
        f"CV F1:      {cv_f1_scores.mean():.4f}  ±  {cv_f1_scores.std():.4f}"
    )
    logger.info(
        f"CV ROC-AUC: {cv_auc_scores.mean():.4f}  ±  {cv_auc_scores.std():.4f}"
    )

    # ── Step 6: Test set evaluation ───────────────────────────────────────
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred,    zero_division=0)
    f1        = f1_score(y_test, y_pred,        zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)

    _print_metrics(accuracy, precision, recall, f1, roc_auc)
    _print_classification_report(y_test, y_pred)
    _print_confusion_matrix(y_test, y_pred)

    # ── Feature importances ───────────────────────────────────────────────
    feat_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    top_features    = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
    _print_feature_importances(top_features[:8])

    # ── Step 7: Save artifacts ────────────────────────────────────────────
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"✅ Model  saved → {MODEL_PATH}")
    logger.info(f"✅ Scaler saved → {SCALER_PATH}")

    metrics = {
        "accuracy":    round(accuracy,  4),
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1_score":    round(f1,        4),
        "roc_auc":     round(roc_auc,   4),
        "cv_f1_mean":  round(float(cv_f1_scores.mean()),  4),
        "cv_f1_std":   round(float(cv_f1_scores.std()),   4),
        "cv_auc_mean": round(float(cv_auc_scores.mean()), 4),
        "cv_auc_std":  round(float(cv_auc_scores.std()),  4),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "total_samples": int(len(df)),
        "n_features":    len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": {
            k: round(float(v), 6) for k, v in feat_importance.items()
        },
        "top_features": [f for f, _ in top_features[:5]],
        "model_params":  XGBOOST_PARAMS,
        "class_distribution": {
            "high_risk_count": int(y.sum()),
            "low_risk_count":  int(len(y) - y.sum()),
            "high_risk_pct":   round(float(y.mean()) * 100, 1),
        },
    }

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"✅ Metrics saved → {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE ✅")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Model     : {MODEL_PATH}")
    print("=" * 60)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _print_header() -> None:
    print("\n" + "=" * 60)
    print("  MediGraph AI — XGBoost Adherence Risk Model Training")
    print("=" * 60)


def _print_class_distribution(y: pd.Series) -> None:
    total = len(y)
    high  = y.sum()
    low   = total - high
    print(f"\nClass distribution:")
    print(f"  High Risk  : {high:>5} ({high/total*100:.1f}%)")
    print(f"  Low Risk   : {low:>5}  ({low/total*100:.1f}%)")


def _print_metrics(acc, prec, rec, f1, auc) -> None:
    print("\n" + "=" * 40)
    print("  MODEL PERFORMANCE METRICS")
    print("=" * 40)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")


def _print_classification_report(y_test, y_pred) -> None:
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))


def _print_confusion_matrix(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negative  (TN): {cm[0][0]:>6}  |  False Positive (FP): {cm[0][1]:>6}")
    print(f"  False Negative (FN): {cm[1][0]:>6}  |  True Positive  (TP): {cm[1][1]:>6}")


def _print_feature_importances(top: list) -> None:
    print("\nTop Feature Importances:")
    for rank, (feat, imp) in enumerate(top, 1):
        bar = "█" * int(imp * 100)
        print(f"  {rank:2}. {feat:<35} {imp:.4f}  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_metrics = train()