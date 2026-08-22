"""
train_model_v2.py

TWO-STAGE version of the ML stage, built to address the finding from
train_model.py's first attempt: predicting qt/fs from Vs+depth ALONE
performed poorly (qt R^2 = -0.016 on the independent test set) because
soil type is a strong confound -- the same Vs value maps to very
different qt depending on soil type (e.g. sand vs. clay), and Vs+depth
alone cannot see soil type.

STAGE A: Classify soil type from Vs+depth (XGBoost multiclass).
STAGE B: Predict qt/fs from Vs + depth + Stage A's PREDICTED soil-type
         PROBABILITIES (not a hard label -- see rationale below),
         with qt/fs log-transformed before training (both are strongly
         right-skewed).

WHY PROBABILITIES, NOT A HARD PREDICTED LABEL: Stage A's classification
accuracy is only ~40% (5-fold CV, 7 classes) -- meaningfully better
than chance (~14%) but unreliable. Passing a full probability vector
(e.g. "60% sand, 30% silt, 10% clay") to Stage B lets it use the
classifier's UNCERTAINTY as information, rather than forcing a
potentially-wrong hard guess. This mirrors the physics stage's
approach of carrying uncertainty forward rather than collapsing it
prematurely.

CRITICAL LEAKAGE-AVOIDANCE NOTE: Stage B's soil-type-probability
features for the TRAINING set are generated via OUT-OF-FOLD prediction
(sklearn's cross_val_predict) -- i.e. each training row's soil-type
probabilities come from a Stage A classifier that was NOT trained on
that row. Using in-sample predictions here would leak information
(the classifier would appear to "know" the answer for rows it was
literally trained on), artificially inflating Stage B's apparent
performance in a way that would not hold up on genuinely new data.
For the actual TEST set, no such leakage risk exists (the final Stage
A classifier, trained on the full training set, has never seen test
rows), so ordinary predict_proba is used there.

REALISTIC EXPECTATIONS (set explicitly, per project conversation):
even with this improvement, qt R^2 is expected to land well below the
0.76 upper bound achievable with TRUE soil type (verified separately
via a quick diagnostic) -- because Stage A's ~40% classification
accuracy means we can only partially recover the soil-type signal.
Landing meaningfully above the original -0.016 (e.g. into the
0.15-0.35 range) would already be a real, legitimate improvement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
)

from load_data import load_training_data, clean_training_data, load_testing_data


# ---------------------------------------------------------------------------
# Configuration. Trial counts reduced from train_model.py's 50 -- we now
# train THREE models (1 classifier + 2 regressors) instead of 2, so
# runtime adds up quickly. This is a documented trade-off, not a
# silent shortfall -- increase if you want a more thorough search and
# have the time to let it run.
# ---------------------------------------------------------------------------
N_OPTUNA_TRIALS_CLASSIFIER = 30
N_OPTUNA_TRIALS_REGRESSOR = 30
CV_N_FOLDS = 5
MAX_BOOSTING_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 20
VAL_SPLIT_FRACTION = 0.05
RANDOM_SEED = 42

FEATURE_COLUMNS_STAGE_A = ["vs_mps", "depth_m"]
TARGET_COLUMNS_STAGE_B = ["qt_mpa", "fs_mpa"]

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "models"


# ---------------------------------------------------------------------------
# STAGE A: soil-type classifier
# ---------------------------------------------------------------------------

def _classifier_optuna_objective(trial: optuna.Trial, dtrain: xgb.DMatrix, n_classes: int) -> float:
    """
    Optuna objective for the Stage A classifier: propose hyperparameters,
    evaluate via xgboost.cv on multiclass error rate (merror = 1 -
    accuracy), minimized.

    Parameters
    ----------
    trial : optuna.Trial
    dtrain : xgb.DMatrix
        Training data with integer-encoded soil_type labels.
    n_classes : int
        Number of distinct soil type classes (required by XGBoost's
        multiclass objective).

    Returns
    -------
    float
        Mean CV merror at the early-stopping point.
    """
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "merror",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "seed": RANDOM_SEED,
    }

    cv_results = xgb.cv(
        params=params, dtrain=dtrain, num_boost_round=MAX_BOOSTING_ROUNDS,
        nfold=CV_N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        metrics="merror", seed=RANDOM_SEED, verbose_eval=False,
    )
    return float(cv_results["test-merror-mean"].iloc[-1])


def train_soil_type_classifier(
    train_df: pd.DataFrame, n_trials: int = N_OPTUNA_TRIALS_CLASSIFIER
) -> Tuple[xgb.XGBClassifier, LabelEncoder]:
    """
    Train the Stage A soil-type classifier (Vs + depth -> soil type),
    with Optuna-tuned hyperparameters.

    Parameters
    ----------
    train_df : pd.DataFrame
        Cleaned training data (output of clean_training_data), must
        include 'soil_type'.
    n_trials : int, optional

    Returns
    -------
    (model, label_encoder) : (xgb.XGBClassifier, LabelEncoder)
        The fitted classifier, and the encoder used to map soil_type
        strings <-> integer class labels (needed later to interpret
        predict_proba's column order).
    """
    X = train_df[FEATURE_COLUMNS_STAGE_A]
    y_raw = train_df["soil_type"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    print(f"Soil type classes ({n_classes}): {list(label_encoder.classes_)}")

    dtrain_full = xgb.DMatrix(X, label=y)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\nTuning Stage A classifier ({n_trials} trials, {CV_N_FOLDS}-fold CV)...")
    study.optimize(
        lambda trial: _classifier_optuna_objective(trial, dtrain_full, n_classes),
        n_trials=n_trials, show_progress_bar=True,
    )
    print(f"Best CV merror: {study.best_value:.4f}  (accuracy: {1 - study.best_value:.4f})")
    print(f"Best hyperparameters: {study.best_params}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT_FRACTION, random_state=RANDOM_SEED, stratify=y,
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=n_classes, eval_metric="merror",
        n_estimators=MAX_BOOSTING_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_SEED, **study.best_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Stage A final model -- held-out validation accuracy: {val_acc:.4f}")

    return model, label_encoder


def get_out_of_fold_soil_probs(
    train_df: pd.DataFrame, label_encoder: LabelEncoder, best_params: dict
) -> np.ndarray:
    """
    Generate OUT-OF-FOLD soil-type probability predictions for every
    training row, to avoid leakage when these probabilities are used
    as Stage B features (see module docstring, "CRITICAL
    LEAKAGE-AVOIDANCE NOTE").

    Parameters
    ----------
    train_df : pd.DataFrame
    label_encoder : LabelEncoder
        Fitted on the same data, from train_soil_type_classifier.
    best_params : dict
        Hyperparameters to use for each fold's classifier (reuses
        Stage A's tuned hyperparameters rather than re-tuning per
        fold, which would be prohibitively slow).

    Returns
    -------
    np.ndarray, shape (n_train_rows, n_classes)
        Out-of-fold predicted class probabilities, column order
        matching label_encoder.classes_.
    """
    X = train_df[FEATURE_COLUMNS_STAGE_A]
    y = label_encoder.transform(train_df["soil_type"])
    n_classes = len(label_encoder.classes_)

    oof_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=n_classes,
        n_estimators=300,  # fixed, modest round count for OOF folds --
                           # no early stopping needed here since we're
                           # not evaluating this model directly, just
                           # generating features
        random_state=RANDOM_SEED, **best_params,
    )

    print("\nGenerating out-of-fold soil-type probabilities for training set "
          f"({CV_N_FOLDS}-fold, leakage-safe)...")
    oof_probs = cross_val_predict(
        oof_model, X, y, cv=StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        method="predict_proba",
    )
    return oof_probs


# ---------------------------------------------------------------------------
# STAGE B: qt / fs regressors (log-transformed target)
# ---------------------------------------------------------------------------

def _regressor_optuna_objective(trial: optuna.Trial, dtrain: xgb.DMatrix) -> float:
    """Same structure as train_model.py's objective -- see that file's
    docstring for full rationale. Operates on log1p-transformed target."""
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "seed": RANDOM_SEED,
    }
    cv_results = xgb.cv(
        params=params, dtrain=dtrain, num_boost_round=MAX_BOOSTING_ROUNDS,
        nfold=CV_N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        metrics="rmse", seed=RANDOM_SEED, verbose_eval=False,
    )
    return float(cv_results["test-rmse-mean"].iloc[-1])


def train_stage_b_model(
    X_train_full: pd.DataFrame, y_train_full_raw: pd.Series, target_name: str,
    n_trials: int = N_OPTUNA_TRIALS_REGRESSOR,
) -> xgb.XGBRegressor:
    """
    Train one Stage B regressor (qt or fs) on log1p-transformed target.

    Parameters
    ----------
    X_train_full : pd.DataFrame
        Features: vs_mps, depth_m, + soil-type probability columns.
    y_train_full_raw : pd.Series
        RAW (not yet log-transformed) target values.
    target_name : str
        For print statements.
    n_trials : int, optional

    Returns
    -------
    xgb.XGBRegressor
        Trained on log1p(target) -- remember to np.expm1() its
        predictions before interpreting them in original MPa units.
    """
    y_log = np.log1p(y_train_full_raw)

    dtrain_full = xgb.DMatrix(X_train_full, label=y_log)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\nTuning Stage B regressor for {target_name} "
          f"({n_trials} trials, {CV_N_FOLDS}-fold CV, log1p-transformed target)...")
    study.optimize(
        lambda trial: _regressor_optuna_objective(trial, dtrain_full),
        n_trials=n_trials, show_progress_bar=True,
    )
    print(f"Best CV RMSE (log1p space): {study.best_value:.4f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_log, test_size=VAL_SPLIT_FRACTION, random_state=RANDOM_SEED,
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse",
        n_estimators=MAX_BOOSTING_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_SEED, **study.best_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Stage B model trained for {target_name}. Best iteration: {model.best_iteration}")

    return model


def evaluate_stage_b_model(
    model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test_raw: pd.Series, target_name: str
) -> dict:
    """
    Evaluate a Stage B model: predict in log1p space, convert back to
    original units via expm1, then compute metrics in original MPa
    units so results are directly comparable to train_model.py's
    single-stage baseline.
    """
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # inverse of log1p

    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2 = r2_score(y_test_raw, y_pred)

    print(f"\n{target_name} -- test set performance (two-stage model):")
    print(f"  MAE:  {mae:.3f} MPa")
    print(f"  RMSE: {rmse:.3f} MPa")
    print(f"  R^2:  {r2:.3f}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def run_two_stage_pipeline():
    """
    Run the complete two-stage ML pipeline: Stage A soil-type
    classifier, out-of-fold probability generation, Stage B qt/fs
    regressors (log-transformed), evaluated against the independent
    test set.
    """
    print("=" * 70)
    print("LOADING AND CLEANING DATA")
    print("=" * 70)
    train_df = clean_training_data(load_training_data())
    test_df = load_testing_data()

    print()
    print("=" * 70)
    print("STAGE A: SOIL TYPE CLASSIFIER (Vs + depth -> soil type)")
    print("=" * 70)
    classifier, label_encoder = train_soil_type_classifier(train_df)

    # Recover the tuned hyperparameters actually used, for the OOF folds
    # (re-extract from the fitted model's get_params to avoid re-running
    # the Optuna search twice).
    classifier_params = {
        k: v for k, v in classifier.get_params().items()
        if k in ["learning_rate", "max_depth", "min_child_weight", "subsample",
                  "colsample_bytree", "reg_lambda"]
    }

    oof_probs = get_out_of_fold_soil_probs(train_df, label_encoder, classifier_params)
    soil_prob_cols = [f"soil_prob_{cls}" for cls in label_encoder.classes_]
    oof_probs_df = pd.DataFrame(oof_probs, columns=soil_prob_cols, index=train_df.index)

    # For the TEST set, use the FINAL classifier's ordinary predict_proba
    # -- no leakage risk here, since the classifier never saw test rows.
    test_probs = classifier.predict_proba(test_df[FEATURE_COLUMNS_STAGE_A])
    test_probs_df = pd.DataFrame(test_probs, columns=soil_prob_cols, index=test_df.index)

    train_augmented = pd.concat([train_df, oof_probs_df], axis=1)
    test_augmented = pd.concat([test_df, test_probs_df], axis=1)

    stage_b_features = FEATURE_COLUMNS_STAGE_A + soil_prob_cols

    print()
    print("=" * 70)
    print("STAGE B: qt / fs REGRESSORS (Vs + depth + predicted soil-type probs)")
    print("=" * 70)

    models_b = {}
    results_b = {}
    for target in TARGET_COLUMNS_STAGE_B:
        print(f"\n--- Target: {target} ---")
        model = train_stage_b_model(
            train_augmented[stage_b_features], train_augmented[target], target
        )
        results = evaluate_stage_b_model(
            model, test_augmented[stage_b_features], test_augmented[target], target
        )
        models_b[target] = model
        results_b[target] = results

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / f"vs_to_{target}_two_stage_xgboost_model.joblib")

    joblib.dump(classifier, MODELS_DIR / "vs_to_soiltype_xgboost_classifier.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "soiltype_label_encoder.joblib")

    print()
    print("=" * 70)
    print("SUMMARY: two-stage model vs. original single-stage baseline")
    print("=" * 70)
    print("                original (Vs+depth only)      two-stage (+ soil-type probs, log-target)")
    print(f"qt_mpa R^2:     -0.016                         {results_b['qt_mpa']['r2']:.3f}")
    print(f"fs_mpa R^2:      0.167                         {results_b['fs_mpa']['r2']:.3f}")
    print(
        "\nNote: even with this improvement, results are expected to remain "
        "well below the ~0.76 upper bound achievable with TRUE soil type "
        "(verified separately), since Stage A's classification accuracy "
        "is only ~40% -- see module docstring."
    )

    return classifier, models_b, results_b


if __name__ == "__main__":
    run_two_stage_pipeline()
    