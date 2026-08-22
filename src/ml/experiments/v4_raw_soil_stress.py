"""
train_model_v3.py

Best-informed combination of everything learned so far:
    - Soil-type probability features (Stage A classifier, OOF-safe) --
      confirmed via ablation to genuinely help (train_model_ablation.py).
    - Effective overburden stress (sigma_v0_kpa) as a new engineered
      feature, physically motivated by the Robertson & Cabal-based
      formula in Peuchen et al. (2024) -- see project conversation.
    - RAW target (no log1p transform) -- the ablation showed log1p
      introduces retransformation bias (Jensen's inequality: converting
      log-space predictions back via expm1 systematically underpredicts
      for right-skewed targets like qt/fs), which was actively hurting
      performance despite looking fine in log-space CV RMSE. Reverting
      to raw-target training, matching train_model.py's original
      approach.

EFFECTIVE STRESS FEATURE: sigma_v0_kpa = submerged_unit_weight_kNm3 *
depth_m. Uses a constant assumed submerged unit weight (9 kN/m^3,
representative of soft-to-medium marine sediment) since site-specific
unit weight isn't available at inference time -- a documented
simplification, same category as the Vp/Vs=2.0 default used in the
physics stage's layered_model.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from load_data import load_training_data, clean_training_data, load_testing_data
from train_model import (
    train_soil_type_classifier, get_out_of_fold_soil_probs,
    N_OPTUNA_TRIALS_REGRESSOR, CV_N_FOLDS, MAX_BOOSTING_ROUNDS,
    EARLY_STOPPING_ROUNDS, VAL_SPLIT_FRACTION, RANDOM_SEED,
    FEATURE_COLUMNS_STAGE_A, TARGET_COLUMNS_STAGE_B,
)

SUBMERGED_UNIT_WEIGHT_KNM3 = 9.0  # see module docstring
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "models"


def add_effective_stress_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sigma_v0_kpa (effective overburden stress, kPa) computed from
    depth_m using a constant assumed submerged unit weight. See module
    docstring for the physical rationale and the documented
    simplification.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'depth_m'.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'sigma_v0_kpa' added.
    """
    df = df.copy()
    df["sigma_v0_kpa"] = SUBMERGED_UNIT_WEIGHT_KNM3 * df["depth_m"]
    return df


def _raw_target_optuna_objective(trial: optuna.Trial, dtrain: xgb.DMatrix) -> float:
    """Same structure as train_model.py's objective -- raw (untransformed) target."""
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


def train_and_evaluate_raw_target(
    X_train_full: pd.DataFrame, y_train_full: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, target_name: str,
) -> dict:
    """
    Tune, train, and evaluate a Stage B regressor on the RAW
    (untransformed) target -- see module docstring for why we reverted
    from log1p.
    """
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\nTuning {target_name} ({N_OPTUNA_TRIALS_REGRESSOR} trials, "
          f"{CV_N_FOLDS}-fold CV, RAW target)...")
    study.optimize(
        lambda trial: _raw_target_optuna_objective(trial, dtrain_full),
        n_trials=N_OPTUNA_TRIALS_REGRESSOR, show_progress_bar=True,
    )
    print(f"Best CV RMSE: {study.best_value:.3f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SPLIT_FRACTION, random_state=RANDOM_SEED,
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse",
        n_estimators=MAX_BOOSTING_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_SEED, **study.best_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Trained. Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{target_name} -- test set performance (v3: soil-probs + stress, raw target):")
    print(f"  MAE:  {mae:.3f} MPa")
    print(f"  RMSE: {rmse:.3f} MPa")
    print(f"  R^2:  {r2:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"vs_to_{target_name}_v3_xgboost_model.joblib")

    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def run_v3_pipeline():
    print("=" * 70)
    print("LOADING AND CLEANING DATA")
    print("=" * 70)
    train_df = clean_training_data(load_training_data())
    test_df = load_testing_data()

    print()
    print("=" * 70)
    print("STAGE A: SOIL TYPE CLASSIFIER (reused from train_model_v2.py)")
    print("=" * 70)
    classifier, label_encoder = train_soil_type_classifier(train_df)
    classifier_params = {
        k: v for k, v in classifier.get_params().items()
        if k in ["learning_rate", "max_depth", "min_child_weight", "subsample",
                  "colsample_bytree", "reg_lambda"]
    }
    oof_probs = get_out_of_fold_soil_probs(train_df, label_encoder, classifier_params)
    soil_prob_cols = [f"soil_prob_{cls}" for cls in label_encoder.classes_]
    oof_probs_df = pd.DataFrame(oof_probs, columns=soil_prob_cols, index=train_df.index)
    test_probs = classifier.predict_proba(test_df[FEATURE_COLUMNS_STAGE_A])
    test_probs_df = pd.DataFrame(test_probs, columns=soil_prob_cols, index=test_df.index)

    print()
    print("=" * 70)
    print("ADDING EFFECTIVE OVERBURDEN STRESS FEATURE")
    print("=" * 70)
    train_augmented = add_effective_stress_feature(
        pd.concat([train_df, oof_probs_df], axis=1)
    )
    test_augmented = add_effective_stress_feature(
        pd.concat([test_df, test_probs_df], axis=1)
    )
    stage_b_features = FEATURE_COLUMNS_STAGE_A + ["sigma_v0_kpa"] + soil_prob_cols
    print(f"Stage B features: {stage_b_features}")

    print()
    print("=" * 70)
    print("STAGE B: qt / fs REGRESSORS (Vs + depth + stress + soil-probs, RAW target)")
    print("=" * 70)

    results = {}
    for target in TARGET_COLUMNS_STAGE_B:
        print(f"\n--- Target: {target} ---")
        results[target] = train_and_evaluate_raw_target(
            train_augmented[stage_b_features], train_augmented[target],
            test_augmented[stage_b_features], test_augmented[target],
            target,
        )

    print()
    print("=" * 70)
    print("FOUR-WAY COMPARISON")
    print("=" * 70)
    print("               v1(orig)   ablation(log)   v2(log+soil)   v3(raw+soil+stress)")
    print(f"qt_mpa R^2:    -0.016     -0.284           -0.102          {results['qt_mpa']['r2']:.3f}")
    print(f"fs_mpa R^2:     0.167      0.111            0.205          {results['fs_mpa']['r2']:.3f}")

    return results


if __name__ == "__main__":
    run_v3_pipeline()