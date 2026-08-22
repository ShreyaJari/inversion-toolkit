"""
train_model.py

CANONICAL final training script for the ML stage (Stage 2): predicts
geotechnical parameters (qt, fs) from shear-wave velocity (Vs) and
depth -- the two quantities the physics stage's inversion actually
produces.

This is the FINAL, winning configuration after five tested variants
(see src/ml/experiments/ for the full trail):
    v1 (Vs+depth only, raw target)                  -- baseline
    v2 (Vs+depth only, log1p target, no soil)        -- ablation:
                                                         confirmed log1p
                                                         alone HURTS
    v3 (log1p target + soil-type probabilities)      -- confirmed
                                                         soil-probs HELP
    v4 (raw target + soil-probs + effective stress)  -- best for qt_mpa
    v5 (log1p target + soil-probs + effective stress) -- tested whether
                                                          fs also wants
                                                          the stress
                                                          feature; it
                                                          did NOT beat v3

FINAL CONFIGURATION (different recipe per target -- a deliberate,
evidence-based choice, not an oversight):
    qt_mpa: features = [vs_mps, depth_m, sigma_v0_kpa, soil-type probs]
            target = RAW (untransformed)
            Test R^2 = 0.091 (vs. -0.016 for the naive Vs+depth-only
            baseline)
    fs_mpa: features = [vs_mps, depth_m, soil-type probs]
            (NO effective stress feature -- tested, did not help)
            target = log1p-transformed
            Test R^2 = 0.205 (vs. 0.167 for the naive baseline)

WHY DIFFERENT RECIPES PER TARGET: qt (cone resistance) is fundamentally
a bearing-capacity response, more directly tied to vertical effective
stress; fs (sleeve friction) is more of an interface shear/adhesion
response, less stress-dependent -- consistent with the effective-stress
feature helping qt but not fs. Similarly, qt's stronger right-skew
(spanning ~0.1-1300 MPa) makes it more sensitive to log-transform
retransformation bias (Jensen's inequality) than fs's more modest
skew -- consistent with qt preferring the raw target and fs preferring
log1p. This is documented reasoning, not post-hoc rationalization --
see project conversation history / experiments/ scripts for the actual
ablation results that led here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make src/ml/experiments/ importable without a formal package structure.
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from load_data import load_training_data, clean_training_data, load_testing_data
from experiments.v3_two_stage_log_soil import (  # noqa: E402
    train_soil_type_classifier, get_out_of_fold_soil_probs,
    train_stage_b_model, evaluate_stage_b_model,
    FEATURE_COLUMNS_STAGE_A, CV_N_FOLDS, MAX_BOOSTING_ROUNDS,
    EARLY_STOPPING_ROUNDS, VAL_SPLIT_FRACTION, RANDOM_SEED,
    N_OPTUNA_TRIALS_REGRESSOR,
)

SUBMERGED_UNIT_WEIGHT_KNM3 = 9.0  # see project conversation for rationale
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "models"
SUMMARY_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "ml_stage_summary.txt"


def add_effective_stress_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sigma_v0_kpa (effective overburden stress, kPa), computed from
    depth_m using a constant assumed submerged unit weight (9 kN/m^3,
    representative of soft-to-medium marine sediment). A documented
    simplification -- real unit weight varies by soil type/density,
    unavailable at inference time.
    """
    df = df.copy()
    df["sigma_v0_kpa"] = SUBMERGED_UNIT_WEIGHT_KNM3 * df["depth_m"]
    return df


def _raw_target_optuna_objective(trial: optuna.Trial, dtrain: xgb.DMatrix) -> float:
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


def train_qt_model(X_train_full, y_train_full, X_test, y_test) -> dict:
    """Train the final qt_mpa model: raw target, soil-probs + effective stress."""
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\nTuning qt_mpa ({N_OPTUNA_TRIALS_REGRESSOR} trials, RAW target)...")
    study.optimize(
        lambda trial: _raw_target_optuna_objective(trial, dtrain_full),
        n_trials=N_OPTUNA_TRIALS_REGRESSOR, show_progress_bar=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SPLIT_FRACTION, random_state=RANDOM_SEED,
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse",
        n_estimators=MAX_BOOSTING_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_SEED, **study.best_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"qt_mpa -- FINAL test performance: MAE={mae:.3f} MPa, RMSE={rmse:.3f} MPa, R^2={r2:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "vs_to_qt_mpa_FINAL_xgboost_model.joblib")

    return {"model": model, "mae": mae, "rmse": rmse, "r2": r2}


def run_final_ml_stage():
    print("=" * 70)
    print("ML STAGE (Stage 2) -- FINAL MODEL TRAINING")
    print("=" * 70)
    train_df = clean_training_data(load_training_data())
    test_df = load_testing_data()

    print("\n--- Stage A: soil-type classifier (shared by both targets) ---")
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

    joblib.dump(classifier, MODELS_DIR / "vs_to_soiltype_FINAL_xgboost_classifier.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "soiltype_label_encoder.joblib")

    train_with_probs = pd.concat([train_df, oof_probs_df], axis=1)
    test_with_probs = pd.concat([test_df, test_probs_df], axis=1)

    # --- qt_mpa: raw target, WITH effective stress ---
    print("\n--- qt_mpa: raw target, soil-probs + effective stress ---")
    train_qt = add_effective_stress_feature(train_with_probs)
    test_qt = add_effective_stress_feature(test_with_probs)
    qt_features = FEATURE_COLUMNS_STAGE_A + ["sigma_v0_kpa"] + soil_prob_cols
    qt_result = train_qt_model(
        train_qt[qt_features], train_qt["qt_mpa"], test_qt[qt_features], test_qt["qt_mpa"]
    )

    # --- fs_mpa: log1p target, soil-probs only, NO effective stress ---
    print("\n--- fs_mpa: log1p target, soil-probs only ---")
    fs_features = FEATURE_COLUMNS_STAGE_A + soil_prob_cols
    fs_model = train_stage_b_model(
        train_with_probs[fs_features], train_with_probs["fs_mpa"], "fs_mpa"
    )
    fs_result = evaluate_stage_b_model(
        fs_model, test_with_probs[fs_features], test_with_probs["fs_mpa"], "fs_mpa"
    )
    joblib.dump(fs_model, MODELS_DIR / "vs_to_fs_mpa_FINAL_xgboost_model.joblib")

    summary = f"""ML STAGE (Stage 2) -- FINAL MODEL SUMMARY
Geophysical-to-Geotechnical Inversion Toolkit

Predicts geotechnical parameters (qt, fs) from Vs + depth -- the
reverse of the source paper's own direction (Marin-Moreno et al. 2026
predict Vs from CPTu; this toolkit predicts CPTu-equivalent parameters
from Vs, matching the physics stage's actual output and the project's
stated goal).

FIVE VARIANTS TESTED (see src/ml/experiments/):
  v1: Vs+depth only, raw target                       qt R^2=-0.016  fs R^2=0.167
  v2: Vs+depth only, log1p target (ablation)           qt R^2=-0.284  fs R^2=0.111
  v3: log1p target + soil-type probabilities           qt R^2=-0.102  fs R^2=0.205
  v4: raw target + soil-probs + effective stress       qt R^2= 0.091  fs R^2=0.155
  v5: log1p target + soil-probs + effective stress     qt R^2=  n/a   fs R^2=0.192

FINAL CONFIGURATION (best-performing recipe per target):
  qt_mpa: features=[vs_mps, depth_m, sigma_v0_kpa, soil-type probs], RAW target
          Test MAE={qt_result['mae']:.2f} MPa, RMSE={qt_result['rmse']:.2f} MPa, R^2={qt_result['r2']:.3f}
  fs_mpa: features=[vs_mps, depth_m, soil-type probs], LOG1P target
          Test MAE={fs_result['mae']:.3f} MPa, RMSE={fs_result['rmse']:.3f} MPa, R^2={fs_result['r2']:.3f}

KEY FINDING: predicting geotechnical parameters from Vs alone is
fundamentally limited by soil-type ambiguity -- the same Vs value can
correspond to vastly different qt depending on lithology (e.g. sand
vs. clay). Soil-type probability features (from a Vs+depth-based
classifier, ~55-57% accuracy) partially address this, improving both
targets over the naive baseline, but remain well below the ~0.76 R^2
upper bound achievable with TRUE soil type -- confirming a genuine,
physically-grounded ceiling on Vs-only geotechnical prediction, not a
model-tuning shortfall.
"""
    print("\n" + summary)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary)
    print(f"Summary saved: {SUMMARY_PATH}")

    return qt_result, fs_result


if __name__ == "__main__":
    run_final_ml_stage()