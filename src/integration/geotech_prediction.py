"""
geotech_prediction.py

Core logic for Stage 3 (integration): given a Vs(z) profile, builds
the features the ML stage's trained models expect, and predicts
qt(z) and fs(z).

Two input modes are supported:
    1. predict_geotech_profile / predict_ensemble_profiles: a LAYERED
       (piecewise-constant) Vs(z), as produced by the physics stage's
       inversion (4 layer values + known layer thicknesses).
    2. predict_qt_fs_from_vs_depth: an arbitrary CONTINUOUS Vs(z) and
       depth(z) array pair -- e.g. a REAL measured SCPTu profile,
       where Vs varies gradually and continuously rather than being
       held constant across a wide depth range.

WHY THE DISTINCTION MATTERS: the ML models were trained on real,
continuous Vs profiles. Evaluating them on the physics stage's
piecewise-constant layered input pushes them into a region of input
space unlike anything in training (Vs held perfectly constant across
a wide depth range), which can produce locally unstable, oscillating
predictions -- a genuine, documented limitation, not a bug (see
project notebook, Stage 3 discussion). predict_qt_fs_from_vs_depth
avoids this by operating on real, naturally continuous input.

This module only USES the trained models saved by src/ml/train_model.py
-- it does not retrain anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "models"

# Must match src/ml/train_model.py exactly.
SUBMERGED_UNIT_WEIGHT_KNM3 = 9.0
FEATURE_COLUMNS_STAGE_A = ["vs_mps", "depth_m"]

DEFAULT_DEPTH_GRID_M = np.arange(0.0, 35.5, 0.5)


def _load_final_models():
    """Load all four trained artifacts saved by src/ml/train_model.py."""
    return {
        "soiltype_classifier": joblib.load(
            MODELS_DIR / "vs_to_soiltype_FINAL_xgboost_classifier.joblib"
        ),
        "soiltype_label_encoder": joblib.load(
            MODELS_DIR / "soiltype_label_encoder.joblib"
        ),
        "qt_model": joblib.load(MODELS_DIR / "vs_to_qt_mpa_FINAL_xgboost_model.joblib"),
        "fs_model": joblib.load(MODELS_DIR / "vs_to_fs_mpa_FINAL_xgboost_model.joblib"),
    }


def vs_at_depths(
    layer_vs_mps: np.ndarray, layer_depths_top_m: np.ndarray, depth_grid_m: np.ndarray
) -> np.ndarray:
    """
    Evaluate a layered Vs(z) step function at arbitrary depths.
    See predict_geotech_profile for how this is used.
    """
    layer_indices = np.searchsorted(layer_depths_top_m, depth_grid_m, side="right") - 1
    layer_indices = np.clip(layer_indices, 0, len(layer_vs_mps) - 1)
    return layer_vs_mps[layer_indices]


def predict_qt_fs_from_vs_depth(
    vs_mps_array: np.ndarray, depth_m_array: np.ndarray, models: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict qt(z), fs(z) directly from an arbitrary (continuous) Vs(z)
    and depth(z) array pair -- e.g. a REAL measured Vs profile. See
    module docstring for why this is kept distinct from the layered/
    step-function prediction path.

    Parameters
    ----------
    vs_mps_array, depth_m_array : np.ndarray, same length
        Vs (m/s) and depth (m) at each measurement point.
    models : dict
        Output of _load_final_models().

    Returns
    -------
    (qt_pred_mpa, fs_pred_mpa) : tuple of np.ndarray
    """
    vs_mps_array = np.asarray(vs_mps_array)
    depth_m_array = np.asarray(depth_m_array)
    sigma_v0_kpa = SUBMERGED_UNIT_WEIGHT_KNM3 * depth_m_array

    base_df = pd.DataFrame({
        "vs_mps": vs_mps_array,
        "depth_m": depth_m_array,
        "sigma_v0_kpa": sigma_v0_kpa,
    })

    soil_probs = models["soiltype_classifier"].predict_proba(base_df[FEATURE_COLUMNS_STAGE_A])
    soil_prob_cols = [f"soil_prob_{cls}" for cls in models["soiltype_label_encoder"].classes_]
    soil_probs_df = pd.DataFrame(soil_probs, columns=soil_prob_cols)
    features_df = pd.concat([base_df, soil_probs_df], axis=1)

    qt_features = ["vs_mps", "depth_m", "sigma_v0_kpa"] + soil_prob_cols
    qt_pred = models["qt_model"].predict(features_df[qt_features])

    fs_features = ["vs_mps", "depth_m"] + soil_prob_cols
    fs_pred_log = models["fs_model"].predict(features_df[fs_features])
    fs_pred = np.expm1(fs_pred_log)

    return qt_pred, fs_pred


def predict_geotech_profile(
    layer_vs_mps: np.ndarray,
    layer_depths_top_m: np.ndarray,
    models: dict,
    depth_grid_m: np.ndarray = DEFAULT_DEPTH_GRID_M,
) -> pd.DataFrame:
    """
    Predict a full synthetic qt(z), fs(z) geotechnical log from a
    LAYERED (piecewise-constant) Vs(z) profile -- e.g. the physics
    stage's inversion output. Internally evaluates Vs at each grid
    depth via the step function, then delegates to
    predict_qt_fs_from_vs_depth.

    Parameters
    ----------
    layer_vs_mps : np.ndarray
        One ensemble member's recovered Vs profile (4 layer values).
    layer_depths_top_m : np.ndarray
        Layer top depths (shared across all ensemble members).
    models : dict
        Output of _load_final_models().
    depth_grid_m : np.ndarray, optional

    Returns
    -------
    pd.DataFrame
        Columns: depth_m, vs_mps, sigma_v0_kpa, qt_mpa_pred, fs_mpa_pred.
    """
    vs_mps = vs_at_depths(layer_vs_mps, layer_depths_top_m, depth_grid_m)
    qt_pred, fs_pred = predict_qt_fs_from_vs_depth(vs_mps, depth_grid_m, models)

    return pd.DataFrame({
        "depth_m": depth_grid_m,
        "vs_mps": vs_mps,
        "sigma_v0_kpa": SUBMERGED_UNIT_WEIGHT_KNM3 * depth_grid_m,
        "qt_mpa_pred": qt_pred,
        "fs_mpa_pred": fs_pred,
    })


def predict_ensemble_profiles(
    ensemble_vs_arrays: List[np.ndarray],
    layer_depths_top_m: np.ndarray,
    models: dict,
    depth_grid_m: np.ndarray = DEFAULT_DEPTH_GRID_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run predict_geotech_profile for every ensemble member, returning
    stacked qt and fs prediction arrays for percentile-band computation.

    Returns
    -------
    (qt_stack, fs_stack) : tuple of np.ndarray, each shape
        (n_ensemble_members, len(depth_grid_m))
    """
    qt_stack = []
    fs_stack = []
    for vs_array in ensemble_vs_arrays:
        profile = predict_geotech_profile(vs_array, layer_depths_top_m, models, depth_grid_m)
        qt_stack.append(profile["qt_mpa_pred"].values)
        fs_stack.append(profile["fs_mpa_pred"].values)
    return np.array(qt_stack), np.array(fs_stack)