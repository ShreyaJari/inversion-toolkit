"""
train_model_ablation.py

Ablation study: isolates the effect of the soil-type-probability
features from the log1p target transform, by training Stage B (qt/fs
regressors) on Vs + depth ONLY (log1p-transformed target), with
identical hyperparameter search settings to train_model_v2.py's Stage
B -- so any difference from the two-stage result is attributable
specifically to the soil-type probability features, not to unrelated
settings differences.

Reuses train_stage_b_model / evaluate_stage_b_model from
train_model_v2.py directly rather than duplicating that logic.
"""

from __future__ import annotations

from load_data import load_training_data, clean_training_data, load_testing_data
from train_model import train_stage_b_model, evaluate_stage_b_model, TARGET_COLUMNS_STAGE_B

FEATURE_COLUMNS_ABLATION = ["vs_mps", "depth_m"]


def run_ablation():
    print("=" * 70)
    print("ABLATION: Vs + depth ONLY, log1p-transformed target, no soil-type features")
    print("=" * 70)

    train_df = clean_training_data(load_training_data())
    test_df = load_testing_data()

    results = {}
    for target in TARGET_COLUMNS_STAGE_B:
        print(f"\n--- Target: {target} ---")
        model = train_stage_b_model(
            train_df[FEATURE_COLUMNS_ABLATION], train_df[target], target
        )
        result = evaluate_stage_b_model(
            model, test_df[FEATURE_COLUMNS_ABLATION], test_df[target], target
        )
        results[target] = result

    print()
    print("=" * 70)
    print("THREE-WAY COMPARISON")
    print("=" * 70)
    print("                          original(v1)   ablation(log only)   two-stage(v2)")
    print(f"qt_mpa R^2:               -0.016         {results['qt_mpa']['r2']:.3f}                -0.102")
    print(f"fs_mpa R^2:                0.167         {results['fs_mpa']['r2']:.3f}                0.205")
    print(
        "\nNote: original(v1) used 50 Optuna trials, no log-transform, no soil "
        "features. ablation and two-stage(v2) both use 30 trials + log-transform, "
        "differing ONLY in the presence of soil-type probability features -- so "
        "the ablation-vs-two-stage comparison isolates the soil-feature effect "
        "cleanly. The v1-vs-ablation comparison is a rougher signal (confounded "
        "by trial count) but still informative directionally."
    )

    return results


if __name__ == "__main__":
    run_ablation()