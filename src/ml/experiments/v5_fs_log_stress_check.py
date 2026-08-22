"""
train_model_v4_fs_log_check.py

Final check: does fs_mpa benefit from v3's feature set (soil-type
probabilities + effective overburden stress) when combined with the
log1p target transform it seemed to prefer (per the v2 vs. v3
comparison)? Only trains fs_mpa -- qt_mpa's preference for the raw
target is already well-established from the ablation and v3 results,
so re-testing it here would not add information.

Reuses Stage A classifier + OOF probability generation from
train_model_v2.py, the effective-stress feature from train_model_v3.py,
and the log1p training/evaluation logic from train_model_v2.py's
train_stage_b_model / evaluate_stage_b_model.
"""

from __future__ import annotations

import pandas as pd

from load_data import load_training_data, clean_training_data, load_testing_data
from train_model_v2 import (
    train_soil_type_classifier, get_out_of_fold_soil_probs,
    train_stage_b_model, evaluate_stage_b_model,
    FEATURE_COLUMNS_STAGE_A,
)
from train_model_v2 import add_effective_stress_feature


def run_check():
    print("=" * 70)
    print("CHECK: fs_mpa with v3 features (soil-probs + stress) + LOG target")
    print("=" * 70)

    train_df = clean_training_data(load_training_data())
    test_df = load_testing_data()

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

    train_augmented = add_effective_stress_feature(pd.concat([train_df, oof_probs_df], axis=1))
    test_augmented = add_effective_stress_feature(pd.concat([test_df, test_probs_df], axis=1))

    features = FEATURE_COLUMNS_STAGE_A + ["sigma_v0_kpa"] + soil_prob_cols
    print(f"Features: {features}")

    model = train_stage_b_model(
        train_augmented[features], train_augmented["fs_mpa"], "fs_mpa"
    )
    result = evaluate_stage_b_model(
        model, test_augmented[features], test_augmented["fs_mpa"], "fs_mpa"
    )

    print()
    print("=" * 70)
    print("FIVE-WAY COMPARISON, fs_mpa")
    print("=" * 70)
    print("v1(orig, raw, no extras):        0.167")
    print("ablation(log, no extras):        0.111")
    print("v2(log + soil-probs only):       0.205")
    print("v3(raw + soil-probs + stress):   0.155")
    print(f"v4(log + soil-probs + stress):   {result['r2']:.3f}   <- this run")

    return result


if __name__ == "__main__":
    run_check()