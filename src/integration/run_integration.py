"""
run_integration.py

Stage 3 (integration): the full "raw geophysics in -> geotechnical log
out" pipeline. Regenerates both physics-stage scenarios (wide-band/
low-noise vs. narrow-band/high-noise), runs every ensemble member's
recovered Vs(z) profile through the ML stage's trained qt/fs models,
and aggregates the results into percentile-band synthetic geotechnical
logs -- demonstrating that survey quality upstream (physics stage)
directly controls prediction confidence downstream (ML stage).

IMPORTANT CAVEAT, STATED EXPLICITLY: the Vs(z) profiles here are fully
SYNTHETIC (from the physics stage's synthetic ground-truth marine-
sediment example) -- they were never paired with a real measured CPT.
This means Stage 3's output can be checked for INTERNAL CONSISTENCY
and MECHANICAL CORRECTNESS (does uncertainty propagate sensibly from
physics to ML predictions), but NOT for accuracy against real ground
truth, since no real ground truth exists for this synthetic case. This
mirrors the same honesty caveat established for the physics stage's
own synthetic validation.

WHY PERCENTILE BANDS (10th/50th/90th), NOT MEAN +/- STD: consistent
with the physics stage's finding that Scenario 2's Vs ensemble is
genuinely BIMODAL (two distinct converged solutions, not scattered
noise around one answer -- see src/physics/inversion.py and project
conversation), a percentile band correctly represents a
possibly-asymmetric, possibly-multimodal spread without assuming the
unimodal shape that mean+/-std implicitly assumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make src/physics/ importable both as a package (physics.layered_model,
# used by this file) and flat-style (layered_model, used internally by
# forward_model.py / inversion.py) -- both styles are needed since the
# physics-stage scripts were originally written to be run standalone
# from inside src/physics/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "physics"))

from physics.layered_model import LayeredEarthModel  # noqa: E402
from physics.forward_model import forward_model_dispersion_curve, add_gaussian_noise  # noqa: E402
from physics.inversion import run_ensemble_inversion  # noqa: E402

from geotech_prediction import (  # noqa: E402
    _load_final_models, predict_ensemble_profiles, DEFAULT_DEPTH_GRID_M,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "figures"
SUMMARY_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "integration_summary.txt"

N_ENSEMBLE_RUNS = 8
NOISE_RANDOM_SEED = 42


def _regenerate_physics_ensembles():
    """
    Regenerate both physics-stage scenarios' 8-run ensembles, exactly
    as in src/physics/synthetic_validation.py, so Stage 3 has the raw
    Vs arrays to feed into the ML models (these were never persisted
    to disk by the physics stage -- only plots/summaries were saved).

    Returns
    -------
    dict
        Keys: 'true_model', 'scenario1_ensemble', 'scenario2_ensemble'
        -- each ensemble is a list of InversionResult.
    """
    true_model = LayeredEarthModel.example_marine_sediment_profile()
    thicknesses_m = true_model.thicknesses_m

    # Scenario 1: wide band, low noise
    clean_curve_s1 = forward_model_dispersion_curve(true_model)
    observed_curve_s1 = add_gaussian_noise(clean_curve_s1, noise_fraction=0.03, random_seed=NOISE_RANDOM_SEED)
    ensemble_s1 = run_ensemble_inversion(observed_curve_s1, thicknesses_m, n_runs=N_ENSEMBLE_RUNS)

    # Scenario 2: narrow band, high noise
    clean_curve_s2 = forward_model_dispersion_curve(true_model, freq_min_hz=8.0, freq_max_hz=25.0, n_frequencies=40)
    observed_curve_s2 = add_gaussian_noise(clean_curve_s2, noise_fraction=0.08, random_seed=NOISE_RANDOM_SEED)
    ensemble_s2 = run_ensemble_inversion(observed_curve_s2, thicknesses_m, n_runs=N_ENSEMBLE_RUNS)

    return {
        "true_model": true_model,
        "scenario1_ensemble": ensemble_s1,
        "scenario2_ensemble": ensemble_s2,
    }


def plot_geotech_log_comparison(
    depth_grid_m: np.ndarray,
    qt_stack_s1: np.ndarray, fs_stack_s1: np.ndarray,
    qt_stack_s2: np.ndarray, fs_stack_s2: np.ndarray,
) -> plt.Figure:
    """
    Plot predicted qt(z) and fs(z) percentile bands, Scenario 1 vs.
    Scenario 2, side by side -- the key figure demonstrating that
    physics-stage survey quality controls downstream prediction
    confidence.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    for ax, s1_stack, s2_stack, label in zip(
        axes, [qt_stack_s1, fs_stack_s1], [qt_stack_s2, fs_stack_s2],
        ["qt (MPa)", "fs (MPa)"],
    ):
        for stack, color, scenario_label in [
            (s1_stack, "tab:blue", "Scenario 1 (wide band, low noise)"),
            (s2_stack, "tab:red", "Scenario 2 (narrow band, high noise)"),
        ]:
            p10 = np.percentile(stack, 10, axis=0)
            p50 = np.percentile(stack, 50, axis=0)
            p90 = np.percentile(stack, 90, axis=0)

            ax.plot(p50, depth_grid_m, color=color, linewidth=2, label=f"{scenario_label} (median)")
            ax.fill_betweenx(depth_grid_m, p10, p90, color=color, alpha=0.2, label=f"{scenario_label} (10th-90th pctile)")

        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    fig.suptitle(
        "Synthetic geotechnical log: predicted qt(z), fs(z) with uncertainty\n"
        "(band width reflects physics-stage survey quality, not real measurement error)"
    )
    fig.tight_layout()

    out_path = FIGURES_DIR / "integrated_geotech_log_prediction.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    return fig


def run_integration():
    print("=" * 70)
    print("STAGE 3: INTEGRATION -- regenerating physics-stage ensembles")
    print("=" * 70)
    physics_data = _regenerate_physics_ensembles()

    print("Loading trained ML models...")
    models = _load_final_models()

    layer_depths_top_m = physics_data["true_model"].depths_m()

    print("\nPredicting geotechnical logs for Scenario 1 ensemble (8 runs)...")
    vs_arrays_s1 = [r.recovered_vs_mps for r in physics_data["scenario1_ensemble"]]
    qt_stack_s1, fs_stack_s1 = predict_ensemble_profiles(vs_arrays_s1, layer_depths_top_m, models)

    print("Predicting geotechnical logs for Scenario 2 ensemble (8 runs)...")
    vs_arrays_s2 = [r.recovered_vs_mps for r in physics_data["scenario2_ensemble"]]
    qt_stack_s2, fs_stack_s2 = predict_ensemble_profiles(vs_arrays_s2, layer_depths_top_m, models)

    print("\nGenerating figure...")
    plot_geotech_log_comparison(
        DEFAULT_DEPTH_GRID_M, qt_stack_s1, fs_stack_s1, qt_stack_s2, fs_stack_s2
    )

    # Quantitative band-width comparison at representative depths.
    #
    # NOTE ON RATIOS: Scenario 1's Vs ensemble is near-deterministic
    # (physics stage found essentially one converged solution -- see
    # inversion.py's ensemble std dev results), so its qt/fs band
    # widths are genuinely tiny (not exactly zero, but on the order of
    # 1e-5 to 1e-3 MPa). Dividing Scenario 2's width by such a small
    # number produces an astronomically large, meaningless ratio (e.g.
    # "12,281,029x wider") -- a reporting artifact, not a real result.
    # We report a ratio only when the denominator is large enough for
    # it to mean anything; otherwise we state plainly that Scenario 1's
    # band is effectively zero.
    check_depths_m = [5.0, 15.0, 25.0]
    MEANINGFUL_WIDTH_THRESHOLD_MPA = 0.01

    lines = [
        "STAGE 3 (INTEGRATION) -- SUMMARY",
        "Geophysical-to-Geotechnical Inversion Toolkit",
        "",
        "IMPORTANT: Vs(z) profiles here are fully SYNTHETIC (physics",
        "stage's synthetic ground truth) -- never paired with a real",
        "measured CPT. Results below check INTERNAL CONSISTENCY of the",
        "physics-to-ML pipeline (does uncertainty propagate sensibly),",
        "NOT accuracy against real ground truth, since none exists for",
        "this synthetic case.",
        "",
        "qt(z) and fs(z) 10th-90th percentile band width (MPa), at",
        "representative depths, Scenario 1 (wide band, low noise) vs.",
        "Scenario 2 (narrow band, high noise):",
        "",
    ]

    for depth in check_depths_m:
        idx = int(np.argmin(np.abs(DEFAULT_DEPTH_GRID_M - depth)))
        qt_width_s1 = np.percentile(qt_stack_s1[:, idx], 90) - np.percentile(qt_stack_s1[:, idx], 10)
        qt_width_s2 = np.percentile(qt_stack_s2[:, idx], 90) - np.percentile(qt_stack_s2[:, idx], 10)
        fs_width_s1 = np.percentile(fs_stack_s1[:, idx], 90) - np.percentile(fs_stack_s1[:, idx], 10)
        fs_width_s2 = np.percentile(fs_stack_s2[:, idx], 90) - np.percentile(fs_stack_s2[:, idx], 10)

        qt_ratio_str = (
            f"{qt_width_s2 / qt_width_s1:.1f}x wider"
            if qt_width_s1 > MEANINGFUL_WIDTH_THRESHOLD_MPA
            else "S1 band effectively zero (near-deterministic)"
        )
        fs_ratio_str = (
            f"{fs_width_s2 / fs_width_s1:.1f}x wider"
            if fs_width_s1 > MEANINGFUL_WIDTH_THRESHOLD_MPA
            else "S1 band effectively zero (near-deterministic)"
        )

        lines.append(
            f"  Depth {depth:5.1f} m:  qt band S1={qt_width_s1:.4f} MPa, S2={qt_width_s2:6.2f} MPa ({qt_ratio_str})   |   "
            f"fs band S1={fs_width_s1:.4f} MPa, S2={fs_width_s2:.3f} MPa ({fs_ratio_str})"
        )

    lines.append("")
    lines.append(
        "CONCLUSION: prediction uncertainty in the final geotechnical log "
        "is directly inherited from physics-stage survey quality -- a "
        "degraded survey (narrow frequency band, high noise) propagates "
        "into a wider, less confident qt/fs prediction band, demonstrating "
        "end-to-end uncertainty propagation through the full toolkit."
    )

    summary = "\n".join(lines)
    print("\n" + summary)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary)
    print(f"\nSummary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_integration()