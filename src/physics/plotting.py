"""
plotting.py

Visualization helpers for the physics stage: plotting layered Vs(z)
profiles, dispersion-curve fits, and ensemble non-uniqueness results.
Figures are saved to results/figures/ at the project root.

WHY A DEDICATED STEP-PLOT HELPER, NOT plt.plot()
-----------------------------------------------------
A layered-earth model is a STEP FUNCTION: Vs is constant within each
layer and changes DISCONTINUOUSLY at layer boundaries. Plotting
(depth, Vs) points at layer tops with an ordinary line would draw a
smooth, gradually-sloping line between layers -- which misrepresents
the model, implying a gradational transition where the idealization
actually has a sharp boundary. _to_step_arrays() below constructs the
doubled-up coordinate arrays needed to draw a proper staircase.

WHY THE HALF-SPACE IS DRAWN WITH A DASHED/FADED EXTENSION
----------------------------------------------------------------
The deepest layer is a half-space (infinite thickness) -- see
layered_model.py. We cannot plot infinity, so we extend it downward by
its stored NOMINAL thickness (a plotting-only convention, not a
physical value -- see LayeredEarthModel docstring) and visually mark
that this is an artificial cutoff, not a real layer boundary, using a
dashed linestyle and an explicit annotation.

DEPTH AXIS CONVENTION
------------------------
Depth increases DOWNWARD on the page (standard geoscience convention,
matching how a borehole log or CPT log is read), so the y-axis is
inverted relative to matplotlib's default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt

from layered_model import LayeredEarthModel
from forward_model import DispersionCurve
from inversion import InversionResult


# ---------------------------------------------------------------------------
# Figures are saved relative to the project root's results/figures/
# directory. This resolves that path relative to THIS file's location
# (src/physics/plotting.py -> ../../results/figures), so it works
# regardless of what directory you happen to run the script from.
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "figures"


def _to_step_arrays(
    depths_top_m: np.ndarray, vs_mps: np.ndarray, bottom_depth_m: float
) -> tuple:
    """
    Convert per-layer (top depth, Vs) values into doubled-up coordinate
    arrays suitable for a proper staircase plot (see module docstring
    for why this is needed instead of a plain line plot).

    Parameters
    ----------
    depths_top_m : np.ndarray, shape (n_layers,)
        Depth to the TOP of each layer, m (as returned by
        LayeredEarthModel.depths_m()).
    vs_mps : np.ndarray, shape (n_layers,)
        Vs value for each layer, m/s.
    bottom_depth_m : float
        Depth at which to terminate the plot for the deepest
        (half-space) layer -- a plotting-only cutoff, not a physical
        boundary.

    Returns
    -------
    (depth_plot, vs_plot) : tuple of np.ndarray, each shape (2 * n_layers,)
        Coordinate arrays that, when plotted with ax.plot(vs_plot,
        depth_plot), produce a correct staircase: horizontal segments
        at each layer's Vs, vertical jumps at each layer boundary.
    """
    depth_edges = np.concatenate([depths_top_m, [bottom_depth_m]])
    depth_plot = np.repeat(depth_edges, 2)[1:-1]
    vs_plot = np.repeat(vs_mps, 2)
    return depth_plot, vs_plot


def plot_vs_profile_comparison(
    true_model: LayeredEarthModel,
    recovered_vs_mps: Optional[np.ndarray] = None,
    recovered_label: str = "Recovered (inverted)",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the true synthetic Vs(z) profile as a staircase, optionally
    overlaid with a single recovered profile for direct comparison.

    Parameters
    ----------
    true_model : LayeredEarthModel
        The known synthetic ground-truth model.
    recovered_vs_mps : np.ndarray, optional
        A single recovered Vs array (same layer thicknesses as
        true_model assumed) to overlay. If None, only the true profile
        is plotted.
    recovered_label : str, optional
        Legend label for the recovered profile.
    save_path : Path, optional
        Where to save the figure. Defaults to
        FIGURES_DIR / "synthetic_vs_profile_true_vs_recovered.png".
    show : bool, optional
        Whether to call plt.show() (useful when running interactively;
        leave False for headless/script runs).

    Returns
    -------
    matplotlib.figure.Figure
    """
    depths_top = true_model.depths_m()
    # Half-space plotting cutoff: extend by its own nominal thickness
    # below the top of the half-space (a visual choice, not physical --
    # see module docstring).
    bottom_depth = depths_top[-1] + true_model.thicknesses_m[-1]

    fig, ax = plt.subplots(figsize=(6, 8))

    true_depth_plot, true_vs_plot = _to_step_arrays(
        depths_top, true_model.vs_profile_mps, bottom_depth
    )
    ax.plot(
        true_vs_plot, true_depth_plot,
        color="black", linewidth=2.5, label="True (synthetic ground truth)",
    )

    if recovered_vs_mps is not None:
        rec_depth_plot, rec_vs_plot = _to_step_arrays(
            depths_top, recovered_vs_mps, bottom_depth
        )
        ax.plot(
            rec_vs_plot, rec_depth_plot,
            color="tab:red", linewidth=2, linestyle="--", label=recovered_label,
        )

    # Mark the half-space cutoff explicitly, so it reads as "this
    # continues below" rather than "this is where the layer ends."
    ax.axhline(
        depths_top[-1] + true_model.thicknesses_m[-1] * 0.15,
        color="grey", linewidth=0, alpha=0,  # invisible; placeholder for
                                               # any future shading logic
    )
    ax.annotate(
        "half-space continues\n(not a real boundary)",
        xy=(true_model.vs_profile_mps[-1], bottom_depth),
        xytext=(true_model.vs_profile_mps[-1] * 1.05, bottom_depth * 0.97),
        fontsize=8, color="grey", ha="left",
    )

    ax.invert_yaxis()  # depth increases downward -- see module docstring
    ax.set_xlabel("Shear-wave velocity, Vs (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Layered Vs(z) profile: true vs. recovered")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = save_path or (FIGURES_DIR / "synthetic_vs_profile_true_vs_recovered.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    return fig


def plot_dispersion_fit(
    observed_curve: DispersionCurve,
    true_clean_curve: DispersionCurve,
    recovered_curve: DispersionCurve,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the observed (noisy) dispersion curve alongside the true
    noise-free curve and the curve forward-modeled from the inversion's
    recovered Vs profile -- the direct visual check of "does the
    recovered model actually explain the data."

    Parameters
    ----------
    observed_curve : DispersionCurve
        The noisy curve that was fed into the inversion.
    true_clean_curve : DispersionCurve
        The noise-free curve forward-modeled from the true synthetic
        model (for reference -- shows how much the noise perturbed the
        "real" signal).
    recovered_curve : DispersionCurve
        The curve forward-modeled from the inversion's recovered Vs
        profile (InversionResult.recovered_curve).
    save_path, show : see plot_vs_profile_comparison.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        observed_curve.frequency_hz, observed_curve.phase_velocity_mps,
        color="tab:grey", s=20, label="Observed (noisy)", zorder=3,
    )
    ax.plot(
        true_clean_curve.frequency_hz, true_clean_curve.phase_velocity_mps,
        color="black", linewidth=1.5, linestyle=":",
        label="True (noise-free)", zorder=2,
    )
    ax.plot(
        recovered_curve.frequency_hz, recovered_curve.phase_velocity_mps,
        color="tab:red", linewidth=2, label="Recovered model fit", zorder=2,
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Rayleigh phase velocity (m/s)")
    ax.set_title("Dispersion curve: observed data vs. recovered model fit")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = save_path or (FIGURES_DIR / "dispersion_curve_fit.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    return fig


def _cluster_ensemble_runs(
    ensemble: List[InversionResult], round_to_mps: float = 2.0
) -> List[List[InversionResult]]:
    """
    Group ensemble runs that converged to (numerically) the same
    recovered Vs profile, so distinct local minima can be color-coded
    separately when plotted rather than visually overlapping into an
    indistinguishable single line.

    METHOD: round each run's recovered_vs_mps to the nearest
    round_to_mps and group runs with identical rounded tuples. This is
    a simple exact-match grouping, not a general clustering algorithm
    (e.g. k-means) -- appropriate here because runs converging to the
    "same" local minimum typically agree to within a fraction of a
    m/s (see e.g. the scenario 2 runs discussed in-conversation, which
    agreed to within ~0.5 m/s within a cluster but differed by 100+
    m/s across clusters -- an unambiguous gap, not a borderline case
    a fuzzier method would be needed for).

    Parameters
    ----------
    ensemble : list[InversionResult]
        The ensemble to group.
    round_to_mps : float, optional
        Rounding resolution, m/s, for the grouping key. Defaults to
        2.0 -- coarse enough to tolerate small floating-point/solver
        convergence differences between runs that found the "same"
        minimum, fine enough to keep genuinely different minima
        (typically tens-to-hundreds of m/s apart) separate.

    Returns
    -------
    list[list[InversionResult]]
        Clusters, ordered by ascending MEAN misfit (best-fitting
        cluster first).
    """
    groups: dict = {}
    for result in ensemble:
        key = tuple(np.round(result.recovered_vs_mps / round_to_mps) * round_to_mps)
        groups.setdefault(key, []).append(result)

    clusters = list(groups.values())
    clusters.sort(key=lambda cluster: np.mean([r.misfit_rmse_mps for r in cluster]))
    return clusters


def plot_ensemble_profiles(
    true_model: LayeredEarthModel,
    ensemble_scenario1: List[InversionResult],
    ensemble_scenario2: List[InversionResult],
    scenario1_label: str = "Scenario 1: wide band, low noise",
    scenario2_label: str = "Scenario 2: narrow band, high noise",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot every individual recovered profile from two ensembles,
    side-by-side, overlaid as thin lines against the true profile.
    Runs that converged to the SAME local minimum (within
    _cluster_ensemble_runs' rounding tolerance) are grouped and
    color-coded together, so distinct competing solutions are visually
    distinguishable rather than overlapping into an indistinguishable
    single trace.

    WHY CLUSTER-COLORING, NOT ALL RUNS THE SAME COLOR: as observed when
    reviewing this project's own scenario 2 results, multiple runs can
    converge to EXACTLY the same alternate (and sometimes lower-misfit
    but further-from-truth) solution, while a minority converge
    elsewhere. Plotting all runs in one color makes this invisible
    (identical lines simply overlap); color-coding by cluster makes the
    bimodal/multimodal structure directly visible, and the legend
    reports each cluster's size and mean misfit so the reader can see,
    e.g., that the BEST-FITTING cluster is not necessarily the
    cluster closest to the true profile.

    Parameters
    ----------
    true_model : LayeredEarthModel
        The known synthetic ground truth (same for both scenarios).
    ensemble_scenario1, ensemble_scenario2 : list[InversionResult]
        The ensemble results from run_ensemble_inversion() for each
        scenario.
    scenario1_label, scenario2_label : str, optional
        Subplot titles.
    save_path, show : see plot_vs_profile_comparison.

    Returns
    -------
    matplotlib.figure.Figure
    """
    depths_top = true_model.depths_m()
    bottom_depth = depths_top[-1] + true_model.thicknesses_m[-1]

    # Distinct colors for up to 4 clusters per panel -- more than 4
    # genuinely distinct local minima in a 4-layer search would be
    # unusual, but the color list can be extended if needed.
    cluster_colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    for ax, ensemble, title in zip(
        axes, [ensemble_scenario1, ensemble_scenario2],
        [scenario1_label, scenario2_label],
    ):
        clusters = _cluster_ensemble_runs(ensemble)

        for cluster_idx, cluster in enumerate(clusters):
            color = cluster_colors[cluster_idx % len(cluster_colors)]
            mean_misfit = np.mean([r.misfit_rmse_mps for r in cluster])
            cluster_label = (
                f"Solution {chr(65 + cluster_idx)} "
                f"({len(cluster)}/{len(ensemble)} runs, "
                f"misfit\u2248{mean_misfit:.1f} m/s)"
            )

            for i, result in enumerate(cluster):
                depth_plot, vs_plot = _to_step_arrays(
                    depths_top, result.recovered_vs_mps, bottom_depth
                )
                ax.plot(
                    vs_plot, depth_plot,
                    color=color, linewidth=1.5, alpha=0.7,
                    label=cluster_label if i == 0 else None,
                )

        true_depth_plot, true_vs_plot = _to_step_arrays(
            depths_top, true_model.vs_profile_mps, bottom_depth
        )
        ax.plot(
            true_vs_plot, true_depth_plot,
            color="black", linewidth=2.5, label="True", zorder=5,
        )

        ax.set_xlabel("Vs (m/s)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    fig.suptitle(
        "Non-uniqueness: distinct converged solutions, color-coded by cluster\n"
        "(lowest-misfit solution is not necessarily closest to truth)"
    )
    fig.tight_layout()

    out_path = save_path or (FIGURES_DIR / "inversion_nonuniqueness.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    return fig

if __name__ == "__main__":
    # Reproduce scenario 1 and scenario 2 from inversion.py's smoke
    # test, then generate all three figures. This duplicates a small
    # amount of setup from inversion.py's __main__ block deliberately,
    # so plotting.py can be run standalone without needing inversion.py
    # to have been run first in the same session.
    from forward_model import forward_model_dispersion_curve, add_gaussian_noise
    from inversion import invert_dispersion_curve, run_ensemble_inversion

    true_model = LayeredEarthModel.example_marine_sediment_profile()
    thicknesses_m = true_model.thicknesses_m

    # --- Scenario 1 (wide band, low noise) ---
    clean_curve_s1 = forward_model_dispersion_curve(true_model)
    observed_curve_s1 = add_gaussian_noise(
        clean_curve_s1, noise_fraction=0.03, random_seed=42
    )
    single_result_s1 = invert_dispersion_curve(
        observed_curve_s1, thicknesses_m, random_seed=0
    )
    ensemble_s1 = run_ensemble_inversion(observed_curve_s1, thicknesses_m, n_runs=8)

    # --- Scenario 2 (narrow band, high noise) ---
    clean_curve_s2 = forward_model_dispersion_curve(
        true_model, freq_min_hz=8.0, freq_max_hz=25.0, n_frequencies=40
    )
    observed_curve_s2 = add_gaussian_noise(
        clean_curve_s2, noise_fraction=0.08, random_seed=42
    )
    ensemble_s2 = run_ensemble_inversion(observed_curve_s2, thicknesses_m, n_runs=8)

    print("Generating figures...")
    plot_vs_profile_comparison(true_model, single_result_s1.recovered_vs_mps)
    plot_dispersion_fit(observed_curve_s1, clean_curve_s1, single_result_s1.recovered_curve)
    plot_ensemble_profiles(true_model, ensemble_s1, ensemble_s2)
    print("Done.")

    print("\nAll 8 individual recovered Vs profiles, scenario 2:")
    for i, r in enumerate(ensemble_s2):
        print(f"  Run {i} (seed={r.random_seed}): {np.round(r.recovered_vs_mps, 1)}  misfit={r.misfit_rmse_mps:.2f}")