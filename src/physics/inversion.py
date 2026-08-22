"""
inversion.py

Inverts a (noisy) Rayleigh-wave fundamental-mode dispersion curve to
recover a layered Vs(z) profile, using disba's forward model as the
engine inside a global optimization search.

PHYSICAL BACKGROUND: WHY INVERSION IS HARD
---------------------------------------------
Forward modeling (previous script) is well-posed: one subsurface model
produces exactly one dispersion curve. Inversion is the reverse and is
ILL-POSED: multiple different Vs(z) profiles can produce dispersion
curves that are nearly indistinguishable from each other, especially
for deeper layers (which surface waves are inherently less sensitive
to than shallow layers -- see forward_model.py's depth-of-investigation
discussion). This is a well-known, unavoidable property of surface-wave
inversion, not a flaw in this implementation.

Because of this, we do NOT just find "an" answer and report it. We:
    1. Use a GLOBAL optimizer (differential_evolution) rather than a
       local one, so we are less likely to silently converge to just
       one of several similarly-good answers.
    2. Run the inversion MULTIPLE TIMES with different random seeds and
       report the SPREAD of recovered profiles, not just one result --
       this spread is the empirical signature of non-uniqueness, and it
       is the uncertainty signal that Stage 3 (integration) will
       propagate forward into the ML predictions.

SCOPE SIMPLIFICATIONS (STATED EXPLICITLY, PER PROJECT SCOPE)
-----------------------------------------------------------------
1. Layer THICKNESSES are treated as KNOWN and fixed -- only the 4 Vs
   values are inverted for. Jointly inverting for thickness AND
   velocity is standard practice to avoid in a basic workflow like this,
   because it multiplies the number of free parameters chasing the same
   amount of data, making the non-uniqueness problem far worse. Real
   MASW practice typically fixes layer boundaries from prior geological
   knowledge (e.g. borehole control), then inverts velocities only --
   which is what we do here.

2. Vp/Vs ratio and the density-Vs relation are treated as KNOWN
   (the same empirical relations from layered_model.py are reused when
   the optimizer proposes candidate Vs profiles). In reality, being
   wrong about these would bias the recovered Vs -- we are not modeling
   that additional source of uncertainty in this toolkit, and say so
   explicitly here rather than leave it as a silent assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from scipy.optimize import differential_evolution

from layered_model import Layer, LayeredEarthModel
from forward_model import DispersionCurve, forward_model_dispersion_curve
from disba._exception import DispersionError


# ---------------------------------------------------------------------------
# Default search bounds for each layer's Vs, m/s. Chosen to comfortably
# contain the synthetic true profile (100-400 m/s) without being so wide
# that the optimizer wastes effort in physically implausible territory
# for near-surface marine sediment.
# ---------------------------------------------------------------------------
DEFAULT_VS_BOUNDS_MPS = (50.0, 600.0)


@dataclass
class InversionResult:
    """
    The result of a single inversion run.

    Attributes
    ----------
    recovered_vs_mps : np.ndarray
        The best-fitting Vs value found for each layer, m/s.
    misfit_rmse_mps : float
        Root-mean-square error, in m/s, between the observed dispersion
        curve and the curve forward-modeled from recovered_vs_mps. Lower
        is a better fit. This is the quantity differential_evolution
        minimized.
    recovered_curve : DispersionCurve
        The dispersion curve forward-modeled from recovered_vs_mps, for
        plotting against the observed curve to visually check the fit.
    random_seed : Optional[int]
        The seed used for this run's optimizer, if any -- kept so that
        an ensemble of results can be traced back to which run produced
        which answer.
    """

    recovered_vs_mps: np.ndarray
    misfit_rmse_mps: float
    recovered_curve: DispersionCurve
    random_seed: Optional[int] = None


def _build_candidate_model(
    vs_mps_array: np.ndarray, thicknesses_m: np.ndarray
) -> LayeredEarthModel:
    """
    Build a LayeredEarthModel from a candidate Vs array and known
    (fixed) thicknesses. Vp and density are auto-estimated from Vs via
    the same empirical relations used to build the synthetic ground
    truth in layered_model.py -- see module docstring, scope
    simplification #2.

    Parameters
    ----------
    vs_mps_array : np.ndarray, shape (n_layers,)
        Candidate Vs value for each layer, m/s.
    thicknesses_m : np.ndarray, shape (n_layers,)
        Fixed, known layer thicknesses, m. The last value is the
        half-space's nominal thickness (not physically used, but
        required by the Layer/LayeredEarthModel constructor -- see
        layered_model.py).

    Returns
    -------
    LayeredEarthModel
    """
    layers = [
        Layer(thickness_m=float(t), vs_mps=float(vs))
        for t, vs in zip(thicknesses_m, vs_mps_array)
    ]
    return LayeredEarthModel(layers=layers)


from disba._exception import DispersionError

# ---------------------------------------------------------------------------
# Large penalty misfit (m/s) returned when a candidate Vs profile causes
# disba's root-finding to fail outright (rather than just fit poorly).
# This is a common, expected occurrence during a GLOBAL search: the
# optimizer explores the full bounds, and some candidate velocity
# combinations are numerically pathological for the dispersion solver
# (e.g. certain velocity-inversion geometries create trapped-mode
# effects where no fundamental-mode root exists at some periods). We
# treat these candidates as simply "very bad" rather than letting the
# exception crash the search -- this is standard practice in surface-
# wave inversion codes generally, not a workaround specific to disba.
# The value must be much larger than any realistic misfit (which will
# be on the order of tens of m/s given our noise level) so the
# optimizer reliably steers away from these regions.
# ---------------------------------------------------------------------------
FAILED_FORWARD_MODEL_PENALTY_MPS = 1.0e5


def _misfit_rmse_mps(
    vs_mps_array: np.ndarray,
    thicknesses_m: np.ndarray,
    observed_curve: DispersionCurve,
) -> float:
    """
    Compute the RMSE (m/s) between a candidate Vs profile's forward-
    modeled dispersion curve and the observed dispersion curve.

    This is the objective function differential_evolution minimizes.

    ROBUSTNESS NOTE: during a global search, some candidate Vs
    combinations cause disba's root-finding to fail outright (raise
    DispersionError) rather than return a poor-but-valid dispersion
    curve. We catch this and return a large fixed penalty instead of
    letting the exception propagate -- see FAILED_FORWARD_MODEL_PENALTY_MPS
    comment above for why this is standard practice, not a hack.

    Parameters
    ----------
    vs_mps_array : np.ndarray, shape (n_layers,)
        Candidate Vs values proposed by the optimizer.
    thicknesses_m : np.ndarray, shape (n_layers,)
        Fixed, known thicknesses.
    observed_curve : DispersionCurve
        The (noisy) observed dispersion curve to fit against.

    Returns
    -------
    float
        RMSE, m/s, between observed and candidate-model-predicted phase
        velocities -- or FAILED_FORWARD_MODEL_PENALTY_MPS if the
        forward model failed to evaluate for this candidate.
    """
    candidate_model = _build_candidate_model(vs_mps_array, thicknesses_m)

    try:
        candidate_curve = forward_model_dispersion_curve(
            candidate_model,
            freq_min_hz=float(observed_curve.frequency_hz.min()),
            freq_max_hz=float(observed_curve.frequency_hz.max()),
            n_frequencies=len(observed_curve.frequency_hz),
        )
    except DispersionError:
        return FAILED_FORWARD_MODEL_PENALTY_MPS

    residuals = (
        candidate_curve.phase_velocity_mps - observed_curve.phase_velocity_mps
    )
    return float(np.sqrt(np.mean(residuals**2)))

def invert_dispersion_curve(
    observed_curve: DispersionCurve,
    thicknesses_m: np.ndarray,
    vs_bounds_mps: tuple = DEFAULT_VS_BOUNDS_MPS,
    random_seed: Optional[int] = None,
    max_iterations: int = 200,
) -> InversionResult:
    """
    Invert a single observed dispersion curve for the best-fitting Vs
    profile, given fixed/known layer thicknesses, using a global
    (differential evolution) search.

    WHY differential_evolution AND NOT A LOCAL OPTIMIZER: see module
    docstring. Briefly -- the misfit surface for this class of problem
    is known to be multimodal (multiple distinct Vs combinations can
    produce similarly low misfit), and a local optimizer would silently
    converge to whichever local minimum is nearest its starting guess,
    hiding that ambiguity. A population-based global search explores
    more broadly and is less prone to that failure mode.

    Parameters
    ----------
    observed_curve : DispersionCurve
        The (typically noisy) dispersion curve to invert.
    thicknesses_m : np.ndarray, shape (n_layers,)
        Fixed, known layer thicknesses, m (including the half-space's
        nominal thickness as the last entry).
    vs_bounds_mps : tuple(float, float), optional
        (min, max) Vs search bounds, m/s, applied to EVERY layer.
        Defaults to DEFAULT_VS_BOUNDS_MPS -- see module-level constant
        comment for rationale.
    random_seed : int, optional
        Seed for differential_evolution's stochastic population
        initialization. Different seeds can converge to different
        answers when the problem is non-unique -- this is intentional
        and is exploited by run_ensemble_inversion() below, not a bug
        to be seeded away.
    max_iterations : int, optional
        Maximum number of generations for differential_evolution.
        200 is generous for a 4-parameter search; increase if
        convergence looks incomplete (check InversionResult misfit
        against forward-model noise floor as a rough diagnostic).

    Returns
    -------
    InversionResult
    """
    n_layers = len(thicknesses_m)
    bounds = [vs_bounds_mps] * n_layers

    result = differential_evolution(
        func=_misfit_rmse_mps,
        bounds=bounds,
        args=(thicknesses_m, observed_curve),
        seed=random_seed,
        maxiter=max_iterations,
        tol=1e-4,
        polish=True,  # final local refinement step after global search
    )

    recovered_vs_mps = result.x
    recovered_model = _build_candidate_model(recovered_vs_mps, thicknesses_m)
    recovered_curve = forward_model_dispersion_curve(
        recovered_model,
        freq_min_hz=float(observed_curve.frequency_hz.min()),
        freq_max_hz=float(observed_curve.frequency_hz.max()),
        n_frequencies=len(observed_curve.frequency_hz),
    )

    return InversionResult(
        recovered_vs_mps=recovered_vs_mps,
        misfit_rmse_mps=float(result.fun),
        recovered_curve=recovered_curve,
        random_seed=random_seed,
    )


def run_ensemble_inversion(
    observed_curve: DispersionCurve,
    thicknesses_m: np.ndarray,
    n_runs: int = 8,
    vs_bounds_mps: tuple = DEFAULT_VS_BOUNDS_MPS,
) -> List[InversionResult]:
    """
    Run the inversion multiple times with different random seeds, to
    empirically characterize non-uniqueness.

    WHY THIS MATTERS: if the inverse problem were perfectly well-posed,
    every run would converge to (approximately) the same Vs profile
    regardless of starting seed. The SPREAD across runs -- how much the
    recovered Vs values vary from run to run -- is a direct, empirical
    signature of how ill-posed the problem is for THIS particular
    dispersion curve and layer geometry. This spread is what Stage 3
    (integration) will use as the uncertainty band on the physics
    stage's output, before it gets fed into the ML model.

    Parameters
    ----------
    observed_curve : DispersionCurve
        The dispersion curve to invert repeatedly.
    thicknesses_m : np.ndarray
        Fixed, known layer thicknesses, m.
    n_runs : int, optional
        Number of independent inversion runs (each with a different
        random seed: 0, 1, 2, ..., n_runs-1). Defaults to 8 -- enough
        to see a meaningful spread without excessive runtime, since
        each run itself involves many forward-model evaluations inside
        differential_evolution's search.
    vs_bounds_mps : tuple(float, float), optional
        Passed through to each individual invert_dispersion_curve call.

    Returns
    -------
    list[InversionResult]
        One InversionResult per run, in seed order.
    """
    return [
        invert_dispersion_curve(
            observed_curve,
            thicknesses_m,
            vs_bounds_mps=vs_bounds_mps,
            random_seed=seed,
        )
        for seed in range(n_runs)
    ]


if __name__ == "__main__":
    from layered_model import LayeredEarthModel
    from forward_model import add_gaussian_noise

    # --- Set up the same synthetic ground truth and noisy observed
    # curve as forward_model.py's smoke test, so this script can be run
    # standalone and still validate against a known answer.
    true_model = LayeredEarthModel.example_marine_sediment_profile()
    thicknesses_m = true_model.thicknesses_m
    true_vs_mps = true_model.vs_profile_mps

    clean_curve = forward_model_dispersion_curve(true_model)
    observed_curve = add_gaussian_noise(clean_curve, noise_fraction=0.03, random_seed=42)

    print("=" * 70)
    print("SINGLE INVERSION RUN")
    print("=" * 70)
    single_result = invert_dispersion_curve(
        observed_curve, thicknesses_m, random_seed=0
    )
    print(f"True Vs (m/s):      {np.round(true_vs_mps, 1)}")
    print(f"Recovered Vs (m/s): {np.round(single_result.recovered_vs_mps, 1)}")
    print(f"Misfit RMSE (m/s):  {single_result.misfit_rmse_mps:.2f}")
    print(
        f"Per-layer error (%): "
        f"{np.round(100 * (single_result.recovered_vs_mps - true_vs_mps) / true_vs_mps, 1)}"
    )

    print()
    print("=" * 70)
    print("ENSEMBLE INVERSION (non-uniqueness check, 8 runs)")
    print("=" * 70)
    ensemble = run_ensemble_inversion(observed_curve, thicknesses_m, n_runs=8)
    ensemble_vs = np.array([r.recovered_vs_mps for r in ensemble])  # shape (8, 4)

    print(f"True Vs (m/s):           {np.round(true_vs_mps, 1)}")
    print(f"Ensemble mean Vs (m/s):  {np.round(ensemble_vs.mean(axis=0), 1)}")
    print(f"Ensemble std dev (m/s):  {np.round(ensemble_vs.std(axis=0), 1)}")
    print(
        "(Larger std dev per layer = less well-resolved by this "
        "dispersion curve -- typically expect the DEEPEST layer to be "
        "the least well-resolved, per surface-wave sensitivity "
        "physics.)"
    )

    print()
    print("=" * 70)
    print("SCENARIO 2: HARDER CASE (narrow frequency band + higher noise)")
    print("=" * 70)
    print(
        "Purpose: scenario 1's near-zero ensemble spread indicates a "
        "well-constrained problem, not an absence of non-uniqueness in "
        "general. This scenario deliberately reduces the information "
        "available to the inversion -- narrower frequency coverage "
        "(simulating a shorter/sparser real geophone array) and higher "
        "noise (simulating a lower-quality survey) -- to test whether "
        "non-uniqueness emerges under more realistic field constraints."
    )
    print()

    # Narrower frequency band: 8-25 Hz instead of the full 3-50 Hz.
    # This removes the very-low-frequency information that best
    # constrains the half-space, and the very-high-frequency
    # information that best constrains the shallowest layer.
    HARD_FREQ_MIN_HZ = 8.0
    HARD_FREQ_MAX_HZ = 25.0
    HARD_NOISE_FRACTION = 0.08  # 8%, vs. 3% in scenario 1

    hard_clean_curve = forward_model_dispersion_curve(
        true_model,
        freq_min_hz=HARD_FREQ_MIN_HZ,
        freq_max_hz=HARD_FREQ_MAX_HZ,
        n_frequencies=len(true_model.vs_profile_mps) * 10,  # keep similar
                                                              # point density
                                                              # over the
                                                              # narrower band
    )
    hard_observed_curve = add_gaussian_noise(
        hard_clean_curve, noise_fraction=HARD_NOISE_FRACTION, random_seed=42
    )

    print(
        f"Frequency band: {HARD_FREQ_MIN_HZ}-{HARD_FREQ_MAX_HZ} Hz "
        f"(vs. {clean_curve.frequency_hz.min():.0f}-"
        f"{clean_curve.frequency_hz.max():.0f} Hz in scenario 1)"
    )
    print(
        f"Noise level: {HARD_NOISE_FRACTION * 100:.0f}% "
        f"(vs. 3% in scenario 1)"
    )
    print()

    hard_ensemble = run_ensemble_inversion(
        hard_observed_curve, thicknesses_m, n_runs=8
    )
    hard_ensemble_vs = np.array([r.recovered_vs_mps for r in hard_ensemble])

    print(f"True Vs (m/s):           {np.round(true_vs_mps, 1)}")
    print(f"Ensemble mean Vs (m/s):  {np.round(hard_ensemble_vs.mean(axis=0), 1)}")
    print(f"Ensemble std dev (m/s):  {np.round(hard_ensemble_vs.std(axis=0), 1)}")
    print()
    print(
        "COMPARISON -- ensemble std dev per layer, scenario 1 vs. scenario 2:"
    )
    print(f"  Scenario 1 (wide band, low noise):  {np.round(ensemble_vs.std(axis=0), 2)}")
    print(f"  Scenario 2 (narrow band, high noise): {np.round(hard_ensemble_vs.std(axis=0), 2)}")
    print(
        "If scenario 2's std devs are meaningfully larger -- especially "
        "for the deepest layer(s) -- this confirms non-uniqueness "
        "emerging under reduced information, consistent with surface-"
        "wave sensitivity physics (deep layers are inherently less "
        "constrained by the available wavelength range)."
    )