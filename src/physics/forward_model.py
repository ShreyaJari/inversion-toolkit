"""
forward_model.py

Forward-models the Rayleigh-wave fundamental-mode dispersion curve for a
given LayeredEarthModel, using the `disba` package (Thomson-Haskell /
Dunkin propagator-matrix method).

PHYSICAL BACKGROUND
--------------------
"Forward modeling" here means: given a KNOWN subsurface Vs(z) [plus Vp
and density], compute what a MASW survey would measure -- the Rayleigh
phase velocity as a function of frequency. This is the well-posed,
unique direction of the problem. The next script (inversion.py) solves
the ill-posed, non-unique REVERSE direction: given a measured dispersion
curve, recover Vs(z).

Rayleigh waves are dispersive in a layered medium because different
frequencies (equivalently, different wavelengths) sample different
depths: low frequencies (long wavelengths) penetrate deep and are
sensitive to deep, generally stiffer, layers -- so they travel faster.
High frequencies (short wavelengths) are confined near the surface and
are sensitive mainly to the near-surface, generally softer, layers -- so
they travel slower. This frequency-dependence of phase velocity (i.e.
dispersion) is the signal that lets us infer Vs(z) at all.

DEPTH OF INVESTIGATION AND DEFAULT FREQUENCY RANGE
-----------------------------------------------------
Rule of thumb in surface-wave methods: a wavelength lambda "feels" the
subsurface down to roughly lambda / 2 to lambda / 3 (this is the
"depth of investigation" heuristic used throughout MASW literature).

Given our synthetic profile extends to ~15 m before the half-space
(Vs 100-400 m/s):
    - Resolving near-surface (~1 m) needs SHORT wavelengths -> HIGH
      frequencies (~40-50 Hz)
    - Resolving down to ~30-35 m (a bit beyond the last finite layer,
      so the inversion has some visibility into the half-space
      contrast) needs LONG wavelengths -> LOW frequencies (~3-5 Hz)

This is why DEFAULT_FREQ_RANGE_HZ below is (3.0, 50.0) -- it is chosen
to match the depth range of the synthetic model, not picked arbitrarily,
and is also realistic for what geophones in a real shallow-MASW field
array actually record.

FUNDAMENTAL MODE ONLY -- A STATED SIMPLIFICATION
----------------------------------------------------
A layered medium supports multiple dispersion "modes" (fundamental,
first higher mode, etc.) -- multiple frequency-dependent solutions can
satisfy the Rayleigh secular equation at a given frequency. Real MASW
field data sometimes contains higher-mode energy, and mode
misidentification is a well-known practical difficulty in the field.

This toolkit restricts itself to the FUNDAMENTAL MODE ONLY, which is
the standard simplifying assumption for a basic surface-wave inversion
workflow. This keeps the inverse problem (next script) tractable and
matches how most introductory/standard MASW processing is done. It is
a genuine scope limitation, not a hidden one -- see project README,
"out of scope" section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from disba import PhaseDispersion

from layered_model import LayeredEarthModel


# ---------------------------------------------------------------------------
# Default frequency sampling -- see module docstring for the depth-of-
# investigation reasoning behind these bounds.
# ---------------------------------------------------------------------------
DEFAULT_FREQ_MIN_HZ = 3.0
DEFAULT_FREQ_MAX_HZ = 50.0
DEFAULT_N_FREQUENCIES = 40


@dataclass
class DispersionCurve:
    """
    A Rayleigh-wave fundamental-mode dispersion curve: phase velocity as
    a function of frequency (equivalently, period).

    Storing both frequency and period (rather than just one, with the
    other derived on demand) is a deliberate convenience choice --
    frequency is the more natural axis for describing/plotting MASW
    results, but period is what disba's API actually consumes/returns
    internally, so having both readily available avoids repeated
    conversions scattered through downstream code.

    Attributes
    ----------
    frequency_hz : np.ndarray
        Frequency, Hz. Ascending order.
    period_s : np.ndarray
        Period, s. Equal to 1 / frequency_hz, descending order
        (since period and frequency are inversely related, ascending
        frequency means descending period).
    phase_velocity_mps : np.ndarray
        Rayleigh-wave fundamental-mode phase velocity, m/s, one value
        per frequency/period.
    """

    frequency_hz: np.ndarray
    period_s: np.ndarray
    phase_velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        # All three arrays must be the same length and line up index-
        # for-index -- catch mismatches early rather than let them
        # surface confusingly downstream (e.g. in a misfit calculation).
        lengths = {
            len(self.frequency_hz),
            len(self.period_s),
            len(self.phase_velocity_mps),
        }
        if len(lengths) != 1:
            raise ValueError(
                "frequency_hz, period_s, and phase_velocity_mps must all "
                f"have the same length; got lengths {lengths}."
            )


def forward_model_dispersion_curve(
    model: LayeredEarthModel,
    freq_min_hz: float = DEFAULT_FREQ_MIN_HZ,
    freq_max_hz: float = DEFAULT_FREQ_MAX_HZ,
    n_frequencies: int = DEFAULT_N_FREQUENCIES,
) -> DispersionCurve:
    """
    Compute the Rayleigh-wave fundamental-mode dispersion curve for a
    given layered-earth model, using disba's PhaseDispersion solver.

    Parameters
    ----------
    model : LayeredEarthModel
        The subsurface model to forward-model. Converted internally to
        disba's expected [thickness_km, vp_km/s, vs_km/s, density_g/cm3]
        array via model.to_disba_units().
    freq_min_hz, freq_max_hz : float, optional
        Frequency range to compute the dispersion curve over. Defaults
        chosen to match this toolkit's synthetic depth range -- see
        module docstring. Override if you build a model with a very
        different depth range (e.g. a deeper or shallower target).
    n_frequencies : int, optional
        Number of frequency points to sample, log-spaced (log-spacing
        because MASW/surface-wave dispersion curves are typically
        smoother and more evenly informative in log-frequency space --
        equal spacing in linear Hz over-samples the high-frequency end
        and under-samples the low-frequency end relative to how much
        NEW depth information each additional point adds).

    Returns
    -------
    DispersionCurve

    Notes
    -----
    disba's PhaseDispersion is called with mode=0, which by convention
    is the FUNDAMENTAL mode (mode=1 would be the first higher mode, and
    so on) -- see module docstring, "fundamental mode only" section.
    """
    # disba works in period (s), not frequency (Hz) -- log-spaced in
    # frequency also gives log-spacing in period, so we sample frequency
    # first (more intuitive) and derive period from it.
    frequency_hz = np.logspace(
        np.log10(freq_min_hz), np.log10(freq_max_hz), n_frequencies
    )
    period_s = 1.0 / frequency_hz

    # disba requires periods in ASCENDING order for internal root-
    # tracking to work reliably (it walks from long to short period,
    # using the previous solution as a starting guess for the next --
    # this is a standard root-continuation trick to avoid mode-jumping).
    # Since period is inversely related to frequency, ascending
    # frequency means DESCENDING period, so we sort explicitly here
    # rather than assume the log-space array is already in the order
    # disba wants.
    sort_idx = np.argsort(period_s)
    period_s_sorted = period_s[sort_idx]

    disba_model = model.to_disba_units()
    thickness_km, vp_kms, vs_kms, density_gcm3 = disba_model.T

    pd_solver = PhaseDispersion(thickness_km, vp_kms, vs_kms, density_gcm3)

    result = pd_solver(period_s_sorted, mode=0, wave="rayleigh")
    # disba's return object exposes .period (s) and .velocity (km/s),
    # already in fundamental-mode order matching period_s_sorted.

    phase_velocity_mps = np.asarray(result.velocity) * 1000.0  # km/s -> m/s
    period_s_out = np.asarray(result.period)
    frequency_hz_out = 1.0 / period_s_out

    # Re-sort back into ascending-FREQUENCY order for the returned
    # DispersionCurve, since that is the more natural axis for plotting
    # and for downstream misfit calculations against "observed" data
    # (which will also be frequency-indexed).
    freq_sort_idx = np.argsort(frequency_hz_out)

    return DispersionCurve(
        frequency_hz=frequency_hz_out[freq_sort_idx],
        period_s=period_s_out[freq_sort_idx],
        phase_velocity_mps=phase_velocity_mps[freq_sort_idx],
    )


def add_gaussian_noise(
    curve: DispersionCurve,
    noise_fraction: float = 0.03,
    random_seed: Optional[int] = None,
) -> DispersionCurve:
    """
    Add Gaussian noise to a dispersion curve's phase velocities, to
    simulate realistic MASW picking uncertainty.

    WHY THIS NOISE LEVEL: real MASW dispersion-curve picking (whether
    manual or automated, from a frequency-velocity spectrum image)
    typically carries on the order of 2-5% uncertainty in picked phase
    velocity, driven by spectral peak width, geophone spacing/array
    geometry, and picking method. We default to 3% (noise_fraction=0.03)
    as a representative mid-range value -- NOT zero (which would make
    the inversion trivially recover the exact truth and not test
    anything realistic) and not an extreme value either.

    This function does NOT modify the input curve in place -- it returns
    a new DispersionCurve, so you always retain the noise-free "true"
    curve for comparison/validation (this matters for the validation
    step: we need both the noisy "observed" curve fed to the inversion,
    AND the clean curve to check the inversion's fit against).

    Parameters
    ----------
    curve : DispersionCurve
        The (presumably noise-free, forward-modeled) curve to add noise
        to.
    noise_fraction : float, optional
        Standard deviation of the Gaussian noise, as a fraction of each
        point's phase velocity (i.e. noise scales with signal magnitude,
        not a fixed absolute value in m/s -- this matches how picking
        uncertainty behaves in practice, since faster/low-frequency
        picks and slower/high-frequency picks aren't equally uncertain
        in absolute terms). Defaults to 0.03 (3%) -- see rationale above.
    random_seed : int, optional
        Seed for reproducibility. If None, noise will differ each call
        (useful later for the non-uniqueness/ensemble analysis, where we
        WANT different noise realizations across runs).

    Returns
    -------
    DispersionCurve
        A new curve with the same frequency/period arrays but noisy
        phase_velocity_mps.
    """
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(
        loc=0.0,
        scale=noise_fraction * curve.phase_velocity_mps,
    )
    return DispersionCurve(
        frequency_hz=curve.frequency_hz.copy(),
        period_s=curve.period_s.copy(),
        phase_velocity_mps=curve.phase_velocity_mps + noise,
    )


if __name__ == "__main__":
    # Quick smoke test: forward-model the example marine-sediment
    # profile from layered_model.py, then add noise, and print a
    # summary so we can sanity-check before wiring in the inversion.
    model = LayeredEarthModel.example_marine_sediment_profile()

    clean_curve = forward_model_dispersion_curve(model)
    noisy_curve = add_gaussian_noise(clean_curve, noise_fraction=0.03, random_seed=42)

    print("Frequency (Hz):        ", np.round(clean_curve.frequency_hz, 2))
    print("Clean phase vel (m/s): ", np.round(clean_curve.phase_velocity_mps, 2))
    print("Noisy phase vel (m/s): ", np.round(noisy_curve.phase_velocity_mps, 2))
    print()
    print(
        "Sanity check: phase velocity should generally INCREASE with "
        "increasing period / decreasing frequency (long wavelengths feel "
        "the deeper, stiffer half-space)."
    )
    print(
        "Low-freq (long-period) phase vel:",
        round(clean_curve.phase_velocity_mps[0], 2),
        "m/s  vs.  High-freq (short-period) phase vel:",
        round(clean_curve.phase_velocity_mps[-1], 2),
        "m/s",
    )