"""
layered_model.py

Defines a 1-D layered-earth model for MASW-style surface-wave dispersion
analysis, as used in Stage 1 (physics) of the Geophysical-to-Geotechnical
Inversion Toolkit.

PHYSICAL BACKGROUND
--------------------
Surface-wave (Rayleigh-wave) dispersion is controlled by how shear-wave
velocity (Vs) varies with depth. To compute a synthetic dispersion curve
from a known subsurface, we need to fully specify, for each layer:

    - thickness      (m)
    - Vs             (m/s)  -- the quantity we ultimately want to recover
    - Vp             (m/s)  -- needed because Rayleigh waves are a coupled
                                 P-SV phenomenon; dispersion depends on the
                                 Vp/Vs ratio (equivalently, Poisson's ratio),
                                 not on Vs alone
    - density (rho)  (kg/m^3) -- needed because the elastic moduli that
                                 govern wave propagation are rho-dependent

The deepest layer is always a HALF-SPACE (effectively infinite thickness).
This is standard in surface-wave theory: no finite survey can resolve
depth beyond roughly (dominant wavelength / 2 to 3), and mathematically
the half-space provides the correct radiation boundary condition (energy
propagates downward and away, rather than artificially reflecting off a
fake bottom boundary).

UNIT CONVENTIONS
-----------------
This class stores everything in SI units (m, m/s, kg/m^3) since that is
the natural unit system for geotechnical work and keeps the model
human-readable. A conversion method (`to_disba_units`) is provided
separately for when we hand this off to the forward-modeling library in
the next script, which expects km, km/s, and g/cm^3.

DEFAULT Vp/Vs AND DENSITY
---------------------------
If Vp and density are not explicitly supplied for a layer, we estimate
them from Vs using simple empirical relations, clearly documented below.
These are SIMPLIFICATIONS -- real Vp/Vs and density depend on saturation,
lithology, stress state, and more. We flag this explicitly rather than
silently treating the defaults as ground truth, because if you (or a
reader) later change these defaults, the resulting dispersion curves
will change too, and that dependency should be visible, not hidden.

    - Vp/Vs ratio: defaults to 2.0, appropriate for shallow, water-
      saturated, unconsolidated marine sediment. (Pore water strongly
      raises Vp but has little effect on Vs, since shear waves cannot
      propagate through fluid -- this is why saturated soils have much
      higher Vp/Vs than typical dry rock, where 1.6-1.8 is more common.)

    - Density: estimated via a Gardner-type power-law relation adapted
      for soils,
          rho (kg/m^3) = 1000 + 0.8 * Vs (m/s)
      This is a rough approximation calibrated to give ~1400-1900 kg/m^3
      over the 100-400 m/s Vs range typical of marine sediments -- it is
      NOT a rigorously derived geotechnical correlation, just a
      physically-reasonable placeholder. Override with real values
      wherever you have them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Empirical defaults used when Vp / density are not explicitly specified.
# Pulled out as module-level constants (rather than buried in a method) so
# they are easy to find, cite, and override.
# ---------------------------------------------------------------------------
DEFAULT_VP_VS_RATIO = 2.0          # dimensionless, saturated marine sediment
DENSITY_INTERCEPT_KGM3 = 1000.0    # kg/m^3, ~density of water/very soft mud
DENSITY_VS_SLOPE = 0.8             # kg/m^3 per (m/s) of Vs


def estimate_vp_from_vs(vs_mps: float, vp_vs_ratio: float = DEFAULT_VP_VS_RATIO) -> float:
    """
    Estimate Vp from Vs using a fixed Vp/Vs ratio.

    Parameters
    ----------
    vs_mps : float
        Shear-wave velocity, m/s.
    vp_vs_ratio : float, optional
        Vp/Vs ratio to assume. Defaults to DEFAULT_VP_VS_RATIO (2.0),
        appropriate for shallow saturated marine sediment.

    Returns
    -------
    float
        Estimated Vp, m/s.
    """
    return vs_mps * vp_vs_ratio


def estimate_density_from_vs(vs_mps: float) -> float:
    """
    Estimate bulk density from Vs using a simple linear (Gardner-type)
    relation calibrated for soft marine sediment. See module docstring
    for the caveat that this is a rough placeholder, not a rigorously
    derived correlation.

    Parameters
    ----------
    vs_mps : float
        Shear-wave velocity, m/s.

    Returns
    -------
    float
        Estimated bulk density, kg/m^3.
    """
    return DENSITY_INTERCEPT_KGM3 + DENSITY_VS_SLOPE * vs_mps


@dataclass
class Layer:
    """
    A single horizontal layer in the 1-D layered-earth model.

    Attributes
    ----------
    thickness_m : float
        Layer thickness in metres. Must be strictly positive for all
        layers except the final (half-space) layer -- see
        LayeredEarthModel.is_halfspace_layer for how the half-space is
        represented.
    vs_mps : float
        Shear-wave velocity, m/s. This is the target quantity of the
        whole physics stage.
    vp_mps : Optional[float]
        Compressional-wave velocity, m/s. If not supplied, is
        auto-estimated from vs_mps via estimate_vp_from_vs() at
        construction time (see LayeredEarthModel.__post_init__).
    density_kgm3 : Optional[float]
        Bulk density, kg/m^3. If not supplied, is auto-estimated from
        vs_mps via estimate_density_from_vs().
    """

    thickness_m: float
    vs_mps: float
    vp_mps: Optional[float] = None
    density_kgm3: Optional[float] = None

    def __post_init__(self) -> None:
        # Basic physical sanity checks. These are deliberately simple
        # (not a full validation framework) -- the goal is to catch
        # obvious unit-confusion mistakes (e.g. accidentally passing Vs
        # in km/s) early, not to build a general-purpose validator.
        if self.vs_mps <= 0:
            raise ValueError(
                f"vs_mps must be positive, got {self.vs_mps}. "
                "(If this looks like a km/s value, convert to m/s.)"
            )
        if self.vs_mps > 5000:
            raise ValueError(
                f"vs_mps = {self.vs_mps} is implausibly high for a "
                "near-surface sediment/rock layer (>5000 m/s is in the "
                "range of dense crystalline rock, not typical MASW "
                "targets). Check units."
            )

        # Auto-fill Vp and density from Vs if not explicitly given.
        if self.vp_mps is None:
            self.vp_mps = estimate_vp_from_vs(self.vs_mps)
        if self.density_kgm3 is None:
            self.density_kgm3 = estimate_density_from_vs(self.vs_mps)

        if self.vp_mps <= self.vs_mps:
            # Vp must exceed Vs for a physically valid Poisson solid
            # (Poisson's ratio must be positive and less than 0.5).
            raise ValueError(
                f"vp_mps ({self.vp_mps}) must be greater than vs_mps "
                f"({self.vs_mps}) -- Vp < Vs is not physically valid."
            )


@dataclass
class LayeredEarthModel:
    """
    A full 1-D layered-earth model: an ordered stack of Layer objects
    from the surface downward, where the LAST layer is always treated
    as a half-space (infinite thickness) regardless of what
    thickness_m is set to for it.

    Why is the half-space's thickness_m not simply ignored/set to zero?
    We still store a nominal thickness_m for the last layer (used only
    for plotting a sensible "extend the last layer downward by this much
    on the figure" cutoff) -- but every physics/forward-modeling
    calculation treats it as extending to infinity. This is called out
    explicitly in `depths_m` and in the forward-model script, so there
    is no ambiguity about which number means what.

    Attributes
    ----------
    layers : list[Layer]
        Ordered list of layers, surface (index 0) to depth. Must
        contain at least 2 layers (at least one finite layer plus the
        half-space) -- a single-half-space model has no dispersion to
        speak of, since dispersion arises from velocity CONTRASTS
        between layers.
    """

    layers: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.layers) < 2:
            raise ValueError(
                "A LayeredEarthModel needs at least 2 layers (one finite "
                "layer plus the half-space) -- surface-wave dispersion "
                "arises from velocity contrasts between layers, so a "
                "single homogeneous half-space produces no dispersion."
            )

    @property
    def n_layers(self) -> int:
        """Total number of layers, including the half-space."""
        return len(self.layers)

    @property
    def vs_profile_mps(self) -> np.ndarray:
        """Array of Vs values, one per layer, surface to depth (m/s)."""
        return np.array([layer.vs_mps for layer in self.layers])

    @property
    def vp_profile_mps(self) -> np.ndarray:
        """Array of Vp values, one per layer, surface to depth (m/s)."""
        return np.array([layer.vp_mps for layer in self.layers])

    @property
    def density_profile_kgm3(self) -> np.ndarray:
        """Array of density values, one per layer, surface to depth (kg/m^3)."""
        return np.array([layer.density_kgm3 for layer in self.layers])

    @property
    def thicknesses_m(self) -> np.ndarray:
        """
        Array of layer thicknesses (m), surface to depth. Note that the
        LAST value is the half-space's *nominal* thickness (used only
        for plotting), not a physically meaningful finite thickness --
        see class docstring.
        """
        return np.array([layer.thickness_m for layer in self.layers])

    def depths_m(self) -> np.ndarray:
        """
        Cumulative depth to the TOP of each layer (m), surface to depth.
        depths_m()[0] is always 0.0 (the ground/seabed surface).

        This is what you plot Vs against on a step-plot y-axis (inverted,
        since it's a subsurface depth axis).

        Returns
        -------
        np.ndarray, shape (n_layers,)
        """
        # Cumulative sum of all thicknesses EXCEPT we want the depth to
        # the TOP of each layer, so we prepend a 0 and drop the last
        # cumulative value (which would be the depth to the bottom of
        # the half-space's nominal thickness -- not physically meaningful).
        top_depths = np.concatenate([[0.0], np.cumsum(self.thicknesses_m)[:-1]])
        return top_depths

    def to_disba_units(self) -> np.ndarray:
        """
        Convert this model into the units/array format expected by the
        `disba` forward-modeling library (used in the next script):
        columns [thickness_km, vp_kms, vs_kms, density_gcm3], one row
        per layer, surface to depth.

        disba convention: the half-space's thickness is conventionally
        given as 0.0 (a flag disba interprets as "extends to infinity"),
        NOT the nominal thickness_m we store for plotting purposes. This
        is the one place that distinction actually matters numerically,
        so it is handled explicitly here rather than left implicit.

        Returns
        -------
        np.ndarray, shape (n_layers, 4)
            Columns: [thickness_km, vp_km_s, vs_km_s, density_g_cm3]
        """
        n = self.n_layers
        model = np.zeros((n, 4))

        # thickness: m -> km, except the half-space row gets 0.0
        model[:, 0] = self.thicknesses_m / 1000.0
        model[-1, 0] = 0.0  # disba's "infinite half-space" flag

        model[:, 1] = self.vp_profile_mps / 1000.0        # m/s -> km/s
        model[:, 2] = self.vs_profile_mps / 1000.0         # m/s -> km/s
        model[:, 3] = self.density_profile_kgm3 / 1000.0   # kg/m^3 -> g/cm^3

        return model

    @classmethod
    def example_marine_sediment_profile(cls) -> "LayeredEarthModel":
        """
        Construct a representative synthetic 4-layer marine-sediment Vs
        profile, used as the "known ground truth" for validating the
        forward-model + inversion round trip in this stage.

        Profile rationale (not arbitrary numbers):
        Layer 1 (0-3 m):    Vs = 100 m/s   -- very soft near-surface mud/clay
        Layer 2 (3-8 m):    Vs = 180 m/s   -- soft-to-firm clay/silt
        Layer 3 (8-15 m):   Vs = 280 m/s   -- denser silty sand
        Half-space (15+ m): Vs = 400 m/s   -- stiffer sand/consolidated clay

        This increasing-with-depth trend is typical of normally
        consolidated marine sediment (Vs generally increases with
        effective overburden stress, i.e. with depth), which is why we
        chose it as the default synthetic case for a project framed
        around offshore/seabed characterization -- see project README
        for the broader motivation.

        Returns
        -------
        LayeredEarthModel
        """
        return cls(layers=[
            Layer(thickness_m=3.0, vs_mps=100.0),
            Layer(thickness_m=5.0, vs_mps=180.0),
            Layer(thickness_m=7.0, vs_mps=280.0),
            Layer(thickness_m=20.0, vs_mps=400.0),  # half-space; thickness_m
                                                      # is nominal (plotting only)
        ])


if __name__ == "__main__":
    # Quick smoke test: build the example profile and print a summary,
    # so you can run this file directly (`python layered_model.py`) to
    # sanity-check it before wiring it into the forward model.
    model = LayeredEarthModel.example_marine_sediment_profile()

    print(f"Number of layers (incl. half-space): {model.n_layers}")
    print(f"Depths to top of each layer (m):     {model.depths_m()}")
    print(f"Vs profile (m/s):                     {model.vs_profile_mps}")
    print(f"Vp profile (m/s, auto-estimated):     {model.vp_profile_mps}")
    print(f"Density profile (kg/m^3, auto-est.):  {model.density_profile_kgm3}")
    print()
    print("disba-format array [thickness_km, vp_km/s, vs_km/s, density_g/cm3]:")
    print(model.to_disba_units())