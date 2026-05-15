"""POSEIDON setup API skeleton — Phase 0.5.2a.

Native ports of POSEIDON's setup-time helpers
(`create_star`, `create_planet`, `wl_grid_constant_R`) plus
thin orchestrators for `define_model`, `read_opacities`, and
`make_atmosphere` that delegate to POSEIDON in this sub-phase.
Subsequent phases (0.5.6 / 0.5.7 / 0.5.8 / 0.5.9 / 0.5.11 /
0.5.12 / 0.5.14 / 0.5.15) replace the POSEIDON delegations
with jaxPOSEIDON-native implementations.

Setup-only module: numpy / scipy / file I/O permitted. Allow-listed
by the v1 source-grep gate (see `CLAUDE.md`).

End-state (v0.5 complete): a user can `import jaxposeidon as jpo`
and run the full POSEIDON forward model + log-posterior boundary
without `import POSEIDON` at runtime. Phase 0.5.2a establishes the
public-API surface; deferred branches surface descriptive
`NotImplementedError` rather than `TypeError`.
"""

import numpy as np
import scipy.constants as sc


# ---------------------------------------------------------------------------
# wl_grid_constant_R — native port (POSEIDON core.py:783-815)
# ---------------------------------------------------------------------------
def wl_grid_constant_R(wl_min, wl_max, R):
    """Wavelength array with constant spectral resolution R = wl/dwl.

    Native port of POSEIDON `core.py:783-815`. Pure arithmetic.
    """
    delta_log_wl = 1.0 / R
    N_wl = (np.log(wl_max) - np.log(wl_min)) / delta_log_wl
    N_wl = np.around(N_wl).astype(np.int64)
    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), N_wl)
    wl = np.exp(log_wl)
    wl[0] = wl_min
    wl[-1] = wl_max
    return wl


# ---------------------------------------------------------------------------
# create_star — v0.5.2a skeleton (blackbody path only; non-blackbody to 0.5.11)
# ---------------------------------------------------------------------------
def create_star(
    R_s,
    T_eff,
    log_g,
    Met,
    T_eff_error=100.0,
    log_g_error=0.1,
    stellar_grid="blackbody",
    stellar_contam=None,
    f_het=None,
    T_het=None,
    log_g_het=None,
    f_spot=None,
    f_fac=None,
    T_spot=None,
    T_fac=None,
    log_g_spot=None,
    log_g_fac=None,
    wl=(),
    interp_backend="pysynphot",
    user_spectrum=(),
    user_wl=(),
):
    """Initialise the stellar dictionary used by POSEIDON.

    Phase 0.5.2a skeleton: only `stellar_grid='blackbody'` with
    `stellar_contam=None` is implemented natively. Non-blackbody grids
    (pysynphot / pymsg) and stellar contamination (one_spot /
    two_spots) are Phase 0.5.11.

    Native port of the blackbody path of POSEIDON `core.py:80-310`.
    Accepts the full POSEIDON kwarg surface; non-v0-skeleton paths
    raise descriptive `NotImplementedError`.
    """
    if stellar_contam is not None and stellar_contam not in (
        "one_spot",
        "two_spots",
        "three_spots",
    ):
        raise NotImplementedError(
            f"stellar_contam={stellar_contam!r} not in "
            "('one_spot', 'two_spots', 'three_spots')."
        )
    if stellar_grid not in ("blackbody", "custom"):
        raise NotImplementedError(
            f"stellar_grid={stellar_grid!r}: non-blackbody stellar grids "
            f"(pysynphot ICAT / PyMSG) are Phase 0.5.11."
        )
    if stellar_grid == "custom":
        raise NotImplementedError(
            "stellar_grid='custom' user-spectrum interpolation is Phase 0.5.11."
        )

    # Default stellar wavelength grid (POSEIDON core.py:152-166).
    if len(wl) == 0:
        wl_star = wl_grid_constant_R(0.2, 5.4, 20000)
    else:
        wl_star = np.asarray(wl)

    # Blackbody photosphere intensity via Planck function.
    I_phot = _planck_lambda(T_eff, wl_star)

    # Uniform stellar surface: F_star = π · I_phot.
    F_star = np.pi * I_phot

    # Stellar contamination heterogeneity intensities (blackbody only at
    # v1-D; pysynphot / PyMSG grids are a follow-up).
    I_het = None
    I_spot = None
    I_fac = None
    if stellar_contam == "one_spot" and T_het is not None:
        I_het = _planck_lambda(T_het, wl_star)
    elif stellar_contam in ("two_spots", "three_spots"):
        het_intensities = []
        if T_spot is not None:
            I_spot = _planck_lambda(T_spot, wl_star)
            het_intensities.append(I_spot)
        if T_fac is not None:
            I_fac = _planck_lambda(T_fac, wl_star)
            het_intensities.append(I_fac)
        if stellar_contam == "three_spots" and T_het is not None:
            het_intensities.append(_planck_lambda(T_het, wl_star))
        if het_intensities:
            I_het = np.stack(het_intensities, axis=0)

    return {
        "R_s": R_s,
        "T_eff": T_eff,
        "T_eff_error": T_eff_error,
        "log_g_error": log_g_error,
        "Met": Met,
        "log_g": log_g,
        "F_star": F_star,
        "wl_star": wl_star,
        "f_het": f_het,
        "T_het": T_het,
        "log_g_het": log_g_het,
        "f_spot": f_spot,
        "T_spot": T_spot if T_spot is not None else T_het,
        "log_g_spot": log_g_spot if log_g_spot is not None else log_g_het,
        "f_fac": f_fac,
        "T_fac": T_fac if T_fac is not None else T_het,
        "log_g_fac": log_g_fac if log_g_fac is not None else log_g_het,
        "I_phot": I_phot,
        "I_het": I_het,
        "I_spot": I_spot,
        "I_fac": I_fac,
        "stellar_grid": stellar_grid,
        "stellar_interp_backend": interp_backend,
        "stellar_contam": stellar_contam,
    }


def _planck_lambda(T, wl_um):
    """Planck spectral radiance B_λ(T) on a wavelength grid in μm.

    Returns intensity in W / (m² · sr · m), matching POSEIDON.
    """
    wl_m = wl_um * 1e-6
    h, c, k = sc.h, sc.c, sc.k
    return (2.0 * h * c**2) / (wl_m**5) / (np.exp(h * c / (wl_m * k * T)) - 1.0)


# ---------------------------------------------------------------------------
# create_planet — v0.5.2a skeleton (fixed gravity/mass; free → 0.5.2b)
# ---------------------------------------------------------------------------
def create_planet(
    planet_name,
    R_p,
    mass=None,
    gravity=None,
    log_g=None,
    T_eq=None,
    d=None,
    d_err=None,
    b_p=0.0,
    a_p=None,
):
    """Initialise the planet dictionary used by POSEIDON.

    Native port of POSEIDON `core.py:313-378` for the fixed-gravity
    / fixed-mass path. Free `gravity_setting` / `mass_setting`
    parameterizations are Phase 0.5.2b.
    """
    if (gravity is None) and (log_g is None) and (mass is None):
        raise Exception("At least one of Mass or gravity must be specified.")

    if gravity is None:
        if log_g is not None:
            gravity = np.power(10.0, log_g) / 100.0
        elif mass is not None:
            gravity = (sc.G * mass) / (R_p**2)

    if (mass is None) and (gravity is not None):
        mass = (gravity * R_p**2) / sc.G

    return {
        "planet_name": planet_name,
        "planet_radius": R_p,
        "planet_mass": mass,
        "planet_gravity": gravity,
        "planet_T_eq": T_eq,
        "planet_impact_parameter": b_p,
        "system_distance": d,
        "system_distance_error": d_err,
        "planet_semi_major_axis": a_p,
    }


# ---------------------------------------------------------------------------
# define_model, read_opacities, make_atmosphere — 0.5.2a thin orchestrators
# ---------------------------------------------------------------------------
# These three delegate to POSEIDON in 0.5.2a. Subsequent phases lift
# the POSEIDON dependency:
#   - define_model: 0.5.6 + 0.5.7 + 0.5.9 + 0.5.12 + 0.5.14 + 0.5.16
#     extend the kwarg surface as guards are lifted; the parameter
#     ordering lives in _parameter_setup.assign_free_params.
#   - read_opacities: 0.5.4 (ff/bf), 0.5.12 (Mie/aerosol), 0.5.15 (LBL)
#     extend; v0 opacity-sampling base path is already in
#     _opacity_precompute.
#   - make_atmosphere: 0.5.6 (non-v0 PT), 0.5.7 (gradient chem),
#     0.5.8 (FastChem), 0.5.9 (2D/3D) extend; v0 is _atmosphere.profiles.


def define_model(*args, **kwargs):
    """Thin POSEIDON delegator. Replaced by jaxPOSEIDON-native
    implementation across Phases 0.5.6 - 0.5.16."""
    from POSEIDON.core import define_model as _p

    return _p(*args, **kwargs)


def read_opacities(*args, **kwargs):
    """Thin POSEIDON delegator. Replaced by jaxPOSEIDON-native
    implementation across Phases 0.5.4 (ff/bf), 0.5.12 (Mie),
    0.5.15 (LBL)."""
    from POSEIDON.core import read_opacities as _p

    return _p(*args, **kwargs)


def make_atmosphere(*args, **kwargs):
    """Thin POSEIDON delegator. Replaced by jaxPOSEIDON-native
    implementation across Phases 0.5.6 (non-v0 PT), 0.5.7 (gradient
    chem), 0.5.8 (FastChem), 0.5.9 (2D/3D)."""
    from POSEIDON.core import make_atmosphere as _p

    return _p(*args, **kwargs)
