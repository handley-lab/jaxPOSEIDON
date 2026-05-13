"""POSEIDON-vs-jaxposeidon comparison harness.

Mirrors the canonical setup in `POSEIDON/tests/test_TRIDENT.py::test_Rayleigh`
so any phase test can construct an identical POSEIDON config + jaxposeidon
config and compare outputs to the phase's tolerance target.

This module imports POSEIDON unconditionally. If POSEIDON isn't installed
the harness will fail at import — that is intentional (POSEIDON is the
numerical oracle and a hard test-time dependency).

Source mapping (POSEIDON repo at $POSEIDON_ROOT/POSEIDON/):
  create_star          core.py:80-310
  create_planet        core.py:313-378
  define_model         core.py:381-780
  read_opacities       core.py:854-993
  make_atmosphere      core.py:996-1252
  compute_spectrum     core.py:1303-2132
"""

import numpy as np


def canonical_rayleigh_config():
    """Reproduce POSEIDON/tests/test_TRIDENT.py::test_Rayleigh setup.

    Returns a dict with all the POSEIDON state objects needed for a
    forward-model call:
        star, planet, model, P, P_ref, R_p_ref, PT_params, log_X_params,
        wl, T_fine, log_P_fine, atmosphere.

    The Rayleigh-only configuration is intentional for Phase 0: it uses
    `testing=True` so it does not require the 70+ GB opacity database, only
    the CIA HDF5 (which POSEIDON unconditionally opens, see
    absorption.py:810-812).
    """
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (
        create_star, create_planet, define_model,
        read_opacities, make_atmosphere, wl_grid_constant_R,
    )

    # Star — POSEIDON signature: R_s, T_eff, log_g, Met
    star = create_star(R_Sun, 5000.0, 4.0, 0.0)

    planet = create_planet("Example Planet", R_J, mass=M_J, T_eq=1000.0)

    model = define_model("Only_Rayleigh", ["H2"], [], PT_profile="isotherm")

    # Atmosphere pressure grid: log-uniform from P_max (deepest) to P_min (top).
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    P_ref = 100.0
    R_p_ref = R_J

    PT_params = np.array([1000.0])
    log_X_params = np.array([])

    atmosphere = make_atmosphere(
        planet, model, P, P_ref, R_p_ref,
        PT_params, log_X_params,
        constant_gravity=True,
    )

    wl = wl_grid_constant_R(0.2, 10.0, 10000)

    T_fine = np.arange(900, 1110, 10)
    log_P_fine = np.arange(-6.0, 2.2, 0.2)

    opac = read_opacities(
        model, wl, "opacity_sampling", T_fine, log_P_fine,
        testing=True,
    )
    # Mirror POSEIDON/tests/test_TRIDENT.py:96-97 — zero CIA so the oracle is
    # a pure-H2 Rayleigh-scattering atmosphere as the test expects.
    opac["CIA_stored"] *= 0.0

    return dict(
        star=star, planet=planet, model=model,
        P=P, P_ref=P_ref, R_p_ref=R_p_ref,
        PT_params=PT_params, log_X_params=log_X_params,
        atmosphere=atmosphere, wl=wl,
        T_fine=T_fine, log_P_fine=log_P_fine,
        opac=opac,
    )


def poseidon_transmission_spectrum(cfg):
    """Run POSEIDON's compute_spectrum on a config dict from canonical_*."""
    from POSEIDON.core import compute_spectrum
    return compute_spectrum(
        cfg["planet"], cfg["star"], cfg["model"],
        cfg["atmosphere"], cfg["opac"], cfg["wl"],
        spectrum_type="transmission",
    )


def paired_transmission_spectra(jax_compute_spectrum, cfg=None):
    """Run POSEIDON and a jaxposeidon-side callable on the same config.

    Phase 0 calls this with `jax_compute_spectrum` raising
    NotImplementedError because the JAX forward model is not yet ported.
    Each subsequent phase progressively wires the JAX side until the
    paired call returns two spectra whose difference is within the
    phase's tolerance target.

    Args:
        jax_compute_spectrum: callable accepting the config dict and
            returning a 1D spectrum array.
        cfg: optional config dict; defaults to canonical_rayleigh_config().

    Returns:
        (spectrum_poseidon, spectrum_jax) — both 1D arrays on cfg['wl'].
    """
    if cfg is None:
        cfg = canonical_rayleigh_config()
    spectrum_pos = poseidon_transmission_spectrum(cfg)
    spectrum_jax = jax_compute_spectrum(cfg)
    return spectrum_pos, spectrum_jax
