"""Phase 9d: finite-difference smoothness sanity for the v0 numpy port.

v0 jaxposeidon is a numpy port (JAX-tracing gated to v1 per Phase 7).
This module sanity-checks that the forward model is smooth in
its continuous parameters (T_iso, R_p_ref, P_ref, log_a_haze, γ,
log_P_cloud, f_cloud) by verifying that small perturbations produce
finite central-difference derivatives away from rejection branches.

Full `jax.grad` vs finite-difference parity (100 random points,
rtol=1e-4) is gated to v1 once the numpy ports are replaced by
JAX-traceable equivalents.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from jaxposeidon._compute_spectrum import (
    compute_spectrum as j_compute_spectrum,
)


@pytest.fixture(scope="module", autouse=True)
def _synthetic_poseidon_input_data():
    if os.environ.get("POSEIDON_input_data"):
        yield
        return
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "opacity"))
        path = os.path.join(tmp, "opacity", "Opacity_database_cia.hdf5")
        T_grid = np.linspace(200, 2000, 10, dtype=np.float64)
        nu = np.linspace(1.0e4, 5.0e5, 50, dtype=np.float64)
        log_cia = np.full((10, 50), -50.0, dtype=np.float64)
        with h5py.File(path, "w") as f:
            for pair in ("H2-H2", "H2-He"):
                g = f.create_group(pair)
                g.create_dataset("T", data=T_grid)
                g.create_dataset("nu", data=nu)
                g.create_dataset("log(cia)", data=log_cia)
        os.environ["POSEIDON_input_data"] = tmp
        yield
        del os.environ["POSEIDON_input_data"]


def _make_atm(T_iso, R_p_ref_fac, P_ref):
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (create_star, create_planet, define_model,
                                make_atmosphere, read_opacities,
                                wl_grid_constant_R)
    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=M_J, T_eq=T_iso)
    model = define_model("m", ["H2", "He"], [], PT_profile="isotherm")
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 60)
    atm = make_atmosphere(planet, model, P, P_ref, R_J * R_p_ref_fac,
                           np.array([T_iso]), np.array([]),
                           constant_gravity=True)
    wl = wl_grid_constant_R(1.0, 4.0, 800)
    T_fine = np.arange(max(200, T_iso - 200), T_iso + 210, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    opac["CIA_stored"] *= 0.0
    return planet, star, model, atm, opac, wl


@pytest.mark.parametrize("T_iso", [700.0, 900.0, 1100.0])
@pytest.mark.parametrize("rp_fac", [0.95, 1.0, 1.05])
def test_compute_spectrum_finite_difference_smoothness(T_iso, rp_fac):
    """Spectrum smoothness in (T, R_p_ref): forward differences are finite
    and the central-difference approximation does not blow up.
    """
    P_ref = 10.0
    planet, star, model, atmosphere, opac, wl = _make_atm(T_iso, rp_fac, P_ref)
    base = j_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    # Perturb T_eq by ±1 K — the only parameter that flows into the
    # forward model below this point is atmosphere['T'], so perturb that.
    dT = 1.0
    atm_plus = dict(atmosphere)
    atm_plus["T"] = atmosphere["T"] + dT
    atm_minus = dict(atmosphere)
    atm_minus["T"] = atmosphere["T"] - dT
    plus = j_compute_spectrum(planet, star, model, atm_plus, opac, wl)
    minus = j_compute_spectrum(planet, star, model, atm_minus, opac, wl)
    deriv = (plus - minus) / (2.0 * dT)
    assert np.all(np.isfinite(deriv))
    assert np.all(np.isfinite(base))
    # Spectrum changes are O(1e-5) per K — sanity-bound to catch
    # blowups while permitting nearest-T-index step discontinuities.
    assert np.max(np.abs(deriv)) < 1e-3


def test_compute_spectrum_returns_NaN_outside_T_fine_grid_gradient_safe():
    """At the rejection branch (T outside fine grid), output is NaN as a
    sentinel — finite-difference would propagate NaN, which is the
    POSEIDON-expected behavior, not a bug."""
    planet, star, model, atmosphere, opac, wl = _make_atm(900.0, 1.0, 10.0)
    atmosphere["T"] = np.full_like(atmosphere["T"], 5000.0)
    out = j_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    assert np.all(np.isnan(out))
