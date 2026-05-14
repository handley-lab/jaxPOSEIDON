"""Phase 10b: retrieval driver wiring smoke tests.

Verifies that:
  - `_retrieval.make_loglikelihood` returns a callable that runs the
    full prior → spectrum → bin → loglikelihood chain end-to-end on
    the canonical K2-18b-style v0 envelope.
  - `_retrieval.run_NSS` raises NotImplementedError until the JAX-
    traceable forward model lands (v1).
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from jaxposeidon import _retrieval, _parameters


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


def test_run_NSS_is_gated_to_v1():
    with pytest.raises(NotImplementedError, match="v1"):
        _retrieval.run_NSS(lambda u: 0.0, n_dim=3)


def test_make_loglikelihood_end_to_end():
    """End-to-end: build a logp closure and call it on a unit cube. The
    result must be a finite float (or POSEIDON's -1e100 sentinel)."""
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (create_star, create_planet, define_model,
                                make_atmosphere, read_opacities,
                                wl_grid_constant_R)
    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=M_J, T_eq=900.0)
    model = define_model("m", ["H2", "He"], [], PT_profile="isotherm")
    wl = wl_grid_constant_R(1.0, 3.0, 500)
    T_fine = np.arange(700, 1110, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    opac["CIA_stored"] *= 0.0

    # Fake "binned data" properties — single dataset, no offsets, no inflation.
    n_bins = 20
    bin_cent = np.linspace(int(0.1 * len(wl)), int(0.9 * len(wl)),
                            n_bins).astype(np.int64)
    bin_left = np.maximum(bin_cent - 4, 0)
    bin_right = np.minimum(bin_cent + 4, len(wl) - 1)
    sigma = np.full(n_bins, 1.2)
    sens = np.ones(len(wl))
    norm = np.array([np.trapezoid(sens[bl:br], wl[bl:br])
                     for bl, br in zip(bin_left, bin_right)])
    ydata = np.full(n_bins, 2.7e-3)
    err_data = np.full(n_bins, 1.0e-4)
    data_properties = dict(
        datasets=["d"], instruments=["JWST_NIRSpec_PRISM"],
        wl_data=wl[bin_cent], half_bin=np.full(n_bins, 0.02),
        ydata=ydata, err_data=err_data,
        sens=sens, psf_sigma=sigma, fwhm=np.full(n_bins, 0.02),
        bin_left=bin_left, bin_cent=bin_cent, bin_right=bin_right,
        norm=norm, len_data_idx=np.array([0, n_bins]),
        offset_start=0, offset_end=0,
        offset_1_start=0, offset_1_end=0,
        offset_2_start=0, offset_2_end=0,
        offset_3_start=0, offset_3_end=0,
    )

    # Trivial 2-parameter wrapper: (R_p_ref, T_iso); both uniform.
    param_names = ["R_p_ref", "T"]
    prior_types = {"R_p_ref": "uniform", "T": "uniform"}
    prior_ranges = {"R_p_ref": [0.9 * R_J, 1.1 * R_J],
                    "T": [800.0, 1000.0]}

    def split_params(physical, N_params_cum=None):
        R_p_ref = physical[0]
        T_iso = physical[1]
        physical_params = np.array([R_p_ref])
        PT_params = np.array([T_iso])
        log_X_params = np.array([])
        cloud_params = np.array([])
        geometry_params = np.array([])
        rest = (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))
        return (physical_params, PT_params, log_X_params, cloud_params,
                geometry_params, *rest)

    def make_atm(planet, model, P, P_ref, R_p_ref, PT_params, log_X_params,
                  cloud_params=None, geometry_params=None,
                  constant_gravity=True):
        return make_atmosphere(planet, model, P, P_ref, R_p_ref,
                                PT_params, log_X_params,
                                constant_gravity=constant_gravity)

    logp = _retrieval.make_loglikelihood(
        planet, star, model, opac, wl, data_properties,
        split_params, make_atm,
        param_names, prior_types, prior_ranges,
    )
    val = logp(np.array([0.5, 0.5]))
    assert np.isfinite(val) or val == -1.0e100
