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
    from POSEIDON.core import (
        create_star,
        create_planet,
        define_model,
        make_atmosphere,
        read_opacities,
        wl_grid_constant_R,
    )

    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=M_J, T_eq=900.0)
    model = define_model("m", ["H2", "He"], [], PT_profile="isotherm")
    wl = wl_grid_constant_R(1.0, 3.0, 500)
    T_fine = np.arange(700, 1110, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(
        model, wl, "opacity_sampling", T_fine, log_P_fine, testing=True
    )
    opac["CIA_stored"] *= 0.0

    # Fake "binned data" properties — single dataset, no offsets, no inflation.
    n_bins = 20
    bin_cent = np.linspace(int(0.1 * len(wl)), int(0.9 * len(wl)), n_bins).astype(
        np.int64
    )
    bin_left = np.maximum(bin_cent - 4, 0)
    bin_right = np.minimum(bin_cent + 4, len(wl) - 1)
    sigma = np.full(n_bins, 1.2)
    sens = np.ones(len(wl))
    norm = np.array(
        [np.trapezoid(sens[bl:br], wl[bl:br]) for bl, br in zip(bin_left, bin_right)]
    )
    ydata = np.full(n_bins, 2.7e-3)
    err_data = np.full(n_bins, 1.0e-4)
    data_properties = dict(
        datasets=["d"],
        instruments=["JWST_NIRSpec_PRISM"],
        wl_data=wl[bin_cent],
        half_bin=np.full(n_bins, 0.02),
        ydata=ydata,
        err_data=err_data,
        sens=sens,
        psf_sigma=sigma,
        fwhm=np.full(n_bins, 0.02),
        bin_left=bin_left,
        bin_cent=bin_cent,
        bin_right=bin_right,
        norm=norm,
        len_data_idx=np.array([0, n_bins]),
        offset_start=0,
        offset_end=0,
        offset_1_start=0,
        offset_1_end=0,
        offset_2_start=0,
        offset_2_end=0,
        offset_3_start=0,
        offset_3_end=0,
    )

    # Trivial 2-parameter wrapper: (R_p_ref, T_iso); both uniform.
    param_names = ["R_p_ref", "T"]
    prior_types = {"R_p_ref": "uniform", "T": "uniform"}
    prior_ranges = {"R_p_ref": [0.9 * R_J, 1.1 * R_J], "T": [800.0, 1000.0]}

    def split_params(physical, N_params_cum=None):
        R_p_ref = physical[0]
        T_iso = physical[1]
        physical_params = np.array([R_p_ref])
        PT_params = np.array([T_iso])
        log_X_params = np.array([])
        cloud_params = np.array([])
        geometry_params = np.array([])
        rest = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        return (
            physical_params,
            PT_params,
            log_X_params,
            cloud_params,
            geometry_params,
            *rest,
        )

    def make_atm(
        planet,
        model,
        P,
        P_ref,
        R_p_ref,
        PT_params,
        log_X_params,
        cloud_params=None,
        geometry_params=None,
        constant_gravity=True,
    ):
        return make_atmosphere(
            planet,
            model,
            P,
            P_ref,
            R_p_ref,
            PT_params,
            log_X_params,
            constant_gravity=constant_gravity,
        )

    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 60)
    logp = _retrieval.make_loglikelihood(
        planet,
        star,
        model,
        opac,
        wl,
        data_properties,
        split_params,
        make_atm,
        param_names,
        prior_types,
        prior_ranges,
        P=P,
        reference_parameter="R_p_ref",
        log_P_ref_fixed=1.0,
    )
    val = logp(np.array([0.5, 0.5]))
    assert np.isfinite(val) or val == -1.0e100


# ---------------------------------------------------------------------------
# reference_parameter handling
# ---------------------------------------------------------------------------
def _trivial_components():
    """Minimal stand-ins for make_atmosphere/split_params used to test the
    reference_parameter branching in make_loglikelihood without involving
    POSEIDON. Records the (R_p_ref, P_ref) the closure computed."""
    captured = {}

    def split_params(physical, N_params_cum=None):
        return (
            np.asarray(physical),
            np.array([1000.0]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    def make_atmosphere(
        planet,
        model,
        P,
        P_ref,
        R_p_ref,
        PT,
        log_X,
        cloud_params=None,
        geometry_params=None,
        constant_gravity=True,
    ):
        captured["P_ref"] = P_ref
        captured["R_p_ref"] = R_p_ref
        # Return an atmosphere that's-physical=False so compute_spectrum
        # short-circuits to NaN and we don't need a real opac.
        raise StopIteration  # caller handles

    return split_params, make_atmosphere, captured


@pytest.mark.parametrize(
    "reference_parameter,prior_ranges,unit_cube,expected_R_p_ref,expected_P_ref",
    [
        # cube=0.25 over [6e7, 8e7] → physical = 6.5e7; log_P_ref_fixed=1 → P_ref=10
        ("R_p_ref", {"R_p_ref": [6.0e7, 8.0e7]}, [0.25], 6.5e7, 10.0),
        # cube=0.75 over [-3, -1] → physical log_P = -1.5 → P_ref = 10**-1.5
        ("P_ref", {"log_P_ref": [-3.0, -1.0]}, [0.75], 7.0e7, 10.0**-1.5),
        # cube=[0.25, 0.75] → R_p_ref=6.5e7, log_P_ref=-1.5
        (
            "R_p_ref+P_ref",
            {"R_p_ref": [6.0e7, 8.0e7], "log_P_ref": [-3.0, -1.0]},
            [0.25, 0.75],
            6.5e7,
            10.0**-1.5,
        ),
    ],
)
def test_make_loglikelihood_reference_parameter(
    reference_parameter,
    prior_ranges,
    unit_cube,
    expected_R_p_ref,
    expected_P_ref,
):
    sp, ma, captured = _trivial_components()
    param_names = list(prior_ranges.keys())
    prior_types = {p: "uniform" for p in param_names}
    logp = _retrieval.make_loglikelihood(
        planet=object(),
        star=None,
        model=None,
        opac=None,
        wl=np.zeros(1),
        data_properties={},
        split_params=sp,
        make_atmosphere=ma,
        param_names=param_names,
        prior_types=prior_types,
        prior_ranges=prior_ranges,
        P=np.logspace(2, -7, 10),
        reference_parameter=reference_parameter,
        R_p_ref_fixed=7.0e7,
        log_P_ref_fixed=1.0,
    )
    with pytest.raises(StopIteration):
        logp(np.array(unit_cube))
    np.testing.assert_allclose(
        captured["R_p_ref"], expected_R_p_ref, atol=0, rtol=1e-12
    )
    np.testing.assert_allclose(captured["P_ref"], expected_P_ref, atol=0, rtol=1e-12)


def test_make_loglikelihood_rejects_unknown_reference_parameter():
    with pytest.raises(NotImplementedError, match="reference_parameter"):
        _retrieval.make_loglikelihood(
            planet=None,
            star=None,
            model=None,
            opac=None,
            wl=np.zeros(1),
            data_properties={},
            split_params=lambda x, n=None: tuple(),
            make_atmosphere=lambda *a, **kw: None,
            param_names=[],
            prior_types={},
            prior_ranges={},
            P=np.zeros(1),
            reference_parameter="bogus",
        )


def test_make_loglikelihood_rejects_missing_fixed_values():
    with pytest.raises(ValueError, match="R_p_ref_fixed"):
        _retrieval.make_loglikelihood(
            planet=None,
            star=None,
            model=None,
            opac=None,
            wl=np.zeros(1),
            data_properties={},
            split_params=lambda x, n=None: tuple(),
            make_atmosphere=lambda *a, **kw: None,
            param_names=[],
            prior_types={},
            prior_ranges={},
            P=np.zeros(1),
            reference_parameter="P_ref",
        )
    with pytest.raises(ValueError, match="log_P_ref_fixed"):
        _retrieval.make_loglikelihood(
            planet=None,
            star=None,
            model=None,
            opac=None,
            wl=np.zeros(1),
            data_properties={},
            split_params=lambda x, n=None: tuple(),
            make_atmosphere=lambda *a, **kw: None,
            param_names=[],
            prior_types={},
            prior_ranges={},
            P=np.zeros(1),
            reference_parameter="R_p_ref",
        )


# ---------------------------------------------------------------------------
# Offsets / error inflation propagation through the closure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "offsets_applied",
    [
        None,
        "single_dataset",
        "two_datasets",
        "three_datasets",
    ],
)
@pytest.mark.parametrize(
    "error_inflation",
    [
        None,
        "Line15",
        "Piette20",
        "Line15+Piette20",
    ],
)
def test_make_loglikelihood_offsets_inflation_combinatorial(
    monkeypatch,
    offsets_applied,
    error_inflation,
):
    """The closure must forward offsets_applied / error_inflation /
    offset_params / err_inflation_params / offset_start / offset_end to
    _data.loglikelihood. Inject stub compute_spectrum and
    bin_spectrum_to_data so the test does not depend on POSEIDON
    opacities, then compare the closure output to an independent
    loglikelihood call with the same args.
    """
    from jaxposeidon import _retrieval as ret_module
    from jaxposeidon._data import loglikelihood

    n = 24
    ymodel_target = np.full(n, 2.7e-3)
    rng = np.random.default_rng(0)
    ydata = ymodel_target + rng.normal(0, 5e-5, size=n)
    err_data = np.full(n, 1e-4)

    if offsets_applied == "single_dataset":
        offset_params = np.array([50.0])
        data_off = dict(offset_start=0, offset_end=n)
    elif offsets_applied == "two_datasets":
        offset_params = np.array([50.0, -30.0])
        data_off = dict(offset_start=[0, n // 2], offset_end=[n // 2, n])
    elif offsets_applied == "three_datasets":
        offset_params = np.array([50.0, -30.0, 20.0])
        t = n // 3
        data_off = dict(offset_start=[0, t, 2 * t], offset_end=[t, 2 * t, n])
    else:
        offset_params = np.array([])
        data_off = dict(offset_start=0, offset_end=0)
    if error_inflation == "Line15":
        err_inflation_params = np.array([-8.5])
    elif error_inflation == "Piette20":
        err_inflation_params = np.array([0.2])
    elif error_inflation == "Line15+Piette20":
        err_inflation_params = np.array([-9.0, 0.15])
    else:
        err_inflation_params = np.array([])

    data_properties = dict(
        ydata=ydata,
        err_data=err_data,
        offset_1_start=0,
        offset_1_end=0,
        offset_2_start=0,
        offset_2_end=0,
        offset_3_start=0,
        offset_3_end=0,
        **data_off,
    )

    def split_params(physical, N_params_cum=None):
        # physical = [R_p_ref]; pad offset_params/err_inflation_params into rest
        return (
            np.array([physical[0]]),  # physical_params
            np.array([1000.0]),  # PT
            np.array([]),
            np.array([]),
            np.array([]),  # log_X, cloud, geom
            np.array([]),  # stellar (rest[0])
            offset_params,  # offsets (rest[1])
            err_inflation_params,  # err_inflation (rest[2])
            np.array([]),
            np.array([]),
        )  # high_res, surface

    def make_atm_stub(*a, **kw):
        return {"_stub": True}  # never used, compute_spectrum is monkeypatched

    monkeypatch.setattr(ret_module, "compute_spectrum", lambda *a, **kw: ymodel_target)
    monkeypatch.setattr(
        ret_module, "bin_spectrum_to_data", lambda spectrum, wl, dp: spectrum
    )

    param_names = ["R_p_ref"]
    prior_types = {"R_p_ref": "uniform"}
    prior_ranges = {"R_p_ref": [6.5e7, 7.5e7]}
    logp = ret_module.make_loglikelihood(
        planet=None,
        star=None,
        model=None,
        opac=None,
        wl=np.zeros(n),
        data_properties=data_properties,
        split_params=split_params,
        make_atmosphere=make_atm_stub,
        param_names=param_names,
        prior_types=prior_types,
        prior_ranges=prior_ranges,
        P=np.logspace(2, -7, 10),
        reference_parameter="R_p_ref",
        log_P_ref_fixed=1.0,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
    )

    actual = logp(np.array([0.5]))
    expected = loglikelihood(
        ymodel_target,
        ydata,
        err_data,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
        offset_1_start=0,
        offset_1_end=0,
        offset_2_start=0,
        offset_2_end=0,
        offset_3_start=0,
        offset_3_end=0,
        **data_off,
    )
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
