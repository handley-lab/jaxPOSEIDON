"""Phase 0.5.13b: Toon two-stream emission + reflection parity.

Ports POSEIDON `emission.py:181-1700` Toon solver bits:
- emission_single_stream_w_albedo
- determine_photosphere_radii
- slice_gt / setup_tri_diag / tri_diag_solve / numba_cumsum
- emission_Toon (full PICASO source-function method)
- reflection_Toon (TTHG+Rayleigh multi-scattering)
- reflection_bare_surface
"""

import numpy as np
import pytest

from jaxposeidon import _emission


def test_slice_gt_matches_poseidon():
    from POSEIDON.emission import slice_gt as p_slice

    rng = np.random.default_rng(0)
    arr = rng.uniform(0.0, 50.0, size=(20, 30))
    np.testing.assert_array_equal(
        _emission.slice_gt(arr.copy(), 35.0),
        p_slice(arr.copy(), 35.0),
    )


def test_numba_cumsum_matches_poseidon():
    from POSEIDON.emission import numba_cumsum as p_cs

    rng = np.random.default_rng(0)
    mat = rng.standard_normal((20, 30))
    np.testing.assert_allclose(
        np.asarray(_emission.numba_cumsum(mat)), p_cs(mat), rtol=1e-13, atol=1e-15
    )


def test_determine_photosphere_radii_matches_poseidon():
    from POSEIDON.emission import determine_photosphere_radii as p_dpr

    rng = np.random.default_rng(0)
    N_layers, N_wl = 50, 100
    dtau = rng.uniform(1e-4, 1e-1, size=(N_layers, N_wl))
    r_low = np.linspace(7.0e7, 8.0e7, N_layers)
    wl = np.linspace(1.0, 10.0, N_wl)
    np.testing.assert_array_equal(
        _emission.determine_photosphere_radii(dtau, r_low, wl),
        p_dpr(dtau, r_low, wl),
    )


@pytest.mark.parametrize("Gauss_quad", [2, 3])
def test_emission_single_stream_w_albedo_matches_poseidon(Gauss_quad):
    from POSEIDON.emission import emission_single_stream_w_albedo as p_emw

    rng = np.random.default_rng(0)
    N_layers, N_wl = 50, 100
    T = np.linspace(1500.0, 800.0, N_layers)
    dz = 1.0e5 * np.ones(N_layers)
    wl = np.linspace(1.0, 10.0, N_wl)
    kappa = rng.uniform(1e-7, 1e-5, size=(N_layers, N_wl))
    surf_reflect = 0.3 * np.ones(N_wl)
    F_ours, dtau_ours = _emission.emission_single_stream_w_albedo(
        T,
        dz,
        wl,
        kappa,
        Gauss_quad,
        surf_reflect,
        0,
    )
    F_theirs, dtau_theirs = p_emw(T, dz, wl, kappa, Gauss_quad, surf_reflect, 0)
    np.testing.assert_allclose(F_ours, F_theirs, atol=0, rtol=1e-13)
    np.testing.assert_array_equal(dtau_ours, dtau_theirs)


def test_setup_tri_diag_matches_poseidon():
    from POSEIDON.emission import setup_tri_diag as p_setup

    rng = np.random.default_rng(0)
    N_layer, N_wl = 30, 20
    c_plus_up = rng.standard_normal((N_layer, N_wl))
    c_minus_up = rng.standard_normal((N_layer, N_wl))
    c_plus_down = rng.standard_normal((N_layer, N_wl))
    c_minus_down = rng.standard_normal((N_layer, N_wl))
    b_top = rng.standard_normal(N_wl)
    b_surface = rng.standard_normal(N_wl)
    surf_reflect = 0.3 * np.ones(N_wl)
    gamma = rng.uniform(0.1, 0.9, size=(N_layer, N_wl))
    dtau = rng.uniform(1e-4, 1e-1, size=(N_layer, N_wl))
    exptrm_pos = rng.uniform(1.0, 5.0, size=(N_layer, N_wl))
    exptrm_min = 1.0 / exptrm_pos
    A1, B1, C1, D1 = _emission.setup_tri_diag(
        N_layer,
        N_wl,
        c_plus_up,
        c_minus_up,
        c_plus_down,
        c_minus_down,
        b_top,
        b_surface,
        surf_reflect,
        gamma,
        dtau,
        exptrm_pos,
        exptrm_min,
    )
    A2, B2, C2, D2 = p_setup(
        N_layer,
        N_wl,
        c_plus_up,
        c_minus_up,
        c_plus_down,
        c_minus_down,
        b_top,
        b_surface,
        surf_reflect,
        gamma,
        dtau,
        exptrm_pos,
        exptrm_min,
    )
    np.testing.assert_array_equal(A1, A2)
    np.testing.assert_array_equal(B1, B2)
    np.testing.assert_array_equal(C1, C2)
    np.testing.assert_array_equal(D1, D2)


def test_tri_diag_solve_matches_poseidon():
    from POSEIDON.emission import tri_diag_solve as p_tds

    rng = np.random.default_rng(0)
    L = 20
    a = rng.uniform(-1.0, 1.0, size=L)
    b = rng.uniform(2.0, 4.0, size=L)
    c = rng.uniform(-1.0, 1.0, size=L)
    d = rng.standard_normal(L)
    np.testing.assert_allclose(
        np.asarray(_emission.tri_diag_solve(L, a, b, c, d)),
        p_tds(L, a, b, c, d),
        rtol=1e-13,
        atol=1e-15,
    )


def _toon_fixture(rng, N_layers=30, N_wl=20):
    """Build a self-consistent inputs bundle for emission_Toon / reflection_Toon."""
    P = np.logspace(np.log10(100.0), np.log10(1.0e-6), N_layers)
    T = np.linspace(1500.0, 800.0, N_layers)
    wl = np.linspace(1.0, 10.0, N_wl)
    dtau_tot = rng.uniform(1e-3, 0.5, size=(N_layers, 1, 1, N_wl))
    kappa_Ray = rng.uniform(1e-30, 1e-26, size=(N_layers, 1, 1, N_wl))
    kappa_cloud = rng.uniform(1e-30, 1e-26, size=(N_layers, 1, 1, N_wl))
    kappa_tot = (kappa_Ray + kappa_cloud)[:, 0, 0, :] + 1e-25
    N_aerosol = 1
    w_cloud = 0.5 * np.ones((N_aerosol, N_layers, 1, 1, N_wl))
    g_cloud = 0.3 * np.ones((N_aerosol, N_layers, 1, 1, N_wl))
    surf_reflect = np.zeros(N_wl)
    kappa_cloud_separate = kappa_cloud[None, :, :, :, :].copy()
    return dict(
        P=P,
        T=T,
        wl=wl,
        dtau_tot=dtau_tot[:, 0, 0, :],
        kappa_Ray=kappa_Ray,
        kappa_cloud=kappa_cloud,
        kappa_tot=kappa_tot,
        w_cloud=w_cloud,
        g_cloud=g_cloud,
        zone_idx=0,
        surf_reflect=surf_reflect,
        kappa_cloud_seperate=kappa_cloud_separate,
    )


def test_emission_Toon_matches_poseidon():
    from POSEIDON.emission import emission_Toon as p_eT

    rng = np.random.default_rng(0)
    cfg = _toon_fixture(rng)
    # POSEIDON mutates w_cloud in place; pass copies to keep parity inputs
    # identical.
    F_ours, _ = _emission.emission_Toon(
        cfg["P"],
        cfg["T"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
    )
    F_theirs, _ = p_eT(
        cfg["P"],
        cfg["T"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
    )
    np.testing.assert_allclose(F_ours, F_theirs, atol=0, rtol=1e-10)


def test_reflection_Toon_matches_poseidon():
    from POSEIDON.emission import reflection_Toon as p_rT

    rng = np.random.default_rng(0)
    cfg = _toon_fixture(rng)
    A_ours = _emission.reflection_Toon(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
    )
    A_theirs = p_rT(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
    )
    np.testing.assert_allclose(A_ours, A_theirs, atol=0, rtol=1e-10)


@pytest.mark.parametrize("hard_surface", [0, 1])
def test_emission_Toon_hard_surface_matches_poseidon(hard_surface):
    """emission_Toon parity for both gas-giant (hard_surface=0) and terrestrial
    (hard_surface=1) boundary conditions."""
    from POSEIDON.emission import emission_Toon as p_eT

    rng = np.random.default_rng(0)
    cfg = _toon_fixture(rng)
    surf_reflect = (
        0.3 * np.ones(len(cfg["wl"])) if hard_surface else np.zeros(len(cfg["wl"]))
    )
    F_ours, _ = _emission.emission_Toon(
        cfg["P"],
        cfg["T"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        surf_reflect.copy(),
        cfg["kappa_cloud_seperate"].copy(),
        hard_surface=hard_surface,
        T_surf=1500.0,
    )
    F_theirs, _ = p_eT(
        cfg["P"],
        cfg["T"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        surf_reflect.copy(),
        cfg["kappa_cloud_seperate"].copy(),
        hard_surface=hard_surface,
        T_surf=1500.0,
    )
    np.testing.assert_allclose(F_ours, F_theirs, atol=0, rtol=1e-10)


@pytest.mark.parametrize("toon_coefficients", [0, 1])
def test_reflection_Toon_toon_coefficients_matches_poseidon(toon_coefficients):
    """reflection_Toon parity for quadrature (0) and Eddington (1) Toon coefficients."""
    from POSEIDON.emission import reflection_Toon as p_rT

    rng = np.random.default_rng(0)
    cfg = _toon_fixture(rng)
    A_ours = _emission.reflection_Toon(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
        toon_coefficients=toon_coefficients,
    )
    A_theirs = p_rT(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
        toon_coefficients=toon_coefficients,
    )
    np.testing.assert_allclose(A_ours, A_theirs, atol=0, rtol=1e-10)


@pytest.mark.parametrize("multi_phase", [0, 1])
def test_reflection_Toon_multi_phase_matches_poseidon(multi_phase):
    """reflection_Toon parity for N=2 (0) and N=1 (1) Legendre multi-scattering."""
    from POSEIDON.emission import reflection_Toon as p_rT

    rng = np.random.default_rng(0)
    cfg = _toon_fixture(rng)
    A_ours = _emission.reflection_Toon(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
        multi_phase=multi_phase,
    )
    A_theirs = p_rT(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        cfg["surf_reflect"].copy(),
        cfg["kappa_cloud_seperate"].copy(),
        multi_phase=multi_phase,
    )
    np.testing.assert_allclose(A_ours, A_theirs, atol=0, rtol=1e-10)


def test_reflection_Toon_nonzero_surf_reflect_matches_poseidon():
    """reflection_Toon parity with a non-zero (terrestrial) surface."""
    from POSEIDON.emission import reflection_Toon as p_rT

    rng = np.random.default_rng(0)
    cfg = _toon_fixture(rng)
    surf_reflect = 0.4 * np.ones(len(cfg["wl"]))
    A_ours = _emission.reflection_Toon(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        surf_reflect.copy(),
        cfg["kappa_cloud_seperate"].copy(),
    )
    A_theirs = p_rT(
        cfg["P"],
        cfg["wl"],
        cfg["dtau_tot"].copy(),
        cfg["kappa_Ray"].copy(),
        cfg["kappa_cloud"].copy(),
        cfg["kappa_tot"].copy(),
        cfg["w_cloud"].copy(),
        cfg["g_cloud"].copy(),
        cfg["zone_idx"],
        surf_reflect.copy(),
        cfg["kappa_cloud_seperate"].copy(),
    )
    np.testing.assert_allclose(A_ours, A_theirs, atol=0, rtol=1e-10)


def test_reflection_bare_surface_matches_poseidon():
    from POSEIDON.emission import reflection_bare_surface as p_rb

    wl = np.linspace(0.5, 5.0, 100)
    surf_reflect = 0.3 + 0.1 * np.sin(wl)
    np.testing.assert_allclose(
        _emission.reflection_bare_surface(wl, surf_reflect),
        p_rb(wl, surf_reflect),
        atol=0,
        rtol=1e-13,
    )
