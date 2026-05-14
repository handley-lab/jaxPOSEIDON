"""Phase 0.5.13a (subset): Planck + single-stream emission + bare surface parity.

Ports POSEIDON `emission.py:30-178, 1576-1609`. Toon two-stream solvers
(`emission_Toon`, `reflection_Toon`) are the follow-up Phase 0.5.13b.
"""

import numpy as np
import pytest

from jaxposeidon import _emission


def test_planck_lambda_arr_matches_poseidon():
    """Planck function parity. POSEIDON precomputes c_2 / wl_m once per k
    inside numba; numpy reorders trivially → 1-ULP diff at rtol=1e-13."""
    from POSEIDON.emission import planck_lambda_arr as p_planck

    T = np.array([300.0, 800.0, 1500.0, 2500.0])
    wl = np.linspace(0.5, 20.0, 200)
    np.testing.assert_allclose(
        _emission.planck_lambda_arr(T, wl),
        p_planck(T, wl),
        atol=0,
        rtol=1e-13,
    )


@pytest.mark.parametrize("Gauss_quad", [2, 3])
def test_emission_single_stream_matches_poseidon(Gauss_quad):
    from POSEIDON.emission import emission_single_stream as p_em

    rng = np.random.default_rng(0)
    N_layers, N_wl = 50, 100
    T = np.linspace(1500.0, 800.0, N_layers)  # decreasing with altitude (deep is hot)
    dz = 1.0e5 * np.ones(N_layers)  # 100 km layers
    wl = np.linspace(1.0, 10.0, N_wl)
    kappa = rng.uniform(1e-7, 1e-5, size=(N_layers, N_wl))
    F_ours, dtau_ours = _emission.emission_single_stream(T, dz, wl, kappa, Gauss_quad)
    F_theirs, dtau_theirs = p_em(T, dz, wl, kappa, Gauss_quad)
    np.testing.assert_allclose(F_ours, F_theirs, atol=0, rtol=1e-13)
    np.testing.assert_array_equal(dtau_ours, dtau_theirs)


def test_emission_bare_surface_matches_poseidon():
    """Bare-surface parity (same 1-ULP Planck reorder as above)."""
    from POSEIDON.emission import emission_bare_surface as p_bare

    wl = np.linspace(0.5, 20.0, 200)
    surf_reflect = 0.3 + 0.1 * np.sin(wl)
    T_surf = 800.0
    np.testing.assert_allclose(
        _emission.emission_bare_surface(T_surf, wl, surf_reflect),
        p_bare(T_surf, wl, surf_reflect),
        atol=0,
        rtol=1e-13,
    )


def test_emission_single_stream_rejects_unknown_quadrature():
    T = np.array([1000.0])
    dz = np.array([1e5])
    wl = np.array([1.0, 2.0])
    kappa = np.zeros((1, 2))
    with pytest.raises(NotImplementedError, match="Gauss_quad"):
        _emission.emission_single_stream(T, dz, wl, kappa, Gauss_quad=5)
