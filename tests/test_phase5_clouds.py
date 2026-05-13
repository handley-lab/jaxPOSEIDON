"""Phase 5 MacMad17 cloud parameter unpacking — POSEIDON parity."""

import numpy as np
import pytest

from jaxposeidon._clouds import unpack_MacMad17_cloud_params


def _poseidon_unpack(*, cloud_model="MacMad17", cloud_type, cloud_dim,
                     clouds_in, cloud_param_names,
                     TwoD_type=None):
    """Wrapper that calls POSEIDON's `unpack_cloud_params` and returns the
    MacMad17-relevant subset for comparison.

    POSEIDON signature (parameters.py:1916-1917):
        unpack_cloud_params(param_names, clouds_in, cloud_model,
                            cloud_dim, N_params_cumulative, TwoD_type)
    POSEIDON return order (parameters.py:2475-2476):
        (kappa_cloud_0, P_cloud, f_cloud, phi_0, theta_0, a, gamma, ...)

    `param_names` must contain `cloud_param_names` in the slice
    `[N_params_cumulative[2]:N_params_cumulative[3]]`. We construct a
    minimal `param_names` with the first three groups empty and the
    cloud group at indices 0:len(cloud_param_names).
    """
    from POSEIDON.parameters import unpack_cloud_params as p_unpack
    enable_haze = 1 if "haze" in cloud_type else 0
    enable_deck = 1 if "deck" in cloud_type else 0
    # Build N_params_cumulative so slice [2]:[3] = [0:len(cloud_param_names)]
    n_cp = len(cloud_param_names)
    N_params_cumulative = np.array([0, 0, 0, n_cp, n_cp, n_cp, n_cp,
                                     n_cp, n_cp, n_cp])
    param_names = np.array(list(cloud_param_names))  # only the cloud names
    out = p_unpack(param_names, np.asarray(clouds_in, dtype=float),
                   cloud_model, cloud_dim, N_params_cumulative, TwoD_type)
    kappa_cloud_0, P_cloud, f_cloud, phi_0, theta_0, a, gamma = out[:7]
    return dict(
        a=float(a), gamma=float(gamma),
        P_cloud=float(P_cloud),
        kappa_cloud_0=float(kappa_cloud_0),
        f_cloud=float(f_cloud), phi_0=float(phi_0), theta_0=float(theta_0),
        enable_haze=enable_haze, enable_deck=enable_deck,
    )


def _assert_dicts_close(d1, d2):
    assert d1.keys() == d2.keys()
    for k in d1:
        a, b = d1[k], d2[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            np.testing.assert_array_equal(np.atleast_1d(a), np.atleast_1d(b))
        else:
            assert a == b, f"key {k}: {a} != {b}"


@pytest.mark.parametrize("cloud_type,cloud_dim,clouds_in,names", [
    ("deck", 1, [-1.0], ["log_P_cloud"]),
    ("haze", 1, [3.0, -10.0], ["log_a", "gamma"]),
    ("deck_haze", 1, [4.0, -8.0, 0.0], ["log_a", "gamma", "log_P_cloud"]),
    ("deck_haze", 2, [4.0, -8.0, 0.0, 0.3],
     ["log_a", "gamma", "log_P_cloud", "phi_cloud"]),
    ("deck", 2, [-2.0, 0.4], ["log_P_cloud", "phi_cloud"]),
    ("haze", 2, [5.0, -12.0, 0.7], ["log_a", "gamma", "phi_cloud"]),
])
def test_unpack_MacMad17_matches_poseidon(cloud_type, cloud_dim, clouds_in,
                                          names):
    ours = unpack_MacMad17_cloud_params(
        clouds_in=np.array(clouds_in),
        cloud_param_names=np.array(names),
        cloud_type=cloud_type, cloud_dim=cloud_dim,
    )
    theirs = _poseidon_unpack(
        cloud_type=cloud_type, cloud_dim=cloud_dim,
        clouds_in=clouds_in, cloud_param_names=names,
    )
    _assert_dicts_close(ours, theirs)


def test_unpack_rejects_shiny():
    with pytest.raises(NotImplementedError, match="cloud_type"):
        unpack_MacMad17_cloud_params(
            clouds_in=np.array([0.0]),
            cloud_param_names=np.array(["log_P_cloud"]),
            cloud_type="shiny_deck", cloud_dim=1,
        )


def test_unpack_rejects_cloud_dim_3():
    with pytest.raises(NotImplementedError, match="cloud_dim"):
        unpack_MacMad17_cloud_params(
            clouds_in=np.array([0.0]),
            cloud_param_names=np.array(["log_P_cloud"]),
            cloud_type="deck", cloud_dim=3,
        )
