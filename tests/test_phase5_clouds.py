"""Phase 5 MacMad17 cloud parameter unpacking — POSEIDON parity."""

import numpy as np
import pytest

from jaxposeidon._clouds import unpack_MacMad17_cloud_params


def _poseidon_unpack(*, cloud_model="MacMad17", cloud_type, cloud_dim,
                     clouds_in, cloud_param_names,
                     TwoD_type=None, n_aerosol_array=None,
                     aerosol_species=()):
    """Wrapper that calls POSEIDON's `unpack_cloud_params` and returns the
    MacMad17-relevant subset for comparison."""
    from POSEIDON.parameters import unpack_cloud_params as p_unpack
    enable_haze = 1 if "haze" in cloud_type else 0
    enable_deck = 1 if "deck" in cloud_type else 0
    enable_shiny_deck = 0
    enable_Mie = 0
    out = p_unpack(
        np.asarray(clouds_in), np.asarray(cloud_param_names),
        cloud_model, cloud_dim, cloud_type, TwoD_type,
        enable_haze, enable_deck, enable_shiny_deck,
        np.array(aerosol_species), enable_Mie,
        False, n_aerosol_array,
    )
    # POSEIDON returns a large tuple. We extract only the MacMad17 fields.
    # Order per parameters.py:unpack_cloud_params return statement.
    # Pull from the docstring/source — we'll just take the first few:
    # a, gamma, P_cloud, kappa_cloud_0, f_cloud, phi_0, theta_0, ...
    a, gamma, P_cloud, kappa_cloud_0, f_cloud, phi_0, theta_0 = out[:7]
    return dict(
        a=float(a), gamma=float(gamma),
        P_cloud=np.atleast_1d(np.asarray(P_cloud, dtype=float)),
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
