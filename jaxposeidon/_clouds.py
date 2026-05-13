"""MacMad17 cloud / haze parameter unpacking — v0.

Mirrors the `cloud_model == 'MacMad17'` branch of
`POSEIDON/POSEIDON/parameters.py:unpack_cloud_params` (lines 2010-2065),
filtered to the v0 envelope:

- `cloud_type in {'deck', 'haze', 'deck_haze'}` → no shiny decks
- `cloud_dim in {1, 2}`

The actual extinction additions for deck/haze are in
`_opacities.extinction(...)` (Phase 4). This module's job is converting
the typed `clouds_drawn` parameter vector returned by
`_parameters.split_params(...)` into the named scalars that extinction()
expects, plus the partial-cloud geometry parameters
`(f_cloud, phi_0, theta_0)` that TRIDENT consumes in Phase 7.

Iceberg / Mie / eddysed are deferred (`_parameters.assert_v0_model_config`
already rejects them with NotImplementedError).
"""

import numpy as np


def unpack_MacMad17_cloud_params(*, clouds_in, cloud_param_names,
                                  cloud_type, cloud_dim):
    """v0 MacMad17 cloud parameter unpacking.

    Args:
        clouds_in: array of MacMad17 cloud-parameter values (in the order
            given by `cloud_param_names`).
        cloud_param_names: numpy array of POSEIDON cloud parameter names
            for this configuration (e.g. ['log_a', 'gamma', 'log_P_cloud',
            'phi_cloud']).
        cloud_type: 'deck' / 'haze' / 'deck_haze'.
        cloud_dim: 1 or 2.

    Returns:
        dict with keys:
            a, gamma, P_cloud, kappa_cloud_0,
            f_cloud, phi_0, theta_0,
            enable_haze, enable_deck.
    """
    if cloud_type not in {"deck", "haze", "deck_haze"}:
        raise NotImplementedError(
            f"cloud_type={cloud_type!r}: v0 supports only "
            "{'deck','haze','deck_haze'} for MacMad17"
        )
    if cloud_dim not in (1, 2):
        raise NotImplementedError(
            f"cloud_dim={cloud_dim}: v0 supports only 1 or 2 for MacMad17"
        )

    cloud_param_names = np.asarray(cloud_param_names)
    enable_haze = 1 if "haze" in cloud_type else 0
    enable_deck = 1 if "deck" in cloud_type else 0

    kappa_cloud_0 = 1.0e250

    if enable_haze == 1:
        a = 10.0 ** clouds_in[
            int(np.where(cloud_param_names == "log_a")[0][0])
        ]
        gamma = clouds_in[
            int(np.where(cloud_param_names == "gamma")[0][0])
        ]
    else:
        a, gamma = 1.0, -4.0

    if enable_deck == 1:
        P_cloud = 10.0 ** clouds_in[
            int(np.where(cloud_param_names == "log_P_cloud")[0][0])
        ]
    else:
        P_cloud = 100.0

    if cloud_dim != 1:
        phi_c = clouds_in[
            int(np.where(cloud_param_names == "phi_cloud")[0][0])
        ]
        phi_0 = 0.0
        f_cloud = phi_c
        theta_0 = -90.0
    else:
        if enable_deck == 1:
            f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0
        else:
            f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0

    return dict(
        a=a, gamma=gamma,
        P_cloud=np.array([P_cloud]),  # POSEIDON wraps scalar in length-1 array
        kappa_cloud_0=kappa_cloud_0,
        f_cloud=f_cloud, phi_0=phi_0, theta_0=theta_0,
        enable_haze=enable_haze, enable_deck=enable_deck,
    )
