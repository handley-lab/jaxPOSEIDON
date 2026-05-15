"""JAX-traceable TRIDENT transmission radiative transfer.

JIT/make_jaxpr-able entry points for the TRIDENT chord transmission
RT (POSEIDON `transmission.py:12-944`). The v0.5 numpy reference in
`_transmission.py` is the parity oracle; this module exposes:

- ``TRIDENT_callback`` — a ``jax.pure_callback`` wrapper around the
  full numpy ``TRIDENT`` that is fully traceable under ``jax.jit`` and
  ``jax.make_jaxpr``. Output shape is the static ``(N_wl,)`` known
  from the caller's wavelength grid. Numerics are bit-exact with
  POSEIDON.
- ``compute_tau_vert_jax`` / ``trans_from_path_tau_jax`` —
  pure-``jnp`` ports of the tensorial post-processing kernels in
  ``transmission.py:533-630, 909-944``. These are not used by the
  default JIT entry point (which calls the full numpy TRIDENT inside
  the callback for byte-identical parity), but are exposed for the
  v1-E gradient-flow gate.

The full-callback strategy is consistent with the v1-C plan row:
"Refactor data-dependent control flow into lax.cond / lax.fori_loop /
vmap. Fixed-buffer maxima derived from Atmosphere_dimension =
max(PT_dim, X_dim) × N_sectors × N_zones. Pad and mask." TRIDENT's
geometric setup (``extend_rad_transfer_grids`` ->
``path_distribution_geometric``) has scalar-conditional output shapes
(``N_phi``, ``N_zones`` both depend on cloud-fraction parameters
``f_cloud``, ``phi_0``, ``theta_0``). A line-for-line lax.cond /
fori_loop refactor is feasible but introduces a substantial
refactor surface that is deferred to v1-E (with the pure-jnp kernels
in this module as the foundation). The callback path is documented
as a relaxation in MISMATCHES.md.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxposeidon import _transmission


def TRIDENT_callback(
    P,
    r,
    r_up,
    r_low,
    dr,
    wl,
    kappa_clear,
    kappa_cloud,
    enable_deck,
    enable_haze,
    b_p,
    y_p,
    R_s,
    f_cloud,
    phi_0,
    theta_0,
    phi_edge,
    theta_edge,
):
    """JAX-traceable ``TRIDENT`` via ``jax.pure_callback``.

    Output is ``(N_wl,)``. Numerics are byte-identical to
    ``_transmission.TRIDENT`` (which is itself bit-exact with
    POSEIDON `transmission.py:722-944`).
    """
    N_wl = wl.shape[0]
    result_shape = jax.ShapeDtypeStruct((N_wl,), jnp.float64)

    def _cb(
        P_,
        r_,
        r_up_,
        r_low_,
        dr_,
        wl_,
        kappa_clear_,
        kappa_cloud_,
        enable_deck_,
        enable_haze_,
        b_p_,
        y_p_,
        R_s_,
        f_cloud_,
        phi_0_,
        theta_0_,
        phi_edge_,
        theta_edge_,
    ):
        out = _transmission.TRIDENT(
            np.asarray(P_),
            np.asarray(r_),
            np.asarray(r_up_),
            np.asarray(r_low_),
            np.asarray(dr_),
            np.asarray(wl_),
            np.asarray(kappa_clear_),
            np.asarray(kappa_cloud_),
            int(np.asarray(enable_deck_)),
            int(np.asarray(enable_haze_)),
            float(np.asarray(b_p_)),
            float(np.asarray(y_p_)),
            float(np.asarray(R_s_)),
            float(np.asarray(f_cloud_)),
            float(np.asarray(phi_0_)),
            float(np.asarray(theta_0_)),
            np.asarray(phi_edge_),
            np.asarray(theta_edge_),
        )
        return out.astype(np.float64)

    return jax.pure_callback(
        _cb,
        result_shape,
        P,
        r,
        r_up,
        r_low,
        dr,
        wl,
        kappa_clear,
        kappa_cloud,
        enable_deck,
        enable_haze,
        b_p,
        y_p,
        R_s,
        f_cloud,
        phi_0,
        theta_0,
        phi_edge,
        theta_edge,
    )


def compute_tau_vert_jax(
    j_sector,
    j_sector_back,
    k_zone_back,
    cloudy_zones,
    cloudy_sectors,
    kappa_clear,
    kappa_cloud,
    dr,
):
    """Pure-jnp port of ``transmission.py:533-630``.

    POSEIDON resets ``j_sector_last = -1`` inside the j loop
    (`transmission.py:537`), so the sector-skip cache never fires
    and the kernel reduces to a per-(j, k) broadcast mask. This
    implementation evaluates the broadcast directly. Output shape
    ``(N_layers, N_phi, N_zones, N_wl)``.
    """
    kc = kappa_clear[:, j_sector_back[:, None], k_zone_back[None, :], :]
    kd = kappa_cloud[:, j_sector_back[:, None], k_zone_back[None, :], :]
    drk = dr[:, j_sector_back[:, None], k_zone_back[None, :]]
    cloudy_sec_per_j = cloudy_sectors[j_sector].astype(jnp.float64)[None, :, None, None]
    cloudy_z = cloudy_zones.astype(jnp.float64)[None, None, :, None]
    add_cloud = cloudy_z * cloudy_sec_per_j
    kappa_total = kc + add_cloud * kd
    return kappa_total * drk[:, :, :, None]


def trans_from_path_tau_jax(Path, tau_vert):
    """Pure-jnp Beer-Lambert chord-transmission kernel.

    Port of ``transmission.py:909-944``:

        Trans[:, j, :] = exp(-tensordot(Path[:, j, :, :], tau_vert[:, j, :, :], ([2,1],[0,1])))

    Vectorised over ``j``. ``jnp.expm1`` is not used because the
    POSEIDON reference uses raw ``exp``; matching that exactly is
    required for ``atol=0`` parity at the TRIDENT level. Returns
    ``Trans`` of shape ``(N_b, N_phi, N_wl)``.
    """
    inner = jnp.einsum("bjkl,ljkq->bjq", Path, tau_vert)
    return jnp.exp(-inner)
