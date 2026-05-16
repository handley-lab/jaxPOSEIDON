"""JAX-traceable TRIDENT transmission radiative transfer.

Two entry points to the TRIDENT chord transmission RT
(POSEIDON `transmission.py:12-944`):

- ``TRIDENT_kernel_jit`` — pure-``jnp`` kernel for the tensor-math
  half of TRIDENT (``transmission.py:476-492``): computes ``tau_vert``,
  ``Trans = exp(-einsum(Path, tau_vert))``, and
  ``transit_depth = (A_overlap - <Trans, dA>) / (π R_s²)``. **All
  inputs are precomputed-geometry tensors**; the kernel is fully
  ``jit``-able and ``jax.grad`` flows through ``kappa_clear``,
  ``kappa_cloud``, ``dr``, ``Path``, ``dA_atm_overlap``, and
  ``A_overlap``.

- ``setup_TRIDENT_geometry`` — numpy setup-only orchestrator that
  runs ``extend_rad_transfer_grids`` + ``path_distribution_geometric``
  + ``delta_ray_geom`` and produces the fixed-shape arrays the JIT
  kernel consumes. Runs **outside** ``jit``. The geometric output
  shapes depend on cloud-morphology parameters (``f_cloud``,
  ``phi_0``, ``theta_0``) which is why this stays setup-only.

- ``TRIDENT_callback`` — legacy ``jax.pure_callback`` wrapper around
  the full numpy ``TRIDENT``. Still used by
  ``compute_transmission_spectrum_jit`` for backward compatibility
  with the v1-C parity surface (which wraps the public function in
  ``jax.jit`` directly — that pattern only works with the callback
  path, since the setup orchestrator cannot itself be traced).
  ``compute_transmission_spectrum_real_jit`` is the real-JAX entry
  point.

- ``compute_tau_vert_jax`` / ``trans_from_path_tau_jax`` — pure-``jnp``
  ports of the tensorial subroutines in ``transmission.py:533-630,
  909-944``. ``TRIDENT_kernel_jit`` chains them with the final
  ``transit_depth`` assembly.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxposeidon import _transmission  # noqa: E402


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
        # Host-side numpy is permitted inside the pure_callback body; not
        # in the jit trace. Skip the v1 source-grep gate for this line.
        import numpy as np  # v1-grep-skip

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


def TRIDENT_kernel_jit(
    j_sector,
    j_sector_back,
    k_zone_back,
    cloudy_zones,
    cloudy_sectors,
    Path,
    dA_atm_overlap,
    A_overlap,
    R_s_sq,
    kappa_clear,
    kappa_cloud,
    dr,
):
    """Pure-jnp TRIDENT post-setup kernel — full transit_depth.

    Chains ``compute_tau_vert_jax`` + ``trans_from_path_tau_jax`` and
    the final ``transit_depth`` assembly (POSEIDON
    `transmission.py:476-492`). All inputs are fixed-shape tensors
    produced by ``setup_TRIDENT_geometry``; the kernel is ``jit``-able
    and ``jax.grad`` flows through ``kappa_clear``, ``kappa_cloud``,
    ``dr``.

    Returns ``transit_depth`` of shape ``(N_wl,)``.
    """
    tau_vert = compute_tau_vert_jax(
        j_sector,
        j_sector_back,
        k_zone_back,
        cloudy_zones,
        cloudy_sectors,
        kappa_clear,
        kappa_cloud,
        dr,
    )
    Trans = trans_from_path_tau_jax(Path, tau_vert)
    A_atm_overlap_eff = jnp.tensordot(Trans, dA_atm_overlap, axes=([0, 1], [0, 1]))
    return (A_overlap - A_atm_overlap_eff) / (jnp.pi * R_s_sq)


def TRIDENT_real_jit(
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
    """Real JAX TRIDENT — setup-only numpy outside jit, jnp kernel inside.

    Replaces ``TRIDENT_callback`` for the typical retrieval case where
    cloud-morphology parameters (``f_cloud``, ``phi_0``, ``theta_0``,
    ``enable_deck``) are **fixed** across the jit boundary (i.e., the
    sampler varies physics parameters that flow through ``kappa_*``
    and atmosphere arrays, not cloud topology). For those callers,
    ``jax.grad`` flows through ``kappa_clear``, ``kappa_cloud``, and
    ``dr``.

    The numpy geometry setup is delegated to the allow-listed
    ``_jax_transmission_setup.setup_TRIDENT_geometry`` (called outside
    the jit boundary); the pure-jnp ``TRIDENT_kernel_jit`` runs the
    tensor compute under jit.
    """
    from jaxposeidon._jax_transmission_setup import setup_TRIDENT_geometry

    geom = setup_TRIDENT_geometry(
        P,
        r,
        r_up,
        r_low,
        dr,
        wl,
        int(enable_deck),
        int(enable_haze),
        float(b_p),
        float(y_p),
        float(R_s),
        float(f_cloud),
        float(phi_0),
        float(theta_0),
        phi_edge,
        theta_edge,
    )
    if geom["geometry_empty"]:
        return jnp.zeros(geom["N_wl"])

    return TRIDENT_kernel_jit(
        jnp.asarray(geom["j_sector"]),
        jnp.asarray(geom["j_sector_back"]),
        jnp.asarray(geom["k_zone_back"]),
        jnp.asarray(geom["cloudy_zones"]),
        jnp.asarray(geom["cloudy_sectors"]),
        jnp.asarray(geom["Path"]),
        jnp.asarray(geom["dA_atm_overlap"]),
        jnp.asarray(geom["A_overlap"]),
        jnp.asarray(geom["R_s_sq"]),
        kappa_clear,
        kappa_cloud,
        dr,
    )
