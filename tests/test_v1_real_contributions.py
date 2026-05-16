"""Real-JAX spectral contribution kernel — active-molecule branch.

Verifies the new ``_jax_contributions.spectral_contribution_kernel_jit``
against the numpy ``_contributions.extinction_spectral_contribution``
for the most common call pattern: contribution of one active
molecule with ``bulk_species=False``, ``cloud_contribution=False``,
``enable_haze=0``, ``enable_deck=0``, ``enable_surface=0``,
``enable_Mie=0``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxposeidon import _contributions
from jaxposeidon._jax_contributions import (
    setup_spectral_contribution_indices,
    spectral_contribution_kernel_jit,
)


def _fixture(N_sectors=1, N_zones=1, N_ff_pairs=0, P_deep=1000.0, include_active_cia=False):
    """Small realistic input set for active-molecule contribution.

    ``include_active_cia=True`` adds a CIA pair involving the active
    species H2O (e.g. 'H2-H2O') to exercise the
    `cia_mask | bulk_cia_mask` selector path. ``N_ff_pairs > 0``
    exercises the free-free accumulation.
    """
    rng = np.random.default_rng(0)
    N_layers = 20
    N_wl = 12
    N_T_fine = 8
    N_P_fine = 9

    chemical_species = np.array(["H2", "He", "H2O", "CH4"])
    active_species = np.array(["H2O", "CH4"])
    if include_active_cia:
        cia_pairs = np.array(["H2-H2", "H2-He", "H2-H2O"])
    else:
        cia_pairs = np.array(["H2-H2", "H2-He"])
    N_cia_pairs = len(cia_pairs)

    P = np.logspace(2.0, -7.0, N_layers)
    T = 1000.0 + 200.0 * rng.standard_normal((N_layers, N_sectors, N_zones))
    n = rng.uniform(1e15, 1e22, (N_layers, N_sectors, N_zones))
    wl = np.linspace(1.0, 5.0, N_wl)

    X = np.zeros((4, N_layers, N_sectors, N_zones))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 1e-3
    X[3] = 1e-4
    X_active = X[2:]  # (N_species_active, N_layers, N_sectors, N_zones)

    X_cia = np.zeros((2, N_cia_pairs, N_layers, N_sectors, N_zones))
    X_cia[0, 0] = X[0]  # H2 in H2-H2
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]  # H2 in H2-He
    X_cia[1, 1] = X[1]
    if include_active_cia:
        X_cia[0, 2] = X[0]  # H2 in H2-H2O
        X_cia[1, 2] = X[2]  # H2O

    # ff_pairs (mock): give them arbitrary names that don't match species
    ff_pairs = np.array([f"ff{q}" for q in range(N_ff_pairs)])
    X_ff = np.zeros((2, N_ff_pairs, N_layers, N_sectors, N_zones))
    for q in range(N_ff_pairs):
        X_ff[0, q] = X[0] * 0.5
        X_ff[1, q] = X[1] * 0.5
    X_bf = np.zeros((0, N_layers, N_sectors, N_zones))

    T_fine = np.linspace(500.0, 2000.0, N_T_fine)
    log_P_fine = np.linspace(-6.0, 2.0, N_P_fine)
    sigma_stored = rng.uniform(1e-23, 1e-21, (2, N_P_fine, N_T_fine, N_wl))
    cia_stored = rng.uniform(1e-45, 1e-43, (N_cia_pairs, N_T_fine, N_wl))
    Rayleigh_stored = rng.uniform(1e-28, 1e-26, (4, N_wl))
    ff_stored = rng.uniform(1e-46, 1e-44, (N_ff_pairs, N_T_fine, N_wl))
    bf_stored = np.zeros((0, N_wl))

    return dict(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=ff_pairs,
        bf_species=np.array([], dtype=str),
        aerosol_species=np.array([], dtype=str),
        n=n,
        T=T,
        P=P,
        wl=wl,
        X=X,
        X_active=X_active,
        X_cia=X_cia,
        X_ff=X_ff,
        X_bf=X_bf,
        a=1.0,
        gamma=-4.0,
        P_cloud=np.array([1e-3]),
        kappa_cloud_0=1e-30,
        sigma_stored=sigma_stored,
        cia_stored=cia_stored,
        Rayleigh_stored=Rayleigh_stored,
        ff_stored=ff_stored,
        bf_stored=bf_stored,
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        N_sectors=N_sectors,
        N_zones=N_zones,
        T_fine=T_fine,
        log_P_fine=log_P_fine,
        P_surf=1e-30,
        enable_Mie=0,
        n_aerosol_array=np.zeros((0, N_layers, N_sectors, N_zones)),
        sigma_Mie_array=np.zeros((0, N_wl)),
        P_deep=P_deep,
    )


@pytest.mark.parametrize(
    "label,contribution_species,fixture_kwargs",
    [
        ("H2O_1D_basic", "H2O", {}),
        ("CH4_1D_basic", "CH4", {}),
        ("H2O_with_active_cia_H2-H2O", "H2O", {"include_active_cia": True}),
        ("H2O_with_ff_pairs", "H2O", {"N_ff_pairs": 2}),
        ("H2O_multi_sector_2", "H2O", {"N_sectors": 2}),
        ("H2O_multi_zone_2", "H2O", {"N_zones": 2}),
        ("H2O_P_deep_inside_grid", "H2O", {"P_deep": 0.1}),
        ("He_1D_basic", "He", {}),
    ],
)
def test_spectral_contribution_kernel_jit_parity_with_numpy(
    label, contribution_species, fixture_kwargs
):
    cfg = _fixture(**fixture_kwargs)
    kg_t, kR_t, _kc_t = _contributions.extinction_spectral_contribution(
        contribution_species=contribution_species,
        bulk_species=False,
        cloud_contribution=False,
        cloud_species="",
        cloud_total_contribution=False,
        **cfg,
    )

    (csi, cai, cia_mask, bulk_cia_mask, bulk_species_mask) = (
        setup_spectral_contribution_indices(
            cfg["chemical_species"],
            cfg["active_species"],
            cfg["cia_pairs"],
            contribution_species,
        )
    )
    is_He = contribution_species == "He"

    # POSEIDON's "bulk_species=False, cloud_contribution=False" CIA branch:
    #   include CIA pair q if (q in bulk_cia_indices) OR (q in cia_indices)
    cia_combined_mask = cia_mask | bulk_cia_mask

    kg_o, kR_o = spectral_contribution_kernel_jit(
        jnp.asarray(cfg["n"]),
        jnp.asarray(cfg["T"]),
        jnp.asarray(cfg["P"]),
        jnp.asarray(cfg["X"]),
        jnp.asarray(cfg["X_active"]),
        jnp.asarray(cfg["X_cia"]),
        jnp.asarray(cfg["X_ff"]),
        jnp.asarray(cfg["sigma_stored"]),
        jnp.asarray(cfg["cia_stored"]),
        jnp.asarray(cfg["ff_stored"]),
        jnp.asarray(cfg["Rayleigh_stored"]),
        jnp.asarray(cfg["T_fine"]),
        jnp.asarray(cfg["log_P_fine"]),
        csi,
        cai,
        jnp.asarray(cia_combined_mask),
        jnp.asarray(bulk_species_mask),
        is_He,
        P_deep=cfg["P_deep"],
    )

    np.testing.assert_allclose(np.asarray(kg_o), kg_t, rtol=1e-13, atol=1e-50)
    np.testing.assert_allclose(np.asarray(kR_o), kR_t, rtol=1e-13, atol=1e-50)


def test_spectral_contribution_kernel_jit_is_jit_traceable():
    cfg = _fixture()
    (csi, cai, cia_mask, bulk_cia_mask, bulk_species_mask) = (
        setup_spectral_contribution_indices(
            cfg["chemical_species"],
            cfg["active_species"],
            cfg["cia_pairs"],
            "H2O",
        )
    )
    cia_combined_mask = cia_mask | bulk_cia_mask

    @jax.jit
    def _f(
        n, T, P, X, X_active, X_cia, X_ff,
        sigma_stored, cia_stored, ff_stored, Rayleigh_stored,
    ):
        return spectral_contribution_kernel_jit(
            n, T, P, X, X_active, X_cia, X_ff,
            sigma_stored, cia_stored, ff_stored, Rayleigh_stored,
            jnp.asarray(cfg["T_fine"]), jnp.asarray(cfg["log_P_fine"]),
            csi, cai,
            jnp.asarray(cia_combined_mask), jnp.asarray(bulk_species_mask),
            False,
        )

    kg, kR = _f(
        jnp.asarray(cfg["n"]),
        jnp.asarray(cfg["T"]),
        jnp.asarray(cfg["P"]),
        jnp.asarray(cfg["X"]),
        jnp.asarray(cfg["X_active"]),
        jnp.asarray(cfg["X_cia"]),
        jnp.asarray(cfg["X_ff"]),
        jnp.asarray(cfg["sigma_stored"]),
        jnp.asarray(cfg["cia_stored"]),
        jnp.asarray(cfg["ff_stored"]),
        jnp.asarray(cfg["Rayleigh_stored"]),
    )
    assert kg.shape == (20, 1, 1, 12)
    assert kR.shape == (20, 1, 1, 12)
    assert jnp.all(jnp.isfinite(kg))
    assert jnp.all(jnp.isfinite(kR))


def test_grad_through_jit_spectral_contribution_kernel_kappa_via_X_active():
    """jax.grad through X_active produces finite, nonzero gradient via the
    active-species kappa_gas accumulation path."""
    cfg = _fixture()
    (csi, cai, cia_mask, bulk_cia_mask, bulk_species_mask) = (
        setup_spectral_contribution_indices(
            cfg["chemical_species"],
            cfg["active_species"],
            cfg["cia_pairs"],
            "H2O",
        )
    )
    cia_combined_mask = cia_mask | bulk_cia_mask

    def _loss(X_active):
        kg, _kR = spectral_contribution_kernel_jit(
            jnp.asarray(cfg["n"]),
            jnp.asarray(cfg["T"]),
            jnp.asarray(cfg["P"]),
            jnp.asarray(cfg["X"]),
            X_active,
            jnp.asarray(cfg["X_cia"]),
            jnp.asarray(cfg["X_ff"]),
            jnp.asarray(cfg["sigma_stored"]),
            jnp.asarray(cfg["cia_stored"]),
            jnp.asarray(cfg["ff_stored"]),
            jnp.asarray(cfg["Rayleigh_stored"]),
            jnp.asarray(cfg["T_fine"]),
            jnp.asarray(cfg["log_P_fine"]),
            csi, cai,
            jnp.asarray(cia_combined_mask),
            jnp.asarray(bulk_species_mask),
            False,
        )
        return jnp.sum(kg)

    g = jax.grad(jax.jit(_loss))(jnp.asarray(cfg["X_active"]))
    assert g.shape == cfg["X_active"].shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0), "jax.grad through X_active is identically zero"


def test_make_jaxpr_spectral_contribution_kernel_succeeds():
    cfg = _fixture()
    (csi, cai, cia_mask, bulk_cia_mask, bulk_species_mask) = (
        setup_spectral_contribution_indices(
            cfg["chemical_species"],
            cfg["active_species"],
            cfg["cia_pairs"],
            "H2O",
        )
    )
    cia_combined_mask = cia_mask | bulk_cia_mask

    def _f(X_active):
        return spectral_contribution_kernel_jit(
            jnp.asarray(cfg["n"]),
            jnp.asarray(cfg["T"]),
            jnp.asarray(cfg["P"]),
            jnp.asarray(cfg["X"]),
            X_active,
            jnp.asarray(cfg["X_cia"]),
            jnp.asarray(cfg["X_ff"]),
            jnp.asarray(cfg["sigma_stored"]),
            jnp.asarray(cfg["cia_stored"]),
            jnp.asarray(cfg["ff_stored"]),
            jnp.asarray(cfg["Rayleigh_stored"]),
            jnp.asarray(cfg["T_fine"]),
            jnp.asarray(cfg["log_P_fine"]),
            csi, cai,
            jnp.asarray(cia_combined_mask),
            jnp.asarray(bulk_species_mask),
            False,
        )

    jaxpr = jax.make_jaxpr(_f)(jnp.asarray(cfg["X_active"]))
    assert jaxpr is not None
    assert "pure_callback" not in str(jaxpr)
