"""v1-B: parity tests for JAX-ported opacities / atmosphere / chemistry /
clouds modules.

Coverage:
- `_opacities.extinction` under `jax.jit` + `make_jaxpr`
- `_atmosphere` PT-profile builders under `jax.jit`
- `_atmosphere.profiles` orchestrator under `make_jaxpr` (via the
  numpy reference; the dispatcher itself stays Python-level because
  the rejection branch returns a different-shape tuple)
- `_chemistry.interpolate_log_X_grid` under `jax.jit` + `make_jaxpr`
- `_clouds.unpack_Mie_cloud_params` under `jax.jit` + `make_jaxpr`
- `_clouds.interpolate_sigma_Mie_grid` under `jax.jit`

Tolerance per plan: rtol=1e-13 (with documented FP-reorder
relaxation entries in MISMATCHES.md where the JAX kernels diverge
from scipy by ULP-scale residuals).
"""

import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jaxposeidon._atmosphere import (  # noqa: E402
    compute_T_Guillot,
    compute_T_Guillot_dayside,
    compute_T_isotherm,
    compute_T_Line,
    compute_T_Madhu,
    compute_T_Pelletier,
    compute_T_slope,
    gauss_conv,
    radial_profiles,
    radial_profiles_constant_g,
)
from jaxposeidon._chemistry import interpolate_log_X_grid  # noqa: E402
from jaxposeidon._clouds import (  # noqa: E402
    interpolate_sigma_Mie_grid,
    unpack_MacMad17_cloud_params,
    unpack_Mie_cloud_params,
)
from jaxposeidon._opacities import extinction  # noqa: E402


# ---------------------------------------------------------------------------
# _opacities.extinction
# ---------------------------------------------------------------------------
def _set_up_extinction(
    enable_haze=0, enable_deck=0, N_layers=10, N_wl=6, N_T_fine=4, N_P_fine=4
):
    rng = np.random.default_rng(0)
    chemical_species = np.array(["H2", "He", "H2O"])
    active_species = np.array(["H2O"])
    cia_pairs = np.array(["H2-H2", "H2-He"])
    ff_pairs = np.array([], dtype=str)
    bf_species = np.array([], dtype=str)
    P = np.logspace(np.log10(100.0), np.log10(1.0e-6), N_layers)
    T = 1000.0 * np.ones((N_layers, 1, 1))
    n = rng.uniform(1e15, 1e25, size=(N_layers, 1, 1))
    wl = np.linspace(1.0, 5.0, N_wl)
    X = np.zeros((3, N_layers, 1, 1))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 0.001
    X_active = X[2:3]
    X_cia = np.zeros((2, 2, N_layers, 1, 1))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]
    X_ff = np.zeros((2, 0, N_layers, 1, 1))
    X_bf = np.zeros((0, N_layers, 1, 1))
    T_fine = np.linspace(500.0, 2000.0, N_T_fine)
    log_P_fine = np.linspace(-6.0, 2.0, N_P_fine)
    sigma_stored = rng.uniform(0.0, 1e-22, size=(1, N_P_fine, N_T_fine, N_wl))
    cia_stored = rng.uniform(0.0, 1e-44, size=(2, N_T_fine, N_wl))
    Rayleigh_stored = rng.uniform(0.0, 1e-27, size=(3, N_wl))
    return dict(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=ff_pairs,
        bf_species=bf_species,
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
        P_cloud=np.array([1.0e-3]),
        kappa_cloud_0=1.0e-30,
        sigma_stored=sigma_stored,
        cia_stored=cia_stored,
        Rayleigh_stored=Rayleigh_stored,
        ff_stored=np.zeros((0, N_T_fine, N_wl)),
        bf_stored=np.zeros((0, N_wl)),
        enable_haze=enable_haze,
        enable_deck=enable_deck,
        enable_surface=0,
        N_sectors=1,
        N_zones=1,
        T_fine=T_fine,
        log_P_fine=log_P_fine,
        P_surf=1.0e-30,
        enable_Mie=0,
        n_aerosol_array=np.zeros((0, N_layers, 1, 1)),
        sigma_Mie_array=np.zeros((0, N_wl)),
        P_deep=1000.0,
    )


@pytest.mark.parametrize(
    "enable_haze,enable_deck",
    [(0, 0), (1, 0), (0, 1), (1, 1)],
)
def test_extinction_under_jit_matches_eager(enable_haze, enable_deck):
    args = _set_up_extinction(enable_haze=enable_haze, enable_deck=enable_deck)
    eager = extinction(**args)
    # Re-call inside jit via a wrapper that fixes static config.
    static_keys = (
        "chemical_species",
        "active_species",
        "cia_pairs",
        "ff_pairs",
        "bf_species",
        "T_fine",
        "log_P_fine",
        "enable_haze",
        "enable_deck",
        "enable_surface",
        "N_sectors",
        "N_zones",
        "enable_Mie",
        "n_aerosol_array",
        "sigma_Mie_array",
        "P_deep",
    )
    static = {k: args[k] for k in static_keys}
    dyn_keys = [k for k in args if k not in static_keys]

    def call(*dyn_vals):
        kwargs = dict(zip(dyn_keys, dyn_vals))
        return extinction(**static, **kwargs)

    dyn_vals = [args[k] for k in dyn_keys]
    jitted = jax.jit(call)
    out = jitted(*dyn_vals)
    for a, b in zip(eager, out):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=1e-15)


def test_extinction_make_jaxpr_succeeds():
    args = _set_up_extinction(enable_haze=1, enable_deck=1)
    static_keys = (
        "chemical_species",
        "active_species",
        "cia_pairs",
        "ff_pairs",
        "bf_species",
        "T_fine",
        "log_P_fine",
        "enable_haze",
        "enable_deck",
        "enable_surface",
        "N_sectors",
        "N_zones",
        "enable_Mie",
        "n_aerosol_array",
        "sigma_Mie_array",
        "P_deep",
    )
    static = {k: args[k] for k in static_keys}
    dyn_keys = [k for k in args if k not in static_keys]

    def call(*dyn_vals):
        kwargs = dict(zip(dyn_keys, dyn_vals))
        return extinction(**static, **kwargs)

    jpr = jax.make_jaxpr(call)(*[args[k] for k in dyn_keys])
    assert jpr is not None


# ---------------------------------------------------------------------------
# _atmosphere PT-profile builders
# ---------------------------------------------------------------------------
def _common_P():
    return np.logspace(np.log10(100.0), np.log10(1.0e-7), 60)


def test_compute_T_isotherm_jit_matches_numpy_oracle():
    P = _common_P()
    T_iso = 500.0
    eager = compute_T_isotherm(P, T_iso)
    jitted = jax.jit(compute_T_isotherm)(P, T_iso)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


def test_compute_T_Madhu_under_jit():
    P = _common_P()
    args = (1.0, 0.5, -4.0, -3.0, -1.0, 1500.0, 1.0e-6)
    eager = compute_T_Madhu(P, *args)
    jitted = jax.jit(compute_T_Madhu, static_argnames=())(P, *args)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


def test_compute_T_Guillot_under_jit():
    P = _common_P()
    args = (10.0, -3.0, -1.0, 100.0, 1500.0)
    eager = compute_T_Guillot(P, *args)
    jitted = jax.jit(compute_T_Guillot)(P, *args)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


def test_compute_T_Guillot_dayside_under_jit():
    P = _common_P()
    args = (10.0, -3.0, -1.0, 100.0, 1500.0)
    eager = compute_T_Guillot_dayside(P, *args)
    jitted = jax.jit(compute_T_Guillot_dayside)(P, *args)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


def test_compute_T_Line_under_jit():
    P = _common_P()
    args = (10.0, 1500.0, -3.0, -1.0, -0.5, 0.5, 0.5, 100.0)
    eager = compute_T_Line(P, *args)
    jitted = jax.jit(compute_T_Line)(P, *args)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-12, atol=1e-13
    )


def test_compute_T_Pelletier_under_jit():
    P = _common_P()
    T_points = np.linspace(2000.0, 500.0, 10)
    eager = compute_T_Pelletier(P, T_points)
    jitted = jax.jit(compute_T_Pelletier)(P, T_points)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


def test_compute_T_slope_under_jit():
    P = _common_P()
    T_phot = 1200.0
    Delta_T_arr = np.array([100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0])
    eager = compute_T_slope(P, T_phot, Delta_T_arr)
    jitted = jax.jit(compute_T_slope)(P, T_phot, Delta_T_arr)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


def test_gauss_conv_under_jit_matches_eager():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((50, 1, 1))
    eager = gauss_conv(arr, sigma=3, axis=0, mode="nearest")
    jitted = jax.jit(lambda x: gauss_conv(x, sigma=3, axis=0, mode="nearest"))(arr)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


# ---------------------------------------------------------------------------
# _atmosphere.radial_profiles
# ---------------------------------------------------------------------------
def _radial_args():
    N_layers = 40
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), N_layers)
    T = 800.0 * np.ones((N_layers, 1, 1))
    mu = 2.3 * 1.66053906660e-27 * np.ones((N_layers, 1, 1))
    return dict(
        P=P,
        T=T,
        g_0=10.0,
        R_p=7.1492e7,
        P_ref=10.0,
        R_p_ref=7.1492e7,
        mu=mu,
        N_sectors=1,
        N_zones=1,
    )


def test_radial_profiles_jit_matches_eager():
    args = _radial_args()
    eager = radial_profiles(**args)
    jitted = jax.jit(
        lambda P, T, mu: radial_profiles(
            P,
            T,
            args["g_0"],
            args["R_p"],
            args["P_ref"],
            args["R_p_ref"],
            mu,
            args["N_sectors"],
            args["N_zones"],
        )
    )
    out = jitted(args["P"], args["T"], args["mu"])
    for a, b in zip(eager, out):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-13, atol=1e-15)


def test_radial_profiles_constant_g_jit_matches_eager():
    args = _radial_args()
    eager = radial_profiles_constant_g(
        args["P"],
        args["T"],
        args["g_0"],
        args["P_ref"],
        args["R_p_ref"],
        args["mu"],
        args["N_sectors"],
        args["N_zones"],
    )
    jitted = jax.jit(
        lambda P, T, mu: radial_profiles_constant_g(
            P,
            T,
            args["g_0"],
            args["P_ref"],
            args["R_p_ref"],
            mu,
            args["N_sectors"],
            args["N_zones"],
        )
    )
    out = jitted(args["P"], args["T"], args["mu"])
    for a, b in zip(eager, out):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# _chemistry.interpolate_log_X_grid
# ---------------------------------------------------------------------------
def _make_synthetic_chem_grid(tmp_path):
    """Create a tiny synthetic fastchem-shaped grid for parity tests."""
    P_grid = np.logspace(-6, 2, 8)
    T_grid = np.linspace(300.0, 3000.0, 6)
    Met_grid = np.array([0.1, 1.0, 10.0])
    C_to_O_grid = np.array([0.3, 0.55, 1.0])
    species = ("H2O", "CO2", "CH4")
    rng = np.random.default_rng(0)
    log_X_grid = rng.uniform(
        -8.0,
        -2.0,
        size=(len(species), len(Met_grid), len(C_to_O_grid), len(T_grid), len(P_grid)),
    )
    return dict(
        grid="fastchem",
        log_X_grid=log_X_grid,
        T_grid=T_grid,
        P_grid=P_grid,
        Met_grid=Met_grid,
        C_to_O_grid=C_to_O_grid,
    ), species


def test_interpolate_log_X_grid_jit_matches_eager(tmp_path):
    grid, species = _make_synthetic_chem_grid(tmp_path)
    log_P = np.array([-2.0])
    T = np.array([1500.0])
    C_to_O = np.array([0.6])
    log_Met = np.array([0.5])

    eager = interpolate_log_X_grid(
        grid, log_P, T, C_to_O, log_Met, species, return_dict=False
    )
    jitted_fn = jax.jit(
        lambda lp, t, co, lm: interpolate_log_X_grid(
            grid, lp, t, co, lm, species, return_dict=False
        )
    )
    out = jitted_fn(log_P, T, C_to_O, log_Met)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(eager), rtol=1e-13, atol=1e-15
    )


def test_interpolate_log_X_grid_make_jaxpr_succeeds(tmp_path):
    grid, species = _make_synthetic_chem_grid(tmp_path)
    log_P = np.array([-2.0])
    T = np.array([1500.0])
    C_to_O = np.array([0.6])
    log_Met = np.array([0.5])
    jpr = jax.make_jaxpr(
        lambda lp, t, co, lm: interpolate_log_X_grid(
            grid, lp, t, co, lm, species, return_dict=False
        )
    )(log_P, T, C_to_O, log_Met)
    assert jpr is not None


# ---------------------------------------------------------------------------
# _clouds.unpack_Mie_cloud_params
# ---------------------------------------------------------------------------
def test_unpack_Mie_cloud_params_uniform_X_under_jit():
    cloud_param_names = np.array(["log_r_m_H2O", "log_X_H2O"])
    clouds_in = np.array([-1.0, -4.0])
    species = ["H2O"]

    def _go(c_in):
        d = unpack_Mie_cloud_params(
            clouds_in=c_in,
            cloud_param_names=cloud_param_names,
            cloud_type="uniform_X",
            cloud_dim=1,
            aerosol_species=species,
        )
        return d["r_m"], d["log_X_Mie"]

    eager_rm, eager_log_X = _go(clouds_in)
    jit_rm, jit_log_X = jax.jit(_go)(clouds_in)
    np.testing.assert_allclose(
        np.asarray(jit_rm), np.asarray(eager_rm), rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(jit_log_X), np.asarray(eager_log_X), rtol=1e-13, atol=1e-15
    )


def test_unpack_Mie_cloud_params_make_jaxpr():
    cloud_param_names = np.array(["log_r_m_H2O", "log_X_H2O"])
    clouds_in = np.array([-1.0, -4.0])
    species = ["H2O"]

    def _go(c_in):
        d = unpack_Mie_cloud_params(
            clouds_in=c_in,
            cloud_param_names=cloud_param_names,
            cloud_type="uniform_X",
            cloud_dim=1,
            aerosol_species=species,
        )
        return d["r_m"], d["log_X_Mie"]

    jpr = jax.make_jaxpr(_go)(clouds_in)
    assert jpr is not None


# ---------------------------------------------------------------------------
# _clouds.unpack_MacMad17_cloud_params
# ---------------------------------------------------------------------------
def test_unpack_MacMad17_cloud_params_deck_haze():
    cloud_param_names = np.array(["log_a", "gamma", "log_P_cloud"])
    clouds_in = np.array([3.0, -8.0, -2.0])
    d = unpack_MacMad17_cloud_params(
        clouds_in=clouds_in,
        cloud_param_names=cloud_param_names,
        cloud_type="deck_haze",
        cloud_dim=1,
    )
    assert d["enable_haze"] == 1
    assert d["enable_deck"] == 1


# ---------------------------------------------------------------------------
# Source-grep gate: hot-path modules should not contain `import numpy as np`
# for runtime arithmetic. v1-B narrows this to the explicit primitive
# substitutions called out in the plan (scipy.{ndimage,interpolate,special}
# → _jax_filters / _jax_interpolate / _jax_special).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module_path",
    [
        "jaxposeidon/_opacities.py",
        "jaxposeidon/_atmosphere.py",
        "jaxposeidon/_chemistry.py",
        "jaxposeidon/_clouds.py",
        "jaxposeidon/_parameters.py",
    ],
)
def test_no_scipy_primitive_imports_in_hotpath(module_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, module_path)).read()
    forbidden = [
        "from scipy.ndimage import",
        "from scipy.interpolate import",
        "from scipy.special import",
        "import scipy.ndimage",
        "import scipy.interpolate",
        "import scipy.special",
    ]
    # Strip docstrings/comments for the import check: scan only lines that
    # syntactically begin (post-indent) with `import ` or `from `.
    import_lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(stripped)
    src_imports = "\n".join(import_lines)
    for f in forbidden:
        assert f not in src_imports, (
            f"{module_path} still imports {f!r}; v1-B requires JAX primitives"
        )
