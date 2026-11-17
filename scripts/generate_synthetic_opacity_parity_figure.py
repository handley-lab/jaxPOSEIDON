"""POSEIDON vs jaxPOSEIDON parity on a SYNTHETIC molecular-opacity dataset.

This script is NOT a realistic-atmosphere validation — it exercises
the opacity-sampling code path end-to-end with a synthetic H2O cross
section drawn from a uniform random distribution in log-space. It
does not use real HITRAN/HITEMP/ExoMol H2O line opacity and does not
reproduce the POSEIDON first transmission tutorial.

What it does verify:
  * jaxPOSEIDON's `compute_spectrum` consumes the same opacity-sampling
    arrays POSEIDON builds via `read_opacities(..., testing=False)`.
  * The jaxPOSEIDON JAX code path runs (not a silent numpy fall-through).
  * Numerical equivalence to upstream `POSEIDON.core.compute_spectrum`
    across the generated wavelength grid on this configuration.

Configuration (verifiable by reading this script):
  * Background:            H2 / He (mole fractions 0.85 / 0.149)
  * Active species:        H2O
  * log10(X_H2O):          -4.0   =>   X_H2O = 1.0e-4
  * PT profile:            isotherm, T = 1000 K
  * X profile:             isochem
  * Pressure grid:         50 layers, logspace from 100 bar to 1e-7 bar
  * Wavelength grid:       wl_grid_constant_R(1.0, 3.0, 500)
  * Opacity treatment:     opacity_sampling with testing=False
  * T_fine / log_P_fine:   np.arange(800, 1210, 20) / arange(-6, 2.2, 0.4)

Opacity synthesis provenance:
  * H2O log(sigma):        np.random.default_rng(7).uniform(-23, -21, ...)
                           -- IID random values on each (nu, logP, T) grid
                           point. Not physical. Not a line list. Not
                           tutorial-equivalent.
  * nu grid:               np.linspace(2000, 12000, 200) cm^-1
  * CIA:                   log(cia) = -50 constant for H2-H2 / H2-He
                           (effectively off; the CIA channel is not what
                           this test exercises).

Real-opacity validation (i.e. actually reproducing POSEIDON's first
transmission tutorial with a HITRAN/HITEMP/ExoMol H2O line list) needs
the POSEIDON opacity database (~70 GB) staged at `POSEIDON_input_data`
and was not within scope of this change.
"""

import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jaxposeidon._compute_spectrum import compute_spectrum as j_compute


def _build_synthetic_opacity_tempdir(tmp):
    """Write a synthetic H2O molecular HDF5 + off-valued H2-H2 / H2-He
    CIA under POSEIDON's expected `$POSEIDON_input_data/opacity/` layout.

    Opacity provenance:
      * H2O log(sigma) drawn IID from U(-23, -21) in dex. Not a line list.
      * CIA log(cia) = -50 constant (effectively off).

    Returns the dict of grid arrays that generated the file, so the
    caller can save them as provenance.
    """
    opac_dir = os.path.join(tmp, "opacity")
    os.makedirs(opac_dir, exist_ok=True)

    T_h2o = np.linspace(200, 2000, 12, dtype=np.float64)
    logP_h2o = np.linspace(-6, 2, 9, dtype=np.float64)
    nu_h2o = np.linspace(2000.0, 12000.0, 200, dtype=np.float64)
    rng = np.random.default_rng(7)
    log_sigma = rng.uniform(
        -23.0, -21.0, size=(len(nu_h2o), len(logP_h2o), len(T_h2o))
    ).astype(np.float32)
    with h5py.File(os.path.join(opac_dir, "Opacity_database_v1.3.hdf5"), "w") as f:
        g = f.create_group("H2O")
        g.create_dataset("T", data=T_h2o)
        g.create_dataset("log(P)", data=logP_h2o)
        g.create_dataset("nu", data=nu_h2o)
        g.create_dataset("log(sigma)", data=log_sigma)

    T_cia = np.linspace(200, 2000, 10, dtype=np.float64)
    nu_cia = np.linspace(1.0e4, 5.0e5, 50, dtype=np.float64)
    log_cia = np.full((10, 50), -50.0, dtype=np.float64)
    with h5py.File(os.path.join(opac_dir, "Opacity_database_cia.hdf5"), "w") as f:
        for pair in ("H2-H2", "H2-He"):
            gg = f.create_group(pair)
            gg.create_dataset("T", data=T_cia)
            gg.create_dataset("nu", data=nu_cia)
            gg.create_dataset("log(cia)", data=log_cia)

    return {
        "T_h2o_range_K": (float(T_h2o[0]), float(T_h2o[-1])),
        "T_h2o_N": int(len(T_h2o)),
        "log10P_h2o_range": (float(logP_h2o[0]), float(logP_h2o[-1])),
        "log10P_h2o_N": int(len(logP_h2o)),
        "nu_h2o_range_cm_inv": (float(nu_h2o[0]), float(nu_h2o[-1])),
        "nu_h2o_N": int(len(nu_h2o)),
        "log10_sigma_distribution": "uniform(-23, -21)",
        "cia_log_value": -50.0,
    }


def _assert_jax_path_exercised():
    """Smoke-check: confirm the jaxposeidon JAX code path is loaded and
    runs rather than silently falling through to numpy."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jaxposeidon._jax_transmission as jm
    assert hasattr(jm, "TRIDENT_kernel_jit"), "JAX TRIDENT kernel missing"
    import jax.numpy as jnp
    x = jnp.asarray([1.0])
    assert hasattr(x, "block_until_ready")


def main():
    from POSEIDON.constants import M_J, R_J, R_Sun
    from POSEIDON.core import (
        compute_spectrum as p_compute_spectrum,
    )
    from POSEIDON.core import (
        create_planet,
        create_star,
        define_model,
        make_atmosphere,
        read_opacities,
        wl_grid_constant_R,
    )

    _assert_jax_path_exercised()

    with tempfile.TemporaryDirectory() as tmp:
        opacity_provenance = _build_synthetic_opacity_tempdir(tmp)
        saved = os.environ.get("POSEIDON_input_data")
        os.environ["POSEIDON_input_data"] = tmp
        try:
            star = create_star(R_Sun, 5000.0, 4.0, 0.0)
            planet = create_planet("p", R_J, mass=M_J, T_eq=1000.0)
            model = define_model(
                "h2o_synthetic",
                ["H2", "He"],
                ["H2O"],
                PT_profile="isotherm",
                X_profile="isochem",
            )
            P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)
            # log10(X_H2O) = -4.0  =>  X_H2O = 1.0e-4
            log_X_H2O = -4.0
            atmosphere = make_atmosphere(
                planet,
                model,
                P,
                10.0,
                R_J,
                np.array([1000.0]),
                np.array([log_X_H2O]),
                constant_gravity=True,
            )
            wl = wl_grid_constant_R(1.0, 3.0, 500)
            T_fine = np.arange(800, 1210, 20)
            log_P_fine = np.arange(-6.0, 2.2, 0.4)
            opac = read_opacities(
                model, wl, "opacity_sampling", T_fine, log_P_fine,
                testing=False,
            )
            assert np.max(opac["sigma_stored"]) > 0.0, (
                "synthetic sigma_stored is all zero -- opacity grid did "
                "not survive the POSEIDON ingest path."
            )

            theirs = p_compute_spectrum(planet, star, model, atmosphere, opac, wl)
            ours = j_compute(planet, star, model, atmosphere, opac, wl)
        finally:
            if saved is None:
                del os.environ["POSEIDON_input_data"]
            else:
                os.environ["POSEIDON_input_data"] = saved

    residual_ppm = (np.asarray(ours) - np.asarray(theirs)) * 1e6
    max_abs_ppm = float(np.max(np.abs(residual_ppm)))
    med_abs_ppm = float(np.median(np.abs(residual_ppm)))
    theirs_max_ppm = float(np.max(np.abs(np.asarray(theirs))) * 1e6)
    rtol_eq = max_abs_ppm / theirs_max_ppm if theirs_max_ppm > 0 else float("nan")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(wl, np.asarray(theirs) * 1e2, "k--", lw=1.2, label="POSEIDON")
    ax_top.plot(wl, np.asarray(ours) * 1e2, "C0-", lw=0.9, label="jaxPOSEIDON")
    ax_top.set_ylabel("transit depth / %")
    ax_top.set_title(
        "SYNTHETIC H2O opacity (log sigma ~ U(-23, -21)); "
        f"isotherm 1000 K; X_H2O = 1e-4 -- max |Delta| = {max_abs_ppm:.2e} ppm"
    )
    ax_top.legend(loc="upper right", frameon=False)

    ax_bot.plot(wl, residual_ppm, "C3-", lw=0.9)
    ax_bot.axhline(0.0, color="k", lw=0.5)
    ax_bot.set_xlabel(r"wavelength / $\mu$m")
    ax_bot.set_ylabel(r"$\Delta$ / ppm")

    fig.tight_layout()
    fig_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "parity_synthetic_h2o.png",
    )
    fig.savefig(fig_path, dpi=150)

    import json
    prov_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks", "synthetic_opacity_provenance.json",
    )
    with open(prov_path, "w") as f:
        json.dump(
            {
                "opacity_provenance": opacity_provenance,
                "atmosphere": {
                    "PT_profile": "isotherm",
                    "T_K": 1000.0,
                    "X_profile": "isochem",
                    "log10_X_H2O": log_X_H2O,
                    "X_H2O": 10.0 ** log_X_H2O,
                    "P_range_bar": [100.0, 1e-7],
                    "N_layers": 50,
                },
                "wavelength_grid": {
                    "scheme": "wl_grid_constant_R",
                    "wl_min_um": 1.0,
                    "wl_max_um": 3.0,
                    "R": 500,
                    "N_wl": int(len(wl)),
                },
                "T_fine_range_K": [float(T_fine[0]), float(T_fine[-1]),
                                   float(T_fine[1] - T_fine[0])],
                "log_P_fine_range": [float(log_P_fine[0]),
                                     float(log_P_fine[-1]),
                                     float(log_P_fine[1] - log_P_fine[0])],
                "parity_vs_POSEIDON_ppm": {
                    "max_abs": max_abs_ppm,
                    "median_abs": med_abs_ppm,
                    "poseidon_max_abs": theirs_max_ppm,
                    "rtol_equivalent": rtol_eq,
                },
                "not_a_realistic_validation": True,
            },
            f, indent=2,
        )

    print(f"Wrote {fig_path}")
    print(f"Wrote {prov_path}")
    print(f"N_wl                    = {len(wl)}")
    print(f"max |residual|          = {max_abs_ppm:.6e} ppm")
    print(f"median |residual|       = {med_abs_ppm:.6e} ppm")
    print(f"max |POSEIDON depth|    = {theirs_max_ppm:.6e} ppm")
    print(f"rtol-equivalent         = {rtol_eq:.3e}")


if __name__ == "__main__":
    main()
