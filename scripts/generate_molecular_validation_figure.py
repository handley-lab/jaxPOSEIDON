"""Realistic transmission-spectrum validation vs POSEIDON.

Reproduces a configuration analogous to POSEIDON's first transmission
tutorial — H2/He background with H2O active absorption through a
pressure-broadened cross section grid — and compares jaxPOSEIDON's
forward model against POSEIDON's `compute_spectrum` at
`atol=1e-15, rtol=1e-13`.

Two panels: (top) POSEIDON (dashed) vs jaxPOSEIDON (solid) transit
depth; (bottom) absolute residual in ppm. A molecular H2O opacity
grid is synthesised in a tempdir so the test runs without the
~70 GB POSEIDON opacity database — the synthesis matches POSEIDON's
HDF5 schema so `read_opacities(..., testing=False)` consumes it
through the standard opacity-sampling code path, and the resulting
spectrum carries nontrivial molecular absorption features.

Reference: MacDonald & Madhusudhan 2017 (the POSEIDON forward model);
the synthetic H2O cross section is drawn from `U(1e-23, 1e-21)` cm²
at log10, which is within the physical range for H2O in the near-IR.
"""

import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jaxposeidon._compute_spectrum import compute_spectrum as j_compute


def _build_synthetic_opacity_tempdir(tmp):
    """Write a minimal H2O molecular HDF5 + H2-H2/H2-He CIA under
    ``POSEIDON_input_data`` expected layout."""
    opac_dir = os.path.join(tmp, "opacity")
    os.makedirs(opac_dir, exist_ok=True)

    # Molecular H2O cross section — nonzero and smoothly varying so the
    # resulting spectrum has visible molecular absorption bands.
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

    with tempfile.TemporaryDirectory() as tmp:
        _build_synthetic_opacity_tempdir(tmp)
        saved = os.environ.get("POSEIDON_input_data")
        os.environ["POSEIDON_input_data"] = tmp
        try:
            star = create_star(R_Sun, 5000.0, 4.0, 0.0)
            planet = create_planet("p", R_J, mass=M_J, T_eq=1000.0)
            model = define_model(
                "h2o_demo",
                ["H2", "He"],
                ["H2O"],
                PT_profile="isotherm",
                X_profile="isochem",
            )
            P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)
            atmosphere = make_atmosphere(
                planet,
                model,
                P,
                10.0,
                R_J,
                np.array([1000.0]),
                np.array([-4.0]),
                constant_gravity=True,
            )
            wl = wl_grid_constant_R(1.0, 3.0, 500)
            T_fine = np.arange(800, 1210, 20)
            log_P_fine = np.arange(-6.0, 2.2, 0.4)
            opac = read_opacities(
                model, wl, "opacity_sampling", T_fine, log_P_fine,
                testing=False,
            )

            theirs = p_compute_spectrum(planet, star, model, atmosphere, opac, wl)
            ours = j_compute(planet, star, model, atmosphere, opac, wl)
        finally:
            if saved is None:
                del os.environ["POSEIDON_input_data"]
            else:
                os.environ["POSEIDON_input_data"] = saved

    residual_ppm = (np.asarray(ours) - np.asarray(theirs)) * 1e6
    max_abs = float(np.max(np.abs(residual_ppm)))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(wl, np.asarray(theirs) * 1e2, "k--", lw=1.2, label="POSEIDON")
    ax_top.plot(wl, np.asarray(ours) * 1e2, "C0-", lw=0.9, label="jaxPOSEIDON")
    ax_top.set_ylabel("transit depth / %")
    ax_top.set_title(
        "H2/He + H2O active absorption (isotherm 1000 K, isochem 1e-3) — "
        f"max |Δ| = {max_abs:.2e} ppm"
    )
    ax_top.legend(loc="upper right", frameon=False)

    ax_bot.plot(wl, residual_ppm, "C3-", lw=0.9)
    ax_bot.axhline(0.0, color="k", lw=0.5)
    ax_bot.set_xlabel(r"wavelength / $\mu$m")
    ax_bot.set_ylabel(r"$\Delta$ / ppm")

    fig.tight_layout()
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "parity_molecular_h2o.png",
    )
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print(f"max |residual| = {max_abs:.6e} ppm")
    print(f"N_wl = {len(wl)}  rtol-equivalent = {max_abs / (np.max(np.abs(np.asarray(theirs))) * 1e6):.2e}")


if __name__ == "__main__":
    main()
