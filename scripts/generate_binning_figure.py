"""POSEIDON vs jaxPOSEIDON end-to-end through the instrument model.

Runs compute_spectrum → bin_spectrum_to_data through both stacks and
plots the binned data vectors with simulated NIRSpec-PRISM-like
binning. The binned y-vector is what enters the likelihood.
"""

import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jaxposeidon._compute_spectrum import compute_spectrum as j_compute
from jaxposeidon._instruments import bin_spectrum_to_data as j_bin
from jaxposeidon._instruments import compute_instrument_indices


def _synth_cia(tmp):
    os.makedirs(os.path.join(tmp, "opacity"), exist_ok=True)
    path = os.path.join(tmp, "opacity", "Opacity_database_cia.hdf5")
    T_grid = np.linspace(200, 2000, 10, dtype=np.float64)
    nu = np.linspace(1.0e4, 5.0e5, 50, dtype=np.float64)
    log_cia = np.full((10, 50), -50.0, dtype=np.float64)
    with h5py.File(path, "w") as f:
        for pair in ("H2-H2", "H2-He"):
            g = f.create_group(pair)
            g.create_dataset("T", data=T_grid)
            g.create_dataset("nu", data=nu)
            g.create_dataset("log(cia)", data=log_cia)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        _synth_cia(tmp)
        os.environ["POSEIDON_input_data"] = tmp

        from POSEIDON.constants import M_J, R_J, R_Sun
        from POSEIDON.core import (
            compute_spectrum as p_compute,
            create_planet,
            create_star,
            define_model,
            make_atmosphere,
            read_opacities,
            wl_grid_constant_R,
        )
        from POSEIDON.instrument import bin_spectrum_to_data as p_bin

        star = create_star(R_Sun, 5000.0, 4.0, 0.0)
        planet = create_planet("demo", R_J, mass=M_J, T_eq=900.0)
        model = define_model(
            "demo", ["H2", "He"], [],
            PT_profile="isotherm",
            cloud_model="MacMad17", cloud_type="deck_haze", cloud_dim=1,
        )
        P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
        atmosphere = make_atmosphere(
            planet, model, P, 10.0, R_J,
            np.array([900.0]), np.array([]),
            cloud_params=np.array([1.5, -2.0, -1.0]),
            constant_gravity=True,
        )
        wl = wl_grid_constant_R(0.5, 5.5, 4000)
        T_fine = np.arange(700, 1110, 20)
        log_P_fine = np.arange(-6.0, 2.2, 0.4)
        opac = read_opacities(
            model, wl, "opacity_sampling", T_fine, log_P_fine, testing=True,
        )

        spec_jpo = j_compute(planet, star, model, atmosphere, opac, wl)
        spec_pos = p_compute(planet, star, model, atmosphere, opac, wl,
                              spectrum_type="transmission")

        # Build a JWST NIRSpec-PRISM-like bin layout (R ~ 100).
        N_bins = 40
        wl_data = np.geomspace(0.8, 4.5, N_bins)
        half_width = 0.5 * np.diff(np.concatenate([
            [wl_data[0] * 0.95], wl_data, [wl_data[-1] * 1.05],
        ]))[:-1]
        rng = np.random.default_rng(0)
        sensitivity = rng.uniform(0.3, 1.0, size=len(wl))
        # smooth the sensitivity for realism
        from scipy.ndimage import gaussian_filter1d
        sensitivity = gaussian_filter1d(sensitivity, sigma=20)
        fwhm_um = 0.5 * np.gradient(wl_data) * 2.355
        sigma_grid, bin_left, bin_cent, bin_right, norm = (
            compute_instrument_indices(wl, wl_data, half_width, sensitivity,
                                        fwhm_um)
        )
        N_wl = len(wl)
        dp = dict(
            datasets=["d"], instruments=["JWST_NIRSpec_PRISM"],
            psf_sigma=sigma_grid, sens=sensitivity,
            bin_left=bin_left, bin_cent=bin_cent, bin_right=bin_right,
            norm=norm,
            len_data_idx=np.array([0, N_bins]),
        )
        bin_jpo = j_bin(spec_jpo, wl, dp)
        bin_pos = p_bin(spec_pos, wl, dp)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10, 8),
            gridspec_kw={"height_ratios": [3, 3, 1.4], "hspace": 0.05},
            sharex=True,
        )

        ax1.plot(wl, spec_pos * 1e6, "C0-", lw=0.6, alpha=0.5,
                  label="POSEIDON (high-res)")
        ax1.plot(wl, spec_jpo * 1e6, "C1-", lw=0.6, alpha=0.5,
                  label="jaxPOSEIDON (high-res)")
        ax1.set_ylabel("transit depth (ppm)")
        ax1.legend(loc="upper right", frameon=False, fontsize=9)
        ax1.set_title(
            "End-to-end: high-res spectrum → instrument binning\n"
            "MacMad17 deck+haze, T=900K, R=4000 model grid",
            fontsize=11, loc="left",
        )

        ax2.errorbar(wl_data, bin_pos * 1e6, fmt="o", color="C0",
                      ms=7, mfc="none", mew=1.5, label="POSEIDON")
        ax2.errorbar(wl_data, bin_jpo * 1e6, fmt="x", color="C1",
                      ms=6, mew=1.5, label="jaxPOSEIDON")
        ax2.set_ylabel("binned depth (ppm)")
        ax2.legend(loc="upper right", frameon=False, fontsize=9)

        resid_ppm = (bin_jpo - bin_pos) * 1e6
        ax3.stem(wl_data, resid_ppm, basefmt=" ", linefmt="C3-",
                  markerfmt="C3o")
        ax3.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax3.set_ylabel("binned Δ (ppm)")
        ax3.set_xlabel("wavelength (μm)")
        ax3.set_title(
            f"max binned |Δ| = {np.max(np.abs(resid_ppm)):.2e} ppm   "
            f"(plan target: ≤ 1 ppm)",
            fontsize=9, loc="left",
        )
        ymax = max(np.max(np.abs(resid_ppm)) * 1.5, 1e-15)
        ax3.set_ylim(-ymax, ymax)

        fig.tight_layout()
        out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "parity_binned.png",
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
