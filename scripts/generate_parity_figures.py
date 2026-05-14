"""Generate side-by-side POSEIDON vs jaxPOSEIDON parity figures.

Builds a synthetic POSEIDON_input_data CIA HDF5 in a tempdir (no 70 GB
opacity database needed for the Rayleigh/MacMad17/CIA path), then runs
the full forward model through both POSEIDON's compute_spectrum and
jaxposeidon's port across several v0 configurations representative of
K2-18b-style retrievals:

  1. Clear H2/He Rayleigh (canonical oracle)
  2. Madhu P-T (MS09 6-param)
  3. MacMad17 haze (Rayleigh-enhanced, γ=-4)
  4. MacMad17 deck (grey cloud top)
  5. MacMad17 deck + haze
  6. cloud_dim=2 patchy deck (φ partial coverage)

Each panel: top = both spectra overlaid (POSEIDON dashed, jaxPOSEIDON
solid); bottom = residual in ppm. Tight axes so the eye can resolve
the FP-precision agreement.
"""

import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jaxposeidon._compute_spectrum import compute_spectrum as j_compute


def _synth_cia(tmp):
    """Build a synthetic CIA HDF5 in tmp/opacity so POSEIDON's testing=True
    read_opacities can open it. H2-H2 and H2-He pairs only."""
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
    return tmp


def _build(planet_T_eq, PT_profile, PT_params, cloud_model="cloud-free",
           cloud_type="cloud-free", cloud_dim=1, cloud_params=None,
           wl_R=2000, wl_min=0.6, wl_max=5.0):
    """Build (planet, star, model, atmosphere, opac, wl) for one config."""
    from POSEIDON.constants import M_J, R_J, R_Sun
    from POSEIDON.core import (
        create_planet,
        create_star,
        define_model,
        make_atmosphere,
        read_opacities,
        wl_grid_constant_R,
    )

    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("K2-18 b-ish", R_J, mass=M_J, T_eq=planet_T_eq)
    define_kwargs = {"PT_profile": PT_profile}
    if cloud_model != "cloud-free":
        define_kwargs.update(
            cloud_model=cloud_model,
            cloud_type=cloud_type,
            cloud_dim=cloud_dim,
        )
    model = define_model("demo", ["H2", "He"], [], **define_kwargs)
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    mka_kwargs = {"constant_gravity": True}
    if cloud_params is not None:
        mka_kwargs["cloud_params"] = cloud_params
    atmosphere = make_atmosphere(
        planet, model, P, 10.0, R_J,
        np.asarray(PT_params), np.array([]),
        **mka_kwargs,
    )
    wl = wl_grid_constant_R(wl_min, wl_max, wl_R)
    T_min = max(200, planet_T_eq - 200)
    T_max = planet_T_eq + 200
    T_fine = np.arange(T_min, T_max + 10, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    return planet, star, model, atmosphere, opac, wl


def _both(planet, star, model, atmosphere, opac, wl):
    from POSEIDON.core import compute_spectrum as p_compute
    ours = j_compute(planet, star, model, atmosphere, opac, wl,
                     spectrum_type="transmission")
    theirs = p_compute(planet, star, model, atmosphere, opac, wl,
                       spectrum_type="transmission")
    return ours, theirs


CONFIGS = [
    # (label, build_kwargs)
    ("Clear H2/He isothermal\n(T=900K)",
     dict(planet_T_eq=900.0, PT_profile="isotherm", PT_params=[900.0])),
    ("Madhu MS09 P-T\n(T_set=900, α1=0.7, α2=0.6)",
     dict(planet_T_eq=900.0, PT_profile="Madhu",
          PT_params=[0.7, 0.6, -4.0, -1.5, 1.0, 900.0])),
    ("MacMad17 Rayleigh-enhanced haze\n(log a=2, γ=-4)",
     dict(planet_T_eq=900.0, PT_profile="isotherm", PT_params=[900.0],
          cloud_model="MacMad17", cloud_type="haze", cloud_dim=1,
          cloud_params=[2.0, -4.0])),
    ("MacMad17 grey deck\n(log P_c=-2)",
     dict(planet_T_eq=900.0, PT_profile="isotherm", PT_params=[900.0],
          cloud_model="MacMad17", cloud_type="deck", cloud_dim=1,
          cloud_params=[-2.0])),
    ("MacMad17 deck + haze\n(log a=1.5, γ=-2, log P_c=-1)",
     dict(planet_T_eq=900.0, PT_profile="isotherm", PT_params=[900.0],
          cloud_model="MacMad17", cloud_type="deck_haze", cloud_dim=1,
          cloud_params=[1.5, -2.0, -1.0])),
    ("cloud_dim=2 patchy deck\n(log P_c=-1, φ=0.5)",
     dict(planet_T_eq=900.0, PT_profile="isotherm", PT_params=[900.0],
          cloud_model="MacMad17", cloud_type="deck", cloud_dim=2,
          cloud_params=[-1.0, 0.5])),
]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        _synth_cia(tmp)
        os.environ["POSEIDON_input_data"] = tmp

        fig, axes = plt.subplots(
            len(CONFIGS), 2, figsize=(13, 2.6 * len(CONFIGS)),
            gridspec_kw={"width_ratios": [2.2, 1], "hspace": 0.35, "wspace": 0.3},
        )

        for row, (label, kwargs) in enumerate(CONFIGS):
            print(f"[{row + 1}/{len(CONFIGS)}] {label.replace(chr(10), ' ')}")
            planet, star, model, atmosphere, opac, wl = _build(**kwargs)
            ours, theirs = _both(planet, star, model, atmosphere, opac, wl)

            ax_spec = axes[row, 0]
            ax_resid = axes[row, 1]

            ax_spec.plot(wl, theirs * 1e6, "C0--", lw=1.6, label="POSEIDON")
            ax_spec.plot(wl, ours * 1e6, "C1-", lw=0.9, label="jaxPOSEIDON")
            ax_spec.set_ylabel("transit depth (ppm)")
            ax_spec.set_title(label, fontsize=9, loc="left")
            ax_spec.legend(loc="best", frameon=False, fontsize=8)
            if row == len(CONFIGS) - 1:
                ax_spec.set_xlabel("wavelength (μm)")

            resid_ppm = (ours - theirs) * 1e6
            max_abs = np.max(np.abs(resid_ppm))
            ax_resid.plot(wl, resid_ppm, "C3", lw=0.8)
            ax_resid.axhline(0, color="k", lw=0.4, alpha=0.5)
            ax_resid.set_ylabel("Δ (ppm)")
            ax_resid.set_title(
                f"max |Δ| = {max_abs:.2e} ppm", fontsize=9, loc="left",
            )
            if row == len(CONFIGS) - 1:
                ax_resid.set_xlabel("wavelength (μm)")
            # symmetric tight y-axis around zero
            if max_abs > 0:
                ax_resid.set_ylim(-1.5 * max_abs, 1.5 * max_abs)

        fig.suptitle(
            "jaxPOSEIDON vs POSEIDON — v0 transmission-spectrum parity\n"
            "(R = 2000, λ = 0.6–5 μm, R_J planet, solar-type host)",
            fontsize=12, y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "parity_spectra.png",
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
