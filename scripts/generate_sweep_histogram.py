"""Residual-distribution histogram across the 396-case parametric sweep.

Re-runs the Phase 9 sweep matrix (300 atmosphere + 96 cloud cases) and
collects the max-per-case |Δ(transit depth)| in ppm, then plots the
histogram. The plan target is ≤ 1 ppm on the binned spectrum; this
shows the raw forward-model agreement, which is much tighter.
"""

import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jaxposeidon._compute_spectrum import compute_spectrum as j_compute


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


def _atm_case(T_iso, rp_fac, P_ref, bulk):
    from POSEIDON.constants import M_J, R_J, R_Sun
    from POSEIDON.core import (
        create_planet, create_star, define_model, make_atmosphere,
        read_opacities, wl_grid_constant_R,
    )
    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=M_J, T_eq=T_iso)
    model = define_model("m", bulk, [], PT_profile="isotherm")
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 60)
    atmosphere = make_atmosphere(
        planet, model, P, P_ref, R_J * rp_fac,
        np.array([T_iso]), np.array([]),
        constant_gravity=True,
    )
    wl = wl_grid_constant_R(0.5, 5.0, 1500)
    T_fine = np.arange(max(200, T_iso - 200), T_iso + 210, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    return planet, star, model, atmosphere, opac, wl


def main():
    from POSEIDON.core import compute_spectrum as p_compute

    with tempfile.TemporaryDirectory() as tmp:
        _synth_cia(tmp)
        os.environ["POSEIDON_input_data"] = tmp

        T_VALUES = [500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0]
        RP_FAC = [0.9, 0.95, 1.0, 1.05, 1.1]
        P_REF = [1.0, 5.0, 10.0, 50.0, 100.0]
        BULKS = [["H2"], ["H2", "He"]]

        from POSEIDON.constants import M_J, R_J, R_Sun
        from POSEIDON.core import (
            create_planet, create_star, define_model, make_atmosphere,
            read_opacities, wl_grid_constant_R,
        )

        deltas = []
        labels = []
        for T in T_VALUES:
            for rp in RP_FAC:
                for P_ref in P_REF:
                    for bulk in BULKS:
                        planet, star, model, atm, opac, wl = _atm_case(
                            T, rp, P_ref, bulk,
                        )
                        ours = j_compute(planet, star, model, atm, opac, wl)
                        theirs = p_compute(planet, star, model, atm, opac, wl,
                                             spectrum_type="transmission")
                        if np.any(np.isnan(theirs)):
                            continue
                        deltas.append(np.max(np.abs(ours - theirs)) * 1e6)
                        labels.append("atm")

        # Cloud sweep — 96 MacMad17 cases.
        for cloud_type in ["deck", "haze", "deck_haze"]:
            for cloud_dim in [1, 2]:
                for f_cloud in [0.3, 0.7]:
                    for log_a in [-2.0, 2.0]:
                        for gamma in [-4.0, 0.0]:
                            for log_Pc in [-2.0, 0.0]:
                                star = create_star(R_Sun, 5000.0, 4.0, 0.0)
                                planet = create_planet("p", R_J, mass=M_J,
                                                        T_eq=1000.0)
                                model = define_model(
                                    "m", ["H2", "He"], [],
                                    PT_profile="isotherm",
                                    cloud_model="MacMad17",
                                    cloud_type=cloud_type, cloud_dim=cloud_dim,
                                )
                                P = np.logspace(np.log10(100.0),
                                                 np.log10(1.0e-7), 60)
                                if cloud_dim == 1:
                                    if cloud_type == "deck":
                                        cp = np.array([log_Pc])
                                    elif cloud_type == "haze":
                                        cp = np.array([log_a, gamma])
                                    else:
                                        cp = np.array([log_a, gamma, log_Pc])
                                else:
                                    if cloud_type == "deck":
                                        cp = np.array([log_Pc, f_cloud])
                                    elif cloud_type == "haze":
                                        cp = np.array([log_a, gamma, f_cloud])
                                    else:
                                        cp = np.array(
                                            [log_a, gamma, log_Pc, f_cloud],
                                        )
                                atm = make_atmosphere(
                                    planet, model, P, 10.0, R_J,
                                    np.array([1000.0]), np.array([]),
                                    cloud_params=cp,
                                    constant_gravity=True,
                                )
                                wl = wl_grid_constant_R(0.5, 5.0, 1500)
                                T_fine = np.arange(800, 1210, 20)
                                log_P_fine = np.arange(-6.0, 2.2, 0.4)
                                opac = read_opacities(
                                    model, wl, "opacity_sampling",
                                    T_fine, log_P_fine, testing=True,
                                )
                                ours = j_compute(planet, star, model, atm,
                                                  opac, wl)
                                theirs = p_compute(planet, star, model, atm,
                                                    opac, wl,
                                                    spectrum_type="transmission")
                                if np.any(np.isnan(theirs)):
                                    continue
                                deltas.append(
                                    np.max(np.abs(ours - theirs)) * 1e6,
                                )
                                labels.append("cloud")

        deltas = np.array(deltas)
        labels = np.array(labels)
        n_atm = int(np.sum(labels == "atm"))
        n_cloud = int(np.sum(labels == "cloud"))
        n_bit_exact = int(np.sum(deltas == 0))
        print(f"{len(deltas)} cases (atm={n_atm}, cloud={n_cloud}); "
              f"bit-exact = {n_bit_exact}; "
              f"max |Δ| = {deltas.max():.2e} ppm; "
              f"median = {np.median(deltas):.2e} ppm")

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        # Floor zero deltas to a fixed "<= 1e-12 ppm" bin so they appear.
        floor = 1e-12
        cloud_d = np.clip(deltas[labels == "cloud"], floor, None)
        atm_d = np.clip(deltas[labels == "atm"], floor, None)
        bins = np.logspace(np.log10(floor) - 0.3,
                            np.log10(max(deltas.max() * 5, 1e-8)), 40)
        ax.hist([atm_d, cloud_d], bins=bins, stacked=True,
                 color=["C0", "C2"],
                 label=[f"atmosphere sweep ({n_atm} cases, "
                        f"{int((deltas[labels == 'atm'] == 0).sum())} bit-exact)",
                        f"MacMad17 cloud sweep ({n_cloud} cases)"],
                 edgecolor="black", linewidth=0.5)
        ax.set_xscale("log")
        ax.axvline(1.0, color="C3", ls="--",
                    label="plan target: ≤ 1 ppm binned")
        ax.set_xlabel("max-per-case |Δ(transit depth)|  (ppm)\n"
                       f"(zero-residual cases floored to {floor:g} for display)")
        ax.set_ylabel("# cases")
        ax.set_title(
            f"Forward-model parity across the Phase 9 sweep\n"
            f"({len(deltas)} cases total: "
            f"{n_atm} atmosphere × {n_cloud} MacMad17 cloud)",
            fontsize=11, loc="left",
        )
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "sweep_histogram.png",
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
