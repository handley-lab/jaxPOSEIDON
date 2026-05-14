"""H-minus bound-free + free-free opacity — Phase 0.5.4 port.

Faithful numpy port of POSEIDON `absorption.py:555-691`:
  - `H_minus_bound_free(...)` (John 1988 fit; photodissociation
    threshold at λ = 1.6421 μm)
  - `H_minus_free_free(...)` (John 1988 short-wavelength fit for
    0.182 ≤ λ < 0.3645 μm; long-wavelength fit for λ ≥ 0.3645 μm)

These are continuum opacity contributions for hot H/e- atmospheres
(e.g. ultra-hot Jupiters where H- forms via H + e- ↔ H-).
"""

import numpy as np


def H_minus_bound_free(wl_um):
    """Bound-free cross section α_bf(λ) of the H- ion [m² / n_H-].

    Faithful port of POSEIDON `absorption.py:556-604` (John 1988
    fit). Extinction coefficient per atmosphere layer:
        κ_bf(λ) = α_bf(λ) · n_(H-)

    Returns α_bf shape `(len(wl_um),)`.
    """
    wl_um = np.asarray(wl_um, dtype=np.float64)
    alpha_bf = np.zeros(len(wl_um))

    # John 1988 fit coefficients (p. 191).
    C = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])

    for k in range(len(wl_um)):
        if wl_um[k] <= 1.6421:
            f = 0.0
            for n in range(1, 7):
                f += C[n - 1] * np.power(
                    (1.0 / wl_um[k]) - (1.0 / 1.6421), (n - 1.0) / 2.0
                )
            alpha_bf[k] = (
                1.0e-18
                * wl_um[k] ** 3
                * np.power((1.0 / wl_um[k]) - (1.0 / 1.6421), 3.0 / 2.0)
                * f
            )
            alpha_bf[k] *= 1.0e-4  # cm² -> m²
        else:
            alpha_bf[k] = 1.0e-250  # POSEIDON sentinel (avoid log(0))
    return alpha_bf


def H_minus_free_free(wl_um, T_arr):
    """Free-free cross section α_ff(λ, T) of the H- ion [m⁵ / n_H / n_e-].

    Faithful port of POSEIDON `absorption.py:606-691` (John 1988
    short- + long-wavelength fits). Extinction coefficient per
    atmosphere layer:
        κ_ff(λ, T) = α_ff(λ, T) · n_H · n_(e-)

    Returns α_ff shape `(len(T_arr), len(wl_um))`.
    """
    wl_um = np.asarray(wl_um, dtype=np.float64)
    T_arr = np.asarray(T_arr, dtype=np.float64)
    alpha_ff = np.zeros((len(T_arr), len(wl_um)))

    wl = wl_um
    wl_2 = wl * wl
    wl_3 = wl_2 * wl
    wl_4 = wl_2 * wl_2

    # Short-wavelength fit (0.182 ≤ λ < 0.3645 μm).
    A_s = np.array([518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0])
    B_s = np.array([-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0])
    C_s = np.array([1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
    D_s = np.array([-479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0])
    E_s = np.array([93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0])
    F_s = np.array([-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0])

    # Long-wavelength fit (λ ≥ 0.3645 μm).
    A_l = np.array([0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830])
    B_l = np.array([0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170])
    C_l = np.array([0.0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8640])
    D_l = np.array([0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880])
    E_l = np.array([0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880])
    F_l = np.array([0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850])

    for i in range(len(T_arr)):
        theta = 5040.0 / T_arr[i]
        kT = 1.38066e-16 * T_arr[i]  # erg

        for k in range(len(wl_um)):
            if wl[k] < 0.182:
                alpha_ff[i, k] = 1.0e-250
            elif (wl[k] >= 0.182) and (wl[k] < 0.3645):
                for n in range(1, 7):
                    alpha_ff[i, k] += (
                        1.0e-29
                        * (
                            np.power(theta, (n + 1.0) / 2.0)
                            * (
                                A_s[n - 1] * wl_2[k]
                                + B_s[n - 1]
                                + C_s[n - 1] / wl[k]
                                + D_s[n - 1] / wl_2[k]
                                + E_s[n - 1] / wl_3[k]
                                + F_s[n - 1] / wl_4[k]
                            )
                        )
                        * kT
                    )
                alpha_ff[i, k] *= 1.0e-10  # cm⁵ -> m⁵
            elif wl[k] >= 0.3645:
                for n in range(1, 7):
                    alpha_ff[i, k] += (
                        1.0e-29
                        * (
                            np.power(theta, (n + 1.0) / 2.0)
                            * (
                                A_l[n - 1] * wl_2[k]
                                + B_l[n - 1]
                                + C_l[n - 1] / wl[k]
                                + D_l[n - 1] / wl_2[k]
                                + E_l[n - 1] / wl_3[k]
                                + F_l[n - 1] / wl_4[k]
                            )
                        )
                        * kT
                    )
                alpha_ff[i, k] *= 1.0e-10
    return alpha_ff
