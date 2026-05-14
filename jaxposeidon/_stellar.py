"""Stellar contamination forward model.

Ports POSEIDON `stellar.py:26-794` numerical functions:
- planck_lambda                  — single-T black-body radiance
- stellar_contamination_single_spot — Rackham+17/18 single-spot factor
- stellar_contamination_general — multi-region collective factor

The pysynphot / PyMSG grid loaders live in `_stellar_grid_loader.py`
(setup-only). The runtime `stellar_contamination` wrapper that does
spectres interpolation + single-spot / multi-region dispatch from a
star dict ports POSEIDON's `stellar.py:797-863` and is left as a thin
caller — applications usually call the numeric functions directly.
"""

import numpy as np
import scipy.constants as sc


def planck_lambda(T, wl):
    """Single-temperature black-body radiance B(T, λ) in SI (W/m²/sr/m).

    Bit-equivalent port of POSEIDON `stellar.py:26-60`. Differs from
    `_emission.planck_lambda_arr` only in that T is scalar and the
    returned array is 1D (per wavelength).
    """
    wl_m = wl * 1.0e-6
    c_2 = (sc.h * sc.c) / sc.k
    B_lambda = np.zeros(len(wl))
    for k in range(len(wl)):
        coeff = (2.0 * sc.h * sc.c**2) / (wl_m[k] ** 5)
        B_lambda[k] = coeff * (1.0 / (np.exp(c_2 / (wl_m[k] * T)) - 1.0))
    return B_lambda


def stellar_contamination_single_spot(f, I_het, I_phot):
    """Rackham+17/18 single-region stellar contamination factor.

    Bit-equivalent port of POSEIDON `stellar.py:733-756`.
    `ε(λ) = 1 / (1 - f · (1 - I_het / I_phot))`
    """
    return 1.0 / (1.0 - f * (1.0 - I_het / I_phot))


def stellar_contamination_general(f_het, I_het, I_phot):
    """Multi-region Rackham+17/18 stellar contamination factor.

    Bit-equivalent port of POSEIDON `stellar.py:760-794`. Sums each
    region's `f_i · (1 - I_het,i / I_phot)` then applies the same
    1 / (1 - Σ) form.
    """
    N_wl = np.shape(I_phot)[0]
    total = np.zeros(N_wl)
    for i in range(len(f_het)):
        I_ratio = I_het[i, :] / I_phot
        total += f_het[i] * (1.0 - I_ratio)
    return 1.0 / (1.0 - total)
