"""load_data + init_instrument shim — v0 thin wrappers over POSEIDON.

POSEIDON's `core.py:2135-2363` `load_data(...)` and `instrument.py:146-318`
`init_instrument(...)` are file-I/O wrappers around POSEIDON's shipped
`reference_data/` directory (per-instrument sensitivity tables and JWST
resolution files). Re-porting that dispatch table inside jaxposeidon
would either duplicate POSEIDON's data files or rely on a fragile
fork; instead we delegate to POSEIDON for the reference-data I/O and
return the same dict our `_instruments.bin_spectrum_to_data(...)` and
`_data.loglikelihood(...)` consume.

The pure-JAX hot path (`_instruments.make_model_data`,
`_data.loglikelihood`, etc.) does NOT depend on POSEIDON — only the
one-off setup at retrieval start does.

Photometric instruments (IRAC1, IRAC2) are still v1; the wrapper here
will refuse them via the downstream check in
`_instruments.bin_spectrum_to_data`.
"""

import numpy as np


def init_instrument(wl, wl_data, half_width, instrument):
    """Thin wrapper around POSEIDON `instrument.py:146-318`.

    Delegates instrument-sensitivity + FWHM file I/O to POSEIDON. The
    returned tuple `(sigma, fwhm, sensitivity, bin_left, bin_cent,
    bin_right, norm)` matches POSEIDON's exact return signature.
    """
    from POSEIDON.instrument import init_instrument as p_init
    return p_init(wl, wl_data, half_width, instrument)


def load_data(data_dir, datasets, instruments, wl_model, offset_datasets=None,
              wl_unit="micron", bin_width="half", spectrum_unit="(Rp/Rs)^2",
              skiprows=None, offset_1_datasets=None, offset_2_datasets=None,
              offset_3_datasets=None):
    """Thin wrapper around POSEIDON `core.py:2135-2363` `load_data(...)`.

    Accepts the full POSEIDON kwarg surface; returns the same `data` dict
    `_instruments.bin_spectrum_to_data(...)` and `_data.loglikelihood(...)`
    consume.
    """
    from POSEIDON.core import load_data as p_load_data
    return p_load_data(
        data_dir, datasets, instruments, wl_model,
        offset_datasets=offset_datasets,
        wl_unit=wl_unit, bin_width=bin_width, spectrum_unit=spectrum_unit,
        skiprows=skiprows,
        offset_1_datasets=offset_1_datasets,
        offset_2_datasets=offset_2_datasets,
        offset_3_datasets=offset_3_datasets,
    )
