"""Phase 9 tests for load_data + init_instrument shims.

These wrappers thin-delegate to POSEIDON for the reference_data file
I/O. The tests verify that:
  - init_instrument returns POSEIDON's exact 7-tuple for a known JWST
    spectroscopic instrument (NIRSpec PRISM).
  - load_data returns the same dict POSEIDON would for the
    single/two/three offset_datasets variants, including lumped
    offset_*_datasets bookkeeping.
"""

import os
import tempfile

import numpy as np
import pytest

from jaxposeidon import _loaddata


def _write_data_file(path, wl_centres, half_widths, depth, err):
    with open(path, "w") as f:
        for w, hb, d, e in zip(wl_centres, half_widths, depth, err):
            f.write(f"{w} {hb} {d} {e}\n")


def test_init_instrument_matches_poseidon_NIRSpec_PRISM():
    from POSEIDON.instrument import init_instrument as p_init

    wl = np.linspace(0.6, 5.3, 4000)
    wl_data = np.linspace(0.9, 5.1, 50)
    half_width = np.full(50, 0.02)
    ours = _loaddata.init_instrument(wl, wl_data, half_width, "JWST_NIRSpec_PRISM")
    theirs = p_init(wl, wl_data, half_width, "JWST_NIRSpec_PRISM")
    assert len(ours) == len(theirs) == 7
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "offset_kwargs",
    [
        {},
        {"offset_datasets": ["dsA.dat"]},
        {"offset_datasets": ["dsA.dat", "dsB.dat"]},
        {"offset_datasets": ["dsA.dat", "dsB.dat", "dsC.dat"]},
        {"offset_1_datasets": ["dsA.dat"]},
        {"offset_1_datasets": ["dsA.dat", "dsB.dat"]},
        {"offset_1_datasets": ["dsA.dat"], "offset_2_datasets": ["dsB.dat"]},
        {
            "offset_1_datasets": ["dsA.dat"],
            "offset_2_datasets": ["dsB.dat"],
            "offset_3_datasets": ["dsC.dat"],
        },
    ],
)
def test_load_data_matches_poseidon(offset_kwargs):
    from POSEIDON.core import load_data as p_load_data

    rng = np.random.default_rng(0)
    wl_model = np.linspace(0.6, 5.3, 2000)
    with tempfile.TemporaryDirectory() as tmpdir:
        # three small JWST-style datasets
        names = ["dsA.dat", "dsB.dat", "dsC.dat"]
        bins_per = [8, 6, 5]
        wl_ranges = [(1.0, 2.0), (2.2, 3.4), (3.6, 4.9)]
        for n, nb, (w0, w1) in zip(names, bins_per, wl_ranges):
            wl_c = np.linspace(w0, w1, nb)
            hb = np.full(nb, 0.02)
            d = rng.uniform(2.5e-3, 3.0e-3, size=nb)
            e = rng.uniform(8e-5, 2e-4, size=nb)
            _write_data_file(os.path.join(tmpdir, n), wl_c, hb, d, e)
        instruments = [
            "JWST_NIRSpec_PRISM",
            "JWST_NIRISS_SOSS_Ord1",
            "JWST_NIRISS_SOSS_Ord2",
        ]
        ours = _loaddata.load_data(
            tmpdir, names, instruments, wl_model, **offset_kwargs
        )
        theirs = p_load_data(tmpdir, names, instruments, wl_model, **offset_kwargs)
    assert set(ours.keys()) == set(theirs.keys())
    for k in ours:
        a, b = ours[k], theirs[k]
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, list):
            assert a == b
        else:
            assert a == b
