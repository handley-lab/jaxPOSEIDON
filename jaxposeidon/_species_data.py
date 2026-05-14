"""Atomic / molecular constants — extracted from POSEIDON `species_data.py`.

Build-time-extracted (see `scripts/extract_poseidon_constants.py`) so
jaxposeidon's runtime forward path does NOT depend on POSEIDON being
importable.

Source of truth:
    POSEIDON/POSEIDON/species_data.py
    Copyright (c) 2022, Ryan J. MacDonald (BSD-3).

If POSEIDON updates these constants, re-run the extraction script and
commit the regenerated table. The `inactive_species` array below mirrors
`supported_chemicals.py:121` and must be kept in sync.
"""

import numpy as np

# Masses in atomic mass units. Multiply by scipy.constants.u (1.66053906660e-27)
# to convert to kg, matching POSEIDON's convention.
masses = {
    "H2O": 18.010565,
    "CO2": 43.989830,
    "CH4": 16.031300,
    "CO": 28.010100,
    "Na": 22.989770,
    "K": 39.098300,
    "NH3": 17.026549,
    "HCN": 27.010899,
    "SO2": 63.961901,
    "H2S": 33.987721,
    "PH3": 33.997238,
    "C2H2": 26.015650,
    "OCS": 59.966986,
    "TiO": 63.942861,
    "VO": 66.938871,
    "AlO": 42.976454,
    "SiO": 43.971842,
    "CaO": 55.957506,
    "MgO": 39.979956,
    "NaO": 38.989200,
    "LaO": 154.90490,
    "ZrO": 107.22300,
    "SO": 48.064000,
    "NO": 29.997989,
    "PO": 46.968676,
    "TiH": 48.955771,
    "CrH": 52.948333,
    "FeH": 56.942762,
    "ScH": 45.963737,
    "AlH": 27.989480,
    "SiH": 28.984752,
    "BeH": 10.020007,
    "CaH": 40.970416,
    "MgH": 24.992867,
    "LiH": 8.0238300,
    "NaH": 23.997594,
    "OH": 17.002740,
    "OH+": 17.002740,
    "CH": 13.007825,
    "NH": 15.010899,
    "SH": 32.979896,
    "PN": 44.976836,
    "PS": 62.945833,
    "CS": 43.972071,
    "C2": 24.000000,
    "CH3": 15.023475,
    "H3+": 3.0234750,
    "N2O": 44.001062,
    "NO2": 45.992904,
    "C2H4": 28.031300,
    "C2H6": 30.046950,
    "CH3CN": 41.026549,
    "CH3OH": 32.026215,
    "CH3Cl": 49.992328,
    "GeH4": 77.952478,
    "CS2": 75.944142,
    "O2": 31.989830,
    "O3": 47.984745,
    "C2H6S": 62.019000,
    "C2H6S2": 93.991093,
    "CH3SH": 48.003371,
    "C3H4": 40.031300,
    "Al": 26.981539,
    "Ba": 137.32770,
    "Ba+": 137.32770,
    "Ca": 40.078400,
    "Ca+": 40.078400,
    "Cr": 51.996160,
    "Cs": 132.90545,
    "Fe": 55.845200,
    "Fe+": 55.845200,
    "Li": 6.9675000,
    "Mg": 24.305500,
    "Mg+": 24.305500,
    "Mn": 54.938044,
    "Ni": 58.693440,
    "O": 15.999400,
    "Rb": 85.467830,
    "Sc": 44.955908,
    "Ti": 47.867100,
    "Ti+": 47.867100,
    "V": 50.941510,
    "V+": 50.941510,
    "H2": 2.0156500,
    "He": 4.0026030,
    "H": 1.0078250,
    "N2": 28.006148,
    "H-": 1.0083740,
    "e-": 5.4858e-4,
    "12C-16O": 27.994915,
    "13C-16O": 28.998270,
    "12C-18O": 29.999161,
    "12C-17O": 28.999130,
    "13C-18O": 31.002516,
    "13C-17O": 30.002485,
}


# Species treated as spectrally inactive (no line/cross-section opacity).
# Mirrors `POSEIDON/supported_chemicals.py:121`.
inactive_species = np.array(["H2", "He", "H", "e-", "H-", "N2", "ghost"])


fastchem_supported_species = np.array(
    [
        "H2O",
        "CO2",
        "OH",
        "SO",
        "C2H2",
        "C2H4",
        "H2S",
        "O2",
        "O3",
        "HCN",
        "NH3",
        "SiO",
        "CH4",
        "CO",
        "C2",
        "CaH",
        "CrH",
        "FeH",
        "HCl",
        "K",
        "MgH",
        "N2",
        "Na",
        "NO",
        "NO2",
        "OCS",
        "PH3",
        "SH",
        "SiH",
        "SO2",
        "TiH",
        "TiO",
        "VO",
    ]
)
