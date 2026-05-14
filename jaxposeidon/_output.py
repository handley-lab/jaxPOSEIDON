"""Spectrum-output side effects (setup-only).

Setup-only: file I/O permitted. Must not be called from inside `jit`.
Allow-listed by the v1 source-grep gate.

Ports POSEIDON `utility.py:694-709` (`write_spectrum`).
"""

import os


def write_spectrum(planet_name, model_name, spectrum, wl, output_dir=None):
    """Write a model spectrum to ``./POSEIDON_output/<planet>/spectra/``.

    Bit-equivalent port of POSEIDON `utility.py:694-709`. The output
    file path is
    ``<output_dir>/<planet_name>/spectra/<planet_name>_<model_name>_spectrum.txt``
    with two whitespace-separated columns formatted as ``%.8e``
    (wavelength μm, transit depth (Rp/Rs)²).

    POSEIDON hard-codes ``output_dir='./POSEIDON_output/'``; the
    ``output_dir`` kwarg is a port-only extension that lets tests redirect
    to a temp dir without polluting the cwd.
    """
    if output_dir is None:
        output_dir = "./POSEIDON_output/"
    spectra_dir = os.path.join(output_dir, planet_name, "spectra")
    os.makedirs(spectra_dir, exist_ok=True)
    file_path = os.path.join(spectra_dir, f"{planet_name}_{model_name}_spectrum.txt")
    with open(file_path, "w") as f:
        for i in range(len(wl)):
            f.write(f"{wl[i]:.8e} {spectrum[i]:.8e} \n")
    return file_path
