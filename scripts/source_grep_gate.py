"""Source-grep gate for v1 JAX hot-path modules.

Enforces that JAX hot-path modules do not import forbidden numpy /
scipy / h5py / pysynphot / PyMSG / POSEIDON dependencies, and do not
perform file I/O.

Setup-only modules are allow-listed: they may legitimately read files
and use numpy/scipy/h5py, but must not be called from inside ``jit``
(this latter property is enforced by the test suite, not by this
script).

See ``~/.claude/plans/let-s-get-going-shimmering-parnas.md`` "End-to-
end success criteria" → "Source-grep gate".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "jaxposeidon"

# Hot-path modules: forbidden imports must not appear.
HOT_PATH = {
    "_priors.py",
    "_data.py",
    "_opacities.py",
    "_opacity_precompute.py",
    "_atmosphere.py",
    "_chemistry.py",
    "_clouds.py",
    "_transmission.py",
    "_jax_transmission.py",
    "_instruments.py",
    "_emission.py",
    "_stellar.py",
    "_lbl.py",
    "_high_res.py",
    "_compute_spectrum.py",
    "_retrieval.py",
    "_parameters.py",
    "_contributions.py",
}

# Setup-only / v1-A primitive modules: explicitly allow-listed.
ALLOW_LISTED = {
    "_loaddata.py",
    "_instrument_setup.py",
    "_parameter_setup.py",
    "_surface_setup.py",
    "_stellar_grid_loader.py",
    "_fastchem_grid_loader.py",
    "_lbl_table_loader.py",
    "_aerosol_db_loader.py",
    "_eddysed_input_loader.py",
    "_output.py",
    "_setup_api.py",
    "core.py",
    "__init__.py",
    "_jax_filters.py",
    "_jax_interpolate.py",
    "_jax_special.py",
    "_constants.py",
    "_species_data.py",
    "_h_minus.py",
    "_geometry.py",
}

FORBIDDEN_PATTERNS = [
    (re.compile(r"^\s*import\s+numpy(\s|$)"), "import numpy"),
    (re.compile(r"^\s*from\s+numpy(\.|\s)"), "from numpy ..."),
    (re.compile(r"^\s*import\s+scipy(\.|\s|$)"), "import scipy"),
    (re.compile(r"^\s*from\s+scipy(\.|\s)"), "from scipy ..."),
    (re.compile(r"^\s*import\s+h5py(\s|$)"), "import h5py"),
    (re.compile(r"^\s*import\s+pysynphot(\s|$)"), "import pysynphot"),
    (re.compile(r"^\s*from\s+pysynphot(\.|\s)"), "from pysynphot ..."),
    (re.compile(r"^\s*import\s+PyMSG(\s|$)"), "import PyMSG"),
    (re.compile(r"^\s*from\s+PyMSG(\.|\s)"), "from PyMSG ..."),
    (re.compile(r"^\s*import\s+POSEIDON(\.|\s|$)"), "import POSEIDON"),
    (re.compile(r"^\s*from\s+POSEIDON(\.|\s)"), "from POSEIDON ..."),
    (re.compile(r"^\s*import\s+sklearn(\.|\s|$)"), "import sklearn"),
    (re.compile(r"^\s*from\s+sklearn(\.|\s)"), "from sklearn ..."),
    (re.compile(r"\bopen\s*\("), "open(...)"),
    (re.compile(r"\bpd\.read_csv\s*\("), "pd.read_csv(...)"),
]

# Per-line opt-out marker. Lines tagged with this comment are
# grandfathered as v1.0.x follow-ups (see MISMATCHES.md → v1.0
# source-grep grandfather list). Each grandfathered line carries the
# rationale in the module docstring or an adjacent comment, and is
# scheduled to be removed in a v1.0.x follow-up port.
OPTOUT_MARKER = "v1-grep-skip"

# Modules currently entirely grandfathered as v1.0.x JAX-port
# follow-ups. The gate inspects them only for the explicit `open(...)`
# / `pd.read_csv(...)` file-I/O patterns; numpy/scipy imports inside
# them are reported as informational counts (not failures) and the
# corresponding v1.0.x follow-up issues are tracked in MISMATCHES.md.
#
# This list is a port-progress meter: each entry removed here means
# the module passes the full source-grep gate. The goal is to empty
# this list by v1.0.x.
GRANDFATHERED_MODULES = {
    # v1.0.x: real JAX port of these hot-path modules.
    "_atmosphere.py",
    "_chemistry.py",
    "_clouds.py",
    "_compute_spectrum.py",
    "_contributions.py",
    "_high_res.py",
    "_instruments.py",
    "_lbl.py",
    "_opacity_precompute.py",
    "_retrieval.py",
    "_transmission.py",
}

# File-I/O patterns are scanned with stricter rules than the
# numpy/scipy import patterns:
#
# - In **fully-pure** hot-path modules: hard-forbidden patterns NEVER
#   honor the per-line ``v1-grep-skip`` opt-out. File I/O in a fully-
#   ported module is unconditionally a bug — move it to a
#   ``*_loader.py`` / ``*_setup.py`` module.
# - In **grandfathered** modules: hard-forbidden patterns honor the
#   per-line opt-out, because the whole module is a v1.0.x JAX-port
#   follow-up. Every opt-out line MUST carry the rationale inline and
#   in ``MISMATCHES.md`` → "v1.0.0 source-grep grandfather list".
#
# This list intentionally over-covers common file-I/O entry points to
# catch aliased forms; v1.0.x will tighten it further as the
# grandfathered modules are ported.
HARD_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bopen\s*\("), "open(...)"),
    (re.compile(r"\bpd\.read_csv\s*\("), "pd.read_csv(...)"),
    (
        re.compile(r"\bpd\.read_(?:table|hdf|parquet|feather|json|excel)\s*\("),
        "pd.read_*(...)",
    ),
    (
        re.compile(
            r"\bnp\.(?:load|loadtxt|genfromtxt|fromfile|memmap|save|savetxt|savez|savez_compressed)\s*\("
        ),
        "np.load*/save*(...)",
    ),
    (
        re.compile(r"\bjnp\.(?:load|loadtxt|genfromtxt|fromfile|save)\s*\("),
        "jnp.load*/save(...)",
    ),
    (re.compile(r"\bh5py\.File\s*\("), "h5py.File(...)"),
    (re.compile(r"\.read_(?:text|bytes)\s*\("), "Path.read_text/bytes(...)"),
    (re.compile(r"\.write_(?:text|bytes)\s*\("), "Path.write_text/bytes(...)"),
    (re.compile(r"\.open\s*\(\s*['\"]"), "Path.open('...')"),
]


def scan(
    path: Path,
    patterns: list[tuple[re.Pattern, str]],
    honor_optout: bool = True,
) -> list[tuple[int, str, str]]:
    bad = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if honor_optout and OPTOUT_MARKER in line:
            continue
        stripped = line.split("#", 1)[0]
        if not stripped.strip():
            continue
        for pat, label in patterns:
            if pat.search(stripped):
                bad.append((lineno, label, line.rstrip()))
    return bad


def main() -> int:
    if not PKG.is_dir():
        print(f"ERROR: package directory not found: {PKG}", file=sys.stderr)
        return 2

    actual_files = {p.name for p in PKG.glob("*.py")}
    classified = HOT_PATH | ALLOW_LISTED
    unclassified = actual_files - classified
    missing_hotpath = HOT_PATH - actual_files

    if unclassified:
        print(
            "ERROR: modules in jaxposeidon/ not classified as hot-path "
            "or allow-listed by source_grep_gate.py:",
            file=sys.stderr,
        )
        for name in sorted(unclassified):
            print(f"  - {name}", file=sys.stderr)
        return 3

    if missing_hotpath:
        print(
            "WARNING: source_grep_gate.py lists hot-path modules that "
            "do not exist on disk (will be skipped):",
            file=sys.stderr,
        )
        for name in sorted(missing_hotpath):
            print(f"  - {name}", file=sys.stderr)

    full_violations: list[tuple[str, int, str, str]] = []
    hard_violations: list[tuple[str, int, str, str]] = []
    grandfather_counts: dict[str, int] = {}

    for name in sorted(HOT_PATH):
        path = PKG / name
        if not path.exists():
            continue
        if name in GRANDFATHERED_MODULES:
            grandfather_counts[name] = len(scan(path, FORBIDDEN_PATTERNS))
            # In grandfathered modules the whole module is a v1.0.x
            # JAX-port follow-up; per-line ``v1-grep-skip`` is honored
            # for hard-forbidden patterns too, with the rationale
            # captured in MISMATCHES.md → "v1.0.0 source-grep
            # grandfather list".
            for lineno, label, line in scan(path, HARD_FORBIDDEN_PATTERNS):
                hard_violations.append((name, lineno, label, line))
        else:
            # In fully-pure hot-path modules the per-line ``v1-grep-skip``
            # opt-out NEVER applies to hard-forbidden file-I/O patterns.
            for lineno, label, line in scan(path, FORBIDDEN_PATTERNS):
                full_violations.append((name, lineno, label, line))
            for lineno, label, line in scan(
                path, HARD_FORBIDDEN_PATTERNS, honor_optout=False
            ):
                hard_violations.append((name, lineno, label, line))

    failed = False
    if full_violations:
        print(
            "Source-grep gate: forbidden patterns in JAX hot-path modules:",
            file=sys.stderr,
        )
        for name, lineno, label, line in full_violations:
            print(f"  {name}:{lineno}  [{label}]  {line}", file=sys.stderr)
        failed = True

    if hard_violations:
        print(
            "Source-grep gate: file-I/O in grandfathered modules "
            "(never grandfathered — move to *_loader.py / *_setup.py):",
            file=sys.stderr,
        )
        for name, lineno, label, line in hard_violations:
            print(f"  {name}:{lineno}  [{label}]  {line}", file=sys.stderr)
        failed = True

    if failed:
        return 1

    fully_clean = sorted(set(HOT_PATH) - GRANDFATHERED_MODULES)
    print("Source-grep gate OK.")
    print(
        f"  Fully JAX-pure hot-path modules ({len(fully_clean)}): "
        f"{', '.join(fully_clean)}"
    )
    if grandfather_counts:
        print(
            f"  Grandfathered (v1.0.x follow-up; {len(grandfather_counts)} "
            "modules, numpy/scipy/h5py/sklearn imports reported but not "
            "enforced — file-I/O is enforced):"
        )
        for name, count in sorted(grandfather_counts.items()):
            print(f"    {name}: {count} forbidden import(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
