"""Setup-only string-dispatch helper for the JAX contributions kernel.

Allow-listed by the source-grep gate. Resolves string-based
``contribution_species`` / ``chemical_species`` / ``cia_pairs`` to
the integer indices + boolean masks the pure-jnp
``_jax_contributions.spectral_contribution_kernel_jit`` consumes.
Must NOT be called from inside ``jit``.
"""

import numpy as np


def setup_spectral_contribution_indices(
    chemical_species,
    active_species,
    cia_pairs,
    contribution_species,
):
    """Resolve string-based selectors to integer indices + masks.

    Returns ``(contribution_species_idx, contribution_active_idx,
    cia_mask, bulk_cia_mask, bulk_species_mask)`` as plain numpy
    arrays.

    The contribution_species/active indices default to 0 (matching
    POSEIDON ``contributions.py:126-127``) when the species is not
    found; downstream selectors zero out the contribution via masks
    so the index value at the default is inert.
    """
    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_bulk_species = N_species - N_species_active
    bulk_species_names = chemical_species[:N_bulk_species]

    contribution_species_idx = 0
    for i in range(N_species):
        if contribution_species == chemical_species[i]:
            contribution_species_idx = i
    contribution_active_idx = 0
    for i in range(N_species_active):
        if contribution_species == active_species[i]:
            contribution_active_idx = i

    cia_mask = np.zeros(len(cia_pairs), dtype=bool)
    bulk_cia_mask = np.zeros(len(cia_pairs), dtype=bool)
    for i in range(len(cia_pairs)):
        pair_1, pair_2 = cia_pairs[i].split("-")
        if contribution_species == pair_1 or contribution_species == pair_2:
            cia_mask[i] = True
        bulk1 = any(pair_1 == bs for bs in bulk_species_names)
        bulk2 = any(pair_2 == bs for bs in bulk_species_names)
        if bulk1 and bulk2:
            bulk_cia_mask[i] = True

    bulk_species_mask = np.zeros(N_species, dtype=bool)
    bulk_species_mask[:N_bulk_species] = True

    return (
        contribution_species_idx,
        contribution_active_idx,
        cia_mask,
        bulk_cia_mask,
        bulk_species_mask,
    )
