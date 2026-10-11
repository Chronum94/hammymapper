import os

# Tests only need the CPU; avoids failures from GPU plugins without a visible device.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from hammymapper.hblockmapper import make_mapper_from_elements

# Element -> orbital angular momenta. Covers the plain spd ordering, non-sorted orderings,
# multiple zetas and f orbitals, which exercise different slot layouts.
SPECIES_CONFIGS = {
    "spd_sdp": {6: [0, 1, 2], 74: [0, 2, 1]},
    "reversed_multizeta": {1: [1, 0], 8: [0, 0, 1, 1, 2]},
    "with_f": {3: [0, 0, 1], 9: [0, 1, 2, 3]},
    "s_only_and_spdf": {2: [0], 26: [0, 0, 1, 2, 2, 3]},
}


@pytest.fixture(params=list(SPECIES_CONFIGS), ids=list(SPECIES_CONFIGS))
def species(request):
    return SPECIES_CONFIGS[request.param]


@pytest.fixture
def mapper(species):
    return make_mapper_from_elements(species)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def norb(ells):
    return sum(2 * ell + 1 for ell in ells)


def random_symmetric_hamiltonian(species, rng, atoms_per_species=2):
    """Random real-symmetric H for a structure with `atoms_per_species` atoms of each
    element. Returns (H, atomic numbers, orbital offsets of each atom)."""
    Zs = [Z for Z in species for _ in range(atoms_per_species)]
    offsets = np.concatenate([[0], np.cumsum([norb(species[Z]) for Z in Zs])])
    H = rng.normal(size=(offsets[-1], offsets[-1]))
    return 0.5 * (H + H.T), Zs, offsets


def atom_block(H, offsets, i, j):
    return H[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]][None]


def empty_irreps(mapper, nbatch, dtype=np.float64):
    return np.zeros((nbatch, 2, (mapper.max_ell + 1) ** 2, mapper.nfeatures), dtype=dtype)


def spec(mapper, Z_i, Z_j=None):
    return mapper.mapper[Z_i if Z_j is None else tuple(sorted([Z_i, Z_j]))]


def full_mask(mapper, Z_i, Z_j=None):
    m = spec(mapper, Z_i, Z_j).mask
    out = np.zeros((2, (mapper.max_ell + 1) ** 2, mapper.nfeatures), dtype=np.int8)
    out[:, : m.shape[1], : m.shape[2]] = m
    return out


def to_irreps(mapper, blocks, Z_i, Z_j=None):
    return mapper.hblocks_to_irreps(blocks, empty_irreps(mapper, len(blocks)), Z_i, Z_j)


def orbital_matrix(ells, per_ell):
    """Block-diagonal matrix over orbital shells, with block per_ell(ell)."""
    n = norb(ells)
    M = np.zeros((n, n))
    start = 0
    for ell in ells:
        M[start : start + 2 * ell + 1, start : start + 2 * ell + 1] = per_ell(ell)
        start += 2 * ell + 1
    return M


def all_blocks(species, rng):
    """Yields (label, block, Z_i, Z_j) for every on-site and pair block (both orientations)
    of a random symmetric H."""
    H, Zs, offsets = random_symmetric_hamiltonian(species, rng)
    for i in range(len(Zs)):
        for j in range(len(Zs)):
            if i == j:
                yield f"onsite {Zs[i]}", atom_block(H, offsets, i, i), Zs[i], None
            else:
                kind = "homo" if Zs[i] == Zs[j] else "hetero"
                yield f"{kind} ({Zs[i]},{Zs[j]})", atom_block(H, offsets, i, j), Zs[i], Zs[j]
