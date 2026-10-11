import numpy as np
import pytest

from conftest import all_blocks, empty_irreps, full_mask, to_irreps, norb
from hammymapper.hblockmapper import make_mapper_from_elements

ATOL = 1e-12  # float64 Clebsch-Gordan coefficients


def test_round_trip_all_block_types(species, mapper, rng):
    """Every on-site, homonuclear and heteronuclear block (both orientations) of a random
    symmetric H survives hblocks -> irreps -> hblocks."""
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        rec = mapper.irreps_to_hblocks(to_irreps(mapper, block, Z_i, Z_j), Z_i, Z_j)
        assert rec.shape == block.shape, label
        np.testing.assert_allclose(rec, block, atol=ATOL, err_msg=label)


def test_round_trip_through_mask(species, mapper, rng):
    """Model-like usage: only masked-in irreps are predicted (everything else zero). The
    mask must keep every component that carries information."""
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        irreps = to_irreps(mapper, block, Z_i, Z_j) * full_mask(mapper, Z_i, Z_j)
        np.testing.assert_allclose(mapper.irreps_to_hblocks(irreps, Z_i, Z_j), block, atol=ATOL, err_msg=label)


def test_homonuclear_block_need_not_be_symmetric(rng):
    """H_ij for i != j of the same element is a general (non-symmetric) matrix."""
    mapper = make_mapper_from_elements({6: [0, 1, 2]})
    H = rng.normal(size=(3, 9, 9))
    np.testing.assert_allclose(mapper.irreps_to_hblocks(to_irreps(mapper, H, 6, 6), 6, 6), H, atol=ATOL)


def test_irreps_round_trip(species, mapper, rng):
    """irreps -> hblocks -> irreps is the identity on the masked-in irreps (the map is
    orthogonal on that subspace)."""
    for Z_i, Z_j in [(Z, None) for Z in species] + [(a, b) for a in species for b in species]:
        m = full_mask(mapper, Z_i, Z_j)
        irreps = rng.normal(size=(4,) + m.shape) * m
        blocks = mapper.irreps_to_hblocks(irreps, Z_i, Z_j)
        np.testing.assert_allclose(to_irreps(mapper, blocks, Z_i, Z_j), irreps, atol=ATOL, err_msg=str((Z_i, Z_j)))


def test_norm_preserved(species, mapper, rng):
    """CG maps are orthogonal: ||H||_F == ||irreps|| for heteronuclear blocks, where every
    subblock is mapped exactly once. (On-site and homonuclear maps only store the n2 >= n1
    subblocks, so the norm identity does not hold there.)"""
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        if Z_j is None or Z_i == Z_j:
            continue
        irreps = to_irreps(mapper, block, Z_i, Z_j)
        np.testing.assert_allclose(np.linalg.norm(irreps), np.linalg.norm(block), rtol=1e-12, err_msg=label)


@pytest.mark.parametrize("ells", [[0], [1], [3], [0, 0], [2, 1, 0], [3, 3], [0, 1, 2, 3, 2, 1, 0]])
def test_single_element_edge_cases(ells, rng):
    mapper = make_mapper_from_elements({5: ells})
    n = norb(ells)
    onsite = rng.normal(size=(2, n, n))
    onsite = 0.5 * (onsite + np.swapaxes(onsite, -1, -2))
    pair = rng.normal(size=(2, n, n))
    np.testing.assert_allclose(mapper.irreps_to_hblocks(to_irreps(mapper, onsite, 5), 5), onsite, atol=ATOL)
    np.testing.assert_allclose(mapper.irreps_to_hblocks(to_irreps(mapper, pair, 5, 5), 5, 5), pair, atol=ATOL)
