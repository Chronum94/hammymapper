"""Invariants of the irreps slot layout produced by the mapping specs."""
import numpy as np
import pytest

from hammymapper.hblockmapper import get_mask_dict, make_mapper_from_elements
from hammymapper.utilities import get_pair_mapping_spec


def _ells_of(slices):
    (rows, cols) = slices
    return (rows.stop - rows.start - 1) // 2, (cols.stop - cols.start - 1) // 2


def test_no_two_subblocks_share_a_slot(mapper):
    """Each (parity, irreps component, feature) belongs to at most one subblock, including
    components masked off because they vanish by symmetry."""
    for key, ms in mapper.mapper.items():
        owner = {}
        for k, (p, sl, f) in enumerate(ms.irreps_slices):
            for c in range(sl.start, sl.stop):
                assert (p, c, f) not in owner, f"{key}: subblocks {owner[(p, c, f)]} and {k} share {(p, c, f)}"
                owner[(p, c, f)] = k


def test_slot_ranges_match_subblock(mapper):
    for key, ms in mapper.mapper.items():
        for block_slice, cgc_slice, (p, sl, f) in zip(ms.block_slices, ms.cgc_slices, ms.irreps_slices):
            ell1, ell2 = _ells_of(block_slice)
            assert sl == slice(abs(ell1 - ell2) ** 2, (ell1 + ell2 + 1) ** 2), key
            assert cgc_slice[2] == sl and f < ms.nfeatures, key


def test_heteronuclear_mask_keeps_every_component(mapper):
    """Heteronuclear blocks have no symmetry: nothing in a subblock's range may be masked."""
    for key, ms in mapper.mapper.items():
        if isinstance(key, tuple) and key[0] != key[1]:
            for (p, sl, f) in ms.irreps_slices:
                assert np.all(ms.mask[p, sl, f] == 1), key


def test_mask_marks_exactly_the_allowed_components(mapper):
    """Mask == union of subblock ranges minus the components that vanish by symmetry:
    for a subblock coupling a shell with itself, sym / on-site keep even ell only and
    anti keeps odd ell only."""
    for key, ms in mapper.mapper.items():
        expected = np.zeros_like(ms.mask)
        homo = isinstance(key, tuple) and key[0] == key[1]
        half = len(ms.irreps_slices) // 2
        nrows = ms.nrows // 2 if homo else ms.nrows
        for k, (block_slice, (p, sl, f)) in enumerate(zip(ms.block_slices, ms.irreps_slices)):
            expected[p, sl, f] = 1
            rows, cols = block_slice
            row0 = rows.start % nrows if homo else rows.start
            same_shell = (not isinstance(key, tuple) or homo) and row0 == cols.start
            if same_shell:
                keep_even = not (homo and k >= half)
                ell1, _ = _ells_of(block_slice)
                for ell in range(0, 2 * ell1 + 1):
                    if (ell % 2 == 0) != keep_even:
                        expected[p, ell**2 : (ell + 1) ** 2, f] = 0
        np.testing.assert_array_equal(ms.mask, expected, err_msg=str(key))


def test_get_mask_dict_matches_specs(mapper):
    masks = get_mask_dict(mapper.max_ell, mapper.nfeatures, mapper)
    assert set(masks) == set(mapper.mapper)
    for key, ms in mapper.mapper.items():
        m = masks[key]
        assert m.shape == (2, (mapper.max_ell + 1) ** 2, mapper.nfeatures)
        np.testing.assert_array_equal(m[:, : ms.mask.shape[1], : ms.mask.shape[2]], ms.mask)
        assert m.sum() == ms.mask.sum()


def test_mapper_covers_all_element_combinations(species, mapper):
    Zs = sorted(species)
    expected = set(Zs) | {(a, b) for i, a in enumerate(Zs) for b in Zs[i:]}
    assert set(mapper.mapper) == expected


def test_feature_count_regression():
    """Number of features for common bases (sym and anti channels are separate for
    homonuclear pairs). Guards against silent layout changes."""
    m = make_mapper_from_elements({6: [0, 1, 2], 74: [0, 2, 1]})
    assert {k: v.nfeatures for k, v in m.mapper.items()} == {6: 3, 74: 3, (6, 6): 6, (6, 74): 4, (74, 74): 6}


@pytest.mark.parametrize("ells", [[], [4], [-1], [0, 5]])
def test_invalid_orbitals_rejected(ells):
    with pytest.raises(ValueError):
        make_mapper_from_elements({1: ells})


def test_pair_spec_without_symmetrization_is_full_block():
    block_slices, irreps_slices, _, max_ell, nfeat, mask = get_pair_mapping_spec([0, 1], [1, 2], False)
    assert len(block_slices) == 4 and max_ell == 3
    covered = np.zeros((4, 8), dtype=int)
    for rows, cols in block_slices:
        covered[rows, cols] += 1
    assert np.all(covered == 1)
