import numpy as np
import pytest
from hammymapper.hblockmapper import make_mapper_from_elements

RNG = np.random.default_rng(42)
ATOL = 1e-7


@pytest.fixture(scope="module")
def sp_mapper():
    return make_mapper_from_elements({1: [0, 1]})


@pytest.fixture(scope="module")
def sp_spd_mapper():
    return make_mapper_from_elements({1: [0, 1], 2: [0, 1, 2]})


# ── onsite blocks ────────────────────────────────────────────────────────────

def test_onsite_sp_roundtrip(sp_mapper):
    N = 1 + 3  # s + p
    H = RNG.random((2, N, N))
    H = 0.5 * (H + np.swapaxes(H, -1, -2))  # enforce symmetry as actual onsites are
    H_rec = sp_mapper.irreps_to_hblocks(sp_mapper.hblocks_to_irreps(H, 1), 1)
    assert np.allclose(H, H_rec, atol=ATOL)


def test_onsite_spd_roundtrip():
    mapper = make_mapper_from_elements({6: [0, 1, 2]})
    N = 1 + 3 + 5  # s + p + d
    H = RNG.random((3, N, N))
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    H_rec = mapper.irreps_to_hblocks(mapper.hblocks_to_irreps(H, 6), 6)
    assert np.allclose(H, H_rec, atol=ATOL)


def test_onsite_result_is_symmetric(sp_mapper):
    N = 1 + 3
    H = RNG.random((2, N, N))
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    H_rec = sp_mapper.irreps_to_hblocks(sp_mapper.hblocks_to_irreps(H, 1), 1)
    assert np.allclose(H_rec, np.swapaxes(H_rec, -1, -2), atol=ATOL)


# ── homonuclear pair blocks ───────────────────────────────────────────────────

def test_homonuclear_sp_roundtrip(sp_mapper):
    N = 1 + 3
    H = RNG.random((2, N, N))
    H_rec = sp_mapper.irreps_to_hblocks(sp_mapper.hblocks_to_irreps(H, 1, 1), 1, 1)
    assert np.allclose(H, H_rec, atol=ATOL)


def test_homonuclear_spd_roundtrip():
    mapper = make_mapper_from_elements({6: [0, 1, 2]})
    N = 1 + 3 + 5
    H = RNG.random((3, N, N))
    H_rec = mapper.irreps_to_hblocks(mapper.hblocks_to_irreps(H, 6, 6), 6, 6)
    assert np.allclose(H, H_rec, atol=ATOL)


def test_homonuclear_even_part_roundtrip(sp_mapper):
    N = 1 + 3
    H = RNG.random((2, N, N))
    H_rec = sp_mapper.irreps_to_hblocks(sp_mapper.hblocks_to_irreps(H, 1, 1), 1, 1)
    H_even     = 0.5 * (H     + np.swapaxes(H,     -1, -2))
    H_even_rec = 0.5 * (H_rec + np.swapaxes(H_rec, -1, -2))
    assert np.allclose(H_even, H_even_rec, atol=ATOL)


def test_homonuclear_odd_part_roundtrip(sp_mapper):
    N = 1 + 3
    H = RNG.random((2, N, N))
    H_rec = sp_mapper.irreps_to_hblocks(sp_mapper.hblocks_to_irreps(H, 1, 1), 1, 1)
    H_odd     = 0.5 * (H     - np.swapaxes(H,     -1, -2))
    H_odd_rec = 0.5 * (H_rec - np.swapaxes(H_rec, -1, -2))
    assert np.allclose(H_odd, H_odd_rec, atol=ATOL)


# ── heteronuclear pair blocks ─────────────────────────────────────────────────

def test_heteronuclear_sp_spd_roundtrip(sp_spd_mapper):
    H = RNG.random((3, 1 + 3, 1 + 3 + 5))
    H_rec = sp_spd_mapper.irreps_to_hblocks(
        sp_spd_mapper.hblocks_to_irreps(H, 1, 2), 1, 2
    )
    assert np.allclose(H, H_rec, atol=ATOL)


def test_heteronuclear_sp_s_roundtrip():
    mapper = make_mapper_from_elements({1: [0, 1], 2: [0]})
    H = RNG.random((4, 1 + 3, 1))
    H_rec = mapper.irreps_to_hblocks(mapper.hblocks_to_irreps(H, 1, 2), 1, 2)
    assert np.allclose(H, H_rec, atol=ATOL)
