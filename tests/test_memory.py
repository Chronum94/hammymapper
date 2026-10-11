"""Memory / buffer-handling robustness: in-place contract of hblocks_to_irreps, no
mutation or aliasing of inputs, reuse of stale buffers, dtypes, input validation, and
bounded peak memory / no leaks over repeated calls."""
import gc
import tracemalloc

import numpy as np
import pytest

from conftest import all_blocks, empty_irreps, full_mask, spec, to_irreps
from hammymapper.hblockmapper import make_mapper_from_elements


def test_inputs_not_mutated(species, mapper, rng):
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        block_copy = block.copy()
        irreps = to_irreps(mapper, block, Z_i, Z_j)
        np.testing.assert_array_equal(block, block_copy, err_msg=label)
        irreps_copy = irreps.copy()
        mapper.irreps_to_hblocks(irreps, Z_i, Z_j)
        np.testing.assert_array_equal(irreps, irreps_copy, err_msg=label)


def test_forward_writes_into_given_buffer(mapper, species, rng):
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        buf = empty_irreps(mapper, 1)
        assert mapper.hblocks_to_irreps(block, buf, Z_i, Z_j) is buf, label


def test_inverse_output_does_not_alias_input(species, mapper, rng):
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        irreps = to_irreps(mapper, block, Z_i, Z_j)
        out = mapper.irreps_to_hblocks(irreps, Z_i, Z_j)
        assert not np.shares_memory(out, irreps), label
        assert not np.shares_memory(out, block), label
        assert out.flags.c_contiguous and out.flags.writeable, label


def test_stale_buffer_is_fully_overwritten(species, mapper, rng):
    """Reusing a buffer that holds garbage from earlier calls must give the same result as
    a fresh zero buffer in the written rows, and leave rows beyond the batch untouched."""
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        batch = np.repeat(block, 3, axis=0)
        stale = np.full((5,) + empty_irreps(mapper, 1).shape[1:], np.nan)
        mapper.hblocks_to_irreps(batch, stale, Z_i, Z_j)
        np.testing.assert_array_equal(stale[:3], to_irreps(mapper, batch, Z_i, Z_j), err_msg=label)
        assert np.isnan(stale[3:]).all(), label


def test_garbage_in_masked_slots_is_ignored(species, mapper, rng):
    """A model may output anything in masked-off slots; reconstruction must not see it."""
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        irreps = to_irreps(mapper, block, Z_i, Z_j)
        noisy = irreps + 1e3 * rng.normal(size=irreps.shape) * (1 - full_mask(mapper, Z_i, Z_j))
        np.testing.assert_allclose(mapper.irreps_to_hblocks(noisy, Z_i, Z_j), block, atol=1e-10, err_msg=label)


def test_float32_buffers(species, mapper, rng):
    """float32 irreps buffers (typical for JAX models) work in both directions with float32
    accuracy, and the inverse keeps float32."""
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        buf = mapper.hblocks_to_irreps(block, empty_irreps(mapper, 1, np.float32), Z_i, Z_j)
        assert buf.dtype == np.float32
        rec = mapper.irreps_to_hblocks(buf, Z_i, Z_j)
        assert rec.dtype == np.float32
        np.testing.assert_allclose(rec, block, atol=1e-5, err_msg=label)


def test_non_contiguous_inputs(species, mapper, rng):
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        # Same values, laid out transposed in memory
        fortran = np.asfortranarray(block)
        np.testing.assert_allclose(to_irreps(mapper, fortran, Z_i, Z_j), to_irreps(mapper, block, Z_i, Z_j),
                                   atol=1e-14, err_msg=label)
        irreps = to_irreps(mapper, block, Z_i, Z_j)
        strided = np.repeat(irreps, 2, axis=0)[::2]  # non-contiguous view
        np.testing.assert_allclose(mapper.irreps_to_hblocks(strided, Z_i, Z_j), block, atol=1e-12, err_msg=label)


def test_empty_batch(mapper, species):
    for Z in species:
        ms = spec(mapper, Z, Z)
        out = mapper.hblocks_to_irreps(np.zeros((0, ms.nrows // 2, ms.ncols)), empty_irreps(mapper, 0), Z, Z)
        assert out.shape[0] == 0
        assert mapper.irreps_to_hblocks(out, Z, Z).shape == (0, ms.nrows // 2, ms.ncols)


def test_batch_matches_individual_calls(species, mapper, rng):
    """Batching must not mix samples: each row equals the single-sample result."""
    for Z in species:
        ms = spec(mapper, Z, Z)
        batch = rng.normal(size=(6, ms.nrows // 2, ms.ncols))
        together = to_irreps(mapper, batch, Z, Z)
        for k in range(len(batch)):
            # Only summation-order rounding may differ (BLAS blocking depends on batch size)
            np.testing.assert_allclose(together[k : k + 1], to_irreps(mapper, batch[k : k + 1], Z, Z),
                                       rtol=0, atol=1e-14)


@pytest.mark.parametrize("bad_shape", [(9, 9), (1, 9, 8), (1, 8, 9), (1, 9, 9, 1)])
def test_wrong_block_shape_raises(bad_shape):
    mapper = make_mapper_from_elements({6: [0, 1, 2]})
    with pytest.raises(ValueError):
        mapper.hblocks_to_irreps(np.zeros(bad_shape), empty_irreps(mapper, 1), 6, 6)


def test_too_small_buffer_raises():
    mapper = make_mapper_from_elements({6: [0, 1, 2]})
    blocks = np.zeros((4, 9, 9))
    for buf in [empty_irreps(mapper, 3), np.zeros((4, 2, 4, mapper.nfeatures)), np.zeros((4, 1, 25, mapper.nfeatures))]:
        with pytest.raises(ValueError):
            mapper.hblocks_to_irreps(blocks, buf, 6, 6)


def test_unknown_element_raises_keyerror():
    mapper = make_mapper_from_elements({6: [0, 1, 2]})
    with pytest.raises(KeyError, match="No mapping spec"):
        mapper.irreps_to_hblocks(empty_irreps(mapper, 1), 7, 6)


def _peak_bytes(fn):
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def test_peak_memory_is_bounded():
    """Peak temporary memory scales like the data, not worse: for a large batch of
    homonuclear blocks, the forward pass needs one [sym; anti] array (2x the input) plus
    small per-subblock temporaries, and the inverse a few copies of its output."""
    mapper = make_mapper_from_elements({26: [0, 0, 1, 2, 2, 3]})
    ms = spec(mapper, 26, 26)
    n = ms.nrows // 2
    blocks = np.random.default_rng(0).normal(size=(2000, n, n))
    buf = empty_irreps(mapper, len(blocks))
    fwd = _peak_bytes(lambda: mapper.hblocks_to_irreps(blocks, buf, 26, 26))
    assert fwd < 2.5 * blocks.nbytes, (fwd, blocks.nbytes)
    out_bytes = blocks.nbytes
    inv = _peak_bytes(lambda: mapper.irreps_to_hblocks(buf, 26, 26))
    assert inv < 5 * out_bytes, (inv, out_bytes)


def test_no_memory_growth_over_repeated_calls():
    """Repeated round trips must not accumulate memory (e.g. through caches)."""
    mapper = make_mapper_from_elements({6: [0, 1, 2], 74: [0, 2, 1]})
    rng = np.random.default_rng(0)
    blocks = {k: rng.normal(size=(64, 9, 9)) for k in [(6, None), (6, 6), (6, 74), (74, 6)]}
    buf = empty_irreps(mapper, 64)

    def many():
        for _ in range(50):
            for (Z_i, Z_j), b in blocks.items():
                mapper.irreps_to_hblocks(mapper.hblocks_to_irreps(b, buf, Z_i, Z_j), Z_i, Z_j)

    many()  # warm-up (einsum path caches etc.)
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    many()
    gc.collect()
    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert after - before < 256 * 1024, after - before
