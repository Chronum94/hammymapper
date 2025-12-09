from hammymapper.utilities import get_element_mapping_spec

import numpy as np


def test_spd():
    block_slices, irreps_array_slices, _, max_ell, nfeatures, mask = get_element_mapping_spec([0, 1, 2])
    assert len(block_slices) == 6
    assert (slice(0, 1, None), slice(0, 1, None)) in block_slices
    assert (slice(0, 1, None), slice(1, 4, None)) in block_slices
    assert (slice(1, 4, None), slice(0, 1, None)) not in block_slices
    assert max_ell == 4
    assert nfeatures == 3
    assert mask[..., 0, 0, 0] == 1
    assert np.all(mask[..., 1, 1:4, 0] == 1)
    assert mask[..., 0, 0, 1] == 1
    assert np.all(mask[..., 0, 1:4, 1] == 0)
    assert np.all(mask[..., 0, 4:9, 1] == 1)