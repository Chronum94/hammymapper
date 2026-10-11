import numpy as np


class _SlotAllocator:
    """Assigns each H subblock a (parity, ell-range, feature) slot in an e3x-style
    irreps array of shape (2, (max_ell + 1)**2, num_features).

    A subblock coupling ell1 and ell2 occupies the irreps ell = |ell1 - ell2| .. ell1 + ell2
    of a single feature. The whole range is reserved, even for components that are masked
    off because they vanish by symmetry. Reusing such a masked-off gap for another subblock
    would make two subblocks write into the same components.
    """

    def __init__(self, max_ell: int, max_features: int):
        self.occupied = np.zeros((2, (max_ell + 1) ** 2, max_features), dtype=bool)
        self.mask = np.zeros((2, (max_ell + 1) ** 2, max_features), dtype=np.int8)
        self.num_features = 0

    def allocate(self, parity: int, ellmin: int, ellmax: int, first_feature: int = 0,
                 vanishing_ells: tuple[int, ...] = ()) -> int:
        ell_slice = slice(ellmin**2, (ellmax + 1) ** 2)
        for ifeature in range(first_feature, self.occupied.shape[-1]):
            if not self.occupied[parity, ell_slice, ifeature].any():
                self.occupied[parity, ell_slice, ifeature] = True
                self.mask[parity, ell_slice, ifeature] = 1
                for ell in vanishing_ells:
                    self.mask[parity, ell**2 : (ell + 1) ** 2, ifeature] = 0
                self.num_features = max(self.num_features, ifeature + 1)
                return ifeature
        raise RuntimeError("Ran out of features while assigning irreps slots.")

    def pruned_mask(self) -> np.ndarray:
        return self.mask[..., : self.num_features]


def _subblock_slices(ell1, ell2, rowstart, colstart):
    ellmin, ellmax = abs(ell1 - ell2), ell1 + ell2
    cgc_slice = (
        slice(ell1**2, (ell1 + 1) ** 2),
        slice(ell2**2, (ell2 + 1) ** 2),
        slice(ellmin**2, (ellmax + 1) ** 2),
    )
    block_slice = (
        slice(rowstart, rowstart + 2 * ell1 + 1),
        slice(colstart, colstart + 2 * ell2 + 1),
    )
    return ellmin, ellmax, cgc_slice, block_slice


def _vanishing_ells(ellmin: int, ellmax: int, transpose_sign: int) -> tuple[int, ...]:
    """Irreps that vanish for a subblock coupling an orbital shell with itself.

    Transposing a coupled (ell1, ell2, ell) block multiplies it by (-1)**(ell1 + ell2 - ell),
    which is (-1)**ell for ell1 == ell2. A block that equals transpose_sign times its own
    transpose can therefore only have the irreps with (-1)**ell == transpose_sign
    (arXiv:2109.12083, Sec. III).
    """
    return tuple(ell for ell in range(ellmin, ellmax + 1) if (-1) ** ell != transpose_sign)


def get_element_mapping_spec(ells1: list[int]):
    """Mapping spec for the (real-symmetric) on-site block of one element.

    Only subblocks with n2 >= n1 are mapped; the rest follow from symmetry. Diagonal
    subblocks (n1 == n2) are symmetric, so their odd-ell irreps vanish and are masked off.
    """
    # This is the maximum irrep angular momentum from Wigner-Eckhart
    # https://e3x.readthedocs.io/stable/overview.html#coupling-irreps
    max_ell = 2 * max(ells1)
    slots = _SlotAllocator(max_ell, len(ells1) ** 2)

    block_slices, cgc_slices, irreps_array_slices = [], [], []
    rowstart = 0
    for n1, ell1 in enumerate(ells1):
        colstart = 0
        for n2, ell2 in enumerate(ells1):
            if n2 < n1:
                # We just need to learn one half of the matrix
                colstart += 2 * ell2 + 1
                continue
            ellmin, ellmax, cgc_slice, block_slice = _subblock_slices(ell1, ell2, rowstart, colstart)
            # Spatial (inversion) parity of the coupled irreps is (-1)**(ell1 + ell2)
            parity = (ell1 + ell2) % 2
            vanishing = _vanishing_ells(ellmin, ellmax, +1) if n1 == n2 else ()
            ifeature = slots.allocate(parity, ellmin, ellmax, vanishing_ells=vanishing)

            cgc_slices.append(cgc_slice)
            block_slices.append(block_slice)
            irreps_array_slices.append((parity, slice(ellmin**2, (ellmax + 1) ** 2), ifeature))
            colstart += 2 * ell2 + 1
        rowstart += 2 * ell1 + 1

    assert len(block_slices) == len(irreps_array_slices) == len(cgc_slices)
    return (block_slices, irreps_array_slices, cgc_slices, max_ell, slots.num_features, slots.pruned_mask())


def get_pair_mapping_spec(ells1: list[int], ells2: list[int], permutation_symmetrize: bool):
    """Takes a general hamiltonian block with an arbitrary arrangement of angular
    momentum blocks, and returns indices corresponding to those blocks, the corresponding
    Clebsch-Gordan blocks (in e3x convention) to move between those and irreps, a mask,
    and slices of that mask corresponding to the irreps of each subblock of H.

    Parameters
    ----------
    ells1 : list[int]
        Angular momenta of the orbital shells of the first (row) element.
    ells2 : list[int]
        Angular momenta of the orbital shells of the second (column) element.
    permutation_symmetrize : bool
        True for homonuclear pairs. The block H_ij is then split into its permutation
        symmetric and antisymmetric parts, H_ij +- H_ji = H_ij +- H_ij^T (arXiv:2109.12083,
        Eq. 20), stacked row-wise as [sym; anti] with twice the rows. Each part only needs
        the n2 >= n1 subblocks.

    Notes
    -----
    The e3x parity index of every subblock is its spatial parity, (ell1 + ell2) % 2.
    Exchange symmetry (sym / anti) is an independent label and lives in separate feature
    channels: the anti part's features come after all of the sym part's features.
    """
    # This is the maximum irrep angular momentum from Wigner-Eckhart
    # https://e3x.readthedocs.io/stable/overview.html#coupling-irreps
    max_ell = max(ells1) + max(ells2)
    # A pessimistic, always-valid number of features
    max_features = len(ells1) * len(ells2) * (2 if permutation_symmetrize else 1)
    slots = _SlotAllocator(max_ell, max_features)

    block_slices, cgc_slices, irreps_array_slices = [], [], []
    rowstart = 0
    first_feature = 0
    # There's an implicit row-wise splitting of the sym and anti parts for homonuclear
    # blocks: for a 9x9 block, rows 0-9 are the sym part and rows 9-18 the anti part.
    # transpose_sign is +1 (sym), -1 (anti), or 0 (heteronuclear, no exchange symmetry).
    for transpose_sign in [1, -1] if permutation_symmetrize else [0]:
        for n1, ell1 in enumerate(ells1):
            colstart = 0
            for n2, ell2 in enumerate(ells2):
                if transpose_sign != 0 and n2 < n1:
                    # Follows from the transpose symmetry of the sym / anti part
                    colstart += 2 * ell2 + 1
                    continue
                ellmin, ellmax, cgc_slice, block_slice = _subblock_slices(ell1, ell2, rowstart, colstart)
                parity = (ell1 + ell2) % 2
                # Only for homonuclear blocks does n1 == n2 mean the same orbital shell
                vanishing = (
                    _vanishing_ells(ellmin, ellmax, transpose_sign)
                    if transpose_sign != 0 and n1 == n2
                    else ()
                )
                ifeature = slots.allocate(parity, ellmin, ellmax, first_feature, vanishing)

                cgc_slices.append(cgc_slice)
                block_slices.append(block_slice)
                irreps_array_slices.append((parity, slice(ellmin**2, (ellmax + 1) ** 2), ifeature))
                colstart += 2 * ell2 + 1
            rowstart += 2 * ell1 + 1
        # The anti part gets its own feature channels
        first_feature = slots.num_features

    assert len(block_slices) == len(irreps_array_slices) == len(cgc_slices)
    return (block_slices, irreps_array_slices, cgc_slices, max_ell, slots.num_features, slots.pruned_mask())
