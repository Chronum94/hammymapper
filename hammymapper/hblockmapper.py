import logging
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import e3x
import jax
import numpy as np

from hammymapper.utilities import get_pair_mapping_spec, get_element_mapping_spec

log = logging.getLogger(__name__)

# Orbitals up to MAX_ORBITAL_ELL (f) are supported by the Clebsch-Gordan table below.
MAX_ORBITAL_ELL = 3

# Computed in float64: JAX would otherwise produce float32 coefficients, which limits
# round trips to ~1e-7 relative accuracy.
_enable_x64 = (lambda: jax.enable_x64(True)) if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
with _enable_x64():
    _CGC = np.asarray(
        e3x.so3.clebsch_gordan(MAX_ORBITAL_ELL, MAX_ORBITAL_ELL, 2 * MAX_ORBITAL_ELL, cartesian_order=False),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BlockIrrepMappingSpec:
    # Slices for subblocking a block
    block_slices: list[tuple[slice, slice]]
    # Slices of CG coeffs to transform back and forth
    cgc_slices: list[tuple[slice, slice, slice]]
    # Slices of irreps array corresponding to subblocks
    irreps_slices: list[tuple[int, slice, int]]
    max_ell: int
    nfeatures: int
    mask: np.ndarray
    cgc = _CGC
    nrows: int
    ncols: int

    def __repr__(self):
        return f"Mapper(nblocks={len(self.block_slices)}, max_ell={self.max_ell}, nfeatures={self.nfeatures})"


@dataclass()
class MultiElementPairHBlockMapper:
    # We keep atomic_number_pairs to map onto the hamiltonian block mappers
    mapper: dict[int | tuple[int, int], BlockIrrepMappingSpec]
    max_ell: int = field(init=False)
    nfeatures: int = field(init=False)

    def __post_init__(self):
        self.max_ell = max([x.max_ell for x in self.mapper.values()])
        self.nfeatures = max([x.nfeatures for x in self.mapper.values()])

    def _spec(self, Z_i, Z_j):
        key = Z_i if Z_j is None else tuple(sorted([Z_i, Z_j]))
        try:
            return self.mapper[key]
        except KeyError:
            raise KeyError(f"No mapping spec for {key}; known keys: {sorted(self.mapper, key=str)}") from None

    def hblocks_to_irreps(self, hblocks, irreps_array, Z_i, Z_j = None):
        """Projects a batch of H blocks onto irreps, writing into ``irreps_array`` in place.

        Parameters
        ----------
        hblocks : array of shape (batch, n_i, n_j)
            On-site blocks (Z_j None) or pair blocks H_ij in the (Z_i, Z_j) orientation.
        irreps_array : array of shape (>= batch, 2, >= (max_ell + 1)**2, >= nfeatures)
            Output buffer. Rows [:batch] are overwritten completely (unused slots are
            zeroed), the remaining rows are left untouched. The buffer may be reused
            and may have a lower precision dtype (e.g. float32).

        Returns
        -------
        irreps_array
            The same buffer that was passed in.
        """
        ms = self._spec(Z_i, Z_j)
        hblocks = np.asarray(hblocks)
        n_i = ms.nrows // 2 if Z_i == Z_j else ms.nrows
        expected = (n_i, ms.ncols) if (Z_j is None or Z_i <= Z_j) else (ms.ncols, n_i)
        if hblocks.ndim != 3 or hblocks.shape[1:] != expected:
            raise ValueError(f"Expected hblocks of shape (batch, {expected[0]}, {expected[1]}) "
                             f"for Z_i={Z_i}, Z_j={Z_j}; got {hblocks.shape}")
        nbatch = len(hblocks)
        if (irreps_array.ndim != 4 or len(irreps_array) < nbatch or irreps_array.shape[1] != 2
                or irreps_array.shape[2] < (ms.max_ell + 1) ** 2 or irreps_array.shape[3] < ms.nfeatures):
            raise ValueError(f"irreps_array of shape {irreps_array.shape} cannot hold {nbatch} samples of "
                             f"(2, {(ms.max_ell + 1) ** 2}, {ms.nfeatures})")

        # Learn even/odd combinations for homonuclear blocks, stacked row-wise as
        # [even; odd]. Built in one preallocated array to avoid full-size temporaries.
        if Z_i == Z_j:
            hblocks_t = np.swapaxes(hblocks, -2, -1)
            stacked = np.empty((nbatch, 2 * n_i, ms.ncols), dtype=np.result_type(hblocks.dtype, np.float32))
            np.add(hblocks, hblocks_t, out=stacked[:, :n_i])
            np.subtract(hblocks, hblocks_t, out=stacked[:, n_i:])
            stacked *= 0.5
            hblocks = stacked

        # We order by atomic number, so if the blocks we are handling have Z_j < Z_i,
        # we need to transpose them (a view, no copy)
        if Z_j is not None and Z_i > Z_j:
            hblocks = np.swapaxes(hblocks, -2, -1)

        out = irreps_array[:nbatch]
        out[...] = 0
        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            # Components that vanish by symmetry are masked to exact zeros
            out[(slice(None),) + irreps_slice] = ms.mask[irreps_slice] * np.einsum(
                "...mn,mnl->...l",
                hblocks[(slice(None),) + block_slice],
                ms.cgc[cgc_slice],
                optimize=True,
            )
        return irreps_array

    def irreps_to_hblocks(self, irreps_array, Z_i, Z_j = None):
        """Inverse of ``hblocks_to_irreps``. Returns a new array of H blocks in the
        caller's (Z_i, Z_j) orientation. Masked-off irreps (unused slots and components
        that vanish by symmetry) are ignored, so arbitrary values there do not leak into H.
        """
        ms = self._spec(Z_i, Z_j)
        irreps_array = np.asarray(irreps_array)
        nbatch = len(irreps_array)
        dtype = np.result_type(irreps_array.dtype, np.float32)

        hblocks_raw = np.zeros((nbatch, ms.nrows, ms.ncols), dtype=dtype)
        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            hblocks_raw[(slice(None),) + block_slice] = np.einsum(
                "...l,mnl->...mn",
                irreps_array[(slice(None),) + irreps_slice] * ms.mask[irreps_slice],
                ms.cgc[cgc_slice],
                optimize=True,
            )

        if Z_j is None:
            # Onsite block: real-symmetric (Hermitian).
            # Forward only projected the upper triangle (n2 >= n1).
            # Fill the strictly lower triangle from the upper triangle.
            tril_r, tril_c = np.tril_indices(ms.nrows, k=-1)
            hblocks_raw[:, tril_r, tril_c] = hblocks_raw[:, tril_c, tril_r]
            return hblocks_raw

        elif Z_i == Z_j:
            # Homonuclear pair block.
            # ms.nrows is doubled: rows 0:N are the even (symmetric) part,
            # rows N:2N are the odd (antisymmetric) part, matching what
            # hblocks_to_irreps does when it concatenates hblockseven and hblocksodd.
            nrows_actual = ms.nrows // 2
            hblocks_even = hblocks_raw[:, :nrows_actual, :]
            hblocks_odd  = hblocks_raw[:, nrows_actual:, :]

            tril_r, tril_c = np.tril_indices(nrows_actual, k=-1)
            # Even / symmetric part: H_even[n2,n1] = H_even[n1,n2]^T
            hblocks_even[:, tril_r, tril_c] =  hblocks_even[:, tril_c, tril_r]
            # Odd / antisymmetric part: H_odd[n2,n1] = -H_odd[n1,n2]^T
            hblocks_odd[:, tril_r, tril_c]  = -hblocks_odd[:, tril_c, tril_r]

            return hblocks_even + hblocks_odd

        else:
            # Heteronuclear pair block: no permutation symmetry,
            # all subblocks were projected in the forward pass.
            # Undo the forward transpose to return the caller's (Z_i, Z_j) orientation.
            if Z_i > Z_j:
                return np.ascontiguousarray(np.swapaxes(hblocks_raw, -2, -1))
            return hblocks_raw


def make_mapper_from_elements(species_ells_dict: dict[int, list[int]]):
    for Z, ells in species_ells_dict.items():
        if len(ells) == 0 or min(ells) < 0 or max(ells) > MAX_ORBITAL_ELL:
            raise ValueError(f"Element {Z}: orbital angular momenta must be a non-empty list "
                             f"of integers in [0, {MAX_ORBITAL_ELL}], got {ells}")

    mapper_keys = []
    mappers = []

    atomic_numbers = species_ells_dict.keys()
    for Z_i in atomic_numbers:
        ells1 = species_ells_dict[Z_i]
        (
            block_slices,
            irreps_slices,
            cgc_slices,
            max_ell_for_pair,
            num_features_for_pair,
            irrep_mask,
        ) = get_element_mapping_spec(ells1)

        mapper_keys.append(Z_i)
        mappers.append(BlockIrrepMappingSpec(
                block_slices=block_slices,
                cgc_slices=cgc_slices,
                irreps_slices=irreps_slices,
                max_ell=max_ell_for_pair,
                nfeatures=num_features_for_pair,
                mask=irrep_mask,
                nrows=sum([2 * ell + 1 for ell in ells1]),
                ncols=sum([2 * ell + 1 for ell in ells1])
            )
        )

    for Z_i, Z_j in combinations_with_replacement(sorted(atomic_numbers), 2):
        ells1 = species_ells_dict[Z_i]
        ells2 = species_ells_dict[Z_j]
        (
            block_slices,
            irreps_slices,
            cgc_slices,
            max_ell_for_pair,
            num_features_for_pair,
            irrep_mask,
        ) = get_pair_mapping_spec(ells1, ells2, permutation_symmetrize=Z_i == Z_j)

        log.debug(
            f"Pair: {Z_i}, {Z_j}, max_ell: {max_ell_for_pair}, num_features:{num_features_for_pair}"
        )

        mapper_keys.append((Z_i, Z_j))
        mappers.append(
            BlockIrrepMappingSpec(
                block_slices=block_slices,
                cgc_slices=cgc_slices,
                irreps_slices=irreps_slices,
                max_ell=max_ell_for_pair,
                nfeatures=num_features_for_pair,
                mask=irrep_mask,
                nrows=sum([2 * ell + 1 for ell in ells1]) * (2 if Z_i == Z_j else 1),
                ncols=sum([2 * ell + 1 for ell in ells2])
            )
        )
    return MultiElementPairHBlockMapper(
        dict(zip(mapper_keys, mappers))
    )


def get_mask_dict(
    max_ell: int, nfeatures: int, pairwise_hmap: MultiElementPairHBlockMapper
) -> dict[tuple[int, int], np.ndarray]:
    mask_dict = {}
    for element_pair, blockmapper in pairwise_hmap.mapper.items():
        # This is e3x convention. 2 for parity, angular momentum channels, features
        # Excludes components that vanish by symmetry (see get_*_mapping_spec)
        mask_array = np.zeros((2, (max_ell + 1) ** 2, nfeatures), dtype=np.int8)
        m = blockmapper.mask
        mask_array[:, : m.shape[1], : m.shape[2]] = m
        mask_dict[element_pair] = mask_array

        log.debug(f"Element pair: {element_pair}, mask:\n{mask_array}")
    return mask_dict