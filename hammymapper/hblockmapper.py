import logging
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import e3x
import numpy as np

from hammymapper.utilities import get_pair_mapping_spec, get_element_mapping_spec

log = logging.getLogger(__name__)


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
    cgc = np.array(e3x.so3.clebsch_gordan(3, 3, 6, cartesian_order=False))
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

    def hblocks_to_irreps(self, hblocks, Z_i, Z_j = None):
        # assert len(hblocks) == len(irreps_array)
        mapping_spec = self.mapper[Z_i if Z_j is None else tuple(sorted([Z_i, Z_j]))]

        ms = mapping_spec
        
        irreps_array = np.zeros((len(hblocks), 2, (self.max_ell + 1) ** 2, self.nfeatures))

        # Learn even/odd combinations for homonuclear blocks
        if Z_i == Z_j:
            hblockseven = 0.5 * (hblocks + np.swapaxes(hblocks, -2, -1))
            hblocksodd = 0.5 * (hblocks - np.swapaxes(hblocks, -2, -1))
            hblocks = np.concatenate([hblockseven, hblocksodd], axis=1)

        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            block_slice = (slice(0, len(hblocks)),) + block_slice
            irreps_slice = (slice(0, len(hblocks)),) + irreps_slice

            np.einsum(
                "...mn,mnl->...l",
                hblocks[block_slice],
                ms.cgc[cgc_slice],
                out=irreps_array[irreps_slice],
                optimize=True,
            )

        mask = ms.mask
        mask_slices = [slice(0, len(hblocks)),] + [slice(0, x) for x in mask.shape]
        # irreps_array[*mask_slices] *= mask
        # print(mask_slices)
        return irreps_array

    def irreps_to_hblocks(self, irreps_array, Z_i, Z_j = None):
        mapping_spec = self.mapper[Z_i if Z_j is None else tuple(sorted([Z_i, Z_j]))]

        ms = mapping_spec

        hblocks = np.zeros((len(irreps_array), ))
        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            block_slice = (slice(0, len(hblocks)),) + block_slice
            irreps_slice = (slice(0, len(hblocks)),) + irreps_slice
            hblocks[block_slice] = np.einsum("...l,mnl->...mn", 
                                             irreps_array[irreps_slice], 
                                             ms.cgc[cgc_slice],
                                             optimize = True
                                            )


def make_mapper_from_elements(species_ells_dict: dict[int, list[int]]):
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

    for Z_i, Z_j in combinations_with_replacement(atomic_numbers, 2):
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
        mask_array = np.zeros((2, (max_ell + 1) ** 2, nfeatures), dtype=np.int8)
        for _slice in blockmapper.irreps_slices:
            log.debug(f"Element pair: {element_pair}, mask:\n{_slice}")
            mask_array[_slice] = 1
        mask_dict[element_pair] = mask_array

        log.debug(f"Element pair: {element_pair}, mask:\n{mask_array}")
    return mask_dict