import numpy as np

def get_element_mapping_spec(ells1: list[int]):
    parity_dict = dict(zip(np.arange(7), np.arange(7) % 2))
    # This is the maximum irrep angular momentum from Wigner-Eckhart
    # https://e3x.readthedocs.io/stable/overview.html#coupling-irreps
    max_ell = 2 * max(ells1)
    # The feature direction adds up the two angular momentum numbers. This is a
    # very pessimistic estimate of the number of features required, but is
    # therefore always valid.
    num_features = len(ells1) ** 2
    # This is since we now hook up the peven and podd parts of the block
    # separately
    mask_candidate = np.zeros(
        (2, (max_ell + 1) ** 2, num_features), dtype=np.int8
    )

    # List of tuples, each of which is the indexing for the H blocks
    block_slices = []
    # As above but for CGC blocks
    cgc_slices = []
    # As above but for final equivariant feature block
    irreps_array_slices = []

    ifeaturemax = 0
    rowstart = 0

    for n1, ell1 in enumerate(ells1):
        colstart = 0
        for n2, ell2 in enumerate(ells1):
            if n2 < n1:
                colstart += 2 * ell2 + 1
                # We just need to learn one half of the matrix
                continue
            # We are rigorous about the parity block index of a given product
            parity = int(np.logical_xor(parity_dict[ell1], parity_dict[ell2]))

            # Wigner-Eckhart irreps limits
            ellmin = abs(ell1 - ell2)
            ellmax = ell1 + ell2

            # Indexing for CGC subblock
            cgc_slices.append(
                (
                    slice(ell1**2, (ell1 + 1) ** 2),
                    slice(ell2**2, (ell2 + 1) ** 2),
                    slice(ellmin**2, (ellmax + 1) ** 2),
                )
            )

            # Indexing for this H subblock
            block_slices.append(
                (
                    slice(rowstart, rowstart + 2 * ell1 + 1),
                    slice(colstart, colstart + 2 * ell2 + 1),
                )
            )

            # Find first feature which has all required irreps for this feature block
            # available
            for ifeature in range(mask_candidate.shape[-1]):
                if np.all(
                    mask_candidate[parity, ellmin**2 : (ellmax + 1) ** 2, ifeature] == 0
                ):
                    mask_candidate[parity, ellmin**2 : (ellmax + 1) ** 2, ifeature] = 1

                    # For n1=n2 blocks, these are purely-symmetric square blocks
                    # So they cannot have odd ells
                    if n1 == n2:
                        for ell in range(ellmin, ellmax + 1):
                            if ell % 2 != 0:
                                mask_candidate[parity, ell**2 : (ell + 1) ** 2, ifeature] = 0


                    # Keep track of the maximum feature size we are at
                    # We prune the mask in the feature dimension to this
                    ifeaturemax = max(ifeaturemax, ifeature)

                    break

            # Indexing for this H block's irreps
            irreps_array_slices.append(
                (parity, slice(ellmin**2, (ellmax + 1) ** 2), ifeature)
            )

            colstart += 2 * ell2 + 1
        rowstart += 2 * ell1 + 1

    assert len(block_slices) == len(irreps_array_slices) == len(cgc_slices)
    # Add one to features here because array sizes are exclusive of index=length.
    return (block_slices, irreps_array_slices, cgc_slices, max_ell, ifeaturemax + 1, mask_candidate[..., :ifeaturemax + 1])


def get_pair_mapping_spec(ells1: list[int], ells2: list[int], permutation_symmetrize):
    """I hate this function. It takes a general hamiltonian block with an
    arbitrary arragement of angular momentum blocks, and then spits out indices
    corresponding to those blocks, and the corresponding Clebsch-Gordan blocks
    (in e3x convention) that will be required to move between those and irreps.
    It also provides a mask, and slices of that mask corresponding to the
    irreps of each subblock of H. As far as I can tell, this is strictly all
    the information required to go back and forth between H blocks and irreps.

    Parameters
    ----------
    ells1 : _type_
        _description_
    ells2 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print("Am I permutation symmetrizing?", permutation_symmetrize)
    parity_dict = dict(zip(np.arange(7), np.arange(7) % 2))
    # This is the maximum irrep angular momentum from Wigner-Eckhart
    # https://e3x.readthedocs.io/stable/overview.html#coupling-irreps
    max_ell = max(ells1) + max(ells2)
    # The feature direction adds up the two angular momentum numbers. This is a
    # very pessimistic estimate of the number of features required, but is
    # therefore always valid.
    num_features = len(ells1) * len(ells2)
    # This is since we now hook up the peven and podd parts of the block
    # separately
    num_features *= 2 if permutation_symmetrize else 1
    mask_candidate = np.zeros(
        (2, (max_ell + 1) ** 2, len(ells1) * len(ells2)), dtype=np.int8
    )

    # List of tuples, each of which is the indexing for the H blocks
    block_slices = []
    # As above but for CGC blocks
    cgc_slices = []
    # As above but for final equivariant feature block
    irreps_array_slices = []

    ifeaturemax = 0
    rowstart = 0

    # There's an implicit row-wise splitting of the even and odd parts
    # for a permutation (anti)symmetric hammy block here.
    # For say, for a 9x9 homonuclear block, rows 0-9 are the symmetric
    # part, while 9-18 are the antisymmetric parts (with their corresponding)
    # analytical relations for how n1 = n2 behave (no odd ell irreps for symm
    # and no even ell irreps for antisymm).
    # The 1's and -1's double as indexing hacks.
    for permutation_symmetry in [1, -1] if permutation_symmetrize else [0]:
        for n1, ell1 in enumerate(ells1):
            colstart = 0
            for n2, ell2 in enumerate(ells2):
                if permutation_symmetry != 0 and n2 < n1:
                    # There is some permutation/transpose symmetry here
                    # So we don't learn n2 < n1
                    # So we skip this, and we move the column start index forward.
                    colstart += 2 * ell2 + 1
                    continue
                # We are rigorous about the parity block index of a given product
                # For general blocks, the subblock ells determine the symmetry
                # For permutation (anti)symmetric blocks, that gives us the parities.
                if permutation_symmetry == 0:
                    parity = int(np.logical_xor(parity_dict[ell1], parity_dict[ell2]))
                else:
                    # This is to switch to e3x's parity indices.
                    if permutation_symmetry == 1:
                        parity = 0
                    else:
                        parity = 1
                
                print(n1, n2, rowstart, colstart)
                # Wigner-Eckhart irreps limits
                ellmin = abs(ell1 - ell2)
                ellmax = ell1 + ell2

                # Indexing for CGC subblock
                cgc_slices.append(
                    (
                        slice(ell1**2, (ell1 + 1) ** 2),
                        slice(ell2**2, (ell2 + 1) ** 2),
                        slice(ellmin**2, (ellmax + 1) ** 2),
                    )
                )

                # Indexing for this H subblock
                block_slices.append(
                    (
                        slice(rowstart, rowstart + 2 * ell1 + 1),
                        slice(colstart, colstart + 2 * ell2 + 1),
                    )
                )

                # Find first feature which has all required irreps for this feature block
                # available
                for ifeature in range(mask_candidate.shape[-1]):
                    if np.all(
                        mask_candidate[parity, ellmin**2 : (ellmax + 1) ** 2, ifeature] == 0
                    ):
                        mask_candidate[parity, ellmin**2 : (ellmax + 1) ** 2, ifeature] = 1

                        # For n1=n2 blocks, these are purely-(anti)symmetric square blocks
                        # So they cannot have odd ells if symmetric, and even ells if antisymmetric
                        if n1 == n2:
                            for ell in range(ellmin, ellmax + 1):
                                if (parity == 0 and ell % 2 != 0) or (parity == 1 and ell % 2 == 0):
                                    mask_candidate[parity, ell**2 : (ell + 1) ** 2, ifeature] = 0


                        # Keep track of the maximum feature size we are at
                        # We prune the mask in the feature dimension to this
                        ifeaturemax = max(ifeaturemax, ifeature)
                        break

                # Indexing for this H block's irreps
                irreps_array_slices.append(
                    (parity, slice(ellmin**2, (ellmax + 1) ** 2), ifeature)
                )

                colstart += 2 * ell2 + 1
            rowstart += 2 * ell1 + 1

    assert len(block_slices) == len(irreps_array_slices) == len(cgc_slices)
    # Add one to features here because array sizes are exclusive of index=length.
    return (block_slices, irreps_array_slices, cgc_slices, max_ell, ifeaturemax + 1, mask_candidate[..., :ifeaturemax + 1])