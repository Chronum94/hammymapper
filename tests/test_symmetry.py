"""Equivariance and permutation-symmetry checks (cf. arXiv:2109.12083, Sec. III)."""
import e3x
import jax
import numpy as np

from conftest import all_blocks, full_mask, orbital_matrix, spec, to_irreps

ATOL = 1e-10


def _wigner_d(mapper, seed):
    R = e3x.so3.random_rotation(jax.random.PRNGKey(seed))
    return np.asarray(e3x.so3.wigner_d(R, max_degree=mapper.max_ell, cartesian_order=False), dtype=np.float64)


def test_rotation_equivariance(species, mapper, rng):
    """Rotating the orbitals (H -> D_i^T H D_j with real Wigner-D blocks) rotates the irreps
    by the Wigner-D matrix of each degree, as e3x features do (x -> x @ D)."""
    D = _wigner_d(mapper, seed=7)
    Dl = lambda ell: D[ell**2 : (ell + 1) ** 2, ell**2 : (ell + 1) ** 2]
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        Ri = orbital_matrix(species[Z_i], Dl)
        Rj = orbital_matrix(species[Z_j if Z_j is not None else Z_i], Dl)
        m = full_mask(mapper, Z_i, Z_j)
        rotated = to_irreps(mapper, Ri.T @ block @ Rj, Z_i, Z_j)
        expected = np.einsum("...pcf,cd->...pdf", to_irreps(mapper, block, Z_i, Z_j), D)
        np.testing.assert_allclose(rotated * m, expected * m, atol=1e-6, err_msg=label)  # D is float32


def test_inversion_parity(species, mapper, rng):
    """Inversion multiplies an orbital of degree ell by (-1)**ell. The coupled irreps must
    then transform with the e3x parity index they are stored in: index 0 even, index 1 odd.
    This fails if the exchange label (sym / anti) is stored in the parity axis."""
    P = lambda ell: (-1) ** ell * np.eye(2 * ell + 1)
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        Pi = orbital_matrix(species[Z_i], P)
        Pj = orbital_matrix(species[Z_j if Z_j is not None else Z_i], P)
        irreps = to_irreps(mapper, block, Z_i, Z_j)
        expected = irreps.copy()
        expected[:, 1] *= -1
        np.testing.assert_allclose(to_irreps(mapper, Pi @ block @ Pj, Z_i, Z_j), expected, atol=ATOL, err_msg=label)


def test_parity_index_is_spatial_parity(mapper):
    """Every subblock (ell1, ell2) is stored in parity index (ell1 + ell2) % 2."""
    for key, ms in mapper.mapper.items():
        for (rows, cols), (parity, _, _) in zip(ms.block_slices, ms.irreps_slices):
            ell1, ell2 = (rows.stop - rows.start - 1) // 2, (cols.stop - cols.start - 1) // 2
            assert parity == (ell1 + ell2) % 2, (key, ell1, ell2)


def test_homonuclear_exchange_symmetry(species, mapper, rng):
    """Swapping the atoms of a homonuclear pair, H_ji = H_ij^T, leaves the sym irreps
    unchanged and negates the anti irreps."""
    for Z in species:
        ms = spec(mapper, Z, Z)
        half = len(ms.irreps_slices) // 2
        block = rng.normal(size=(3,) + (ms.nrows // 2, ms.ncols))
        a = to_irreps(mapper, block, Z, Z)
        b = to_irreps(mapper, np.swapaxes(block, -1, -2), Z, Z)
        for k, (p, sl, f) in enumerate(ms.irreps_slices):
            sign = 1 if k < half else -1
            np.testing.assert_allclose(b[:, p, sl, f], sign * a[:, p, sl, f], atol=ATOL)


def test_sym_and_anti_use_disjoint_features(mapper):
    for key, ms in mapper.mapper.items():
        if isinstance(key, tuple) and key[0] == key[1]:
            half = len(ms.irreps_slices) // 2
            sym = {f for _, _, f in ms.irreps_slices[:half]}
            anti = {f for _, _, f in ms.irreps_slices[half:]}
            assert sym.isdisjoint(anti), key


def test_masked_components_vanish(species, mapper, rng):
    """Components masked off inside a subblock's ell range are exactly the ones that vanish
    by symmetry (paper: some symmetrized terms are identically zero), so no information is
    dropped. Checked before masking by projecting with the raw CG coefficients."""
    for label, block, Z_i, Z_j in all_blocks(species, rng):
        ms = spec(mapper, Z_i, Z_j)
        b = block
        if Z_j is not None and Z_i == Z_j:
            b = np.concatenate([0.5 * (b + np.swapaxes(b, -1, -2)), 0.5 * (b - np.swapaxes(b, -1, -2))], axis=1)
        elif Z_j is not None and Z_i > Z_j:
            b = np.swapaxes(b, -1, -2)
        for block_slice, cgc_slice, irreps_slice in zip(ms.block_slices, ms.cgc_slices, ms.irreps_slices):
            raw = np.einsum("...mn,mnl->...l", b[(slice(None),) + block_slice], ms.cgc[cgc_slice])
            dropped = raw * (1 - ms.mask[irreps_slice])
            assert np.abs(dropped).max() < 1e-12, label


def test_heteronuclear_orientations_agree(species, mapper, rng):
    """H_ij and H_ji = H_ij^T map to the same irreps (the mapper stores pairs in sorted
    Z order)."""
    Zs = sorted(species)
    for a, b in [(Zs[0], Zs[1])]:
        block = rng.normal(size=(2, spec(mapper, a, b).nrows, spec(mapper, a, b).ncols))
        np.testing.assert_allclose(to_irreps(mapper, block, a, b),
                                   to_irreps(mapper, np.swapaxes(block, -1, -2), b, a), atol=ATOL)


def test_reconstructed_hamiltonian_is_symmetric(species, mapper, rng):
    """Assembling the full H from irreps of the (i, j) blocks gives the original symmetric
    H, with H_ji == H_ij^T reconstructed independently from each orientation."""
    from conftest import random_symmetric_hamiltonian, atom_block
    H, Zs, off = random_symmetric_hamiltonian(species, rng)
    H_rec = np.zeros_like(H)
    for i in range(len(Zs)):
        for j in range(len(Zs)):
            Z_j = None if i == j else Zs[j]
            irreps = to_irreps(mapper, atom_block(H, off, i, j), Zs[i], Z_j) * full_mask(mapper, Zs[i], Z_j)
            H_rec[off[i] : off[i + 1], off[j] : off[j + 1]] = mapper.irreps_to_hblocks(irreps, Zs[i], Z_j)[0]
    np.testing.assert_allclose(H_rec, H, atol=1e-12)
    np.testing.assert_allclose(H_rec, H_rec.T, atol=1e-12)
