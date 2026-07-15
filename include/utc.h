#pragma once

#include "tensor.h"
#include "mat_decomp.h"   // SVDDecomp

#include <algorithm>
#include <stdexcept>
#include <vector>

// =====================================================================
// utc: tensor-train contraction kernels (Python `utc` module).
//
// TODO: all four routines below are unimplemented stubs. They compile
// and throw at runtime so that MPO::_mul can be written and tested
// around them.
// =====================================================================
namespace utc {

// ---------------------------------------------------------------------
// Helper for zip-up: the one-site contraction.
//
// Given the running environment r(a, l, L), an MPS core x(l, col, rM)
// and an MPO core o(L, p_fused, RO) with p_fused = row * p_in + col,
// builds the matrix
//
//   M[(a + row*A), (rM + RO*n_rM)]
//       = sum_{l, L, col} r(a,l,L) x(l,col,rM) o(L, row*p_in+col, RO)
//
// Index conventions are chosen to line up with Tensor3D's column-major
// layout: the row index (a fastest, row slower) is exactly
// flatten_as_matrix2 of a Tensor3D(A, p_out, *), and the column index
// (rM fastest, RO slower) is exactly flatten_as_matrix1 of a
// Tensor3D(*, n_rM, n_RO). That lets the SVD factors be copied
// straight into the output core / next environment with no reshuffle.
//
// Contraction order: for each (col), first absorb the MPS slice into
// the environment (G), then for each (row) multiply by the MPO slice
// and scatter — O(chi^3) per site, no big temporaries.
// ---------------------------------------------------------------------
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
_zip_up_site_matrix(Tensor3D<T> const& r,
                    Tensor3D<T> const& x,
                    Tensor3D<T> const& o)
{
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    const Eigen::Index A     = r.n_left;
    const Eigen::Index L     = r.n_right;
    const Eigen::Index p_in  = x.n_phys;
    const Eigen::Index p_out = o.n_phys / p_in;
    const Eigen::Index n_rM  = x.n_right;
    const Eigen::Index n_RO  = o.n_right;

    if (o.n_phys % p_in != 0)
        throw std::invalid_argument(
            "utc::zip_up_mpo_mps: MPO physical dimension is not a multiple of the MPS one");
    if (r.n_phys != x.n_left || r.n_right != o.n_left)
        throw std::invalid_argument(
            "utc::zip_up_mpo_mps: bond dimension mismatch between environment and cores");

    MatrixX M = MatrixX::Zero(A * p_out, n_rM * n_RO);
    MatrixX G(A * n_rM, L);

    for (Eigen::Index col = 0; col < p_in; ++col) {
        // G[(a + A*rM), L] = sum_l r(a,l,L) * x(l,col,rM)
        auto B = x.phys_const(col);                        // l x n_rM
        for (Eigen::Index Lidx = 0; Lidx < L; ++Lidx) {
            MatrixX tmp = r.right_const(Lidx) * B;         // A x n_rM (col-major)
            G.col(Lidx) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
                tmp.data(), tmp.size());                   // vec: a fastest
        }

        for (Eigen::Index row = 0; row < p_out; ++row) {
            // W = o(:, row*p_in + col, :) : L x n_RO
            MatrixX P = G * o.phys_const(row * p_in + col); // (A*n_rM) x n_RO

            // scatter P[(a + A*rM), RO] into M[(a + row*A), (rM + RO*n_rM)]
            for (Eigen::Index RO = 0; RO < n_RO; ++RO)
                for (Eigen::Index rM = 0; rM < n_rM; ++rM)
                    M.block(row * A, rM + RO * n_rM, A, 1)
                        += P.block(rM * A, RO, A, 1);
        }
    }
    return M;
}

// MPO @ MPS via the zip-up algorithm.
// Assumes BOTH the MPO and the MPS are left-canonical.
// Returns MPS cores with canonical center at the LAST site (n - 1):
// all cores except the last are left-canonical (they come out of the
// U factor of a truncated SVD), and the last core carries the weight.
//
// Boundary legs are carried through: the output left boundary is the
// product l0*L0 of the two input left boundaries (a dummy identity
// environment), the output right boundary is the
// product of the two right boundaries, with the MPS leg fastest.
// For the usual trivial boundaries all of these are 1.
template<typename T>
std::vector<Tensor3D<T>> zip_up_mpo_mps(
    std::vector<Tensor3D<T>> const& mpo,
    std::vector<Tensor3D<T>> const& mps,
    typename Eigen::NumTraits<T>::Real reltol = 1e-12,
    int max_bond_dim = 0)
{
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    const std::size_t n = mpo.size();
    if (n == 0 || mps.size() != n)
        throw std::invalid_argument(
            "utc::zip_up_mpo_mps: MPO and MPS must have the same (nonzero) length");

    std::vector<Tensor3D<T>> res;
    res.reserve(n);

    // Dummy environment r(a, l, L) = delta(a, l*L0 + L)
    // (Python: eye(l0*L0).reshape(-1, l0, L0))
    const Eigen::Index l0 = mps[0].n_left;
    const Eigen::Index L0 = mpo[0].n_left;
    Tensor3D<T> r(l0 * L0, l0, L0);   // zero-initialized
    for (Eigen::Index l = 0; l < l0; ++l)
        for (Eigen::Index L = 0; L < L0; ++L)
            r(l * L0 + L, l, L) = T(1);

    for (std::size_t i = 0; i + 1 < n; ++i) {
        const Eigen::Index A     = r.n_left;
        const Eigen::Index p_out = mpo[i].n_phys / mps[i].n_phys;
        const Eigen::Index n_rM  = mps[i].n_right;
        const Eigen::Index n_RO  = mpo[i].n_right;

        MatrixX M = _zip_up_site_matrix(r, mps[i], mpo[i]);

        // Truncated SVD: left() = U (orthonormal columns),
        // right() = diag(s) * V^H, truncation handled internally.
        SVDDecomp<T> svd(M, /*leftOrthogonal=*/true, reltol, max_bond_dim);
        const Eigen::Index chi = svd.s.size();

        MatrixX U  = svd.left();      // (A*p_out) x chi
        MatrixX R2 = svd.right();     // chi x (n_rM*n_RO)

        // New core: Tensor3D(A, p_out, chi) with flatten_as_matrix2 == U
        Tensor3D<T> core(A, p_out, chi);
        std::copy(U.data(), U.data() + U.size(), core.data.begin());
        res.push_back(std::move(core));

        // Next environment: Tensor3D(chi, n_rM, n_RO) with
        // flatten_as_matrix1 == R2 (column index = rM + RO*n_rM,
        // matching the column convention of M).
        r = Tensor3D<T>(chi, n_rM, n_RO);
        std::copy(R2.data(), R2.data() + R2.size(), r.data.begin());
    }

    // Last site: contract, no SVD — the whole matrix becomes the core.
    {
        const Eigen::Index A     = r.n_left;
        const Eigen::Index p_out = mpo[n - 1].n_phys / mps[n - 1].n_phys;
        const Eigen::Index n_rM  = mps[n - 1].n_right;
        const Eigen::Index n_RO  = mpo[n - 1].n_right;

        MatrixX M = _zip_up_site_matrix(r, mps[n - 1], mpo[n - 1]);

        Tensor3D<T> core(A, p_out, n_rM * n_RO);
        std::copy(M.data(), M.data() + M.size(), core.data.begin());
        res.push_back(std::move(core));
    }

    return res;
}

// MPO @ MPS via exact contraction with QR re-orthogonalization
// ("zip-up like" forward half-sweep, no truncation).
//
// Same left-to-right sweep as zip_up_mpo_mps, but each site matrix is
// factored with a reduced QR instead of a truncated SVD, so the
// contraction is EXACT: bond dimensions grow as chi_mps * chi_mpo.
// The "svd" half of the name happens downstream — the caller
// (MPO::_mul with method "qrsvd") compresses during the shift back to
// the original working site (shift_w(w0, compress=true)).
//
// Assumes both inputs are left-canonical. Returns MPS cores with
// canonical center at the LAST site: all cores except the last come
// out of a Q factor (left-canonical), the last carries the weight.
// Boundary conventions are identical to zip_up_mpo_mps.
template<typename T>
std::vector<Tensor3D<T>> qrsvd_contract_mpo_mps(
    std::vector<Tensor3D<T>> const& mpo,
    std::vector<Tensor3D<T>> const& mps)
{
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    const std::size_t n = mpo.size();
    if (n == 0 || mps.size() != n)
        throw std::invalid_argument(
            "utc::qrsvd_contract_mpo_mps: MPO and MPS must have the same (nonzero) length");

    std::vector<Tensor3D<T>> res;
    res.reserve(n);

    // Dummy environment r(a, l, L) = delta(a, l*L0 + L)
    const Eigen::Index l0 = mps[0].n_left;
    const Eigen::Index L0 = mpo[0].n_left;
    Tensor3D<T> r(l0 * L0, l0, L0);   // zero-initialized
    for (Eigen::Index l = 0; l < l0; ++l)
        for (Eigen::Index L = 0; L < L0; ++L)
            r(l * L0 + L, l, L) = T(1);

    for (std::size_t i = 0; i + 1 < n; ++i) {
        const Eigen::Index A     = r.n_left;
        const Eigen::Index p_out = mpo[i].n_phys / mps[i].n_phys;
        const Eigen::Index n_rM  = mps[i].n_right;
        const Eigen::Index n_RO  = mpo[i].n_right;

        MatrixX M = _zip_up_site_matrix(r, mps[i], mpo[i]);

        // Reduced QR: M = Q * R2, Q with orthonormal columns.
        auto qr = MatQR<T>{}(M, /*left=*/true);
        MatrixX& Q  = qr[0];      // (A*p_out) x k
        MatrixX& R2 = qr[1];      // k x (n_rM*n_RO)
        const Eigen::Index k = Q.cols();

        // New core: Tensor3D(A, p_out, k) with flatten_as_matrix2 == Q
        Tensor3D<T> core(A, p_out, k);
        std::copy(Q.data(), Q.data() + Q.size(), core.data.begin());
        res.push_back(std::move(core));

        // Next environment: Tensor3D(k, n_rM, n_RO) with
        // flatten_as_matrix1 == R2 (column index = rM + RO*n_rM).
        r = Tensor3D<T>(k, n_rM, n_RO);
        std::copy(R2.data(), R2.data() + R2.size(), r.data.begin());
    }

    // Last site: contract, no factorization.
    {
        const Eigen::Index A     = r.n_left;
        const Eigen::Index p_out = mpo[n - 1].n_phys / mps[n - 1].n_phys;
        const Eigen::Index n_rM  = mps[n - 1].n_right;
        const Eigen::Index n_RO  = mpo[n - 1].n_right;

        MatrixX M = _zip_up_site_matrix(r, mps[n - 1], mpo[n - 1]);

        Tensor3D<T> core(A, p_out, n_rM * n_RO);
        std::copy(M.data(), M.data() + M.size(), core.data.begin());
        res.push_back(std::move(core));
    }

    return res;
}

// MPO @ MPS via variational (DMRG-style) optimization, seeded with
// `previous` as the initial guess.
// Returns MPS cores with canonical center at site 0.
template<typename T>
std::vector<Tensor3D<T>> optimize_dm_generic(
    std::vector<Tensor3D<T>> const& /*previous*/,
    std::vector<Tensor3D<T>> const& /*mpo*/,
    std::vector<Tensor3D<T>> const& /*mps*/,
    typename Eigen::NumTraits<T>::Real /*reltol*/,
    int /*max_bond_dim*/,
    int /*order*/,
    int /*n_sweeps*/)
{
    throw std::logic_error("TODO: utc::optimize_dm_generic not implemented");
}

// MPO @ MPO via the zip-up algorithm.
// Returns MPO cores with canonical center at the LAST site (r - 1).
// template<typename T>
// std::vector<Tensor3D<T>> zip_up_mpo_mpo(
//     std::vector<Tensor3D<T>> const& /*mpo_a*/,
//     std::vector<Tensor3D<T>> const& /*mpo_b*/,
//     typename Eigen::NumTraits<T>::Real /*reltol*/,
//     int /*max_bond_dim*/)
// {
//     throw std::logic_error("TODO: utc::zip_up_mpo_mpo not implemented");
// }

} // namespace utc