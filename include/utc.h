#pragma once

#include "tensor.h"
#include "mat_decomp.h"   // SVDDecomp

#include <algorithm>
#include <stdexcept>
#include <vector>

// =====================================================================
// utc: tensor-train contraction kernels (Python `utc` module).
//
// Implemented: zip_up_mpo_mps, qrsvd_contract_mpo_mps,
//              optimize_dm_generic (with 1-site and 2-site sweeps).
// TODO:        zip_up_mpo_mpo (stub, throws at runtime).
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
// This mirrors the Python's two consecutive tensordot calls, each of
// which maps to a single BLAS GEMM.  We batch the small multiplies
// into a handful of large matrix products and use block operations
// for the final scatter, completely eliminating the previous
// quintuple-nested per-element loops.
//
// Index conventions are chosen to line up with Tensor3D's column-major
// layout: the row index (a fastest, row slower) is exactly
// flatten_as_matrix2 of a Tensor3D(A, p_out, *), and the column index
// (rM fastest, RO slower) is exactly flatten_as_matrix1 of a
// Tensor3D(*, n_rM, n_RO). That lets the SVD factors be copied
// straight into the output core / next environment with no reshuffle.
// ---------------------------------------------------------------------
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
_zip_up_site_matrix(Tensor3D<T> const& r,
                    Tensor3D<T> const& x,
                    Tensor3D<T> const& o)
{
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    const Eigen::Index A     = r.n_left;
    const Eigen::Index l_dim = r.n_phys;      // MPS left bond (same as x.n_left)
    const Eigen::Index L     = r.n_right;     // MPO left bond (same as o.n_left)
    const Eigen::Index p_in  = x.n_phys;      // MPS physical dim (2 for qubits)
    const Eigen::Index p_out = o.n_phys / p_in;
    const Eigen::Index n_rM  = x.n_right;
    const Eigen::Index n_RO  = o.n_right;

    if (o.n_phys % p_in != 0)
        throw std::invalid_argument(
            "utc::zip_up_mpo_mps: MPO physical dimension is not a multiple of the MPS one");
    if (r.n_phys != x.n_left || r.n_right != o.n_left)
        throw std::invalid_argument(
            "utc::zip_up_mpo_mps: bond dimension mismatch between environment and cores");

    // ================================================================
    // Step 1:  r(A,l,L) * x(l,col,rM)  →  X_mat  (batched over col)
    //
    // Stack r.right_const(Lidx) vertically into  Rstack(A*L, l),
    // then multiply by the MPS slice for each col (single GEMM each).
    // X_mat is (A*L*p_in, n_rM)  where the row for (col,L_idx,a) is
    //   a + L_idx*A + col*A*L.
    // ================================================================
    MatrixX Rstack(A * L, l_dim);
    for (Eigen::Index Lidx = 0; Lidx < L; ++Lidx)
        Rstack.middleRows(Lidx * A, A) = r.right_const(Lidx);

    MatrixX X_mat(A * L * p_in, n_rM);
    for (Eigen::Index col = 0; col < p_in; ++col)
        X_mat.middleRows(col * A * L, A * L).noalias()
            = Rstack * x.phys_const(col);

    // ================================================================
    // Step 2:  build  MpoMat(L*p_in, p_out*n_RO)  from the MPO core
    //
    // Element  MpoMat(Lidx + col*L, row + RO*p_out)
    //        = o(Lidx, row*p_in + col, RO)
    // ================================================================
    MatrixX MpoMat(L * p_in, p_out * n_RO);
    for (Eigen::Index col = 0; col < p_in; ++col) {
        for (Eigen::Index row = 0; row < p_out; ++row) {
            auto o_slice = o.phys_const(row * p_in + col);   // (L, n_RO)
            for (Eigen::Index RO = 0; RO < n_RO; ++RO)
                MpoMat.col(row + RO * p_out).segment(col * L, L)
                    = o_slice.col(RO);
        }
    }

    // ================================================================
    // Step 3:  contract over (L, col)  and reshape to final M
    //
    // For each rM we extract an (A, L*p_in) slice from X_mat,
    // multiply by MpoMat → (A, p_out*n_RO), then scatter into M.
    // ================================================================
    MatrixX M = MatrixX::Zero(A * p_out, n_rM * n_RO);
    for (Eigen::Index rM = 0; rM < n_rM; ++rM) {
        // Extract X_rM(A, L*p_in) from X_mat column rM.
        // X_mat(col*A*L + Lidx*A + a, rM)  →  X_rM(a, Lidx + col*L)
        MatrixX X_rM(A, L * p_in);
        for (Eigen::Index col = 0; col < p_in; ++col)
            for (Eigen::Index Lidx = 0; Lidx < L; ++Lidx)
                X_rM.col(Lidx + col * L)
                    = X_mat.col(rM).segment(Lidx * A + col * A * L, A);

        // Single GEMM  (A, L*p_in) × (L*p_in, p_out*n_RO) = (A, p_out*n_RO)
        MatrixX M_rM(A, p_out * n_RO);
        M_rM.noalias() = X_rM * MpoMat;

        // Scatter M_rM(a, row + RO*p_out)  into  M(a + row*A, rM + RO*n_rM)
        for (Eigen::Index RO = 0; RO < n_RO; ++RO)
            for (Eigen::Index row = 0; row < p_out; ++row)
                M.col(rM + RO * n_rM).segment(row * A, A)
                    += M_rM.col(row + RO * p_out);
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
// environment, as in the Python), the output right boundary is the
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

// =====================================================================
//        MPO @ MPS via variational (DMRG-like) fitting sweeps
// =====================================================================
//
// Port of the Python optimize / optimize_2_sites / optimize_dm_generic
// family. Everything below reduces to two "half contraction" kernels:
//
//   half_left(L, B, W)  = Z1M[(rm + nrm*rk), (lz + nzl*pz)]
//                       = sum_{lm,lk,pm} L(lm,lk,lz) B(lm,pm,rm) W4(lk,pz,pm,rk)
//   half_right(R, B, W) = MR[(lm + nlm*lk), (pz + Pz*zr)]
//                       = sum_{rm,rk,pm} R(rm,rk,zr) B(lm,pm,rm) W4(lk,pz,pm,rk)
//
// where B is an MPS core, W an MPO core with fused physical slot
// p = pz * Pm + pm (output index pz, input index pm), W4 its unfolded
// form, and L / R are 3-leg environments stored as Tensor3D(m, k, z).
// The packed index conventions are chosen so that every result maps
// byte-for-byte onto Tensor3D's flatten_as_matrix1/2 layouts (which
// share the same underlying memory order), so cores and environments
// are plain std::copy's of matrix products. Then:
//
//   env_right_step : R_new.flat2 = half_right(R,B,W) * conj(C).flat1^T
//   env_left_step  : L_new.flat2 = half_left(L,B,W)  * conj(C).flat2
//   local 1-site   : T.flat2     = half_left(L,B,W)^T * R.flat2
//   local 2-site   : M2          = half_left(L,B0,W0)^T * half_right(R,B1,W1)
//
// The whole scheme (index packings, conjugation placement, sweep
// logic, boundary augmentation) was verified numerically against
// dense MPO @ MPS and against the reference Python on random complex
// tensor trains, including nontrivial boundary legs on either side.
//
// The environments carry conj(z-side core) on their z leg; the local
// tensor T is then directly the optimal (un-conjugated) core. The
// Python's rr / s*vd absorptions into zd during the sweeps are dead
// code (each local tensor is regenerated in full from the
// environments) and are omitted here.

namespace detail {

template<typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template<typename T>
Tensor3D<T> conj_tensor(Tensor3D<T> const& x)
{
    Tensor3D<T> out(x.n_left, x.n_phys, x.n_right);
    for (std::size_t i = 0; i < x.data.size(); ++i)
        out.data[i] = Eigen::numext::conj(x.data[i]);
    return out;
}

template<typename T>
Tensor3D<T> trivial_env()
{
    Tensor3D<T> e(1, 1, 1);
    e.data[0] = T(1);
    return e;
}

template<typename T>
Mat<T> half_left(Tensor3D<T> const& L, Tensor3D<T> const& B,
                 Tensor3D<T> const& W, Eigen::Index Pz)
{
    const Eigen::Index nlm = L.n_left, nlk = L.n_phys, nzl = L.n_right;
    const Eigen::Index Pm = B.n_phys, nrm = B.n_right, nrk = W.n_right;
    if (B.n_left != nlm || W.n_left != nlk || W.n_phys != Pz * Pm)
        throw std::invalid_argument("utc::optimize: dimension mismatch in half_left");

    Mat<T> Z1M = Mat<T>::Zero(nrm * nrk, nzl * Pz);
    auto Lf1 = L.flatten_as_matrix1_const();               // nlm x (nlk*nzl)

    // Work buffer reused across (pm,pz) iterations
    Mat<T> Y_reorder(nrm * nzl, nlk);

    for (Eigen::Index pm = 0; pm < Pm; ++pm) {
        // Y = B_phys^T * Lf1   (nrm, nlk*nzl) — single GEMM
        Mat<T> Y = B.phys_const(pm).transpose() * Lf1;

        for (Eigen::Index pz = 0; pz < Pz; ++pz) {
            auto Wsl = W.phys_const(pz * Pm + pm);         // nlk x nrk

            // Reshape Y from (nrm, nlk*nzl) to (nrm*nzl, nlk)
            // so we can contract all lz at once with a single GEMM.
            // Layout: Y(rm, lz*nlk + lk) → Y_reorder(rm + lz*nrm, lk)
            for (Eigen::Index lz = 0; lz < nzl; ++lz)
                for (Eigen::Index lk = 0; lk < nlk; ++lk)
                    Y_reorder.col(lk).segment(lz * nrm, nrm)
                        = Y.col(lz * nlk + lk);

            // Single GEMM: (nrm*nzl, nlk) × (nlk, nrk) = (nrm*nzl, nrk)
            Mat<T> YW(nrm * nzl, nrk);
            YW.noalias() = Y_reorder * Wsl;

            // Scatter: YW(rm + lz*nrm, rk) → Z1M(rm + rk*nrm, lz + pz*nzl)
            for (Eigen::Index rk = 0; rk < nrk; ++rk) {
                Eigen::Map<const Mat<T>> YW_slice(
                    YW.col(rk).data(), nrm, nzl);          // (nrm, nzl) col-major
                Z1M.block(rk * nrm, pz * nzl, nrm, nzl) += YW_slice;
            }
        }
    }
    return Z1M;
}

template<typename T>
Mat<T> half_right(Tensor3D<T> const& R, Tensor3D<T> const& B,
                  Tensor3D<T> const& W, Eigen::Index Pz)
{
    const Eigen::Index nrm = R.n_left, nrk = R.n_phys, nzr = R.n_right;
    const Eigen::Index Pm = B.n_phys, nlm = B.n_left, nlk = W.n_left;
    if (B.n_right != nrm || W.n_right != nrk || W.n_phys != Pz * Pm)
        throw std::invalid_argument("utc::optimize: dimension mismatch in half_right");

    Mat<T> MR = Mat<T>::Zero(nlm * nlk, Pz * nzr);
    auto Rf1 = R.flatten_as_matrix1_const();               // nrm x (nrk*nzr)

    // Work buffer reused across (pm,pz) iterations
    Mat<T> Y_reorder(nlm * nzr, nrk);

    for (Eigen::Index pm = 0; pm < Pm; ++pm) {
        // Y = B_phys * Rf1   (nlm, nrk*nzr) — single GEMM
        Mat<T> Y = B.phys_const(pm) * Rf1;

        for (Eigen::Index pz = 0; pz < Pz; ++pz) {
            auto Wsl = W.phys_const(pz * Pm + pm);         // nlk x nrk

            // Reshape Y from (nlm, nrk*nzr) to (nlm*nzr, nrk)
            // Layout: Y(lm, zr*nrk + rk) → Y_reorder(lm + zr*nlm, rk)
            for (Eigen::Index zr = 0; zr < nzr; ++zr)
                for (Eigen::Index rk = 0; rk < nrk; ++rk)
                    Y_reorder.col(rk).segment(zr * nlm, nlm)
                        = Y.col(zr * nrk + rk);

            // Single GEMM: (nlm*nzr, nrk) × (nrk, nlk) = (nlm*nzr, nlk)
            // (Wsl^T is (nrk, nlk))
            Mat<T> YW(nlm * nzr, nlk);
            YW.noalias() = Y_reorder * Wsl.transpose();

            // Scatter: YW(lm + zr*nlm, lk) → MR(lm + lk*nlm, pz + Pz*zr)
            // Each (zr,lk) pair contributes a contiguous segment of length nlm.
            for (Eigen::Index zr = 0; zr < nzr; ++zr)
                for (Eigen::Index lk = 0; lk < nlk; ++lk)
                    MR.col(pz + Pz * zr).segment(lk * nlm, nlm)
                        += YW.col(lk).segment(zr * nlm, nlm);
        }
    }
    return MR;
}

// R_new(lm, lk, lz) = sum R(rm,rk,zr) B(lm,pm,rm) W4(lk,pz,pm,rk) Cbar(lz,pz,zr)
// Cbar must already be conjugated by the caller.
template<typename T>
Tensor3D<T> env_right_step(Tensor3D<T> const& B, Tensor3D<T> const& W,
                           Tensor3D<T> const& Cbar, Tensor3D<T> const& R,
                           Eigen::Index Pz)
{
    Mat<T> out = half_right(R, B, W, Pz)
               * Cbar.flatten_as_matrix1_const().transpose();  // (nlm*nlk) x nlz
    Tensor3D<T> env(B.n_left, W.n_left, Cbar.n_left);
    std::copy(out.data(), out.data() + out.size(), env.data.begin());
    return env;
}

// L_new(rm, rk, rz) = sum L(lm,lk,lz) B(lm,pm,rm) W4(lk,pz,pm,rk) Zbar(lz,pz,rz)
template<typename T>
Tensor3D<T> env_left_step(Tensor3D<T> const& B, Tensor3D<T> const& W,
                          Tensor3D<T> const& Zbar, Tensor3D<T> const& L,
                          Eigen::Index Pz)
{
    Mat<T> out = half_left(L, B, W, Pz)
               * Zbar.flatten_as_matrix2_const();               // (nrm*nrk) x nrz
    Tensor3D<T> env(B.n_right, W.n_right, Zbar.n_right);
    std::copy(out.data(), out.data() + out.size(), env.data.begin());
    return env;
}

// T(lz, pz, rz) = sum L(lm,lk,lz) B(lm,pm,rm) W4(lk,pz,pm,rk) R(rm,rk,rz)
template<typename T>
Tensor3D<T> local_1site(Tensor3D<T> const& L, Tensor3D<T> const& B,
                        Tensor3D<T> const& W, Tensor3D<T> const& R,
                        Eigen::Index Pz)
{
    Mat<T> Tm = half_left(L, B, W, Pz).transpose()
              * R.flatten_as_matrix2_const();                   // (nzl*Pz) x nzr
    Tensor3D<T> out(L.n_right, Pz, R.n_right);
    std::copy(Tm.data(), Tm.data() + Tm.size(), out.data.begin());
    return out;
}

// M2[(lz + nzl*pz0), (pz1 + Pz1*rz)]: the two-site local matrix.
template<typename T>
Mat<T> local_2site(Tensor3D<T> const& L,
                   Tensor3D<T> const& B0, Tensor3D<T> const& W0,
                   Tensor3D<T> const& B1, Tensor3D<T> const& W1,
                   Tensor3D<T> const& R,
                   Eigen::Index Pz0, Eigen::Index Pz1)
{
    return half_left(L, B0, W0, Pz0).transpose()
         * half_right(R, B1, W1, Pz1);
}

template<typename T>
std::vector<Eigen::Index> phys_out_dims(std::vector<Tensor3D<T>> const& k,
                                        std::vector<Tensor3D<T>> const& m)
{
    std::vector<Eigen::Index> Pz(m.size());
    for (std::size_t i = 0; i < m.size(); ++i) {
        if (k[i].n_phys % m[i].n_phys != 0)
            throw std::invalid_argument(
                "utc::optimize: MPO physical dimension is not a multiple of the MPS one");
        Pz[i] = k[i].n_phys / m[i].n_phys;
    }
    return Pz;
}

// One full 1-site fitting sweep (forward QR pass building left
// environments, then backward LQ pass producing the cores).
// Requires trivial (dim 1) boundary legs on all three trains.
// Output: right-canonical with center at site 0.
template<typename T>
std::vector<Tensor3D<T>> optimize_1site_pass(std::vector<Tensor3D<T>> const& z,
                                             std::vector<Tensor3D<T>> const& k,
                                             std::vector<Tensor3D<T>> const& m)
{
    using MatrixX = Mat<T>;
    const int r = static_cast<int>(m.size());
    const std::vector<Eigen::Index> Pz = phys_out_dims(k, m);

    // right environments from the conjugated guess: envR[i] covers i..r-1
    std::vector<Tensor3D<T>> envR(r + 1, trivial_env<T>());
    for (int i = r - 1; i > 0; --i)
        envR[i] = env_right_step(m[i], k[i], conj_tensor(z[i]), envR[i + 1], Pz[i]);

    // forward pass: only the left environments survive
    std::vector<Tensor3D<T>> envL;
    envL.reserve(r);
    envL.push_back(trivial_env<T>());
    for (int w = 0; w + 1 < r; ++w) {
        Tensor3D<T> Tw = local_1site(envL[w], m[w], k[w], envR[w + 1], Pz[w]);
        auto qr = MatQR<T>{}(Tw.flatten_as_matrix2(), /*left=*/true);
        MatrixX& Q = qr[0];                            // (nzl*Pz) x q
        Tensor3D<T> core(Tw.n_left, Pz[w], Q.cols());
        std::copy(Q.data(), Q.data() + Q.size(), core.data.begin());
        envL.push_back(env_left_step(m[w], k[w], conj_tensor(core), envL[w], Pz[w]));
    }

    // backward pass: LQ at each site, weight ends up at site 0
    std::vector<Tensor3D<T>> out(r, trivial_env<T>());  // placeholder-initialized
    Tensor3D<T> curR = trivial_env<T>();
    for (int w = r - 1; w > 0; --w) {
        Tensor3D<T> Tw = local_1site(envL[w], m[w], k[w], curR, Pz[w]);
        auto lq = MatQR<T>{}(Tw.flatten_as_matrix1(), /*left=*/false);
        MatrixX& Q = lq[1];                            // q x (Pz*nzr), orthonormal rows
        Tensor3D<T> core(Q.rows(), Pz[w], Tw.n_right);
        std::copy(Q.data(), Q.data() + Q.size(), core.data.begin());
        curR = env_right_step(m[w], k[w], conj_tensor(core), curR, Pz[w]);
        out[w] = std::move(core);
    }
    out[0] = local_1site(envL[0], m[0], k[0], curR, Pz[0]);
    return out;
}

// One full 2-site fitting sweep with SVD truncation (bond dimensions
// adapt up to reltol / max_bond_dim). Same boundary requirements and
// output canonical form as the 1-site pass.
template<typename T>
std::vector<Tensor3D<T>> optimize_2site_pass(std::vector<Tensor3D<T>> const& z,
                                             std::vector<Tensor3D<T>> const& k,
                                             std::vector<Tensor3D<T>> const& m,
                                             typename Eigen::NumTraits<T>::Real reltol,
                                             int max_bond_dim)
{
    using MatrixX = Mat<T>;
    const int r = static_cast<int>(m.size());
    if (r == 1)
        return optimize_1site_pass(z, k, m);
    const std::vector<Eigen::Index> Pz = phys_out_dims(k, m);

    std::vector<Tensor3D<T>> envR(r + 1, trivial_env<T>());
    for (int i = r - 1; i > 1; --i)
        envR[i] = env_right_step(m[i], k[i], conj_tensor(z[i]), envR[i + 1], Pz[i]);

    std::vector<Tensor3D<T>> envL;
    envL.reserve(r - 1);
    envL.push_back(trivial_env<T>());
    for (int w = 0; w + 2 < r; ++w) {
        MatrixX M2 = local_2site(envL[w], m[w], k[w], m[w + 1], k[w + 1],
                                 envR[w + 2], Pz[w], Pz[w + 1]);
        SVDDecomp<T> svd(M2, /*leftOrthogonal=*/true, reltol, max_bond_dim);
        MatrixX U = svd.left();                        // (nzl*Pz0) x chi
        Tensor3D<T> core(envL[w].n_right, Pz[w], U.cols());
        std::copy(U.data(), U.data() + U.size(), core.data.begin());
        envL.push_back(env_left_step(m[w], k[w], conj_tensor(core), envL[w], Pz[w]));
    }

    std::vector<Tensor3D<T>> out(r, trivial_env<T>());
    Tensor3D<T> curR = trivial_env<T>();
    for (int w = r - 2; w >= 0; --w) {
        MatrixX M2 = local_2site(envL[w], m[w], k[w], m[w + 1], k[w + 1],
                                 curR, Pz[w], Pz[w + 1]);
        // leftOrthogonal=false: left() = U*S, right() = V^H
        SVDDecomp<T> svd(M2, /*leftOrthogonal=*/false, reltol, max_bond_dim);
        const Eigen::Index chi = svd.s.size();

        MatrixX VH = svd.right();                      // chi x (Pz1*nzr)
        Tensor3D<T> core1(chi, Pz[w + 1], curR.n_right);
        std::copy(VH.data(), VH.data() + VH.size(), core1.data.begin());

        if (w > 0) {
            curR = env_right_step(m[w + 1], k[w + 1], conj_tensor(core1),
                                  curR, Pz[w + 1]);
            out[w + 1] = std::move(core1);
        } else {
            out[1] = std::move(core1);
            MatrixX US = svd.left();                   // (nzl*Pz0) x chi
            Tensor3D<T> core0(envL[0].n_right, Pz[0], chi);
            std::copy(US.data(), US.data() + US.size(), core0.data.begin());
            out[0] = std::move(core0);
        }
    }
    return out;
}

} // namespace detail

// MPO @ MPS via variational (DMRG-style) fitting, seeded with
// `previous` as the initial guess (typically the MPS itself).
//
//   order == 1 : 1-site sweeps — bond dimensions are frozen at those
//                of the guess; QR/LQ only, no truncation inside.
//   order != 1 : 2-site sweeps — bond dimensions adapt via truncated
//                SVDs controlled by reltol / max_bond_dim.
//
// All three trains are assumed LEFT-canonical on entry (the guess is
// used to build the initial right environments). Returns MPS cores
// with canonical center at site 0: cores 1..r-1 are right-canonical.
//
// Boundary legs: the inner sweeps require trivial boundaries, so as
// in the Python this wrapper augments the trains with one auxiliary
// site when boundary legs are nontrivial — on the left when all right
// boundaries are 1, on the right when all left boundaries are 1 —
// runs the sweeps, and absorbs the auxiliary site back (with an SVD
// truncation of the boundary leg controlled by max_mu; max_mu = 0
// keeps everything above reltol). The output boundary leg is the
// product of the two input boundary legs, MPO index fastest.
// NOTE: for the right-boundary case the Python keeps only U*S of the
// boundary SVD (returning the leg rotated into the SVD basis and
// dropping V^H); here BOTH branches keep the full truncated U*S*V^H,
// so the boundary stays in the natural product basis — verified
// against dense contraction.
template<typename T>
std::vector<Tensor3D<T>> optimize_dm_generic(
    std::vector<Tensor3D<T>> const& previous,
    std::vector<Tensor3D<T>> const& mpo,
    std::vector<Tensor3D<T>> const& mps,
    typename Eigen::NumTraits<T>::Real reltol,
    int max_bond_dim,
    int order,
    int n_sweeps,
    int max_mu = 0)
{
    using MatrixX = detail::Mat<T>;

    const std::size_t n = mpo.size();
    if (n == 0 || mps.size() != n || previous.size() != n)
        throw std::invalid_argument(
            "utc::optimize_dm_generic: trains must have the same (nonzero) length");

    auto run_sweeps = [&](std::vector<Tensor3D<T>> z2,
                          std::vector<Tensor3D<T>> const& k2,
                          std::vector<Tensor3D<T>> const& m2)
    {
        for (int s = 0; s < n_sweeps; ++s)
            z2 = (order == 1)
                 ? detail::optimize_1site_pass(z2, k2, m2)
                 : detail::optimize_2site_pass(z2, k2, m2, reltol, max_bond_dim);
        return z2;
    };

    auto const& z = previous;
    auto const& k = mpo;
    auto const& m = mps;

    const Eigen::Index mu_lm = m.front().n_left, mu_rm = m.back().n_right;
    const Eigen::Index mu_lk = k.front().n_left, mu_rk = k.back().n_right;
    const Eigen::Index mu_lz = z.front().n_left, mu_rz = z.back().n_right;

    // ------ extra bond on the left (all right boundaries trivial) ------
    if (mu_rm == 1 && mu_rk == 1 && mu_rz == 1) {
        const Eigen::Index N = mu_lm * mu_lk;

        // z2 = [ identity(1,N,N), padded z[0], z[1:] ]
        Tensor3D<T> E(1, N, N);
        for (Eigen::Index a = 0; a < N; ++a) E(0, a, a) = T(1);

        Tensor3D<T> first(N, z.front().n_phys, z.front().n_right);  // zeros
        const Eigen::Index c = std::min(mu_lz, mu_lm);
        for (Eigen::Index a = 0; a < c; ++a)
            for (Eigen::Index p = 0; p < first.n_phys; ++p)
                for (Eigen::Index rr = 0; rr < first.n_right; ++rr)
                    first(a, p, rr) = z.front()(a, p, rr);

        std::vector<Tensor3D<T>> z2;
        z2.reserve(n + 1);
        z2.push_back(std::move(E));
        z2.push_back(std::move(first));
        for (std::size_t i = 1; i < n; ++i) z2.push_back(z[i]);

        // k2 = [ copy tensor, k ]: W(0, (i*mu_lk + j)*mu_lm + i', j') = d(i,i') d(j,j')
        Tensor3D<T> Wcopy(1, N * mu_lm, mu_lk);
        for (Eigen::Index i = 0; i < mu_lm; ++i)
            for (Eigen::Index j = 0; j < mu_lk; ++j)
                Wcopy(0, (i * mu_lk + j) * mu_lm + i, j) = T(1);

        std::vector<Tensor3D<T>> k2;
        k2.reserve(n + 1);
        k2.push_back(std::move(Wcopy));
        for (auto const& x : k) k2.push_back(x);

        // m2 = [ identity(1, mu_lm, mu_lm), m ]
        Tensor3D<T> Em(1, mu_lm, mu_lm);
        for (Eigen::Index a = 0; a < mu_lm; ++a) Em(0, a, a) = T(1);

        std::vector<Tensor3D<T>> m2;
        m2.reserve(n + 1);
        m2.push_back(std::move(Em));
        for (auto const& x : m) m2.push_back(x);

        std::vector<Tensor3D<T>> res = run_sweeps(std::move(z2), k2, m2);

        // absorb the auxiliary site: res[0] is (1, N, chi0) == N x chi0
        MatrixX A = Eigen::Map<const MatrixX>(res[0].data.data(), N, res[0].n_right);
        SVDDecomp<T> svd(A, /*leftOrthogonal=*/true, reltol, max_mu);
        MatrixX A2 = svd.left() * svd.right();                      // N x chi0

        MatrixX nf = A2 * res[1].flatten_as_matrix1();              // N x (P1*r1)
        Tensor3D<T> newfirst(N, res[1].n_phys, res[1].n_right);
        std::copy(nf.data(), nf.data() + nf.size(), newfirst.data.begin());

        std::vector<Tensor3D<T>> out;
        out.reserve(n);
        out.push_back(std::move(newfirst));
        for (std::size_t i = 2; i < res.size(); ++i)
            out.push_back(std::move(res[i]));
        return out;
    }

    // ------ extra bond on the right (all left boundaries trivial) ------
    if (mu_lm == 1 && mu_lk == 1 && mu_lz == 1) {
        const Eigen::Index N = mu_rm * mu_rk;

        // z2 = [ z[:-1], padded z[-1] (ones background, as in the
        //        Python), identity(N,N,1) ]
        Tensor3D<T> last(z.back().n_left, z.back().n_phys, N);
        std::fill(last.data.begin(), last.data.end(), T(1));
        const Eigen::Index c = std::min(mu_rz, mu_rm);
        for (Eigen::Index l = 0; l < last.n_left; ++l)
            for (Eigen::Index p = 0; p < last.n_phys; ++p)
                for (Eigen::Index rr = 0; rr < c; ++rr)
                    last(l, p, rr) = z.back()(l, p, rr);

        Tensor3D<T> E(N, N, 1);
        for (Eigen::Index a = 0; a < N; ++a) E(a, a, 0) = T(1);

        std::vector<Tensor3D<T>> z2;
        z2.reserve(n + 1);
        for (std::size_t i = 0; i + 1 < n; ++i) z2.push_back(z[i]);
        z2.push_back(std::move(last));
        z2.push_back(std::move(E));

        // k2 = [ k, copy tensor ]: W(j, (i*mu_rk + j)*mu_rm + i, 0) = 1
        Tensor3D<T> Wcopy(mu_rk, N * mu_rm, 1);
        for (Eigen::Index i = 0; i < mu_rm; ++i)
            for (Eigen::Index j = 0; j < mu_rk; ++j)
                Wcopy(j, (i * mu_rk + j) * mu_rm + i, 0) = T(1);

        std::vector<Tensor3D<T>> k2;
        k2.reserve(n + 1);
        for (auto const& x : k) k2.push_back(x);
        k2.push_back(std::move(Wcopy));

        Tensor3D<T> Em(mu_rm, mu_rm, 1);
        for (Eigen::Index a = 0; a < mu_rm; ++a) Em(a, a, 0) = T(1);

        std::vector<Tensor3D<T>> m2;
        m2.reserve(n + 1);
        for (auto const& x : m) m2.push_back(x);
        m2.push_back(std::move(Em));

        std::vector<Tensor3D<T>> res = run_sweeps(std::move(z2), k2, m2);

        // absorb the auxiliary site: res[-1] is (chi, N, 1) == chi x N
        auto const& lastres = res.back();
        MatrixX A = Eigen::Map<const MatrixX>(lastres.data.data(), lastres.n_left, N);
        SVDDecomp<T> svd(A, /*leftOrthogonal=*/true, reltol, max_mu);
        MatrixX A2 = svd.left() * svd.right();                      // chi x N

        auto const& prev = res[res.size() - 2];
        MatrixX nl = prev.flatten_as_matrix2_const() * A2;          // (l*p) x N
        Tensor3D<T> newlast(prev.n_left, prev.n_phys, N);
        std::copy(nl.data(), nl.data() + nl.size(), newlast.data.begin());

        std::vector<Tensor3D<T>> out;
        out.reserve(n);
        for (std::size_t i = 0; i + 2 < res.size(); ++i)
            out.push_back(std::move(res[i]));
        out.push_back(std::move(newlast));
        return out;
    }

    throw std::invalid_argument(
        "utc::optimize_dm_generic: not implemented when both boundary sides are nontrivial");
}

// MPO @ MPO via the zip-up algorithm.
// Returns MPO cores with canonical center at the LAST site (r - 1).
template<typename T>
std::vector<Tensor3D<T>> zip_up_mpo_mpo(
    std::vector<Tensor3D<T>> const& /*mpo_a*/,
    std::vector<Tensor3D<T>> const& /*mpo_b*/,
    typename Eigen::NumTraits<T>::Real /*reltol*/,
    int /*max_bond_dim*/)
{
    throw std::logic_error("TODO: utc::zip_up_mpo_mpo not implemented");
}

} // namespace utc