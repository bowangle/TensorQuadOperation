#pragma once

#include "tt_base.h"
#include <type_int128.h>

// Helper declarations for the convolution implementation below.
template<typename T> class MPS;
template<typename T> class MPO;
template<typename T> MPS<T> magic_tensor_product(const MPS<T>& mps);
template<typename T> MPO<T> reshape_magic_res(const MPS<T>& result);

namespace analytic_mps_construction {
template<typename T> MPS<T> make_basis_mps(const std::vector<int>& bits);
}

// Matrix Product State: a TT whose physical leg is the fused
// (row x column) operator index — n_phys = 2 for qubits, i.e. cores
// of shape (chi, 2, chi'), with slot ordering
//   0 : (0,0)   1 : (0,1)   2 : (1,0)   3 : (1,1)

template<typename T>
class MPS : public TT<T> {
    using Base       = TT<T>;
    using VecTT      = std::vector<Tensor3D<T>>;
    using MatrixX    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

public:
    using Base::Base;                       // inherit both TT constructors

    // Allow promoting a TT (e.g. the result of mps1 + mps2, which
    // returns TT) back into an MPS.
    MPS(Base base) : Base(std::move(base)) {}

    // =============================================================
    //            MPS-MPS contraction  (Python __matmul__)
    // =============================================================
    // dot(other) computes  <this|other> :
    // this's cores are conjugated (matching jnp.conjugate(mps_a[k])
    // in the Python), other's are not.
    //
    // Instead of the Python's growing tensordot chain, we carry a
    // chi_a x chi_b "environment" matrix E across the train:
    //
    //   E_0(ra, rb)  = sum_{l,p} conj(A_0(l,p,ra)) B_0(l,p,rb)
    //   E_k          = sum_p  A_k.phys(p)^H * E_{k-1} * B_k.phys(p)
    //
    // which is the same contraction in a different order, at
    // O(chi^3) per site.
private:
    // Full environment: returns the (ra_last x rb_last) matrix.
    // With trivial boundaries this is 1x1; if you later add extra
    // legs on the right boundary, this is the object you want.
    MatrixX contract_env(MPS const& other) const
    {
        auto const& a = this->core;
        auto const& b = other.core;

        if (a.size() != b.size())
            throw std::invalid_argument("MPS::contract_env: trying to contract MPSs of different lengths");
        if (a[0].n_left != 1 || b[0].n_left != 1)
            throw std::invalid_argument("MPS::contract_env: non-trivial left boundary legs");

        // First site: E = A^H * B over the merged (left*phys) index.
        if (a[0].n_phys != b[0].n_phys)
            throw std::invalid_argument("MPS::contract_env: physical dimensions differ at site 0");
        MatrixX E = a[0].flatten_as_matrix2_const().adjoint()
                  * b[0].flatten_as_matrix2_const();          // (ra_0 x rb_0)

        // Sweep: E <- sum_p A_k.phys(p)^H * E * B_k.phys(p)
        for (std::size_t k = 1; k < a.size(); k++) {
            if (a[k].n_phys != b[k].n_phys)
                throw std::invalid_argument("MPS::contract_env: physical dimensions differ");

            MatrixX Enew = MatrixX::Zero(a[k].n_right, b[k].n_right);
            for (Eigen::Index p = 0; p < a[k].n_phys; p++)
                Enew.noalias() += a[k].phys_const(p).adjoint()
                                * (E * b[k].phys_const(p));
            E = std::move(Enew);
        }
        return E;
    }

public:
    // MPS MPS dot product. Connect all the leg and contract
    T dot(MPS const& other) const
    {
        MatrixX E = contract_env(other);
        if (E.rows() != 1 || E.cols() != 1)
            throw std::invalid_argument("MPS::dot: non-trivial right boundary legs, use contract_env instead");
        return E(0, 0);
    }

    // Convenience: norm^2 = <this|this> (real and >= 0 up to rounding)
    RealScalar norm2() const
    {
        return Eigen::numext::real(dot(*this));
    }

    /// Causal discrete convolution: result[m] = sum_{j=0}^m
    /// (*this)[j] * kernel[m-j], without conjugation or periodic wraparound.
    /// Both MPSs must have the same nonzero number of binary, LSB-first cores.
    /// The output retains that bit count (overflow is discarded).
    /// No grid spacing or quadrature weights are applied; neither input changes.
    /// max_bond_dim and reltol override both inputs' stored settings at every
    /// contraction. A nonpositive cap is unlimited; reltol < 0 disables
    /// tolerance truncation. final_compress adds a full SVD sweep with the
    /// same settings. reltol is a per-truncation tolerance, not a total bound.
    MPS conv(MPS const& kernel,
             int max_bond_dim = 0,
             RealScalar reltol = RealScalar(-1),
             bool final_compress = true)
    {
        if (this->core.empty() || this->core.size() != kernel.get_core().size()) {
            throw std::invalid_argument(
                "MPS::conv: expected nonempty MPSs of the same length");
        }
        for (const auto& tensor : this->core) {
            if (tensor.n_phys != 2) {
                throw std::invalid_argument(
                    "MPS::conv: expected signal physical dimension 2");
            }
        }

        // Use the requested settings throughout, without modifying either input.
        MPS configured_signal(*this);
        configured_signal.max_bond_dim_param = max_bond_dim;
        configured_signal.reltol_param = reltol;
        MPS configured_kernel(kernel);
        configured_kernel.max_bond_dim_param = max_bond_dim;
        configured_kernel.reltol_param = reltol;

        // Steps 1-2: build the exact magic tensor and contract the kernel's m' leg.
        const auto magic_result = magic_tensor_product(configured_kernel);
        // Step 3: interpret the remaining (n'*m) cores as an MPO.
        auto convolution_mpo = reshape_magic_res(magic_result);
        // Step 4: contract n' with the signal, retaining the output index m.
        auto result = convolution_mpo._mul(configured_signal, "zip-up");
        if (final_compress) {
            result.compress_svd(reltol, max_bond_dim);
        }
        return result;
    }

    /// Apply T(-nb_grid_point): shift toward larger grid indices with zero
    /// padding, without periodic wraparound. Uses a basis-MPS convolution
    /// kernel at nb_grid_point, encoded least-significant bit first.
    /// Negative shifts are rejected; shifts beyond the grid return zero.
    /// The original MPS is not modified. Its stored bond cap and tolerance
    /// are forwarded to conv, including the final compression sweep.
    MPS neg_translation(util::i128 nb_grid_point)
    {
        if (nb_grid_point < 0) {
            throw std::invalid_argument(
                "MPS::neg_translation: nb_grid_point must be nonnegative");
        }
        if (this->core.empty()) {
            throw std::invalid_argument("MPS::neg_translation: MPS must not be empty");
        }
        for (const auto& tensor : this->core) {
            if (tensor.n_phys != 2) {
                throw std::invalid_argument(
                    "MPS::neg_translation: expected physical dimension 2");
            }
        }

        std::vector<int> bits(this->core.size());
        util::i128 remaining = nb_grid_point;
        for (std::size_t bit = 0; bit < bits.size(); ++bit) {
            bits[bit] = static_cast<int>(remaining & util::i128(1));
            remaining >>= 1;
        }
        if (remaining != 0) {
            MPS zero(*this);
            zero *= T(0);
            return zero;
        }

        const auto kernel = analytic_mps_construction::make_basis_mps<T>(bits);
        return conv(kernel, this->get_max_bond_dim(), this->get_reltol());
    }
};

#include "magic_tensor.hpp"
#include "generic_construction.hpp"