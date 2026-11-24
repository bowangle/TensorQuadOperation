#pragma once

#include "TensorQuadOperation2/tt_base.hpp"
#include <Eigen/Core>
#include <type_int128.h>
#include <xfac_quad/cubemat_helper.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TensorQuadOperation {

// Helper declarations for the convolution implementation below.
template<typename T> class MPS;
template<typename T> class MPO;
template<typename T> MPS<T> magic_tensor_product(const MPS<T>& mps);
template<typename T> MPO<T> reshape_magic_res(const MPS<T>& result);

namespace analytic_mps_construction {
template<typename T> MPS<T> make_basis_mps(const std::vector<int>& bits, int ordering = 1);
}

// Matrix Product State with one physical leg per core.
// The physical dimension can vary from site to site.
// The first core's left bond and the last core's right bond may both be dangling.

template<typename T>
class MPS : public TT<T> {
    using Base       = TT<T>;
    using VecTT      = std::vector<xfac_quad::Tensor3D<T>>;
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
    /// Contract the physical legs and keep all four boundary legs open.
    /// Both MPSs must have the same storage ordering.
    /// The returned dimensions are {left_A, left_B, right_A, right_B}.
    /// Rows fuse left_A and left_B, and columns fuse right_A and right_B.
    /// A's index varies fastest in each pair, and A's entries are conjugated.
    std::pair<MatrixX, std::array<Eigen::Index, 4>> contract_env(MPS const& other) const
    {
        const int order = this->ordering;
        if (order != other.ordering)
            throw std::invalid_argument("MPS::contract_env: MPSs must have the same ordering");
        if (order < 0 || order > 2)
            throw std::invalid_argument("MPS::contract_env: ordering must be 0, 1 or 2");

        const auto& a = this->core;
        const auto& b = other.core;
        if (a.empty() || a.size() != b.size())
            throw std::invalid_argument("MPS::contract_env: MPSs must have the same nonzero length");

        using Core = xfac_quad::Tensor3D<T>;
        using Index = Eigen::Index;
        using Shape = std::array<Index, 3>;
        const auto core_shape = [](const Core& tensor, int ordering) -> Shape {
            const Shape stored{tensor.n_rows, tensor.n_cols, tensor.n_slices};
            return {stored[ordering == 0 ? 1 : 0], stored[ordering], stored[ordering == 2 ? 1 : 2]};
        };

        // Take matrix views in each core's stored layout.
        using Stride = typename Core::Stride_dim;
        using Slice = Eigen::Map<const MatrixX, 0, Stride>;
        const auto physical_slice = [](const Core& tensor, int ordering, Index p) -> Slice {
            if (ordering == 0) return tensor.row_as_mat_const(p);
            if (ordering == 1) return tensor.col_as_mat_const(p);
            const auto view = tensor.slice_const(p);
            return Slice(view.data(), view.rows(), view.cols(), Stride(view.outerStride(), view.innerStride()));
        };

        std::vector<Shape> a_shapes, b_shapes;
        a_shapes.reserve(a.size());
        b_shapes.reserve(b.size());
        for (std::size_t site = 0; site < a.size(); ++site) {
            const auto sa = core_shape(a[site], order);
            const auto sb = core_shape(b[site], order);
            if (sa[1] != sb[1])
                throw std::invalid_argument("MPS::contract_env: physical dimensions differ");
            if (sa[0] <= 0 || sa[1] <= 0 || sa[2] <= 0 || sb[0] <= 0 || sb[2] <= 0)
                throw std::invalid_argument("MPS::contract_env: core dimensions must be positive");
            if (site > 0 && (a_shapes.back()[2] != sa[0] || b_shapes.back()[2] != sb[0]))
                throw std::invalid_argument("MPS::contract_env: adjacent bond dimensions differ");
            a_shapes.push_back(sa);
            b_shapes.push_back(sb);
        }

        const std::array<Index, 4> boundaries{
            a_shapes.front()[0], b_shapes.front()[0], a_shapes.back()[2], b_shapes.back()[2]};
        const auto [left_a, left_b, right_a, right_b] = boundaries;
        MatrixX result(left_a * left_b, right_a * right_b);
        using Row = Eigen::Matrix<T, 1, Eigen::Dynamic>;

        // Sweep a two-bond environment for each fixed pair of left indices.
        for (Index lb = 0; lb < left_b; ++lb) {
            for (Index la = 0; la < left_a; ++la) {
                MatrixX env = MatrixX::Zero(a_shapes.front()[2], b_shapes.front()[2]);
                for (Index p = 0; p < a_shapes.front()[1]; ++p)
                    env.noalias() += physical_slice(a.front(), order, p).row(la).adjoint()
                                   * physical_slice(b.front(), order, p).row(lb);

                for (std::size_t site = 1; site < a.size(); ++site) {
                    MatrixX next = MatrixX::Zero(a_shapes[site][2], b_shapes[site][2]);
                    for (Index p = 0; p < a_shapes[site][1]; ++p)
                        next.noalias() += physical_slice(a[site], order, p).adjoint()
                                        * (env * physical_slice(b[site], order, p));
                    env = std::move(next);
                }
                result.row(la + left_a * lb) = Eigen::Map<const Row>(env.data(), env.size());
            }
        }
        return {std::move(result), boundaries};
    }

    // MPS MPS dot product. Connect all the leg and contract
    T dot(MPS const& other) const
    {
        const auto E = contract_env(other).first;
        if (E.rows() != 1 || E.cols() != 1)
            throw std::invalid_argument("MPS::dot: non-trivial boundary legs, use contract_env instead");
        return E(0, 0);
    }

    // Convenience: norm^2 = <this|this> (real and >= 0 up to rounding)
    RealScalar norm2() const
    {
        return Eigen::numext::real(dot(*this));
    }

    // TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
            if (tensor.n_cols != 2) {
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
            if (tensor.n_cols != 2) {
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

} // namespace TensorQuadOperation
