#pragma once

#include "TensorQuadOperation/tt_base.hpp"
#include "TensorQuadOperation/mps_base.hpp"
#include "TensorQuadOperation/mpo_base.hpp"
#include "TensorQuadOperation/utc.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TensorQuadOperation {

// MPO may still be incomplete when reached through mpo_base.hpp -> mps_base.hpp.
// The dependent template definitions below are instantiated after it is defined.
template<typename T>
class MPO;

/// Construct the reduced magic tensor Theta(m; n', m') = delta(m, n'+m')
/// from Sec. 10.3, Eq. (116), of whocancompete.pdf as an MPS.
///
/// Sites are ordered least-significant bit first. Each core is built in
/// ordering 1 with shape (carry_in, 8, carry_out), then moved to the
/// requested storage ordering. Internal carry dimensions are two and
/// boundary dimensions are one. Both boundary carries are fixed to zero:
/// additions that overflow nbit bits are rejected, not wrapped modulo 2^nbit.
///
/// Interpret this MPS as a rectangular operator with input m' and output
/// group n'*m, with n' fastest within that group. For each site:
///   output = n'_bit + 2*m_bit,
///   physical = 2*output + m'_bit.
/// This matches the low-level MPO-MPS convention physical = output*p_in+input.
/// Contracting m' with a kernel MPS leaves physical = 2*m_bit+n'_bit,
/// ready to be interpreted as an ordinary MPO with row m and column n'.
///
/// The construction is exact, with no canonicalization, truncation, grid
/// spacing, or quadrature weights applied. For nbit == 1 the ordering-1
/// core has shape (1,8,1).
template<typename Scalar>
MPS<Scalar> make_magic_tensor(int nbit, int ordering = 1)
{
    if (nbit <= 0) {
        throw std::invalid_argument("maek_magic_tensor: nbit must be positive");
    }

    std::vector<xfac_quad::Tensor3D<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(nbit));

    for (int site = 0; site < nbit; ++site) {
        const int left_dim = site == 0 ? 1 : 2;
        const int right_dim = site == nbit - 1 ? 1 : 2;
        xfac_quad::Tensor3D<Scalar> core(left_dim, 8, right_dim); // zero-initialized

        for (int carry_in = 0; carry_in < left_dim; ++carry_in) {
            for (int n_prime = 0; n_prime < 2; ++n_prime) {
                for (int m_prime = 0; m_prime < 2; ++m_prime) {
                    const int sum = n_prime + m_prime + carry_in;
                    const int m = sum % 2;
                    const int carry_out = sum / 2;
                    if (carry_out >= right_dim) {
                        continue; // Reject overflow at the most-significant bit.
                    }

                    const int output = n_prime + 2 * m;
                    const int physical = 2 * output + m_prime;
                    core(carry_in, physical, carry_out) = Scalar(1);
                }
            }
        }
        cores.push_back(std::move(core));
    }

    TT<Scalar> magic(std::move(cores), /*max_bond_dim=*/0,
                     /*reltol=*/-1, /*w=*/-1, /*ordering=*/1);
    return MPS<Scalar>(TT<Scalar>::change_ordering(std::move(magic), ordering));
}

/// Compute (m', n'*m) * (m') = (n'*m) using the magic addition tensor.
/// The bit count is inferred from the input MPS's number of cores.
/// The input must have binary cores in least-significant-bit-first order.
/// The returned MPS keeps the input storage ordering and has physical
/// dimension four, fused as n'_bit+2*m_bit.
/// Its entries are sum_{m'} delta(m,n'+m') * mps(m'), i.e. mps(m-n')
/// when m >= n', and zero otherwise (no periodic wraparound).
///
/// Uses the input MPS's tolerance and maximum bond dimension for the zip-up
/// contraction. Both operands are canonicalized with shift_w(0) first,
/// using a copy of the input MPS so neither it nor its cores are modified.
/// No grid spacing or quadrature weights are applied.
template<typename Scalar>
MPS<Scalar> magic_tensor_product(const MPS<Scalar>& mps)
{
    const auto& input_cores = mps.get_core();
    if (input_cores.empty()) {
        throw std::invalid_argument("magic_tensor_product: MPS must not be empty");
    }
    const int ordering = mps.get_ordering();
    if (ordering < 0 || ordering > 2) {
        throw std::invalid_argument("magic_tensor_product: ordering must be 0, 1 or 2");
    }
    const int nbit = static_cast<int>(input_cores.size());
    for (const auto& core : input_cores) {
        const std::array<Eigen::Index, 3> shape{core.n_rows, core.n_cols, core.n_slices};
        if (shape[ordering] != 2) {
            throw std::invalid_argument("magic_tensor_product: expected MPS physical dimension 2");
        }
    }

    MPO<Scalar> magic(make_magic_tensor<Scalar>(nbit, ordering), /*left_phys=*/4, /*right_phys=*/2);
    MPS<Scalar> canonical_mps(mps);
    magic.shift_w(0);
    canonical_mps.shift_w(0);

    auto result_cores = utc::zip_up::zip_up_mpo_mps(
        magic, canonical_mps,
        mps.get_reltol(), mps.get_max_bond_dim());

    return MPS<Scalar>(std::move(result_cores), mps.get_max_bond_dim(), mps.get_reltol(), /*w=*/-1, ordering);
}

/// Interpret the (n'*m) MPS returned by magic_tensor_product as an MPO
/// with input n' and output m. Its fused index n'_bit+2*m_bit already
/// matches the MPO ordering physical = 2*row+column (row m, column n').
/// No permutation or contraction is needed: copy the underlying TT,
/// preserving its cores, truncation settings, and canonical-center metadata.
/// The MPO constructor checks that every core has physical dimension four.
template<typename Scalar>
MPO<Scalar> reshape_magic_res(const MPS<Scalar>& result)
{
    if (result.get_core().empty()) {throw std::invalid_argument("reshape_magic_res: MPS must not be empty");
    }
    return MPO<Scalar>(static_cast<const TT<Scalar>&>(result));
}

} // namespace TensorQuadOperation

