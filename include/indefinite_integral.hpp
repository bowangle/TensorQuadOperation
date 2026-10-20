#pragma once

#include "mpo_base.h"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// Construct one core of the Heaviside MPO H(n,m) = Theta(n-m).
/// The exact MPO has TT/MPO bond rank chi = 2. `c_left` and `c_right`
/// indicate whether this core is connected to another core on the respective
/// side. Thus its shape is (c_left ? 2 : 1, 4, c_right ? 2 : 1), including
/// the boundary cases and the single-bit shape (1, 4, 1).
///
/// The MPO physical index uses the ordering defined by MPO:
///   0: (n_k,m_k) = (0,0)
///   1: (n_k,m_k) = (0,1)
///   2: (n_k,m_k) = (1,0)
///   3: (n_k,m_k) = (1,1)
///
/// Quantics sites in this project are stored least-significant bit first.
/// Consequently, these matrices are the transposes of the MSB-to-LSB
/// comparison tensors II in the usual presentation. The left boundary is the
/// least-significant bit and is where `diagonal_value = Theta(0)` is applied.
template<typename Scalar>
Tensor3D<Scalar> make_heaviside_mpo_core(
    bool c_left,
    bool c_right,
    Scalar diagonal_value = Scalar(0))
{
    // nothing on each side
    if (!c_left && !c_right) {
        Tensor3D<Scalar> core(/*n_left=*/1, /*n_phys=*/4, /*n_right=*/1);
        core(0, 0, 0) = diagonal_value; // n = m = 0
        core(0, 2, 0) = Scalar(1);      // n = 1, m = 0
        core(0, 3, 0) = diagonal_value; // n = m = 1
        return core;
    }

    // nothing on left
    if (!c_left) {
        // Least-significant-bit boundary. If all more-significant bits are
        // equal, this bit supplies Theta(0), 0, or 1. If n > m has already
        // been established, every local bit pair is accepted.
        Tensor3D<Scalar> core(/*n_left=*/1, /*n_phys=*/4, /*n_right=*/2);
        core(0, 0, 0) = diagonal_value;
        core(0, 0, 1) = Scalar(1);
        core(0, 1, 1) = Scalar(1);
        core(0, 2, 0) = Scalar(1);
        core(0, 2, 1) = Scalar(1);
        core(0, 3, 0) = diagonal_value;
        core(0, 3, 1) = Scalar(1);
        return core;
    }

    // middle case: something on left and on right
    Tensor3D<Scalar> middle(/*n_left=*/2, /*n_phys=*/4, /*n_right=*/2);
    // Equal bits leave the comparison state unchanged.
    middle(0, 0, 0) = Scalar(1);
    middle(1, 0, 1) = Scalar(1);
    middle(0, 3, 0) = Scalar(1);
    middle(1, 3, 1) = Scalar(1);
    // n_k < m_k: an undecided comparison is rejected. A comparison already
    // decided by a more-significant bit remains accepted.
    middle(1, 1, 1) = Scalar(1);
    // n_k > m_k: this bit decides an otherwise equal comparison, while an
    // already-decided comparison remains accepted.
    middle(1, 2, 0) = Scalar(1);
    middle(1, 2, 1) = Scalar(1);

    if (c_right)
        return middle;

    // If !c_right, their is nothing on right but something on left
    // Most-significant-bit boundary: initialize the comparison in the
    // undecided state by selecting column zero from each interior slice.
    Tensor3D<Scalar> core(/*n_left=*/2, /*n_phys=*/4, /*n_right=*/1);
    for (Eigen::Index state = 0; state < 2; ++state)
        for (Eigen::Index physical = 0; physical < 4; ++physical)
            core(state, physical, 0) = middle(state, physical, 0);
    return core;
}

/// Construct the n-bit, bond-rank-2 MPO H(n,m) = Theta(n-m).
/// Boundary bonds have dimension one; every internal bond has dimension two.
///
/// `diagonal_value` specifies Theta(0): use zero for an exclusive prefix sum
/// and one for an inclusive prefix sum.  The returned cores follow the
/// least-significant-bit-first ordering used by QuanticsIntegrator1D.
template<typename Scalar>
MPO<Scalar> make_heaviside_mpo(int nbits, Scalar diagonal_value = Scalar(0))
{
    if (nbits <= 0)
        throw std::invalid_argument("make_heaviside_mpo: nbits must be positive");

    std::vector<Tensor3D<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(nbits));

    for (int site = 0; site < nbits; ++site) {
        const bool c_left = site > 0;
        const bool c_right = site + 1 < nbits;
        cores.push_back(make_heaviside_mpo_core<Scalar>(c_left, c_right, diagonal_value));
    }

    return MPO<Scalar>(std::move(cores), /*max_bond_dim=*/0, /*reltol=*/-1, /*working_site=*/-1);
}

/// Apply the exclusive-prefix Heaviside MPO to an MPS.
///
/// For input samples psi_m, the returned MPS represents
///     F_n = sum_{m=0}^{n-1} psi_m.
/// A grid-spacing factor can be applied separately to turn the prefix sum into
/// a numerical indefinite integral. `method` selects the MPO-MPS contraction
/// algorithm and defaults to zip-up.
template<typename Scalar>
MPS<Scalar> make_indefinite_integral(
    const MPS<Scalar>& values,
    int max_chi,
    typename Eigen::NumTraits<Scalar>::Real reltol,
    const std::string& method = "zip-up")
{
    const int nbits = static_cast<int>(values.get_core().size());
    auto heaviside = make_heaviside_mpo<Scalar>(nbits, Scalar(0));

    // MPO::_mul obtains the zip-up truncation parameters from its operands.
    // Keep the exact Heaviside MPO independent of those parameters and attach
    // them to a local copy of the input MPS instead.
    MPS<Scalar> configured_values(values.get_core(), max_chi, reltol, values.get_w());
    return heaviside._mul(configured_values, method);
}

/// Construct the integrated convolution response
///     F_n = dt * sum_{m=0}^{n-1} kernel[n-m] * values[m].
/// values live on [a,b); the kernel lives on [0,b-a), with matching bit
/// counts and spacing. The kernel's first sample must be exactly zero,
/// making the inclusive conv sum identical to the exclusive sum above.
/// Returns {response_mps, value_grid}; the spacing factor is already applied.
/// Contraction limits and final_compress are forwarded to MPS::conv.
/// Neither input is modified. This is a convolution response, not a general
/// primitive whose differences evaluate arbitrary weighted subintervals.
template<typename Scalar, typename ValueGrid, typename KernelGrid>
std::pair<MPS<Scalar>, ValueGrid> make_indefinite_integral_conv(
    const MPS<Scalar>& values,
    const ValueGrid& value_grid,
    const MPS<Scalar>& kernel,
    const KernelGrid& kernel_grid,
    int max_chi = 0,
    typename Eigen::NumTraits<Scalar>::Real reltol =
        typename Eigen::NumTraits<Scalar>::Real(-1),
    bool final_compress = true)
{
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    const auto a = value_grid.get_a();
    const auto b = value_grid.get_b();
    const int nbit = value_grid.get_nBits();

    // Step 1: validate the physical grids and their represented MPSs.
    if (nbit <= 0 || !(b > a) || !(value_grid.get_dx() > RealScalar(0))) {
        throw std::invalid_argument(
            "make_indefinite_integral_conv: invalid value grid");
    }
    if (kernel_grid.get_a() != RealScalar(0)) {
        throw std::invalid_argument(
            "make_indefinite_integral_conv: kernel grid must start at zero");
    }
    if (kernel_grid.get_b() != b - a) {
        throw std::invalid_argument(
            "make_indefinite_integral_conv: kernel grid must end at b-a");
    }
    if (kernel_grid.get_nBits() != nbit) {
        throw std::invalid_argument(
            "make_indefinite_integral_conv: grid bit counts differ");
    }
    if (kernel_grid.get_dx() != value_grid.get_dx()) {
        throw std::invalid_argument(
            "make_indefinite_integral_conv: grid spacings differ");
    }

    // Step 2: require a zero diagonal activation weight.
    const std::vector<int> zero_bits(nbit, 0);
    if (kernel.eval(zero_bits) != Scalar(0)) {
        throw std::invalid_argument(
            "make_indefinite_integral_conv: kernel(0) must be zero");
    }

    // Step 3: conv is non-const, so call it on a copy and scale once by dt.
    MPS<Scalar> signal(values);
    auto out = signal.conv(kernel, max_chi, reltol, final_compress);
    out *= Scalar(value_grid.get_dx());

    // Step 4: the output origin is a+0, so retain the values' grid.
    return {std::move(out), value_grid};
}

/// Evaluate an indefinite-integral MPS at two grid coordinates and return
/// F(x_max) - F(x_min).
template<typename Scalar, typename Grid>
Scalar integrate_indefinite_method(
    const MPS<Scalar>& indefinite_values,
    const Grid& grid,
    const typename Grid::scalar_type& x_min,
    const typename Grid::scalar_type& x_max)
{
    if (static_cast<int>(indefinite_values.get_core().size())!= grid.get_nBits())
        throw std::invalid_argument("integrate_indefinite_method: MPS length differs from grid nBits");

    const std::vector<int> id_min = grid.coord_to_id(x_min);
    const std::vector<int> id_max = grid.coord_to_id(x_max);
    return indefinite_values.eval(id_max) - indefinite_values.eval(id_min);
}

/// Evaluate several coordinate ranges. The result at position i is
/// F(x_max[i]) - F(x_min[i]), and input order is preserved.
template<typename Scalar, typename Grid>
std::vector<Scalar> integrate_indefinite_method(
    const MPS<Scalar>& indefinite_values,
    const Grid& grid,
    const std::vector<typename Grid::scalar_type>& x_min,
    const std::vector<typename Grid::scalar_type>& x_max)
{
    if (x_min.size() != x_max.size())
        throw std::invalid_argument("integrate_indefinite_method: x_min and x_max sizes differ");
    if (static_cast<int>(indefinite_values.get_core().size())!= grid.get_nBits())
        throw std::invalid_argument("integrate_indefinite_method: MPS length differs from grid nBits");

    std::map<std::vector<int>, Scalar> evaluation_cache;
    const auto evaluate_cached =
        [&indefinite_values, &evaluation_cache](const std::vector<int>& id)
        -> const Scalar& {
        auto [entry, inserted] = evaluation_cache.try_emplace(id, Scalar(0));
        if (inserted)
            entry->second = indefinite_values.eval(id);
        return entry->second;
    };

    std::vector<Scalar> results;
    results.reserve(x_min.size());
    for (std::size_t index = 0; index < x_min.size(); ++index) {
        const std::vector<int> id_min = grid.coord_to_id(x_min[index]);
        const std::vector<int> id_max = grid.coord_to_id(x_max[index]);
        results.push_back(evaluate_cached(id_max) - evaluate_cached(id_min));
    }
    return results;
}


