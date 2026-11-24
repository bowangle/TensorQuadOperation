#pragma once

#include "TensorQuadOperation/ttn_base.hpp"
#include "TensorQuadOperation/utc.hpp"
#include <QTgrid-quad/grid.h>

#include <array>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace QuanticsIntegration::simplex_helper {

namespace detail {

template<typename Scalar>
void validate_train(const TensorQuadOperation::TT<Scalar>& values, int bits,
                    Eigen::Index physical_dim, const char* operation, bool allow_open_right)
{
    const auto& cores = values.get_core();
    const int ordering = values.get_ordering();
    if (ordering < 0 || ordering > 2)
        throw std::invalid_argument(std::string(operation) + ": ordering must be 0, 1 or 2");
    if (bits <= 0 || cores.size() != static_cast<std::size_t>(bits))
        throw std::invalid_argument(std::string(operation) +
                                    ": train length must match the positive grid bit count");
    const int left_axis = ordering == 0 ? 1 : 0;
    const int right_axis = ordering == 2 ? 1 : 2;
    Eigen::Index previous_right = 0;
    for (std::size_t site = 0; site < cores.size(); ++site) {
        const auto& core = cores[site];
        const std::array<Eigen::Index, 3> shape{core.n_rows, core.n_cols, core.n_slices};
        const auto left = shape[left_axis];
        const auto right = shape[right_axis];
        if (shape[ordering] != physical_dim || left <= 0 || right <= 0)
            throw std::invalid_argument(std::string(operation) +
                                        ": invalid core dimensions");
        if ((site == 0 && left != 1) ||
            (!allow_open_right && site + 1 == cores.size() && right != 1))
            throw std::invalid_argument(std::string(operation) +
                                        (allow_open_right ? ": left boundary bond must have dimension one"
                                                          : ": boundary bonds must have dimension one"));
        if (site > 0 && previous_right != left)
            throw std::invalid_argument(std::string(operation) +
                                        ": adjacent bond dimensions differ");
        previous_right = right;
    }
}

template<typename Scalar>
void validate_branch_mpo(const TensorQuadOperation::MPO<Scalar>& branch, int bits,
                         const char* operation)
{
    if (branch.get_left_phys() != 2 || branch.get_right_phys() != 2)
        throw std::invalid_argument(std::string(operation) + ": expected a 2 x 2 branch MPO");
    validate_train<Scalar>(branch, bits, 4, operation, /*allow_open_right=*/true);
}

} // namespace detail

template<typename Scalar>
void validate_simplex_mps(const TensorQuadOperation::MPS<Scalar>& values,
                          int bits, const char* operation, bool allow_open_right = false)
{
    detail::validate_train<Scalar>(values, bits, 2, operation, allow_open_right);
}

template<typename Scalar>
void validate_simplex_tree(const xfac_quad::TensorTree<Scalar>& body, int dimension, int bits)
{
    const auto& tree = body.tree;
    const auto expected = xfac_quad::makeTuckerTree(dimension, bits);
    constexpr auto topology_error =
        "QuanticsIntegrator_kD::make_integration_simplex: expected an ordered Tucker tree with one branch per grid dimension";
    if (!tree.branches || tree.branches->size() != expected.branches->size() ||
        tree.nodes != expected.nodes || tree.size() != expected.size() ||
        body.M.size() != tree.size() || !tree.neigh.contains(tree.root))
        throw std::invalid_argument(topology_error);

    // Physical IDs follow the grid's bit order. Junction IDs are taken from the tree.
    auto node_ids = xfac_quad::iota(static_cast<int>(expected.size()));
    for (int variable = 0; variable < dimension; ++variable) {
        const auto& branch = tree.branches->at(variable);
        const auto& expected_branch = expected.branches->at(variable);
        if (branch.nodes != expected_branch.nodes)
            throw std::invalid_argument(topology_error);
        if (dimension > 2)
            node_ids[expected_branch.attachment] = branch.attachment;
    }
    if (std::set<int>(node_ids.begin(), node_ids.end()).size() != node_ids.size())
        throw std::invalid_argument(topology_error);
    for (int variable = 0; variable < dimension; ++variable) {
        const int attachment = expected.branches->at(variable).attachment;
        if (tree.branches->at(variable).attachment != (attachment < 0 ? -1 : node_ids[attachment]))
            throw std::invalid_argument(topology_error);
    }

    // Compare connections by node ID so storage-axis permutations are accepted.
    for (const auto& [expected_node, expected_neighbors] : expected.neigh) {
        const int node = node_ids[expected_node];
        const auto actual = tree.neigh.find(node);
        if (node < 0 || static_cast<std::size_t>(node) >= body.M.size() || actual == tree.neigh.end())
            throw std::invalid_argument(topology_error);
        const auto& neighbors = actual->second;
        const bool physical = tree.nodes.contains(node);
        const std::size_t bond_axes = physical ? 2 : 3;
        if (neighbors.size() > bond_axes ||
            neighbors.size() - neighbors.to_int().count(-1) != expected_neighbors.size())
            throw std::invalid_argument(topology_error);
        for (int neighbor : expected_neighbors.from_int())
            if (!neighbors.to_int().contains(node_ids[neighbor]))
                throw std::invalid_argument(topology_error);

        const auto& core = body.M[node];
        const std::array<Eigen::Index, 3> shape{core.n_rows, core.n_cols, core.n_slices};
        if (physical && core.n_slices != 2)
            throw std::invalid_argument(
                "QuanticsIntegrator_kD::make_integration_simplex: tree cores must have binary physical dimensions");
        for (std::size_t axis = 0; axis < bond_axes; ++axis) {
            const bool open = axis >= neighbors.size() || neighbors.from_int()[axis] == -1;
            if (shape[axis] <= 0 || (open && shape[axis] != 1))
                throw std::invalid_argument(
                    "QuanticsIntegrator_kD::make_integration_simplex: invalid tree bond dimension");
        }
    }
}

template<typename Scalar, typename Real, typename Sint>
void validation_simplex_parameter(
    const TensorQuadOperation::TTN<Scalar>& values,
    const qtgrid::MultQTGrid<Real, Sint>& grid,
    const std::vector<TensorQuadOperation::MPS<Scalar>>& kernels,
    const qtgrid::MultQTGrid<Real, Sint>& kernel_grid)
{
    if (grid.get_dim() <= 0 || grid.get_nBits().empty())
        throw std::invalid_argument(
            "QuanticsIntegrator_kD::make_integration_simplex: expected a nonempty grid");
    const int bits = grid.get_nBits().front();
    const Real upper = grid.get_b().front();
    const Real spacing = grid.get_dx().front();
    for (int variable = 0; variable < grid.get_dim(); ++variable) {
        if (bits <= 0 || !(spacing > Real(0)) || grid.get_a()[variable] != Real(0) ||
            grid.get_b()[variable] != upper || grid.get_nBits()[variable] != bits)
            throw std::invalid_argument(
                "QuanticsIntegrator_kD::make_integration_simplex: all variables must share a [0, b) grid with a positive bit count and spacing");
    }
    validate_simplex_tree(values.get_TTN_body(), grid.get_dim(), bits);

    if (kernels.empty()) return;
    if (kernels.size() != static_cast<std::size_t>(grid.get_dim()))
        throw std::invalid_argument(
            "QuanticsIntegrator_kD::make_integration_simplex: expected one kernel per variable");
    if (kernel_grid.get_a() != grid.get_a() || kernel_grid.get_b() != grid.get_b() ||
        kernel_grid.get_nBits() != grid.get_nBits())
        throw std::invalid_argument(
            "QuanticsIntegrator_kD::make_integration_simplex: kernel grids must match the integration grid");
    for (const auto& kernel : kernels)
        validate_simplex_mps(kernel, bits, "QuanticsIntegrator_kD::make_integration_simplex kernel");
}

//vk-1 branch (MPS_k-1)(vk-1=2 leg dim) * substractor MPO ((vk-1, (uk-1,uk-2))=2*(2*2) leg dim)
//      -> MPS_k-1((uk-1,uk-2)=(2*2) leg dim)
//      convert to MPO_k-1 (uk-1*uk-2=2*2 leg dim)
/// The resulting MPO has fused slot 2 * upper_time_bit + integrated_time_bit.
/// The branch and subtractor use LSB-first sites. The branch's open right
/// attachment stays in its original basis for the contraction with the junction.
template<typename Scalar>
TensorQuadOperation::MPO<Scalar> substract_branch(
    TensorQuadOperation::MPS<Scalar> branch,
    TensorQuadOperation::TT<Scalar> subtractor)
{
    // MPS has leg                vk-1  = 2
    // MPO has leg (vk-1, (uk-1,uk-2))  = 2*(2*2)

    // MPS has a dangling leg (be carefull, get it from extract_branch)
    const int bits = static_cast<int>(std::as_const(subtractor).get_core().size());
    validate_simplex_mps(branch, bits, "substract_branch", /*allow_open_right=*/true);
    detail::validate_train(subtractor, bits, 8, "substract_branch subtractor", /*allow_open_right=*/false);
    const int ordering = branch.get_ordering();
    TensorQuadOperation::MPO<Scalar> subtractor_mpo(
        TensorQuadOperation::TT<Scalar>::change_ordering(std::move(subtractor), ordering),
        /*left_phys=*/4, /*right_phys=*/2);

    // Move the centres along internal bonds only, then contract without truncation.
    branch.shift_w(bits - 1);
    subtractor_mpo.shift_w(bits - 1);
    auto cores = TensorQuadOperation::utc::qrsvd::qrsvd_contract_mpo_mps(subtractor_mpo, branch);
    return TensorQuadOperation::MPO<Scalar>(std::move(cores), branch.get_max_bond_dim(),
                                           branch.get_reltol(), /*w=*/-1,
                                           /*left_phys=*/2, /*right_phys=*/2, ordering);
}

//  if kernel:
//      MPO_N(uN*uN-1=2*2 leg dim) * MPS_kernel(uN=2 leg dim)
//  if no kernel:
//      MPO_N(uN*uN-1=2*2 leg dim) * MPS=1 everywhere (uN=2 leg dim)
/// Sum over u_N with the given kernel. The caller applies the grid spacing.
template<typename Scalar>
TensorQuadOperation::MPS<Scalar> init_integrate_branch(
    TensorQuadOperation::MPO<Scalar> branch,
    TensorQuadOperation::MPS<Scalar> kernel
)
{
    // MPO_N(uN*uN-1 = 2*2 leg dim) has dangling leg
    // MPS (uN = 2 leg dim)

    // if no kernel provided use kernel = MPS( the constant =1 function)

    const int bits = static_cast<int>(std::as_const(branch).get_core().size());
    validate_simplex_mps(kernel, bits, "init_integrate_branch kernel");
    detail::validate_branch_mpo(branch, bits, "init_integrate_branch");
    const int ordering = branch.get_ordering();
    if (kernel.get_ordering() != ordering)
        kernel = TensorQuadOperation::MPS<Scalar>(
            TensorQuadOperation::TT<Scalar>::change_ordering(std::move(kernel), ordering));

    branch.shift_w(bits - 1);
    kernel.shift_w(bits - 1);
    auto cores = TensorQuadOperation::utc::qrsvd::qrsvd_contract_mpo_mps(branch, kernel);

    // return an mps uN-1 with dangling leg (same as the one of the original mpo_N)
    return TensorQuadOperation::MPS<Scalar>(std::move(cores), branch.get_max_bond_dim(),
                                           branch.get_reltol(), /*w=*/-1, ordering);
}

template<typename Scalar>
TensorQuadOperation::MPS<Scalar> init_integrate_branch(TensorQuadOperation::MPO<Scalar> branch)
{
    const int ordering = branch.get_ordering();
    std::array<Eigen::Index, 3> shape{1, 1, 1};
    shape[ordering] = 2;
    xfac_quad::Tensor3D<Scalar> ones_core(shape[0], shape[1], shape[2]);
    ones_core.data[0] = ones_core.data[1] = Scalar(1);
    std::vector<xfac_quad::Tensor3D<Scalar>> cores(std::as_const(branch).get_core().size(), ones_core);
    return init_integrate_branch(std::move(branch), TensorQuadOperation::MPS<Scalar>(
        std::move(cores), /*max_bond_dim=*/0, /*reltol=*/-1, /*w=*/-1, ordering));
}

/// Compose the branch MPO with the diagonal kernel on the input time u_{k-1}.
/// The kernel has unit boundary bonds, so the branch attachment keeps its basis.
template<typename Scalar>
TensorQuadOperation::MPO<Scalar> add_kernel_k_step(
    const TensorQuadOperation::MPO<Scalar>& branch,
    const TensorQuadOperation::MPS<Scalar>& kernel)
{
    const auto& branch_cores = branch.get_core();
    const auto& kernel_cores = kernel.get_core();
    const int bits = static_cast<int>(branch_cores.size());
    validate_simplex_mps(kernel, bits, "add_kernel_k_step kernel");
    detail::validate_branch_mpo(branch, bits, "add_kernel_k_step");
    const int ordering = branch.get_ordering();
    const int kernel_ordering = kernel.get_ordering();
    using TensorQuadOperation::utc::zip_up::detail::zip_up_core_dimensions;
    using TensorQuadOperation::utc::zip_up::detail::zip_up_physical_slice;

    std::vector<xfac_quad::Tensor3D<Scalar>> cores;
    cores.reserve(branch_cores.size());
    for (std::size_t site = 0; site < branch_cores.size(); ++site) {
        const auto& branch_core = branch_cores[site];
        const auto& kernel_core = kernel_cores[site];
        const auto branch_shape = zip_up_core_dimensions(branch_core, ordering);
        const auto kernel_shape = zip_up_core_dimensions(kernel_core, kernel_ordering);
        const auto left_size = branch_shape[0] * kernel_shape[0];
        const auto right_size = branch_shape[2] * kernel_shape[2];
        const std::array<Eigen::Index, 3> shape = ordering == 0
            ? std::array<Eigen::Index, 3>{4, left_size, right_size}
            : ordering == 1 ? std::array<Eigen::Index, 3>{left_size, 4, right_size}
                            : std::array<Eigen::Index, 3>{left_size, right_size, 4};
        xfac_quad::Tensor3D<Scalar> core(shape[0], shape[1], shape[2]);

        // In slot 2 * upper_bit + input_bit, the diagonal selects input_bit.
        for (Eigen::Index slot = 0; slot < 4; ++slot) {
            const auto branch_slice = zip_up_physical_slice(branch_core, ordering, slot);
            const auto kernel_slice = zip_up_physical_slice(kernel_core, kernel_ordering, slot % 2);
            const auto weight_slice = [&](auto weighted_slice) {
                for (Eigen::Index right = 0; right < kernel_slice.cols(); ++right)
                    for (Eigen::Index left = 0; left < kernel_slice.rows(); ++left)
                        weighted_slice.block(left * branch_slice.rows(), right * branch_slice.cols(),
                                             branch_slice.rows(), branch_slice.cols()) =
                            kernel_slice(left, right) * branch_slice;
            };
            if (ordering == 0)
                weight_slice(core.row_as_mat(slot));
            else if (ordering == 1)
                weight_slice(core.col_as_mat(slot));
            else
                weight_slice(core.slice(slot));
        }
        cores.push_back(std::move(core));
    }
    return TensorQuadOperation::MPO<Scalar>(std::move(cores), branch.get_max_bond_dim(),
                                           branch.get_reltol(), /*w=*/-1,
                                           branch.get_left_phys(), branch.get_right_phys(), ordering);
}

template<typename Scalar>
TensorQuadOperation::MPO<Scalar> had_kernel_k_step(TensorQuadOperation::MPO<Scalar> branch)
{
    return branch;
}

} // namespace QuanticsIntegration::simplex_helper
