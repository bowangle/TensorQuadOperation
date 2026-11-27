#pragma once

#include "TensorQuadOperation/ttn_base.hpp"
#include "TensorQuadOperation/generic_construction.hpp"
#include "TensorQuadOperation/magic_tensor.hpp"
#include "TensorQuadOperation/utc.hpp"
#include <QTgrid-quad/grid.h>
#include <spdlog/logger.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace QuanticsIntegration::simplex_helper {

namespace validation {

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

template<typename Scalar>
void validate_simplex_mps(const TensorQuadOperation::MPS<Scalar>& values,
                          int bits, const char* operation, bool allow_open_right = false)
{
    validate_train<Scalar>(values, bits, 2, operation, allow_open_right);
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
    const TensorQuadOperation::MPS<Scalar>* kernel,
    const qtgrid::QTGrid<Real, Sint>& kernel_grid)
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

    if (!kernel) return;
    if (kernel_grid.get_a() != Real(0) || kernel_grid.get_b() != upper ||
        kernel_grid.get_nBits() != bits ||
        kernel_grid.get_x_ref() - Real(kernel_grid.get_k_offset()) * spacing != Real(0))
        throw std::invalid_argument(
            "QuanticsIntegrator_kD::make_integration_simplex: kernel grid must match the common time grid");
    validate_simplex_mps(*kernel, bits, "QuanticsIntegrator_kD::make_integration_simplex kernel");
}

} // namespace validation

/// Return a two-time MPO with output u_{k-1}, input u_k, and an open attachment bond.
/// Move the tree's canonical center to the branch endpoint so the MPO's open
/// bond matches the tree's post-extraction attachment gauge.
template<typename Scalar>
TensorQuadOperation::MPO<Scalar> substract_branch(
    TensorQuadOperation::TTN<Scalar>& tree, int branch_id,
    const std::string& method = "zip-up")
{
    auto branch = tree._extract_branch(branch_id, /*compress=*/false);
    // Branch branch_id encodes v_k with k = branch_id + 1, LSB first.
    // Its left boundary is one; its right bond connects to the attachment.
    // The subtractor enforces u_{k-1} = u_k + v_k without wraparound (u_0 = t).
    // Input: v_k. Fused output slot: u_k_bit + 2 * u_{k-1}_bit.
    // leg dim is ([(u_k=2) * (u_{k-1}=2)] * [v_k=2])= (4*2)
    TensorQuadOperation::MPO<Scalar> subtractor_mpo(
        TensorQuadOperation::make_magic_tensor<Scalar>(branch.get_size(), branch.get_ordering()),
        /*left_phys=*/4, /*right_phys=*/2);

    return TensorQuadOperation::MPO<Scalar>(subtractor_mpo._mul(branch, method),
                                           /*left_phys=*/2, /*right_phys=*/2);
}

/// Apply only to the last gap branch v_N, where N is the total number of branches.
/// Sum over u_N, changing the tree in place from (v_1, ..., v_{N-1}, v_N)
/// to (v_1, ..., v_{N-1}, u_{N-1}). Before insertion, multiply the resulting
/// MPS by the 1D grid spacing h for the last branch.
/// A null kernel uses an all-ones MPS for the unweighted sum.
template<typename Scalar, typename Real, typename Sint>
void _init_integrate_branch(
    TensorQuadOperation::TTN<Scalar>& tree,
    const qtgrid::MultQTGrid<Real, Sint>& grid,
    std::type_identity_t<const TensorQuadOperation::MPS<Scalar>*> kernel = nullptr,
    const std::string& method = "zip-up")
{
    const auto& branches = tree.get_TTN_body().tree.branches;
    if (!branches || branches->empty())
        throw std::invalid_argument("_init_integrate_branch2: expected at least one branch");
    if (branches->size() != static_cast<std::size_t>(grid.get_dim()))
        throw std::invalid_argument("_init_integrate_branch2: expected one grid dimension per branch");
    const int last_branch = static_cast<int>(branches->size() - 1);
    const Real h = grid.get_dx().at(last_branch);
    if (!(h > Real(0)))
        throw std::invalid_argument("_init_integrate_branch2: grid spacing must be positive");
    auto branch_mpo = substract_branch(tree, last_branch, method);
    const int bits = branch_mpo.get_size();

    if (kernel) {
        // Contract the binary u_N input with the kernel, leaving u_{N-1} open.
        validation::validate_simplex_mps(*kernel, bits, "_init_integrate_branch2 kernel");
        const int ordering = branch_mpo.get_ordering();
        auto message = kernel->get_ordering() == ordering
            ? branch_mpo._mul(*kernel, method)
            : branch_mpo._mul(TensorQuadOperation::MPS<Scalar>(
                  TensorQuadOperation::TT<Scalar>::change_ordering(*kernel, ordering)), method);
        message *= Scalar(h);
        tree._insert_branch(last_branch, std::move(message), /*original_w=*/-1, /*compress=*/false);
    } else {
        auto ones = TensorQuadOperation::analytic_mps_construction::make_one_mps<Scalar>(
            bits, /*ordering=*/2);
        auto message = branch_mpo._mul(ones, method);
        message *= Scalar(h);
        tree._insert_branch(last_branch, std::move(message), /*original_w=*/-1, /*compress=*/false);
    }
}

/// Weight the branch MPO on its input time x = u_k, leaving its output y = u_{k-1} open.
/// Composing B(y, x) with D(x, x') = delta(x, x') * kernel(x') gives
/// B'(y, x') = B(y, x') * kernel(x'). The later merge sums over x'.
/// Use the kernel MPS directly instead of forming D. Internal bond dimensions
/// multiply, but the kernel's unit boundary bonds preserve the branch attachment basis.
/// This step does not apply quadrature scaling or truncate the resulting bonds.
template<typename Scalar>
TensorQuadOperation::MPO<Scalar> add_kernel_k_step(
    const TensorQuadOperation::MPO<Scalar>& branch,
    const TensorQuadOperation::MPS<Scalar>& kernel)
{
    const auto& branch_cores = branch.get_core();
    const auto& kernel_cores = kernel.get_core();
    const int bits = static_cast<int>(branch_cores.size());
    validation::validate_simplex_mps(kernel, bits, "add_kernel_k_step kernel");
    validation::validate_branch_mpo(branch, bits, "add_kernel_k_step");
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
            // Form kernel_slice ⊗ branch_slice with branch indices varying fastest.
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

// from (v_1, ..., v_{k-1}, v_k, u_k) to (v_1, ..., v_{k-1}, u_{k-1})
// work on both branch v_k and u_k which are the branch k-1 and k of the tree.
// Update the tree in place; the final step leaves a single branch on t.
template<typename Scalar, typename Real, typename Sint>
void _integrate_k_branch(
    TensorQuadOperation::TTN<Scalar>& tree,
    const qtgrid::MultQTGrid<Real, Sint>& grid,
    int branch_k,
    std::type_identity_t<const TensorQuadOperation::MPS<Scalar>*> kernel = nullptr,
    const std::string& method = "zip-up",
    spdlog::logger* logger = nullptr,
    bool merge_trucation = true)
{
    const bool log_step = logger && logger->should_log(spdlog::level::info);
    if (log_step)
        logger->info("=================================\n_integrate_k_branch: {}\n=================================", branch_k);
    const auto log_done = [&] {
        if (log_step)
            logger->info("=================================\n_integrate_k_branch: {} done\n=================================", branch_k);
    };
    const auto& branches = tree.get_TTN_body().tree.branches;
    if (!branches || branches->size() < 2 || branch_k < 1 || branch_k >= grid.get_dim() ||
        static_cast<std::size_t>(branch_k) != branches->size() - 1)
        throw std::invalid_argument("_integrate_k_branch: expected the last two active branches");
    const Real h = grid.get_dx().at(branch_k - 1);
    if (!(h > Real(0)))
        throw std::invalid_argument("_integrate_k_branch: grid spacing must be positive");

    // ============================================================
    //              Substract
    // ============================================================
    if (log_step)
        logger->info("Tree before subtractor: max_chi = {}", tree.get_chi());
    auto branch_mpo = substract_branch(tree, branch_k - 1, method); // v_k -> uk-1, uk
    if (log_step)
        logger->info("Branch MPO{}: max_chi = {}", kernel ? " before kernel" : "", branch_mpo.get_chi());
    // ============================================================
    //              kernel if provided
    // ============================================================

    // We have the following:
    // branch_mpo: (Ra,b,c) B1_b,b1,x1,y1 B2_b1,b2,x2,y2 B3_b2,1,x3,y3
    // kernel(has a diag MPO):      D1_1,d1,x1,x'1 D2_d1,d2,x2,x'2 D3_d2,1,x3,x'3
    // We want to sum on x to get:
    // (Ra,b,c) B'1_b,b1,x'1,y1 B'2_b1,b2,x'2,y2 B'3_b2,1,x'3,y3
    // x=u_k; y=u_{k-1}; Dk(x,x')= delta_(x, x')Ker_k(x)
    if (kernel) {
        branch_mpo = add_kernel_k_step(branch_mpo, *kernel);
        if (log_step)
            logger->info("Branch MPO after kernel: max_chi = {}", branch_mpo.get_chi());
    }

    // ============================================================
    //              merge on uk
    // ============================================================

    
    auto message = tree._copy_branch(branch_k); // uk
    if (log_step)
        logger->info("Message MPS: max_chi = {}", message.get_chi());

    using RealScalar = typename TensorQuadOperation::TTN<Scalar>::RealScalar;
    const RealScalar mpo_tol = branch_mpo.get_reltol();
    const RealScalar message_tol = message.get_reltol();
    const RealScalar merged_tol = mpo_tol < RealScalar(0) ? message_tol
        : message_tol < RealScalar(0) ? mpo_tol : std::min(mpo_tol, message_tol);
    const int mpo_cap = branch_mpo.get_max_bond_dim();
    const int message_cap = message.get_max_bond_dim();
    const int merged_cap = mpo_cap <= 0 ? std::max(0, message_cap)
        : message_cap <= 0 ? mpo_cap : std::min(mpo_cap, message_cap);

    // If requested, disable truncation while the attachment bonds are open.
    if (!merge_trucation) {
        branch_mpo = TensorQuadOperation::MPO<Scalar>(
            std::as_const(branch_mpo).get_core(), /*max_bond_dim=*/0, /*reltol=*/RealScalar(-1),
            /*w=*/branch_mpo.get_w(), branch_mpo.get_left_phys(), branch_mpo.get_right_phys(), branch_mpo.get_ordering());
        message = TensorQuadOperation::MPS<Scalar>(
            std::as_const(message).get_core(), /*max_bond_dim=*/0, /*reltol=*/RealScalar(-1),
            /*w=*/message.get_w(), message.get_ordering());
    }
    auto merged = branch_mpo._mul(message, method); // Sum over u_k, leaving u_{k-1} open.

    if (log_step)
        logger->info("Merged MPS before post-merge compression: max_chi = {}", merged.get_chi());

    // Close the fused right bond before compression.
    const auto& body = tree.get_TTN_body();
    const auto& topology = body.tree;
    const auto& gap_branch = branches->at(branch_k - 1);
    const auto& message_branch = branches->at(branch_k);
    const int gap_endpoint = gap_branch.nodes.back();
    const int message_endpoint = message_branch.nodes.back();
    const int junction_id = gap_branch.attachment;
    const Eigen::Index gap_dim = std::as_const(branch_mpo).get_core().back().n_cols;
    const Eigen::Index message_dim = std::as_const(message).get_core().back().n_cols;
    auto cores = merged.get_core();
    const auto& last = cores.back(); // Ordering 2: (left, right, physical).
    if (last.n_cols != gap_dim * message_dim)
        throw std::invalid_argument("_integrate_k_branch: invalid fused bond dimension");

    int outgoing = -1;
    xfac_quad::Tensor3D<Scalar> absorbed;
    if (junction_id == message_endpoint && message_branch.attachment == gap_endpoint) {
        // The branches share a bond directly. Trace the two copies of it.
        if (gap_dim != message_dim)
            throw std::invalid_argument("_integrate_k_branch: direct bond dimensions differ");
        absorbed = xfac_quad::Tensor3D<Scalar>(last.n_rows, 1, last.n_slices);
        for (Eigen::Index p = 0; p < last.n_slices; ++p)
            for (Eigen::Index a = 0; a < gap_dim; ++a)
                absorbed.slice(p).col(0) += last.slice_const(p).col(a + message_dim * a);
    } else {
        if (junction_id < 0 || junction_id != message_branch.attachment ||
            !topology.neigh.contains(junction_id) || topology.nodes.contains(junction_id))
            throw std::invalid_argument("_integrate_k_branch: expected a shared junction node");

        const auto& junction = body.M.at(junction_id);
        const auto& neighbors = topology.neigh.at(junction_id);
        const int gap_axis = neighbors.pos(gap_endpoint);
        const int message_axis = neighbors.pos(message_endpoint);
        if (gap_axis < 0 || gap_axis > 2 || message_axis < 0 || message_axis > 2 ||
            gap_axis == message_axis || neighbors.size() > 3)
            throw std::invalid_argument("_integrate_k_branch: invalid junction axes");
        const int outgoing_axis = 3 - gap_axis - message_axis;
        if (static_cast<std::size_t>(outgoing_axis) < neighbors.size())
            outgoing = neighbors.at(outgoing_axis);
        const std::array<Eigen::Index, 3> shape{junction.n_rows, junction.n_cols, junction.n_slices};
        const Eigen::Index outgoing_dim = shape[outgoing_axis];
        if (shape[gap_axis] != gap_dim || shape[message_axis] != message_dim ||
            outgoing_dim <= 0 || (outgoing == -1 && outgoing_dim != 1))
            throw std::invalid_argument("_integrate_k_branch: invalid junction bond dimensions");

        using Matrix = typename xfac_quad::Tensor3D<Scalar>::MatrixX;
        Matrix packed = xfac_quad::cubeToMat_R(junction, outgoing_axis);
        // The unfolding packs the lower axis first; the merged bond needs message fastest.
        if (gap_axis < message_axis) {
            Matrix reordered(packed.rows(), packed.cols());
            for (Eigen::Index s = 0; s < outgoing_dim; ++s) {
                Eigen::Map<const Matrix> src(packed.col(s).data(), gap_dim, message_dim);
                Eigen::Map<Matrix> dst(reordered.col(s).data(), message_dim, gap_dim);
                dst = src.transpose();
            }
            packed.swap(reordered);
        }
        absorbed = xfac_quad::Tensor3D<Scalar>(last.n_rows, outgoing_dim, last.n_slices);
        for (Eigen::Index p = 0; p < last.n_slices; ++p)
            absorbed.slice(p).noalias() = last.slice_const(p) * packed;
    }
    if ((outgoing == -1) != (branches->size() == 2))
        throw std::invalid_argument("_integrate_k_branch: unexpected outgoing bond for this step");

    // Apply this step's quadrature weight and compress again after closing the fused bond.
    for (auto& value : absorbed.data)
        value *= Scalar(h);
    cores.back() = std::move(absorbed);
    merged = TensorQuadOperation::MPS<Scalar>(
        std::move(cores), merged_cap, merged_tol, /*w=*/-1, merged.get_ordering());
    if (merged_cap > 0 || merged_tol >= RealScalar(0)) {
        merged.compress_svd(merged_tol, merged_cap);
        if (log_step)
            logger->info("Merged MPS after SVD compression: max_chi = {}", merged.get_chi());
    }
    if (outgoing == -1) {
        // Store the closed t-MPS as a single leaf-to-endpoint branch.
        xfac_quad::TensorTree<Scalar> result;
        result.M = merged.get_core();
        auto& branch = result.tree.branches.emplace().emplace_back();
        result.tree.neigh[0]; // Also register the isolated one-bit case.
        for (int site = 0; site < merged.get_size(); ++site) {
            result.tree.nodes.insert(site);
            branch.nodes.push_back(site);
            if (site > 0) result.tree.addEdge(site - 1, site);
        }
        // The leaf's first neighbor slot points right, so swap its bond axes.
        if (result.M.size() > 1) {
            const auto& leaf = result.M.front();
            xfac_quad::Tensor3D<Scalar> swapped(leaf.n_cols, leaf.n_rows, leaf.n_slices);
            for (Eigen::Index p = 0; p < leaf.n_slices; ++p)
                swapped.slice(p) = leaf.slice_const(p).transpose();
            result.M.front() = std::move(swapped);
        }
        tree = TensorQuadOperation::TTN<Scalar>(std::move(result),
            tree.get_max_bond_dim(), tree.get_reltol(), /*w=*/-1);
        log_done();
        return;
    }

    // Reuse the gap nodes for the new message. Remove the old message and the
    // absorbed junction, keeping the same neighbor-axis order on surviving nodes.
    std::set<int> removed(message_branch.nodes.begin(), message_branch.nodes.end());
    removed.insert(junction_id);
    std::vector<int> node_map(body.M.size(), -1);
    xfac_quad::TensorTree<Scalar> reduced;
    for (const auto& [node, neighbors] : topology.neigh) {
        if (removed.contains(node)) continue;
        node_map.at(node) = static_cast<int>(reduced.M.size());
        reduced.M.push_back(body.M.at(node));
    }
    const auto remap = [&](int node) {
        if (node == -1) return -1;
        const int new_node = node_map.at(node);
        if (new_node == -1)
            throw std::invalid_argument("_integrate_k_branch: surviving node references a removed node");
        return new_node;
    };
    reduced.tree.root = remap(removed.contains(topology.root) ? gap_endpoint : topology.root);
    for (const auto& [node, neighbors] : topology.neigh) {
        if (removed.contains(node)) continue;
        const int new_node = remap(node);
        if (topology.nodes.contains(node)) reduced.tree.nodes.insert(new_node);
        auto& new_neighbors = reduced.tree.neigh[new_node];
        for (int neighbor : neighbors.from_int()) {
            if (neighbor == junction_id && node == gap_endpoint) neighbor = outgoing;
            else if (neighbor == junction_id && node == outgoing) neighbor = gap_endpoint;
            new_neighbors.push_back(remap(neighbor));
        }
    }
    reduced.tree.branches.emplace();
    for (int i = 0; i < branch_k; ++i) {
        auto branch = branches->at(i);
        if (i == branch_k - 1) branch.attachment = outgoing;
        else if (branch.attachment == junction_id) branch.attachment = gap_endpoint;
        branch.attachment = remap(branch.attachment);
        for (int& node : branch.nodes) node = remap(node);
        reduced.tree.branches->push_back(std::move(branch));
    }

    const auto& merged_cores = std::as_const(merged).get_core();
    const auto& new_branch = reduced.tree.branches->back();
    for (std::size_t i = 0; i < new_branch.nodes.size(); ++i) {
        const int node = new_branch.nodes[i];
        const auto& neighbors = reduced.tree.neigh.at(node);
        const int left_axis = i > 0 ? neighbors.pos(new_branch.nodes[i - 1])
            : 1 - neighbors.pos(new_branch.nodes.size() == 1 ? new_branch.attachment : new_branch.nodes[1]);
        const auto& core = merged_cores[i];
        if (left_axis == 0) {
            reduced.M.at(node) = core;
        } else {
            xfac_quad::Tensor3D<Scalar> swapped(core.n_cols, core.n_rows, core.n_slices);
            for (Eigen::Index p = 0; p < core.n_slices; ++p)
                swapped.slice(p) = core.slice_const(p).transpose();
            reduced.M.at(node) = std::move(swapped);
        }
    }
    tree = TensorQuadOperation::TTN<Scalar>(std::move(reduced),
        tree.get_max_bond_dim(), tree.get_reltol(), /*w=*/-1);
    log_done();
}

template<typename Scalar>
TensorQuadOperation::MPO<Scalar> had_kernel_k_step(TensorQuadOperation::MPO<Scalar> branch)
{
    return branch;
}

} // namespace QuanticsIntegration::simplex_helper
