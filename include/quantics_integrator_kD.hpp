#pragma once
#include "ttn_base.hpp"
#include "quantics_integrator.hpp"
#include <type_int128.h>
#include <xfac_quad_runner/xf-qd-runner.hpp>
#include <xfac_quad_runner/xf-quad-param.hpp>
#include <Eigen/Dense>
#include <spdlog/logger.h>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


template<typename Scalar, typename Sint = util::i128>
class QuanticsIntegrator_kD {

public:

    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using mGrid = qtgrid::MultQTGrid<RealScalar, Sint>;
    using mState = TTN<Scalar>;

    /// Half-open rectangular grid-index range: [begin[i], end[i]) in variable i.
    struct IndexRange {
        std::vector<Sint> begin;
        std::vector<Sint> end;
    };

    struct FitOptions {
        std::vector<std::vector<RealScalar>> additional_pivots;
        bool verbose = false;
        int nb_point_res = 1000;
    };

    // ======================================================================
    // Constructors
    // ======================================================================

    // Grid dimension i corresponds to branch i; branch.nodes gives its bit order.
    QuanticsIntegrator_kD(mGrid grid, mState values)
        : grid_(std::move(grid)), values_(std::move(values))
    {
        validate();
    }

    // Reload the grid and tree, leaving the canonical center undefined until needed.
    QuanticsIntegrator_kD(const std::string& grid_file, const std::string& tree_file)
        : QuanticsIntegrator_kD(
              mGrid(grid_file),
              mState(tree_file, /*max_bond_dim=*/0, /*reltol=*/RealScalar(-1), /*w=*/-1))
    {}

    /// Run TTCI_runner, save its result, then reload the saved grid and tree.
    static QuanticsIntegrator_kD fit(
        mGrid grid,
        std::function<Scalar(const std::vector<RealScalar>&)> function,
        xfac_quad_runner::ttci_runner_param<Scalar> parameters,
        const std::string& save_prefix,
        FitOptions options = {},
        std::shared_ptr<spdlog::logger> logger = nullptr
    ){
        // Validate the grid before fitting or creating output files.
        validate_grid(grid);
        if (!function)
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: function is empty");
        if (save_prefix.empty())
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: save_prefix is empty");
        if (options.nb_point_res <= 0)
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: nb_point_res must be positive");
        if (parameters.nb_iter < 0 || parameters.bondDim <= 0)
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: invalid iteration count or bond dimension");
        if (!parameters.pivot1.empty() &&
            parameters.pivot1.size() != static_cast<std::size_t>(grid.get_tensorLen()))
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: initial pivot length differs from grid");
        for (int bit : parameters.pivot1)
            if (bit != 0 && bit != 1)
                throw std::invalid_argument("QuanticsIntegrator_kD::fit: initial pivot entries must be binary");
        for (const auto& pivot : options.additional_pivots)
            (void)grid.coord_to_id(pivot);

        // Contiguous Tucker node blocks match MultQTGrid's grouped bit ordering.
        auto tree = xfac_quad::makeTuckerTree(grid.get_dim(), grid.get_nBits().front());
        const int max_bond_dim = parameters.bondDim;
        const RealScalar reltol = parameters.reltol;
        // call TTCI_runner to fit the function on the grid and save the result
        // save grid and tree to files
        if (grid.get_tensorLen() == 1) {
            // TensorTreeCI requires a bond to sweep. A single binary site is sampled exactly.
            xfac_quad::TensorTree<Scalar> body(tree);
            body.M.at(0) = xfac_quad::Tensor3D<Scalar>(1, 1, 2);
            for (int bit = 0; bit < 2; ++bit)
                body.M.at(0)(0, 0, bit) = function(grid.id_to_coord({bit}));
            body.save(save_prefix + ".tt");
            grid.save_json(save_prefix);
        } else {
            xfac_quad_runner::TTCI_runner<Scalar, Sint> runner(
                std::move(grid), std::move(tree), std::move(parameters),
                std::move(function), std::move(logger));
            runner.fit(options.additional_pivots, options.verbose, /*do_save=*/true,
                       save_prefix, options.nb_point_res);
        }
        // reload the saved grid and tree and return QuanticsIntegrator_kD instance
        return QuanticsIntegrator_kD(
            mGrid(save_prefix + "_multgrid_E.json"),
            mState(save_prefix + ".tt", max_bond_dim, reltol, /*w=*/-1));
    }

    // ======================================================================
    // methods
    // ======================================================================

    /// Integrate over the full grid by summing all physical legs and scaling
    /// by the grid-cell volume (the product of spacings over all variables).
    [[nodiscard]] Scalar integrate() const
    {
        return values_.get_TTN_body().sum() * Scalar(grid_.delta_volume());
    }

    /// Integrate a rectangular index range without modifying or truncating the TTN.
    [[nodiscard]] Scalar integrate(const IndexRange& range) const
    {
        validate_range(range);
        bool empty = false, full = true;
        for (std::size_t i = 0; i < range.begin.size(); ++i) {
            empty = empty || range.begin[i] == range.end[i];
            full = full && range.begin[i] == Sint(0) && range.end[i] == grid_.get_N()[i];
        }
        if (empty) return Scalar(0);
        if (full) return integrate();

        const auto& body = values_.get_TTN_body();
        xfac_quad::TensorTree<Scalar> contracted(body.tree);
        for (const auto& [node, neighbors] : body.tree.neigh)
            if (!body.tree.nodes.contains(node)) contracted.M.at(node) = body.M.at(node);

        // Collapse each masked branch to its attachment-bond vector. Unit tensors
        // retain the original topology, including physical roots and the two-variable cut.
        for (std::size_t i = 0; i < range.begin.size(); ++i) {
            const auto& branch = body.tree.branches->at(i);
            for (int node : branch.nodes) {
                contracted.M.at(node) = xfac_quad::Tensor3D<Scalar>(1, 1, 1);
                contracted.M.at(node).data[0] = Scalar(1);
            }
            contracted.M.at(branch.nodes.back()) =
                contract_branch_range(i, range.begin[i], range.end[i]);
        }
        return contracted.sum() * Scalar(grid_.delta_volume());
    }

private:
    // ======================================================================
    // Helpers
    // ======================================================================

    void validate_range(const IndexRange& range) const
    {
        const auto dimensions = static_cast<std::size_t>(grid_.get_dim());
        if (range.begin.size() != dimensions || range.end.size() != dimensions)
            throw std::invalid_argument("QuanticsIntegrator_kD: index range needs one begin and end per variable");
        for (std::size_t i = 0; i < dimensions; ++i)
            if (range.begin[i] < Sint(0) || range.end[i] < range.begin[i] ||
                range.end[i] > grid_.get_N()[i])
                throw std::out_of_range("QuanticsIntegrator_kD: index range must lie in [0, grid.N] for every variable");
    }

    /// Contract an interval indicator along a leaf-to-attachment branch.
    /// As in the 1D Heaviside mask, comparison states are EQ, GT, and LT.
    /// With LSB-first bits, each higher unequal bit overrides the lower-bit
    /// comparison. Tracking both boundaries avoids subtraction of prefix sums.
    xfac_quad::Tensor3D<Scalar> contract_branch_range(
        std::size_t dimension, Sint begin, Sint end) const
    {
        using Row = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
        constexpr int equal = 0, greater = 1, less = 2;
        const auto compare = [=](int previous, int bit, int boundary_bit) {
            return bit == boundary_bit ? previous : bit > boundary_bit ? greater : less;
        };
        const bool full_upper = end == grid_.get_N().at(dimension);
        std::array<Row, 9> environment;
        for (auto& row : environment) row = Row::Zero(1);
        std::array<bool, 9> active{};
        const int initial = full_upper ? less : equal;
        environment[initial](0) = Scalar(1);
        active[initial] = true;

        const auto& body = values_.get_TTN_body();
        const auto& branch = body.tree.branches->at(dimension);
        int endpoint_axis = 1;
        for (std::size_t site = 0; site < branch.nodes.size(); ++site) {
            const int node = branch.nodes[site];
            const int previous = site == 0 ? -1 : branch.nodes[site - 1];
            const int next = site + 1 < branch.nodes.size() ? branch.nodes[site + 1] : branch.attachment;
            const auto& neighbors = body.tree.neigh.at(node);
            for (int neighbor : neighbors.from_int())
                if (neighbor != -1 && neighbor != previous && neighbor != next)
                    throw std::invalid_argument("QuanticsIntegrator_kD: branch has an unexpected external connection");
            int left = previous == -1 ? -1 : neighbors.pos(previous);
            int right = next == -1 ? -1 : neighbors.pos(next);
            if ((previous != -1 && (left < 0 || left > 1)) ||
                (next != -1 && (right < 0 || right > 1)))
                throw std::invalid_argument("QuanticsIntegrator_kD: invalid branch bond axis");
            if (left == -1) left = right == -1 ? 0 : 1 - right;
            if (right == -1) right = 1 - left;
            if (left == right)
                throw std::invalid_argument("QuanticsIntegrator_kD: invalid branch ordering");

            const auto& core = body.M.at(node);
            const std::array<Eigen::Index, 2> shape{core.n_rows, core.n_cols};
            if (shape[left] != environment[0].size() || shape[right] <= 0 ||
                (next == -1 && shape[right] != 1))
                throw std::invalid_argument("QuanticsIntegrator_kD: branch bond dimension mismatch");
            if (next != -1) {
                const int reciprocal = body.tree.neigh.at(next).pos(node);
                const auto& neighbor = body.M.at(next);
                const std::array<Eigen::Index, 3> neighbor_shape{
                    neighbor.n_rows, neighbor.n_cols, neighbor.n_slices};
                if (reciprocal < 0 || reciprocal > 2 || shape[right] != neighbor_shape[reciprocal])
                    throw std::invalid_argument("QuanticsIntegrator_kD: attachment bond dimension mismatch");
            }

            std::array<Row, 9> updated;
            for (auto& row : updated) row = Row::Zero(shape[right]);
            std::array<bool, 9> next_active{};
            const int lower_bit = static_cast<int>((begin >> site) & Sint(1));
            const int upper_bit = static_cast<int>((end >> site) & Sint(1));
            for (int state = 0; state < 9; ++state) {
                if (!active[state]) continue;
                for (int bit = 0; bit < 2; ++bit) {
                    const int lower = compare(state / 3, bit, lower_bit);
                    const int upper = full_upper ? less : compare(state % 3, bit, upper_bit);
                    const int target = 3 * lower + upper;
                    // Transpose is an axis permutation, never complex conjugation.
                    if (left == 0)
                        updated[target].noalias() += environment[state] * core.slice_const(bit);
                    else
                        updated[target].noalias() += environment[state] * core.slice_const(bit).transpose();
                    next_active[target] = true;
                }
            }
            environment = std::move(updated);
            active = next_active;
            endpoint_axis = right;
        }

        // Accept begin <= index < end, leaving only the bond to the exterior.
        const Row message = environment[3 * equal + less] + environment[3 * greater + less];
        const auto rank = message.size();
        xfac_quad::Tensor3D<Scalar> result(endpoint_axis == 0 ? rank : 1,
                                          endpoint_axis == 1 ? rank : 1, 1);
        for (Eigen::Index bond = 0; bond < rank; ++bond)
            if (endpoint_axis == 0) result(bond, 0, 0) = message(bond);
            else result(0, bond, 0) = message(bond);
        return result;
    }

    static void validate_grid(const mGrid& grid)
    {
        const auto& bit_counts = grid.get_nBits();
        if (grid.get_dim() <= 0 || bit_counts.empty())
            throw std::invalid_argument("QuanticsIntegrator_kD: grid must have at least one dimension");
        for (int bits : bit_counts) {
            if (bits <= 0)
                throw std::invalid_argument("QuanticsIntegrator_kD: each variable needs at least one bit");
            if (bits != bit_counts.front())
                throw std::invalid_argument("QuanticsIntegrator_kD: all grid variables must have the same number of bits");
        }
    }

    void validate() const
    {
        validate_grid(grid_);
        const auto& body = values_.get_TTN_body();
        const auto& tree = body.tree;
        if (grid_.get_dim() <= 0 || !tree.branches ||
            tree.branches->size() != static_cast<std::size_t>(grid_.get_dim()))
            throw std::invalid_argument(
                "QuanticsIntegrator_kD: requires one tree branch per grid dimension");

        std::set<int> physical_nodes;
        for (std::size_t dimension = 0; dimension < tree.branches->size(); ++dimension) {
            const auto& branch = tree.branches->at(dimension);
            if (branch.nodes.empty() ||
                branch.nodes.size() != static_cast<std::size_t>(grid_.get_nBits().at(dimension)))
                throw std::invalid_argument(
                    "QuanticsIntegrator_kD: branch length differs from grid bit count");
            for (int node : branch.nodes) {
                if (!tree.nodes.contains(node) || !physical_nodes.insert(node).second)
                    throw std::invalid_argument(
                        "QuanticsIntegrator_kD: branches must contain distinct physical nodes");
                if (body.M.at(node).n_slices != 2)
                    throw std::invalid_argument(
                        "QuanticsIntegrator_kD: every physical dimension must be two");
            }
        }
        if (physical_nodes != tree.nodes)
            throw std::invalid_argument(
                "QuanticsIntegrator_kD: branches must cover all physical nodes");
    }

    // ======================================================================
    // Data members
    // ======================================================================
    mGrid grid_;
    mState values_;
};
