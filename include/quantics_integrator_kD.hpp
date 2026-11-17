#pragma once
#include "ttn_base.hpp"
#include "quantics_integrator.hpp"
#include <type_int128.h>
#include <xfac_quad_runner/xf-qd-runner.hpp>
#include <xfac_quad_runner/xf-quad-param.hpp>
#include <Eigen/Dense>
#include <spdlog/logger.h>
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
        // some validation of inputs
        if (!function)
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: function is empty");
        if (save_prefix.empty())
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: save_prefix is empty");
        if (grid.get_dim() <= 0)
            throw std::invalid_argument("QuanticsIntegrator_kD::fit: grid must have at least one dimension");
        for (int bits : grid.get_nBits())
            if (bits <= 0)
                throw std::invalid_argument("QuanticsIntegrator_kD::fit: each dimension needs at least one bit");
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

        // generate tree_topology from grid
        auto tree = make_tree(grid);
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

private:
    // ======================================================================
    // Helpers
    // ======================================================================

    static xfac_quad::TopologyTree make_tree(const mGrid& grid)
    {
        const int dimensions = grid.get_dim();
        const int physical_count = grid.get_tensorLen();
        // Reuse the Tucker junction structure, extending each endpoint into its bit chain.
        const auto skeleton = xfac_quad::makeTuckerTree(dimensions, 1);
        xfac_quad::TopologyTree tree(0);
        tree.branches.emplace(dimensions);
        std::vector<int> endpoints;
        int offset = 0;
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            auto& branch = tree.branches->at(dimension);
            for (int bit = 0; bit < grid.get_nBits().at(dimension); ++bit) {
                const int node = offset + bit;
                tree.nodes.insert(node);
                tree.neigh[node]; // Also register the isolated one-site case.
                branch.nodes.push_back(node);
                if (bit > 0) tree.addEdge(node - 1, node);
            }
            endpoints.push_back(branch.nodes.back());
            offset += grid.get_nBits().at(dimension);
        }
        const auto mapped_node = [&](int node) {
            return node < dimensions ? endpoints.at(node) : physical_count + node - dimensions;
        };
        for (const auto& [node, neighbors] : skeleton.neigh)
            for (int neighbor : neighbors.from_int())
                if (node < neighbor)
                    tree.addEdge(mapped_node(node), mapped_node(neighbor));
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            const int attachment = skeleton.branches->at(dimension).attachment;
            if (attachment != -1)
                tree.branches->at(dimension).attachment = mapped_node(attachment);
        }
        return tree;
    }

    void validate() const
    {
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
