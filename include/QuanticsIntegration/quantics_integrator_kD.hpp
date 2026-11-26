#pragma once
#include "TensorQuadOperation/ttn_base.hpp"
#include "QuanticsIntegration/quantics_integrator.hpp"
#include "QuanticsIntegration/simplex_helper.hpp"
#include "TensorQuadOperation/generic_construction.hpp"
#include <type_int128.h>
#include <xfac_quad_runner/xf-qd-runner.hpp>
#include <xfac_quad_runner/xf-quad-param.hpp>
#include <Eigen/Dense>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace QuanticsIntegration {

template<typename Scalar, typename Sint = util::i128>
class QuanticsIntegrator_kD {

public:

    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using mGrid = qtgrid::MultQTGrid<RealScalar, Sint>;
    using Grid = qtgrid::QTGrid<RealScalar, Sint>;
    using mState = TensorQuadOperation::TTN<Scalar>;
    using State = TensorQuadOperation::MPS<Scalar>;

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
    // Definite Integration:
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

    // ======================================================================
    // Indefinite Integration:
    // ======================================================================

    /// No Kernel indication this is the class indefinite integration
    /// Construct the ordinary exclusive-prefix integral over all variables.
    /// In 3D: F[p,q,r] = dV * sum(i=0..p-1) sum(j=0..q-1) sum(k=0..r-1) f[i,j,k].
    /// Build one Heaviside kernel per branch, zero at the origin and one afterward.
    /// Delegate volume scaling and saving to make_indefinite_integrator_conv.
    /// Neither input is modified; the returned TTN uses the original value grid.
    static mState make_indefinite_integrator_from_conv(
        const mState& values,
        const mGrid& grid,
        const std::string& save_prefix,
        int max_bond_dim = 0,
        RealScalar reltol = RealScalar(-1),
        bool final_compress = true)
    {
        validate_grid(grid);
        const auto dimensions = static_cast<std::size_t>(grid.get_dim());
        std::vector<TensorQuadOperation::MPS<Scalar>> kernels;
        std::vector<RealScalar> kernel_a, kernel_b;
        kernels.reserve(dimensions);
        kernel_a.reserve(dimensions);
        kernel_b.reserve(dimensions);
        for (std::size_t i = 0; i < dimensions; ++i) {
            const RealScalar length = grid.get_b()[i] - grid.get_a()[i];
            auto [kernel, lag_grid] =
                TensorQuadOperation::analytic_mps_construction::make_exclusive_heaviside_mps<Scalar, Sint>(
                    length, grid.get_nBits()[i]);
            kernels.push_back(std::move(kernel));
            kernel_a.push_back(lag_grid.get_a());
            kernel_b.push_back(lag_grid.get_b());
        }
        const mGrid kernel_grid(std::move(kernel_a), std::move(kernel_b), grid.get_nBits());
        return make_indefinite_integrator_conv(
            values, grid, kernels, kernel_grid, save_prefix,
            max_bond_dim, reltol, final_compress);
    }

    /// Kernel need to be provided. It's the raw convolution operation
    /// Convolve values with one independent 1D MPS kernel per variable.
    /// kernels[i] acts on branch i and uses dimension i of kernel_grid.
    /// The private helper applies the grid-cell volume exactly once.
    /// Save to <save_prefix>_conv_indefinite_int.tt and return the response TTN.
    /// Neither input is modified; kernel-dependent responses are not general
    /// primitives whose differences give arbitrary weighted subinterval integrals.
    static mState make_indefinite_integrator_conv(
        const mState& values,
        const mGrid& grid,
        const std::vector<TensorQuadOperation::MPS<Scalar>>& kernels,
        const mGrid& kernel_grid,
        const std::string& save_prefix,
        int max_bond_dim = 0,
        RealScalar reltol = RealScalar(-1),
        bool final_compress = true)
    {
        if (save_prefix.empty())
            throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integrator_conv: save_prefix must not be empty");

        auto response = make_indefinite_integral_conv(
            values, grid, kernels, kernel_grid, max_bond_dim, reltol, final_compress);
        response.first.save(save_prefix + "_conv_indefinite_int.tt");
        return std::move(response.first);
    }

    /// Evaluate a hypercube using the same x_min and x_max in every variable.
    /// Broadcast the bounds and delegate to eval_rect_indefinite_integrator.
    /// In 3D, writing a=x_min, b=x_max, the corner sum is:
    ///   F(b,b,b) - F(a,b,b) - F(b,a,b) - F(b,b,a)
    ///   + F(a,a,b) + F(a,b,a) + F(b,a,a) - F(a,a,a).
    static Scalar eval_cube_indefinite_integrator(
        const mState& indefinite_values,
        const mGrid& grid,
        const RealScalar& x_min,
        const RealScalar& x_max)
    {
        validate_grid(grid);
        const auto dimensions = static_cast<std::size_t>(grid.get_dim());
        return eval_rect_indefinite_integrator(
            indefinite_values, grid, std::vector<RealScalar>(dimensions, x_min),
            std::vector<RealScalar>(dimensions, x_max));
    }

    /// Evaluate one rectangle, with one lower/upper bound per variable, using
    /// inclusion-exclusion over its 2^d corners. A corner's sign is
    /// (-1)^(number of lower bounds). These vectors are not a batch of queries.
    /// For the exclusive-Heaviside primitive, this integrates the rectangle.
    /// For other kernels, it is only a mixed difference of the convolution response.
    /// Bounds are mapped through the grid; the upper domain endpoint is not stored
    /// in the cumulative TTN and is rejected rather than silently clamped.
    /// The primitive already includes quadrature scaling; do not scale again.
    static Scalar eval_rect_indefinite_integrator(
        const mState& indefinite_values,
        const mGrid& grid,
        const std::vector<RealScalar>& x_min,
        const std::vector<RealScalar>& x_max)
    {
        validate_grid(grid);
        const auto dimensions = static_cast<std::size_t>(grid.get_dim());
        if (x_min.size() != dimensions || x_max.size() != dimensions)
            throw std::invalid_argument("QuanticsIntegrator_kD::eval_rect_indefinite_integrator: expected one lower and upper bound per variable");
        for (std::size_t i = 0; i < dimensions; ++i) {
            if (!(x_min[i] <= x_max[i]))
                throw std::invalid_argument("QuanticsIntegrator_kD::eval_rect_indefinite_integrator: require x_min[i] <= x_max[i]");
            if (!(x_min[i] >= grid.get_a()[i]) || !(x_max[i] < grid.get_b()[i]))
                throw std::out_of_range("QuanticsIntegrator_kD::eval_rect_indefinite_integrator: bounds must lie in each grid interval [a_i,b_i)");
        }

        const auto id_min = grid.coord_to_id(x_min);
        const auto id_max = grid.coord_to_id(x_max);
        const auto bits = static_cast<std::size_t>(grid.get_nBits().front());
        for (std::size_t i = 0; i < dimensions; ++i) {
            const auto offset = i * bits;
            if (std::equal(id_min.begin() + offset, id_min.begin() + offset + bits,
                           id_max.begin() + offset))
                return Scalar(0);
        }
        auto id = id_min;

        // Take one upper-minus-lower difference per variable. Expanding these
        // differences evaluates every mixed corner with its inclusion-exclusion sign.
        const auto evaluate_corners = [&](auto&& self, std::size_t dimension) -> Scalar {
            if (dimension == dimensions) return indefinite_values.eval(id);
            const auto offset = dimension * bits;
            std::copy_n(id_min.begin() + offset, bits, id.begin() + offset);
            const Scalar lower = self(self, dimension + 1);
            std::copy_n(id_max.begin() + offset, bits, id.begin() + offset);
            const Scalar upper = self(self, dimension + 1);
            return upper - lower;
        };
        return evaluate_corners(evaluate_corners, 0);
    }

    // ======================================================================
    // Simplex Integration:
    // ======================================================================
    
    // Some assumption are needed:
    // The function provided describe: f(v1, v2, ..., vN)
    // each variable is located on a single branch from left to right.
    // Each variable are described on the exact same grid ([0, b[, nbit)
    // Integration domain:
    //      0 <= u_N <= u_(N-1) <= ... <= u_1 <= t
    // v1 = t-u1, v2 = u1-u2, ..., vk = u(k-1)-uk, ..., vN = u(N-1)-uN
    // Q_n(t) = \int_0^t du1 \int_0^{u_1} du_2 ... \int_0^{u_{N-1}} du_N f(v1, v2, ..., vN)

    /// given a mState and mGrid
    /// verify condition
    /// perform SimplexIntegration with or without (kernels,kernel_grid) (optional)
    /// We thus construct an mps with the tree convention.
    /// convert to MPS and return it with grid (save it with grid if path given)
    /// Multi-branch weights are not supported yet; a single branch may use a kernel.
    /// The returned MPS already includes quadrature scaling. If save_prefix is
    /// nonempty, save <save_prefix>.tt and the time grid JSON files.
    /// If path_logger is nonempty, log the tree and bond dimensions before and
    /// after each step to path_logger + "logger_integration_simplex.log".
    /// Each entry is timestamped and flushed. Use distinct prefixes for concurrent calls.
    /// method selects the initialization, subtractor, and merge contractions.
    /// By default, the merge backend can truncate before the junction is absorbed.
    /// Set merge_trucation to false to defer truncation until after junction absorption.
    static std::pair<State, Grid> make_integration_simplex(
        const mState& values,
        const mGrid& grid,
        const std::vector<TensorQuadOperation::MPS<Scalar>>& kernels,
        const mGrid& kernel_grid,
        const std::string& save_prefix,
        int max_bond_dim = 0,
        RealScalar reltol = RealScalar(-1),
        bool final_compress = true,
        const std::string& path_logger = {},
        const std::string& method = "zip-up",
        bool merge_trucation = true){

            // parameter validation
            simplex_helper::validation::validation_simplex_parameter(values, grid, kernels, kernel_grid);
            if (!kernels.empty() && grid.get_dim() > 1)
                throw std::invalid_argument("QuanticsIntegrator_kD::make_integration_simplex: multi-branch kernels are not supported yet");

            mState working_tree(values.get_TTN_body(), max_bond_dim, reltol, /*w=*/-1);

            // Keep the logger private to this call, without registering it globally.
            std::shared_ptr<spdlog::logger> simplex_logger;
            if (!path_logger.empty()) {
                const std::string filename = path_logger + "logger_integration_simplex.log";
                auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, /*truncate=*/true);
                simplex_logger = std::make_shared<spdlog::logger>(filename, std::move(sink));
                simplex_logger->set_level(spdlog::level::info);
                simplex_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
                simplex_logger->flush_on(spdlog::level::info);
            }
            const auto log_tree = [&](const char* phase, const char* step, int branch_id) {
                if (!simplex_logger || !simplex_logger->should_log(spdlog::level::info))
                    return;
                std::ostringstream diagram;
                working_tree.get_TTN_body().tree.print(diagram);
                const auto chi = working_tree.compute_list_chi();
                diagram << "max_chi: " << (chi.empty() ? 0 : *std::max_element(chi.begin(), chi.end())) << '\n';
                diagram << "list_chi: [";
                for (std::size_t i = 0; i < chi.size(); ++i) {
                    if (i > 0) diagram << ", ";
                    diagram << chi[i];
                }
                diagram << ']';
                simplex_logger->info("Simplex: {} {} (branch {}):\n{}", phase, step, branch_id, diagram.str());
            };

            // first step(last branch): from (v_1, ..., v_{N-1}, v_N) to (v_1, ..., v_{N-1}, u_{N-1}):
            const int last_branch = grid.get_dim() - 1;
            log_tree("before", "_init_integrate_branch", last_branch);
            simplex_helper::_init_integrate_branch(
                working_tree, grid, kernels.empty() ? nullptr : &kernels.back(), method);
            log_tree("after", "_init_integrate_branch", last_branch);

            // other branch: from (v_1, ..., v_{k-1}, v_k, u_k) to (v_1, ..., v_{k-1}, u_{k-1})
            for (int branch_k = last_branch; branch_k > 0; --branch_k)
            {
                log_tree("before", "_integrate_k_branch", branch_k);
                simplex_helper::_integrate_k_branch(
                    working_tree, grid, branch_k, /*kernel=*/nullptr, method, simplex_logger.get(), merge_trucation);
                log_tree("after", "_integrate_k_branch", branch_k);
            }
                
            // convert 1 branch tree to MPS
            auto result = working_tree._copy_branch(0);
            if (final_compress)
                result.compress_svd(reltol, max_bond_dim);

            Grid time_grid(grid.get_a().front(), grid.get_b().front(), grid.get_nBits().front());
            if (!save_prefix.empty()) {
                result.save(save_prefix);
                time_grid.save_json(save_prefix);
            }
            return {std::move(result), std::move(time_grid)};
        }

    /// Evaluate an already constructed simplex-integral MPS at t in [a, b).
    /// The grid maps t to a stored sample; the MPS already includes quadrature scaling.
    [[nodiscard]] static Scalar integrate_simplex(
        const State& simplex_values, const Grid& grid, const RealScalar& t)
    {
        simplex_helper::validation::validate_simplex_mps(
            simplex_values, grid.get_nBits(), "QuanticsIntegrator_kD::integrate_simplex");
        return simplex_values.eval(simplex_sample_id(grid, t));
    }

    /// Evaluate the same MPS in parallel at several times, preserving input order.
    [[nodiscard]] static std::vector<Scalar> integrate_simplex(
        const State& simplex_values, const Grid& grid,
        const std::vector<RealScalar>& times)
    {
        simplex_helper::validation::validate_simplex_mps(
            simplex_values, grid.get_nBits(), "QuanticsIntegrator_kD::integrate_simplex");
        // Validate and convert every time before eval_list enters its OpenMP loop.
        std::vector<std::vector<int>> ids;
        ids.reserve(times.size());
        for (const auto& t : times)
            ids.push_back(simplex_sample_id(grid, t));
        return simplex_values.eval_list(ids);
    }

private:
    // ======================================================================
    // Helpers
    // ======================================================================

    static std::vector<int> simplex_sample_id(const Grid& grid, const RealScalar& t)
    {
        if (!(t >= grid.get_a() && t < grid.get_b()))
            throw std::out_of_range("QuanticsIntegrator_kD::integrate_simplex: t must lie in [a,b)");
        return grid.coord_to_id(t);
    }

    /// Compute the separable-kernel convolution response and scale once by dV.
    /// Each lag grid must be [0, b_i-a_i), with matching bits and spacing.
    /// Require kernels[i](0) == 0 to exclude equal-index samples in every variable.
    /// Return {response_ttn, value_grid}; neither input is modified.
    static std::pair<mState, mGrid> make_indefinite_integral_conv(
        const mState& values,
        const mGrid& value_grid,
        const std::vector<TensorQuadOperation::MPS<Scalar>>& kernels,
        const mGrid& kernel_grid,
        int max_bond_dim = 0,
        RealScalar reltol = RealScalar(-1),
        bool final_compress = true)
    {
        validate_grid(value_grid);
        validate_grid(kernel_grid);
        if (kernel_grid.get_dim() != value_grid.get_dim())
            throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: grid dimensions differ");
        if (kernels.size() != static_cast<std::size_t>(value_grid.get_dim()))
            throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: expected one kernel per variable");

        for (std::size_t i = 0; i < kernels.size(); ++i) {
            if (!(value_grid.get_dx()[i] > RealScalar(0)))
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: grid spacing must be positive");
            if (kernel_grid.get_a()[i] != RealScalar(0))
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: each kernel grid must start at zero");
            if (kernel_grid.get_b()[i] != value_grid.get_b()[i] - value_grid.get_a()[i])
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: each kernel grid must end at b_i-a_i");
            if (kernel_grid.get_nBits()[i] != value_grid.get_nBits()[i])
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: grid bit counts differ");
            if (kernel_grid.get_dx()[i] != value_grid.get_dx()[i])
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: grid spacings differ");

            // Validate before eval(), which otherwise assumes compatible binary cores.
            const auto& cores = kernels[i].get_core();
            if (cores.size() != static_cast<std::size_t>(value_grid.get_nBits()[i]))
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: kernel length differs from grid bit count");
            if (cores.front().n_rows != 1 || cores.back().n_slices != 1)
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: kernel boundary bonds must have dimension one");
            for (std::size_t site = 0; site < cores.size(); ++site) {
                const auto& core = cores[site];
                if (core.n_cols != 2 || core.n_rows <= 0 || core.n_slices <= 0)
                    throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: invalid binary kernel core dimensions");
                if (site > 0 && cores[site - 1].n_slices != core.n_rows)
                    throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: kernel internal bond mismatch");
            }
            const std::vector<int> zero_bits(cores.size(), 0);
            if (kernels[i].eval(zero_bits) != Scalar(0))
                throw std::invalid_argument("QuanticsIntegrator_kD::make_indefinite_integral_conv: each kernel must be zero at the origin");
        }

        // TTN::conv validates signal branches against the kernel lengths and applies
        // the requested contraction/compression settings, but no quadrature weights.
        auto response = values.conv(kernels, max_bond_dim, reltol, final_compress);
        response *= Scalar(value_grid.delta_volume());
        return {std::move(response), value_grid};
    }

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

} // namespace QuanticsIntegration

