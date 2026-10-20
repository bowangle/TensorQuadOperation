#pragma once

#include "indefinite_integral.hpp"
#include "mpo_base.h"

#include <type_int128.h>
#include <xf-qd-runner.hpp>
#include <xf-quad-param.hpp>

#include <Eigen/Dense>
#include <spdlog/logger.h>

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// Parameters required by QuanticsIntegrator1D::fit():
///
/// 1. Grid grid
///      QTGrid<RealScalar, Sint>(a, b, nBits), defining [a,b).
/// 2. Function function
///      std::function<Scalar(RealScalar)> evaluated on the grid.
/// 3. RunnerParam parameters
///      TCI2_1D_runner_param<Scalar>, containing nBits, nb_iter, bondDim,
///      and all TCI2_1D_runner_opts settings.
/// 4. std::string save_prefix
///      Prefix used for the generated TT, grid, and error files.
/// 5. FitOptions options (optional)
///      Additional pivots, verbosity, error-sampling count, and known
///      discontinuities.
/// 6. std::shared_ptr<spdlog::logger> logger (optional)
///      Logger forwarded to TCI2_1D_Runner for fit diagnostics.
/// 7. std::optional<std::string> indefinite_integral_method (optional)
///      If provided, construct and save the indefinite-integral MPS using this
///      MPO-MPS contraction method.
///
/// After fitting, call integrate() for the full domain or
/// integrate({begin, end}) for a half-open grid-index range.
///
/// One-dimensional quantics integration on a uniform QTGrid.
///
/// The MPS stores sampled function values, with site k representing bit k of
/// the grid index (QTGrid uses least-significant bit first). Integration sums
/// the physical legs and multiplies the result by the grid cell volume.
template<typename Scalar, typename Sint = util::i128>
class QuanticsIntegrator1D {
public:
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Grid        = QTGrid<RealScalar, Sint>;
    using State       = MPS<Scalar>;
    using Function    = std::function<Scalar(RealScalar)>;
    using RunnerParam = TCI2_1D_runner_param<Scalar>;
    using LoggerPtr   = std::shared_ptr<spdlog::logger>;
    using MatrixX     = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    /// Half-open grid-index range [begin, end).
    struct IndexRange {
        Sint begin;
        Sint end;
    };

    struct FitOptions {
        std::vector<RealScalar> additional_pivots;
        bool verbose = false;
        int nb_point_res = 1000;
        std::vector<RealScalar> discontinuities;
    };

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    QuanticsIntegrator1D(Grid grid, State values)
        : grid_(std::move(grid)), values_(std::move(values))
    {
        validate();
    }

    /// Reload a grid and MPS previously saved by TCI2_1D_Runner::fit().
    static QuanticsIntegrator1D from_files(const std::string& grid_file, const std::string& mps_file)
    {
        Grid grid(grid_file);
        State values(mps_file, /*max_bond_dim=*/0, /*reltol=*/-1, /*working_site=*/-1);
        return QuanticsIntegrator1D(std::move(grid), std::move(values));
    }

    /// Run TCI2_1D_Runner, save its result, then reload the saved grid and MPS.
    ///
    /// `save_prefix` is passed directly to the runner. In particular, the
    /// runner writes `<save_prefix>.tt` and `<save_prefix>_grid.json`.
    static QuanticsIntegrator1D fit(Grid grid,
                                    Function function,
                                    RunnerParam parameters,
                                    const std::string& save_prefix,
                                    FitOptions options = {},
                                    LoggerPtr logger = nullptr,
                                    std::optional<std::string> indefinite_integral_method = std::nullopt)
    {
        if (!function)
            throw std::invalid_argument(
                "QuanticsIntegrator1D::fit: function is empty");
        if (save_prefix.empty())
            throw std::invalid_argument(
                "QuanticsIntegrator1D::fit: save_prefix is empty");
        if (parameters.nBit != grid.get_nBits())
            throw std::invalid_argument(
                "QuanticsIntegrator1D::fit: grid and runner parameter nBit differ");
        if (options.nb_point_res <= 0)
            throw std::invalid_argument(
                "QuanticsIntegrator1D::fit: nb_point_res must be positive");

        // The runner's file format stores complex cores. MPS loading currently
        // supports that same complex format.
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex == 0) {
            throw std::invalid_argument(
                "QuanticsIntegrator1D::fit: the runner save/reload path "
                "requires a complex Scalar");
        } else {
            const RealScalar reltol = parameters.reltol;
            const int max_bond_dim = parameters.bondDim;
            TCI2_1D_Runner<Scalar, Sint> runner(
                std::move(grid), std::move(parameters), std::move(function),
                std::move(logger));

            runner.fit(options.additional_pivots,
                       options.verbose,
                       /*do_save=*/true,
                       save_prefix,
                       options.nb_point_res,
                       options.discontinuities);

            Grid saved_grid(save_prefix + "_grid.json");
            State saved_values(save_prefix + ".tt", max_bond_dim, reltol, /*working_site=*/-1);

            if (indefinite_integral_method) {
                make_indefinite_integrator(saved_values, saved_grid, save_prefix, max_bond_dim, reltol, *indefinite_integral_method);
            }

            return QuanticsIntegrator1D(std::move(saved_grid), std::move(saved_values));
        }
    }

    
    // -----------------------------------------------------------------
    // Indefinite integral:
    // -----------------------------------------------------------------
    
    static State make_indefinite_integrator(
        State& values,
        Grid& grid,
        const std::string& save_prefix,
        int max_bond_dim=0,
        RealScalar reltol=-1,
        const std::string& method = "zip-up")
    {
        State mps_indefinite_int = make_indefinite_integral(values, max_bond_dim, reltol, method);
        mps_indefinite_int *= Scalar(grid.delta_volume());
        mps_indefinite_int.save(save_prefix + "_indefinite_int");
        return mps_indefinite_int;
    }

    /// Build the integrated convolution response using values/grid and
    /// kernel/kernel_grid. The helper validates the grids and kernel(0) == 0
    /// and already applies the grid spacing; do not multiply by it again.
    /// Saves <save_prefix>_indefinite_int.tt, replacing that file if present,
    /// and returns the response MPS on grid. Neither input is modified.
    /// Evaluate this response at the observation time; differences are not
    /// general weighted subinterval integrals. Uses MPS::conv's zip-up path.
    static State make_indefinite_integrator_conv(
        const State& values,
        const Grid& grid,
        const State& kernel,
        const Grid& kernel_grid,
        const std::string& save_prefix,
        int max_bond_dim = 0,
        RealScalar reltol = RealScalar(-1),
        bool final_compress = true)
    {
        if (save_prefix.empty()) {
            throw std::invalid_argument(
                "make_indefinite_integrator_conv: save_prefix must not be empty");
        }

        auto response = make_indefinite_integral_conv(
            values, grid, kernel, kernel_grid,
            max_bond_dim, reltol, final_compress);
        response.first.save(save_prefix + "_conv_indefinite_int");
        return std::move(response.first);
    }

    static Scalar integrate_indefinite(
        const State& indefinite_values,
        const Grid& grid,
        const RealScalar& x_min,
        const RealScalar& x_max)
    {
        return integrate_indefinite_method(indefinite_values, grid, x_min, x_max);
    }

    static std::vector<Scalar> integrate_indefinite(
        const State& indefinite_values,
        const Grid& grid,
        const std::vector<RealScalar>& x_min,
        const std::vector<RealScalar>& x_max)
    {
        return integrate_indefinite_method(indefinite_values, grid, x_min, x_max);
    }

    // -----------------------------------------------------------------
    // Integration and raw sums
    // -----------------------------------------------------------------

    [[nodiscard]] Scalar integrate() const
    {
        return sum() * Scalar(grid_.delta_volume());
    }

    [[nodiscard]] Scalar integrate(IndexRange range) const
    {
        return sum_v2(range) * Scalar(grid_.delta_volume());
    }

    /// Integrate several half-open index ranges with the batched direct-sum
    /// implementation. Results preserve the input range order.
    [[nodiscard]] std::vector<Scalar> integrate_lrange(const std::vector<IndexRange>& ranges) const
    {
        std::vector<Scalar> results = sum_batch_v2(ranges);
        const Scalar cell_volume = Scalar(grid_.delta_volume());
        for (Scalar& result : results)
            result *= cell_volume;
        return results;
    }

    [[nodiscard]] Scalar sum() const
    {
        return contract_full(values_);
    }

    /// Full-grid counterpart of the direct range-sum API. No binary boundary
    /// processing is needed when every physical index is included.
    [[nodiscard]] Scalar sum_v2() const
    {
        return contract_full(values_);
    }

    [[nodiscard]] Scalar sum(IndexRange range) const
    {
        validate_range(range);
        if (range.begin == range.end)
            return Scalar(0);
        if (range.begin == Sint(0) && range.end == grid_.get_N())
            return sum();
        return contract_full(restrict_to(range));
    }

    /// Sum a range directly from its binary boundaries, without constructing
    /// an interval MPS or applying an MPO. Prefixes and suffixes are contracted
    /// directly; an interior range is the difference of two prefix sums.
    /// Complexity is O(nBits * chi^2).
    [[nodiscard]] Scalar sum_v2(IndexRange range) const
    {
        validate_range(range);
        if (range.begin == range.end)
            return Scalar(0);

        const auto free_left = make_free_left_environments();
        if (range.begin == Sint(0))
            return prefix_sum_v2(range.end, free_left);
        if (range.end == grid_.get_N())
            return suffix_sum_v2(range.begin, free_left);
        return prefix_sum_v2(range.end, free_left) - prefix_sum_v2(range.begin, free_left);
    }

    /// Evaluate several range sums together. Boundary environments are stored
    /// as matrix columns, so every tensor site is handled with matrix-matrix
    /// products instead of one matrix-vector product per range. Prefixes and
    /// suffixes are batched separately; interior ranges use two batched prefix
    /// boundaries. Results preserve the input range order.
    [[nodiscard]] std::vector<Scalar> sum_batch_v2(const std::vector<IndexRange>& ranges) const
    {
        std::vector<Scalar> results(ranges.size(), Scalar(0));
        if (ranges.empty())
            return results;

        struct BoundaryUse {
            std::size_t result_index;
            Scalar coefficient;
        };

        std::vector<Sint> prefix_boundaries;
        std::vector<BoundaryUse> prefix_uses;
        std::vector<Sint> suffix_boundaries;
        std::vector<std::size_t> suffix_result_indices;
        bool need_full_sum = false;

        for (std::size_t index = 0; index < ranges.size(); ++index) {
            const IndexRange range = ranges[index];
            validate_range(range);
            if (range.begin == range.end)
                continue;
            if (range.begin == Sint(0) && range.end == grid_.get_N()) {
                need_full_sum = true;
                continue;
            }
            if (range.begin == Sint(0)) {
                prefix_boundaries.push_back(range.end);
                prefix_uses.push_back({index, Scalar(1)});
                continue;
            }
            if (range.end == grid_.get_N()) {
                suffix_boundaries.push_back(range.begin);
                suffix_result_indices.push_back(index);
                continue;
            }

            prefix_boundaries.push_back(range.end);
            prefix_uses.push_back({index, Scalar(1)});
            prefix_boundaries.push_back(range.begin);
            prefix_uses.push_back({index, Scalar(-1)});
        }

        const auto free_left = make_free_left_environments();
        if (need_full_sum) {
            const Scalar full_sum = free_left.back()(0, 0);
            for (std::size_t index = 0; index < ranges.size(); ++index) {
                if (ranges[index].begin == Sint(0)
                    && ranges[index].end == grid_.get_N())
                    results[index] = full_sum;
            }
        }

        const auto prefix_values =
            prefix_sums_batch_v2(prefix_boundaries, free_left);
        for (std::size_t index = 0; index < prefix_values.size(); ++index) {
            results[prefix_uses[index].result_index] +=
                prefix_uses[index].coefficient * prefix_values[index];
        }

        const auto suffix_values =
            suffix_sums_batch_v2(suffix_boundaries, free_left);
        for (std::size_t index = 0; index < suffix_values.size(); ++index)
            results[suffix_result_indices[index]] = suffix_values[index];

        return results;
    }

private:
    /// Return the sampled MPS multiplied pointwise by 1_[begin,end).
    ///
    /// The interval indicator is first represented as an MPS, promoted to a
    /// diagonal MPO, and then applied to the sampled values.
    [[nodiscard]] State restrict_to(IndexRange range) const
    {
        validate_range(range);
        if (range.begin == Sint(0) && range.end == grid_.get_N())
            return values_;
        if (range.begin == range.end)
            return State(values_ * Scalar(0));

        auto interval_operator =
            MPO<Scalar>::from_mps(make_interval_mps(range));
        return interval_operator._mul(values_, "zip-up");
    }

public:
    // -----------------------------------------------------------------
    // Access
    // -----------------------------------------------------------------

    [[nodiscard]] const Grid& grid() const noexcept { return grid_; }
    [[nodiscard]] const State& values() const noexcept { return values_; }

private:
    // -----------------------------------------------------------------
    // Tensor contractions
    // -----------------------------------------------------------------

    static Scalar contract_full(const State& values)
    {
        MatrixX environment = MatrixX::Identity(1, 1);
        for (const auto& core : values.get_core()) {
            MatrixX transfer = core.phys_const(0) + core.phys_const(1);
            environment = (environment * transfer).eval();
        }
        return environment(0, 0);
    }

    /// free_left[p] sums all assignments of the p least-significant bits.
    [[nodiscard]] std::vector<MatrixX> make_free_left_environments() const
    {
        const auto& cores = values_.get_core();
        std::vector<MatrixX> free_left(cores.size() + 1);
        free_left[0] = MatrixX::Identity(1, 1);
        for (std::size_t site = 0; site < cores.size(); ++site) {
            const MatrixX transfer = cores[site].phys_const(0) + cores[site].phys_const(1);
            free_left[site + 1] = (free_left[site] * transfer).eval();
        }
        return free_left;
    }

    [[nodiscard]] Scalar prefix_sum_v2(
        Sint boundary,
        const std::vector<MatrixX>& free_left) const
    {
        if (boundary == Sint(0))
            return Scalar(0);
        if (boundary == grid_.get_N())
            return free_left.back()(0, 0);

        const auto& cores = values_.get_core();
        MatrixX fixed_right = MatrixX::Identity(1, 1);
        Scalar result = Scalar(0);
        for (int site = static_cast<int>(cores.size()) - 1;
             site >= 0;
             --site) {
            const bool bit =
                ((boundary >> site) & Sint(1)) != Sint(0);
            if (bit) {
                result += (free_left[static_cast<std::size_t>(site)]
                           * cores[static_cast<std::size_t>(site)].phys_const(0)
                           * fixed_right)(0, 0);
            }
            fixed_right =
                (cores[static_cast<std::size_t>(site)].phys_const(bit ? 1 : 0)
                 * fixed_right).eval();
        }
        return result;
    }

    [[nodiscard]] Scalar suffix_sum_v2(
        Sint boundary,
        const std::vector<MatrixX>& free_left) const
    {
        if (boundary == Sint(0))
            return free_left.back()(0, 0);
        if (boundary == grid_.get_N())
            return Scalar(0);

        const auto& cores = values_.get_core();
        MatrixX fixed_right = MatrixX::Identity(1, 1);
        Scalar result = Scalar(0);
        for (int site = static_cast<int>(cores.size()) - 1;
             site >= 0;
             --site) {
            const bool bit =
                ((boundary >> site) & Sint(1)) != Sint(0);
            if (!bit) {
                result += (free_left[static_cast<std::size_t>(site)]
                           * cores[static_cast<std::size_t>(site)].phys_const(1)
                           * fixed_right)(0, 0);
            }
            fixed_right =
                (cores[static_cast<std::size_t>(site)].phys_const(bit ? 1 : 0)
                 * fixed_right).eval();
        }
        // The branch contributions are strictly greater than the boundary;
        // fixed_right is the value at the boundary itself.
        return result + fixed_right(0, 0);
    }

    [[nodiscard]] std::vector<Scalar> prefix_sums_batch_v2(
        const std::vector<Sint>& boundaries,
        const std::vector<MatrixX>& free_left) const
    {
        std::vector<Scalar> results(boundaries.size(), Scalar(0));
        if (boundaries.empty())
            return results;

        const auto& cores = values_.get_core();
        const Eigen::Index count =
            static_cast<Eigen::Index>(boundaries.size());
        MatrixX fixed_right = MatrixX::Ones(1, count);

        for (int site = static_cast<int>(cores.size()) - 1;
             site >= 0;
             --site) {
            const auto& core = cores[static_cast<std::size_t>(site)];
            const MatrixX next_zero =
                (core.phys_const(0) * fixed_right).eval();
            const MatrixX next_one =
                (core.phys_const(1) * fixed_right).eval();
            const MatrixX contributions =
                (free_left[static_cast<std::size_t>(site)] * next_zero).eval();

            MatrixX next(next_zero.rows(), count);
            for (Eigen::Index column = 0; column < count; ++column) {
                const bool bit =
                    ((boundaries[static_cast<std::size_t>(column)] >> site)
                     & Sint(1)) != Sint(0);
                if (bit) {
                    results[static_cast<std::size_t>(column)] +=
                        contributions(0, column);
                    next.col(column) = next_one.col(column);
                } else {
                    next.col(column) = next_zero.col(column);
                }
            }
            fixed_right = std::move(next);
        }
        return results;
    }

    [[nodiscard]] std::vector<Scalar> suffix_sums_batch_v2(
        const std::vector<Sint>& boundaries,
        const std::vector<MatrixX>& free_left) const
    {
        std::vector<Scalar> results(boundaries.size(), Scalar(0));
        if (boundaries.empty())
            return results;

        const auto& cores = values_.get_core();
        const Eigen::Index count =
            static_cast<Eigen::Index>(boundaries.size());
        MatrixX fixed_right = MatrixX::Ones(1, count);

        for (int site = static_cast<int>(cores.size()) - 1;
             site >= 0;
             --site) {
            const auto& core = cores[static_cast<std::size_t>(site)];
            const MatrixX next_zero =
                (core.phys_const(0) * fixed_right).eval();
            const MatrixX next_one =
                (core.phys_const(1) * fixed_right).eval();
            const MatrixX contributions =
                (free_left[static_cast<std::size_t>(site)] * next_one).eval();

            MatrixX next(next_zero.rows(), count);
            for (Eigen::Index column = 0; column < count; ++column) {
                const bool bit =
                    ((boundaries[static_cast<std::size_t>(column)] >> site)
                     & Sint(1)) != Sint(0);
                if (bit) {
                    next.col(column) = next_one.col(column);
                } else {
                    results[static_cast<std::size_t>(column)] +=
                        contributions(0, column);
                    next.col(column) = next_zero.col(column);
                }
            }
            fixed_right = std::move(next);
        }

        for (Eigen::Index column = 0; column < count; ++column)
            results[static_cast<std::size_t>(column)] +=
                fixed_right(0, column);
        return results;
    }

    /// Construct H(x-step) (ascending) or H(step-x) (descending).
    /// Bits are consumed least-significant first.  The three bond states are
    /// EQ, GT and LT; because significance increases from left to right, the
    /// transition matrices propagate the comparison from the right boundary.
    static State make_heaviside_mps(Sint step, int nbits, bool ascending)
    {
        constexpr Eigen::Index chi = 3;
        std::vector<Tensor3D<Scalar>> cores;
        cores.reserve(static_cast<std::size_t>(nbits));

        for (int site = 0; site < nbits; ++site) {
            const int step_bit =
                static_cast<int>((step >> site) & Sint(1));
            Tensor3D<Scalar> core(chi, 2, chi);

            for (int value_bit = 0; value_bit < 2; ++value_bit) {
                core(1, value_bit, 1) = Scalar(1); // GT -> GT
                core(2, value_bit, 2) = Scalar(1); // LT -> LT

                if (value_bit > step_bit)
                    core(1, value_bit, 0) = Scalar(1); // EQ -> GT
                else if (value_bit < step_bit)
                    core(2, value_bit, 0) = Scalar(1); // EQ -> LT
                else
                    core(0, value_bit, 0) = Scalar(1); // EQ -> EQ
            }
            cores.push_back(std::move(core));
        }

        // The right boundary starts in EQ.
        {
            const auto& source = cores.back();
            Tensor3D<Scalar> last(chi, 2, 1);
            for (Eigen::Index left = 0; left < chi; ++left)
                for (Eigen::Index bit = 0; bit < 2; ++bit)
                    last(left, bit, 0) = source(left, bit, 0);
            cores.back() = std::move(last);
        }

        // Ascending accepts EQ and GT; descending accepts LT only.
        {
            const auto& source = cores.front();
            Tensor3D<Scalar> first(1, 2, source.n_right);
            for (Eigen::Index bit = 0; bit < 2; ++bit) {
                for (Eigen::Index right = 0; right < source.n_right; ++right) {
                    if (ascending)
                        first(0, bit, right) = source(0, bit, right)
                                             + source(1, bit, right);
                    else
                        first(0, bit, right) = source(2, bit, right);
                }
            }
            cores.front() = std::move(first);
        }

        return State(std::move(cores), /*max_bond_dim=*/0,
                     /*reltol=*/RealScalar(-1), /*working_site=*/-1);
    }

    /// Construct the indicator MPS which is one on [begin,end) and zero
    /// elsewhere.  For an interior interval this is simply
    /// 1_{x < end} - 1_{x < begin}.
    State make_interval_mps(IndexRange range) const
    {
        const int nbits = grid_.get_nBits();

        if (range.begin == Sint(0))
            return make_heaviside_mps(range.end, nbits,
                                      /*ascending=*/false);
        if (range.end == grid_.get_N())
            return make_heaviside_mps(range.begin, nbits,
                                      /*ascending=*/true);

        auto below_end = make_heaviside_mps(range.end, nbits,
                                            /*ascending=*/false);
        auto below_begin = make_heaviside_mps(range.begin, nbits,
                                              /*ascending=*/false);
        return State(below_end - below_begin);
    }

    // -----------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------

    void validate() const
    {
        const auto& cores = values_.get_core();
        if (cores.empty())
            throw std::invalid_argument(
                "QuanticsIntegrator1D: the MPS must not be empty");
        if (static_cast<int>(cores.size()) != grid_.get_nBits())
            throw std::invalid_argument(
                "QuanticsIntegrator1D: MPS length differs from grid nBits");
        if (cores.front().n_left != 1 || cores.back().n_right != 1)
            throw std::invalid_argument(
                "QuanticsIntegrator1D: MPS boundary bond dimensions must be one");

        for (std::size_t site = 0; site < cores.size(); ++site) {
            if (cores[site].n_phys != 2)
                throw std::invalid_argument(
                    "QuanticsIntegrator1D: every physical dimension must be two");
            if (site + 1 < cores.size()
                && cores[site].n_right != cores[site + 1].n_left)
                throw std::invalid_argument(
                    "QuanticsIntegrator1D: adjacent MPS bond dimensions differ");
        }
    }

    void validate_range(IndexRange range) const
    {
        if (range.begin < Sint(0) || range.end < range.begin
            || range.end > grid_.get_N())
            throw std::out_of_range(
                "QuanticsIntegrator1D: index range must lie in [0, grid.N]");
    }

    Grid grid_;
    State values_;
};

// Compatibility with the name used by the original placeholder.
template<typename Scalar, typename Sint = util::i128>
using QUANTICS_INT_1D = QuanticsIntegrator1D<Scalar, Sint>;
