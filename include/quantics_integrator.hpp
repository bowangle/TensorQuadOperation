#pragma once

#include "mpo_base.h"

#include <type_int128.h>
#include <xf-qd-runner.hpp>
#include <xf-quad-param.hpp>

#include <Eigen/Dense>

#include <cstddef>
#include <functional>
#include <iostream>
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
    using MatrixX     = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::ColMajor>;

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
    static QuanticsIntegrator1D from_files(const std::string& grid_file,
                                           const std::string& mps_file)
    {
        Grid grid(grid_file);
        State values(mps_file, /*max_bond_dim=*/0, /*reltol=*/-1,
                     /*working_site=*/-1);
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
                                    FitOptions options = {})
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
                std::move(grid), std::move(parameters), std::move(function));

            runner.fit(options.additional_pivots,
                       options.verbose,
                       /*do_save=*/true,
                       save_prefix,
                       options.nb_point_res,
                       options.discontinuities);

            Grid saved_grid(save_prefix + "_grid.json");
            State saved_values(save_prefix + ".tt", max_bond_dim, reltol,
                               /*working_site=*/-1);
            return QuanticsIntegrator1D(std::move(saved_grid),
                                        std::move(saved_values));
        }
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
        return sum(range) * Scalar(grid_.delta_volume());
    }

    [[nodiscard]] Scalar sum() const
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
