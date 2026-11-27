#pragma once

#include "fft-tn-quadp/general_integrator.hpp"
#include "QuanticsIntegration/quantics_integrator.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace quad_integration {

namespace general_integrator = fft_tn_quadp::general_integrator;
using QuanticsIntegration::QuanticsIntegrator1D;

template <typename Cscalar>
using RealScalar = typename Eigen::NumTraits<Cscalar>::Real;

template <typename Cscalar>
using Function1D = std::function<Cscalar(RealScalar<Cscalar>)>;

template <typename Real>
struct Interval {
    Real lower;
    Real upper;
};

template <typename Real>
struct QuadParameters {
    general_integrator::options<Real> integration;
};

template <typename Cscalar>
using QuadResult = general_integrator::result<Cscalar, RealScalar<Cscalar>>;

template <typename Cscalar, typename Sint = util::i128>
struct QuanticsFitParameters {
    using Quantics = QuanticsIntegrator1D<Cscalar, Sint>;

    typename Quantics::RunnerParam runner;
    typename Quantics::FitOptions fit_options = {};
    std::optional<std::string> indefinite_integral_method = std::nullopt;
};

using LoggerPtr = std::shared_ptr<spdlog::logger>;

enum class GridBoundaryPolicy {
    require_exact,
    nearest
};

namespace detail {

template <typename Number>
std::string number_to_string(const Number& value)
{
    std::ostringstream output;
    output << std::scientific
           << std::setprecision(std::numeric_limits<Number>::max_digits10)
           << value;
    return output.str();
}

template <typename Real>
std::string real_to_string(const Real& value)
{
    return number_to_string(value);
}

template <typename Real>
Real number_from_json(const nlohmann::json& value)
{
    const std::string text = value.is_string()
        ? value.get<std::string>()
        : value.dump();
    std::istringstream input(text);
    Real result;
    input >> result;
    if (!input) {
        throw std::runtime_error(
            "quad_integration: invalid real parameter '" + text + "'");
    }
    input >> std::ws;
    if (!input.eof()) {
        throw std::runtime_error(
            "quad_integration: trailing characters in real parameter '" +
            text + "'");
    }
    return result;
}

template <typename Real>
Real real_from_json(const nlohmann::json& value)
{
    return number_from_json<Real>(value);
}

template <typename Real>
nlohmann::json real_vector_to_json(const std::vector<Real>& values)
{
    nlohmann::json output = nlohmann::json::array();
    for (const Real& value : values) {
        output.push_back(number_to_string(value));
    }
    return output;
}

template <typename Real>
std::vector<Real> real_vector_from_json(const nlohmann::json& values)
{
    std::vector<Real> output;
    output.reserve(values.size());
    for (const auto& value : values) {
        output.push_back(number_from_json<Real>(value));
    }
    return output;
}

inline nlohmann::json encode_dd128_exact(const dd_128& value)
{
    return {
        {"hi", number_to_string(value._hi())},
        {"lo", number_to_string(value._lo())}
    };
}

inline dd_128 decode_dd128_exact(const nlohmann::json& value)
{
    const double hi = number_from_json<double>(value.at("hi"));
    const double lo = number_from_json<double>(value.at("lo"));
    return dd_128(dd_real(hi, lo));
}

inline nlohmann::json read_json(const std::string& filename)
{
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error(
            "quad_integration: cannot open '" + filename + "'");
    }

    nlohmann::json document;
    input >> document;
    return document;
}

inline void write_json(
    const std::string& filename,
    const nlohmann::json& document)
{
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error(
            "quad_integration: cannot open '" + filename +
            "' for writing");
    }

    output << document.dump(4) << '\n';
    if (!output) {
        throw std::runtime_error(
            "quad_integration: failed to write '" + filename + "'");
    }
}

template <typename Cscalar, typename Sint>
typename QuanticsIntegrator1D<Cscalar, Sint>::IndexRange coordinate_range_impl(
    const QuanticsIntegrator1D<Cscalar, Sint>& fitted_integrand,
    Interval<RealScalar<Cscalar>> coordinates,
    GridBoundaryPolicy policy)
{
    using Quantics = QuanticsIntegrator1D<Cscalar, Sint>;
    using IndexRange = typename Quantics::IndexRange;
    using Grid = typename Quantics::Grid;

    const Grid& grid = fitted_integrand.grid();
    if (coordinates.lower > coordinates.upper) {
        throw std::invalid_argument(
            "coordinate_range requires lower <= upper");
    }
    if (coordinates.lower < grid.get_a() ||
        coordinates.upper > grid.get_b()) {
        throw std::out_of_range(
            "coordinate_range must lie inside the fitted Quantics domain");
    }

    auto coordinate_to_boundary = [&](const RealScalar<Cscalar>& coordinate)
        -> Sint {
        if (coordinate == grid.get_a()) {
            return Sint(0);
        }
        if (coordinate == grid.get_b()) {
            return grid.get_N();
        }

        const std::vector<int> bits = grid.coord_to_id(coordinate);
        if (policy == GridBoundaryPolicy::require_exact &&
            grid.id_to_coord(bits) != coordinate) {
            throw std::invalid_argument(
                "coordinate_range requires coordinates on the Quantics grid");
        }

        Sint index = 0;
        for (std::size_t bit = 0; bit < bits.size(); ++bit) {
            index |= Sint(bits[bit]) << bit;
        }
        return index;
    };

    return IndexRange{
        coordinate_to_boundary(coordinates.lower),
        coordinate_to_boundary(coordinates.upper)};
}

} // namespace detail

/// Directly integrate a function with adaptive quadrature. The complete
/// convergence and error result is returned to the caller.
template <typename Cscalar>
[[nodiscard]] QuadResult<Cscalar> integrate_quad(
    const Function1D<Cscalar>& integrand,
    Interval<RealScalar<Cscalar>> interval,
    const QuadParameters<RealScalar<Cscalar>>& parameters,
    LoggerPtr logger = nullptr)
{
    if (!integrand) {
        throw std::invalid_argument("integrate_quad: integrand is empty");
    }

    auto result = general_integrator::integrate<RealScalar<Cscalar>>(
        integrand,
        interval.lower,
        interval.upper,
        parameters.integration);

    if (logger) {
        logger->info(
            "integrate_quad: lower={}, upper={}, value=({}, {}), "
            "absolute_error={}, evaluations={}, subintervals={}, converged={}",
            detail::number_to_string(interval.lower),
            detail::number_to_string(interval.upper),
            detail::number_to_string(result.value.real()),
            detail::number_to_string(result.value.imag()),
            detail::number_to_string(result.absolute_error),
            result.evaluations,
            result.subintervals,
            result.converged);
    }

    return result;
}

/// Fit an integrand on a Quantics grid. The returned object owns the fitted
/// MPS and can be contracted repeatedly over the full grid or index subranges.
template <typename Cscalar, typename Sint = util::i128>
[[nodiscard]] QuanticsIntegrator1D<Cscalar, Sint> fit_quantics(
    Function1D<Cscalar> integrand,
    Interval<RealScalar<Cscalar>> fit_interval,
    QuanticsFitParameters<Cscalar, Sint> parameters,
    const std::string& save_prefix,
    LoggerPtr logger = nullptr)
{
    using Quantics = QuanticsIntegrator1D<Cscalar, Sint>;
    using Grid = typename Quantics::Grid;

    if (!integrand) {
        throw std::invalid_argument("fit_quantics: integrand is empty");
    }
    if (fit_interval.lower >= fit_interval.upper) {
        throw std::invalid_argument(
            "fit_quantics requires lower < upper");
    }
    if (parameters.runner.nBit <= 0) {
        throw std::invalid_argument(
            "fit_quantics requires nBit > 0");
    }

    Grid grid(
        fit_interval.lower,
        fit_interval.upper,
        parameters.runner.nBit);

    // A midpoint pivot is a generic default. Domain-specific calculators may
    // provide additional pivots and discontinuities before calling this API.
    if (parameters.runner.pivot1.empty()) {
        parameters.runner.pivot1 = grid.coord_to_id(
            (fit_interval.lower + fit_interval.upper) /
            RealScalar<Cscalar>(2));
    }

    if (logger) {
        logger->info(
            "fit_quantics: lower={}, upper={}, nBit={}, save_prefix={}",
            detail::number_to_string(fit_interval.lower),
            detail::number_to_string(fit_interval.upper),
            parameters.runner.nBit,
            save_prefix);
    }

    return Quantics::fit(
        std::move(grid),
        std::move(integrand),
        std::move(parameters.runner),
        save_prefix,
        std::move(parameters.fit_options),
        std::move(logger),
        std::move(parameters.indefinite_integral_method));
}

/// Convert a coordinate interval into the half-open Quantics index range
/// [begin,end). The rounding policy is explicit at the call site.
template <typename Cscalar, typename Sint = util::i128>
[[nodiscard]] typename QuanticsIntegrator1D<Cscalar, Sint>::IndexRange
coordinate_range(
    const QuanticsIntegrator1D<Cscalar, Sint>& fitted_integrand,
    Interval<RealScalar<Cscalar>> coordinates,
    GridBoundaryPolicy policy = GridBoundaryPolicy::require_exact)
{
    return detail::coordinate_range_impl(
        fitted_integrand, coordinates, policy);
}

/// Contract a fitted Quantics integrand over a coordinate interval. Reversed
/// intervals are supported and receive the usual minus sign.
template <typename Cscalar, typename Sint = util::i128>
[[nodiscard]] Cscalar integrate_quantics(
    const QuanticsIntegrator1D<Cscalar, Sint>& fitted_integrand,
    Interval<RealScalar<Cscalar>> interval,
    GridBoundaryPolicy policy = GridBoundaryPolicy::require_exact)
{
    if (interval.lower == interval.upper) {
        return Cscalar(0);
    }

    const bool reversed = interval.upper < interval.lower;
    const Interval<RealScalar<Cscalar>> ordered = reversed
        ? Interval<RealScalar<Cscalar>>{interval.upper, interval.lower}
        : interval;
    const auto range = coordinate_range(fitted_integrand, ordered, policy);
    const Cscalar value = fitted_integrand.integrate(range);
    return reversed ? -value : value;
}

/// Evaluate a previously constructed indefinite-integral MPS over one
/// coordinate interval.
template <typename Cscalar, typename Sint = util::i128>
[[nodiscard]] Cscalar integrate_quantics_indefinite(
    const QuanticsIntegrator1D<Cscalar, Sint>& fitted_integrand,
    const typename QuanticsIntegrator1D<Cscalar, Sint>::State&
        indefinite_values,
    Interval<RealScalar<Cscalar>> interval)
{
    using Quantics = QuanticsIntegrator1D<Cscalar, Sint>;
    return Quantics::integrate_indefinite(
        indefinite_values,
        fitted_integrand.grid(),
        interval.lower,
        interval.upper);
}

/// Evaluate a previously constructed indefinite-integral MPS over several
/// coordinate intervals. Repeated endpoints share the per-call MPS cache.
template <typename Cscalar, typename Sint = util::i128>
[[nodiscard]] std::vector<Cscalar> integrate_quantics_indefinite(
    const QuanticsIntegrator1D<Cscalar, Sint>& fitted_integrand,
    const typename QuanticsIntegrator1D<Cscalar, Sint>::State&
        indefinite_values,
    const std::vector<Interval<RealScalar<Cscalar>>>& intervals)
{
    using Quantics = QuanticsIntegrator1D<Cscalar, Sint>;

    std::vector<RealScalar<Cscalar>> x_min;
    std::vector<RealScalar<Cscalar>> x_max;
    x_min.reserve(intervals.size());
    x_max.reserve(intervals.size());
    for (const auto& interval : intervals) {
        x_min.push_back(interval.lower);
        x_max.push_back(interval.upper);
    }

    return Quantics::integrate_indefinite(
        indefinite_values, fitted_integrand.grid(), x_min, x_max);
}

namespace io {

/// Serialize only the parameters consumed by adaptive quadrature.
template <typename Real>
void save_quad_parameters(
    const QuadParameters<Real>& parameters,
    const std::string& filename)
{
    const auto& integration = parameters.integration;
    nlohmann::json document = {
        {"method", "quad"},
        {"epsabs", detail::number_to_string(integration.epsabs)},
        {"epsrel", detail::number_to_string(integration.epsrel)},
        {"limit", integration.limit},
        {"points", detail::real_vector_to_json(integration.points)}
    };

    if constexpr (std::is_same_v<Real, dd_128>) {
        document["dd128_exact"] = {
            {"epsabs", detail::encode_dd128_exact(integration.epsabs)},
            {"epsrel", detail::encode_dd128_exact(integration.epsrel)}
        };
    }

    detail::write_json(filename, document);
}

template <typename Real>
[[nodiscard]] QuadParameters<Real> load_quad_parameters(
    const std::string& filename)
{
    const nlohmann::json document = detail::read_json(filename);
    if (document.value("method", "quad") != "quad") {
        throw std::runtime_error(
            "quad_integration: parameter file is not a quad "
            "configuration");
    }

    QuadParameters<Real> parameters;
    auto& integration = parameters.integration;
    if constexpr (std::is_same_v<Real, dd_128>) {
        if (document.contains("dd128_exact")) {
            const auto& exact = document.at("dd128_exact");
            integration.epsabs =
                detail::decode_dd128_exact(exact.at("epsabs"));
            integration.epsrel =
                detail::decode_dd128_exact(exact.at("epsrel"));
        } else {
            integration.epsabs =
                detail::number_from_json<Real>(document.at("epsabs"));
            integration.epsrel =
                detail::number_from_json<Real>(document.at("epsrel"));
        }
    } else {
        integration.epsabs =
            detail::number_from_json<Real>(document.at("epsabs"));
        integration.epsrel =
            detail::number_from_json<Real>(document.at("epsrel"));
    }

    integration.limit = document.at("limit").get<std::size_t>();
    if (document.contains("points")) {
        integration.points = detail::real_vector_from_json<Real>(
            document.at("points"));
    }
    return parameters;
}

/// Serialize only the runner and fit parameters consumed by Quantics fitting.
/// The fit domain, output prefix, logger, and contraction ranges belong to
/// individual calls and are therefore not stored here.
template <typename Cscalar, typename Sint>
void save_quantics_parameters(
    const QuanticsFitParameters<Cscalar, Sint>& parameters,
    const std::string& filename)
{
    parameters.runner.save(filename);
    nlohmann::json document = detail::read_json(filename);
    document["method"] = "quantics";
    document["pivot1"] = parameters.runner.pivot1;
    document["fit_options"] = {
        {"additional_pivots", detail::real_vector_to_json(
            parameters.fit_options.additional_pivots)},
        {"verbose", parameters.fit_options.verbose},
        {"nb_point_res", parameters.fit_options.nb_point_res},
        {"discontinuities", detail::real_vector_to_json(
            parameters.fit_options.discontinuities)}
    };
    document["indefinite_integral_method"] =
        parameters.indefinite_integral_method
            ? nlohmann::json(*parameters.indefinite_integral_method)
            : nlohmann::json(nullptr);
    detail::write_json(filename, document);
}

template <typename Cscalar, typename Sint = util::i128>
[[nodiscard]] QuanticsFitParameters<Cscalar, Sint>
load_quantics_parameters(const std::string& filename)
{
    using Quantics = QuanticsIntegrator1D<Cscalar, Sint>;
    using RunnerParam = typename Quantics::RunnerParam;
    using FitOptions = typename Quantics::FitOptions;

    const nlohmann::json document = detail::read_json(filename);
    if (document.value("method", "quantics") != "quantics") {
        throw std::runtime_error(
            "quad_integration: parameter file is not a quantics "
            "configuration");
    }

    std::vector<int> pivot1;
    if (document.contains("pivot1")) {
        pivot1 = document.at("pivot1").get<std::vector<int>>();
    }
    RunnerParam runner(filename, std::move(pivot1));

    FitOptions fit_options;
    if (document.contains("fit_options")) {
        const auto& fit = document.at("fit_options");
        if (fit.contains("additional_pivots")) {
            fit_options.additional_pivots =
                detail::real_vector_from_json<RealScalar<Cscalar>>(
                    fit.at("additional_pivots"));
        }
        fit_options.verbose = fit.value("verbose", false);
        fit_options.nb_point_res = fit.value("nb_point_res", 1000);
        if (fit.contains("discontinuities")) {
            fit_options.discontinuities =
                detail::real_vector_from_json<RealScalar<Cscalar>>(
                    fit.at("discontinuities"));
        }
    }

    std::optional<std::string> indefinite_integral_method;
    if (document.contains("indefinite_integral_method") &&
        !document.at("indefinite_integral_method").is_null()) {
        indefinite_integral_method =
            document.at("indefinite_integral_method").get<std::string>();
    }

    return {
        std::move(runner),
        std::move(fit_options),
        std::move(indefinite_integral_method)
    };
}

} // namespace io

} // namespace quad_integration
