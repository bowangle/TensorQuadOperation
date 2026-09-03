#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>

#include "quantics_integrator.hpp"

namespace {

template<class Real>
inline constexpr int nbits_for = std::is_same_v<Real, double> ? 50 : 99;

template<class Real>
Real exp_value(const Real& value)
{
    using std::exp;
    return Real(exp(value));
}

template<class Real>
Real sin_value(const Real& value)
{
    using std::sin;
    return Real(sin(value));
}

template<class Real>
Real cos_value(const Real& value)
{
    using std::cos;
    return Real(cos(value));
}

template<class Real>
Real test_tolerance()
{
    return Real(10000) * Real(std::numeric_limits<Real>::epsilon());
}

template<class Real>
Real tci_reltol()
{
    return Real(100) * Real(std::numeric_limits<Real>::epsilon());
}

template<class Real>
void require_relative_close(
    const Real& actual,
    const Real& expected,
    Real tolerance = test_tolerance<Real>())
{
    using std::abs;
    const Real scale = abs(expected) < Real(1) ? Real(1) : Real(abs(expected));
    const Real error = abs(actual - expected) / scale;
    std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10)
              << "relative error = " << error
              << ", tolerance = " << tolerance << '\n';
    REQUIRE(error < tolerance);
}

template<class Real>
void require_relative_close(
    const std::complex<Real>& actual,
    const std::complex<Real>& expected,
    Real tolerance = test_tolerance<Real>())
{
    require_relative_close(actual.real(), expected.real(), tolerance);
    require_relative_close(actual.imag(), expected.imag(), tolerance);
}

template<class Real>
struct QuanticsOptions {
    Real reltol = tci_reltol<Real>();
    std::vector<Real> points;
    int limit = 100;
    int bond_dim = 50;
};

class TemporaryFitFiles {
public:
    TemporaryFitFiles()
    {
        static std::atomic<std::size_t> counter{0};
        const std::filesystem::path directory =
            std::filesystem::path(__FILE__).parent_path() / "tmp";
        std::filesystem::create_directories(directory);
        prefix_ = (directory /
                   ("quantics_integrator_" +
                    std::to_string(counter.fetch_add(1)))).string();
    }

    ~TemporaryFitFiles()
    {
        for (const char* suffix : {".tt", "_error.dat", "_grid.json",
                                   "_grid_E.json", "_f_values.json"}) {
            std::error_code error;
            std::filesystem::remove(prefix_ + suffix, error);
        }
    }

    const std::string& prefix() const noexcept { return prefix_; }

private:
    std::string prefix_;
};

template<class Real, class Function>
std::complex<Real> integrate_quantics(Function&& function,
                                      Real a,
                                      Real b,
                                      QuanticsOptions<Real> settings = {})
{
    using Complex = std::complex<Real>;
    using Integrator = QuanticsIntegrator1D<Complex>;

    constexpr int nbits = nbits_for<Real>;
    typename Integrator::Grid grid(a, b, nbits);
    TCI2_1D_runner_opts<Complex> runner_options;
    runner_options.reltol = settings.reltol;
    runner_options.pivot1 = grid.coord_to_id((a + b) / Real(2));
    typename Integrator::RunnerParam parameters(
        nbits, settings.limit, settings.bond_dim, runner_options);

    typename Integrator::FitOptions fit_options;
    fit_options.additional_pivots = std::move(settings.points);
    fit_options.verbose = false;
    fit_options.nb_point_res = 1000;

    TemporaryFitFiles files;
    auto complex_function =
        [function = std::forward<Function>(function)](Real x) mutable -> Complex {
            return Complex(function(x));
        };

    const auto integrator = Integrator::fit(
        std::move(grid), std::move(complex_function), std::move(parameters),
        files.prefix(), std::move(fit_options));
    return integrator.integrate();
}

template<class Real>
void check_real_type()
{
    const auto polynomial = integrate_quantics<Real>(
        [](Real x) -> Real { return x * x; }, Real(0), Real(1));
    require_relative_close(polynomial, std::complex<Real>(Real(1) / Real(3)));
}

template<class Real>
void check_smooth_non_polynomial_type()
{
    const Real frequency = Real(17);
    const auto answer = integrate_quantics<Real>(
        [frequency](Real x) -> Real {
            return exp_value(x) * cos_value(frequency * x);
        },
        Real(0), Real(1));

    const Real expected =
        (exp_value(Real(1)) *
             (cos_value(frequency) + frequency * sin_value(frequency)) -
         Real(1)) /
        (Real(1) + frequency * frequency);
    require_relative_close(answer, std::complex<Real>(expected));
}

template<class Real>
void check_complex_type()
{
    using Complex = std::complex<Real>;
    const auto answer = integrate_quantics<Real>(
        [](Real x) -> Complex {
            return Complex(cos_value(x), sin_value(x));
        },
        Real(0), Real(1));

    const Complex expected(sin_value(Real(1)),
                           Real(1) - cos_value(Real(1)));
    require_relative_close(answer, expected);
}

} // namespace

TEST_CASE("quantics integrator forwards fit diagnostics to spdlog",
          "[integrator][logging]")
{
    using Complex = std::complex<double>;
    using Integrator = QuanticsIntegrator1D<Complex>;

    constexpr int nbits = 8;
    typename Integrator::Grid grid(0.0, 1.0, nbits);
    TCI2_1D_runner_opts<Complex> runner_options;
    runner_options.reltol = 1e-12;
    runner_options.pivot1 = grid.coord_to_id(0.5);
    runner_options.fullPiv = true;
    typename Integrator::RunnerParam parameters(
        nbits, /*nb_iter=*/2, /*bondDim=*/8, runner_options);

    typename Integrator::FitOptions fit_options;
    fit_options.verbose = false;
    fit_options.nb_point_res = 16;

    std::ostringstream log_output;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log_output);
    auto logger = std::make_shared<spdlog::logger>(
        "quantics_integrator_test", std::move(sink));
    logger->set_pattern("%v");

    TemporaryFitFiles files;
    const auto integrator = Integrator::fit(
        std::move(grid),
        [](double x) -> Complex { return Complex(x, 0.0); },
        std::move(parameters), files.prefix(), std::move(fit_options), logger);
    logger->flush();

    const std::string messages = log_output.str();
    REQUIRE(messages.find("========== FIT PARAMETERS ==========") !=
            std::string::npos);
    REQUIRE(messages.find("sweep 1 |pivot error|") != std::string::npos);
    REQUIRE(messages.find("Max bond dim:") != std::string::npos);
}

TEST_CASE("quantics integrator supports all real scalar types", "[integrator]")
{
    check_real_type<double>();
    check_real_type<dd_128>();
    check_real_type<float128>();
}

TEST_CASE("quantics integrator handles smooth non-polynomial functions",
          "[integrator]")
{
    check_smooth_non_polynomial_type<double>();
    check_smooth_non_polynomial_type<dd_128>();
    check_smooth_non_polynomial_type<float128>();
}

TEST_CASE("quantics integrator supports all complex scalar types",
          "[integrator]")
{
    check_complex_type<double>();
    check_complex_type<dd_128>();
    check_complex_type<float128>();
}

TEST_CASE("additional pivots resolve marked discontinuities", "[integrator]")
{
    QuanticsOptions<double> settings;
    settings.points = {0.75, 0.25, 0.25};
    settings.limit = 100;

    const auto answer = integrate_quantics<double>(
        [](double x) {
            return x < 0.25 ? 1.0 : (x < 0.75 ? 2.0 : 4.0);
        },
        0.0, 1.0, settings);
    require_relative_close(answer, std::complex<double>(2.25));
}

TEST_CASE("quantics integrator resolves an unmarked cusp", "[integrator]")
{
    constexpr double cusp = 0.123456789;
    const auto answer = integrate_quantics<double>(
        [](double x) { return std::abs(x - cusp); }, 0.0, 1.0);
    const double expected =
        0.5 * (cusp * cusp + (1.0 - cusp) * (1.0 - cusp));
    require_relative_close(answer, std::complex<double>(expected));
}

TEST_CASE("quantics integrator resolves a narrow localized peak",
          "[integrator]")
{
    constexpr double centre = 0.3712345;
    constexpr double width = 0.01;
    QuanticsOptions<double> settings;
    settings.limit = 200;

    const auto answer = integrate_quantics<double>(
        [](double x) {
            const double offset = x - centre;
            return 1.0 / (offset * offset + width * width);
        },
        0.0, 1.0, settings);
    const double expected =
        (std::atan((1.0 - centre) / width) -
         std::atan(-centre / width)) /
        width;
    require_relative_close(answer, std::complex<double>(expected));
}

TEST_CASE("oscillatory integrals converge", "[integrator]")
{
    QuanticsOptions<double> settings;
    settings.limit = 200;

    const auto oscillatory = integrate_quantics<double>(
        [](double x) { return std::cos(200.0 * x); },
        0.0, 1.0, settings);
    require_relative_close(oscillatory,
                           std::complex<double>(std::sin(200.0) / 200.0));
}
