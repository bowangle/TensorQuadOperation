#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "general_integrator.hpp"

namespace {

template<class Real>
double as_double(const Real& value)
{
    if constexpr (std::is_same_v<Real, double>)
        return value;
    else if constexpr (std::is_same_v<Real, dd_128>)
        return value.x[0];
    else
        return value.template convert_to<double>();
}

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
Real sqrt_value(const Real& value)
{
    using std::sqrt;
    return Real(sqrt(value));
}

template<class Real>
Real atan_value(const Real& value)
{
    using std::atan;
    return Real(atan(value));
}

template<class Real>
Real test_tolerance()
{
    return Real(100) * Real(std::numeric_limits<Real>::epsilon());
}

template<class Real>
void require_relative_close(const Real& actual,
                            const Real& expected,
                            double tolerance)
{
    using std::abs;
    const Real scale = abs(expected) < Real(1) ? Real(1) : Real(abs(expected));
    REQUIRE(as_double(abs(actual - expected) / scale) < tolerance);
}

template<class Real>
void require_within_test_tolerance(const Real& actual, const Real& expected)
{
    require_relative_close(actual, expected, as_double(test_tolerance<Real>()));
}

template<class Real>
void check_real_type()
{
    general_integrator::options<Real> settings;
    settings.epsabs = test_tolerance<Real>();
    settings.epsrel = test_tolerance<Real>();

    const auto polynomial = general_integrator::integrate<Real>(
        [](Real x) -> Real { return x * x; }, Real(0), Real(1), settings);

    REQUIRE(polynomial.converged);
    REQUIRE(polynomial.evaluations == 21);
    REQUIRE(polynomial.subintervals == 1);
    require_within_test_tolerance(polynomial.value, Real(1) / Real(3));

    const auto reversed = general_integrator::integrate<Real>(
        [](Real x) -> Real { return x * x; }, Real(1), Real(0), settings);
    REQUIRE(reversed.converged);
    require_within_test_tolerance(reversed.value, -Real(1) / Real(3));
}

template<class Real>
void check_smooth_non_polynomial_type()
{
    const Real frequency = Real(17);
    general_integrator::options<Real> settings;
    settings.epsabs = test_tolerance<Real>();
    settings.epsrel = test_tolerance<Real>();
    settings.limit = 100;

    const auto answer = general_integrator::integrate<Real>(
        [frequency](Real x) -> Real {
            return exp_value(x) * cos_value(frequency * x);
        },
        Real(0), Real(1), settings);

    const Real expected =
        (exp_value(Real(1)) *
             (cos_value(frequency) + frequency * sin_value(frequency)) -
         Real(1)) /
        (Real(1) + frequency * frequency);

    REQUIRE(answer.converged);
    REQUIRE(answer.subintervals > 1);
    require_within_test_tolerance(answer.value, expected);
}

template<class Real>
void check_complex_type()
{
    using Complex = std::complex<Real>;
    general_integrator::options<Real> settings;
    settings.epsabs = test_tolerance<Real>();
    settings.epsrel = test_tolerance<Real>();
    settings.limit = 100;

    const auto answer = general_integrator::integrate<Real>(
        [](Real x) -> Complex {
            return Complex(cos_value(x), sin_value(x));
        },
        Real(0), Real(1), settings);

    const Complex expected(sin_value(Real(1)),
                           Real(1) - cos_value(Real(1)));
    REQUIRE(answer.converged);
    require_within_test_tolerance(answer.value.real(), expected.real());
    require_within_test_tolerance(answer.value.imag(), expected.imag());
}

} // namespace

TEST_CASE("general integrator supports all real scalar types", "[integrator]")
{
    check_real_type<double>();
    check_real_type<dd_128>();
    check_real_type<float128>();
}

TEST_CASE("general integrator handles smooth non-polynomial functions",
          "[integrator]")
{
    check_smooth_non_polynomial_type<double>();
    check_smooth_non_polynomial_type<dd_128>();
    check_smooth_non_polynomial_type<float128>();
}

TEST_CASE("general integrator splits all complex scalar types", "[integrator]")
{
    check_complex_type<double>();
    check_complex_type<dd_128>();
    check_complex_type<float128>();
}

TEST_CASE("points create initial subintervals and share the global limit",
          "[integrator]")
{
    general_integrator::options<double> settings;
    settings.epsabs = test_tolerance<double>();
    settings.epsrel = test_tolerance<double>();
    settings.points = {0.75, 0.25, 0.25};
    settings.limit = 3;

    const auto answer = general_integrator::integrate<double>(
        [](double x) { return x < 0.25 ? 1.0 : (x < 0.75 ? 2.0 : 4.0); },
        0.0, 1.0, settings);

    REQUIRE(answer.converged);
    REQUIRE(answer.subintervals == 3);
    REQUIRE(answer.evaluations == 63);
    require_within_test_tolerance(answer.value, 2.25);

    settings.limit = 2;
    REQUIRE_THROWS_AS(
        general_integrator::integrate<double>(
            [](double x) { return x; }, 0.0, 1.0, settings),
        std::invalid_argument);
}

TEST_CASE("global refinement resolves an unmarked cusp", "[integrator]")
{
    constexpr double cusp = 0.123456789;
    general_integrator::options<double> settings;
    settings.epsabs = test_tolerance<double>();
    settings.epsrel = test_tolerance<double>();
    settings.limit = 100;

    const auto answer = general_integrator::integrate<double>(
        [](double x) { return std::abs(x - cusp); },
        0.0, 1.0, settings);
    const double expected =
        0.5 * (cusp * cusp + (1.0 - cusp) * (1.0 - cusp));

    REQUIRE(answer.converged);
    REQUIRE(answer.subintervals > 1);
    require_within_test_tolerance(answer.value, expected);
}

TEST_CASE("global refinement resolves a narrow localized peak", "[integrator]")
{
    constexpr double centre = 0.3712345;
    constexpr double width = 0.01;
    general_integrator::options<double> settings;
    settings.epsabs = test_tolerance<double>();
    settings.epsrel = test_tolerance<double>();
    settings.limit = 200;

    const auto answer = general_integrator::integrate<double>(
        [](double x) {
            const double offset = x - centre;
            return 1.0 / (offset * offset + width * width);
        },
        0.0, 1.0, settings);
    const double expected =
        (std::atan((1.0 - centre) / width) -
         std::atan(-centre / width)) / width;

    REQUIRE(answer.converged);
    REQUIRE(answer.subintervals > 4);
    require_within_test_tolerance(answer.value, expected);
}

TEST_CASE("oscillatory and endpoint-singular integrals converge", "[integrator]")
{
    general_integrator::options<double> settings;
    settings.epsabs = test_tolerance<double>();
    settings.epsrel = test_tolerance<double>();
    settings.limit = 200;

    const auto oscillatory = general_integrator::integrate<double>(
        [](double x) { return std::cos(200.0 * x); },
        0.0, 1.0, settings);
    REQUIRE(oscillatory.converged);
    REQUIRE(oscillatory.subintervals > 1);
    require_within_test_tolerance(
        oscillatory.value, std::sin(200.0) / 200.0);

    const auto singular = general_integrator::integrate<double>(
        [](double x) { return 1.0 / std::sqrt(x); },
        0.0, 1.0, settings);
    REQUIRE(singular.converged);
    REQUIRE(singular.subintervals > 1);
    require_within_test_tolerance(singular.value, 2.0);
}

TEST_CASE("limit stops global refinement", "[integrator]")
{
    general_integrator::options<double> settings;
    settings.epsabs = test_tolerance<double>();
    settings.epsrel = test_tolerance<double>();
    settings.limit = 1;

    const auto answer = general_integrator::integrate<double>(
        [](double x) { return std::abs(x - 0.123456789); },
        0.0, 1.0, settings);

    REQUIRE_FALSE(answer.converged);
    REQUIRE(answer.subintervals == 1);
    REQUIRE(answer.evaluations == 21);
}
