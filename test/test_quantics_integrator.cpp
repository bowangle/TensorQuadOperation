#include <algorithm>
#include <atomic>
#include <chrono>
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
std::pair<Real, Real> integration_interval(int index)
{
    const std::vector<std::pair<Real, Real>> intervals{
        {Real(0), Real(1)}, {Real(-9) / Real(10), Real(1)},
        {Real(1) / Real(10), Real(9) / Real(10)}};
    return intervals.at(static_cast<std::size_t>(index));
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
Real test_tolerance()
{
    return Real(13000) * Real(std::numeric_limits<Real>::epsilon());
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
              << "direct/analytical relative error = " << error
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
void require_indefinite_close(
    const std::complex<Real>& indefinite,
    const std::complex<Real>& direct,
    const char* path = "indefinite",
    Real tolerance = Real(10) * test_tolerance<Real>())
{
    using std::abs;
    const Real real_scale =
        abs(direct.real()) < Real(1) ? Real(1) : Real(abs(direct.real()));
    const Real imag_scale =
        abs(direct.imag()) < Real(1) ? Real(1) : Real(abs(direct.imag()));
    const Real real_error = abs(indefinite.real() - direct.real()) / real_scale;
    const Real imag_error = abs(indefinite.imag() - direct.imag()) / imag_scale;
    const Real error = real_error > imag_error ? real_error : imag_error;

    std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10)
              << path << "/direct relative error = " << error
              << ", tolerance = " << tolerance << '\n';
    REQUIRE(error < tolerance);
}

template<class Real>
void require_indefinite_analytic_close(
    const std::complex<Real>& indefinite,
    const std::complex<Real>& analytical,
    const char* path = "indefinite",
    Real tolerance = Real(10) * test_tolerance<Real>())
{
    using std::abs;
    const Real real_scale =
        abs(analytical.real()) < Real(1)
            ? Real(1) : Real(abs(analytical.real()));
    const Real imag_scale =
        abs(analytical.imag()) < Real(1)
            ? Real(1) : Real(abs(analytical.imag()));
    const Real real_error =
        abs(indefinite.real() - analytical.real()) / real_scale;
    const Real imag_error =
        abs(indefinite.imag() - analytical.imag()) / imag_scale;
    const Real error = real_error > imag_error ? real_error : imag_error;

    std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10)
              << path << "/analytical relative error = " << error
              << ", tolerance = " << tolerance << '\n';
    REQUIRE(error < tolerance);
}

template<class Real>
struct QuanticsIntegralResults {
    std::complex<Real> direct;
    std::complex<Real> indefinite;
    std::complex<Real> indefinite_conv;
};

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
        for (const char* suffix : {".tt", "_indefinite_int.tt",
                                   "_conv_indefinite_int.tt",
                                   "_error.dat", "_grid.json",
                                   "_grid_E.json", "_f_values.json"}) {
            std::error_code error;
            std::filesystem::remove(prefix_ + suffix, error);
        }
    }

    const std::string& prefix() const noexcept { return prefix_; }

private:
    std::string prefix_;
};

template<class Real>
typename QuanticsIntegrator1D<std::complex<Real>>::State build_indefinite_conv(
    const QuanticsIntegrator1D<std::complex<Real>>& integrator,
    const std::string& save_prefix,
    int max_bond_dim,
    Real reltol)
{
    using Integrator = QuanticsIntegrator1D<std::complex<Real>>;
    auto values = integrator.values();
    auto grid = integrator.grid();
    auto response = Integrator::make_indefinite_integrator_from_conv(
        values, grid, save_prefix, max_bond_dim, reltol, "zip-up");
    REQUIRE(std::filesystem::is_regular_file(save_prefix + "_conv_indefinite_int.tt"));
    return response;
}

template<class Real, class Function>
QuanticsIntegralResults<Real> integrate_quantics(Function&& function,
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
        files.prefix(), std::move(fit_options), nullptr,
        std::string("zip-up"));

    typename Integrator::State indefinite_values(
        files.prefix() + "_indefinite_int.tt", settings.bond_dim,
        settings.reltol, /*working_site=*/-1);
    const auto indefinite_conv_values = build_indefinite_conv<Real>(
        integrator, files.prefix(), settings.bond_dim, settings.reltol);
    const Complex direct = integrator.integrate();
    const Complex indefinite = Integrator::integrate_indefinite(
        indefinite_values, integrator.grid(), a, b);
    const Complex indefinite_conv = Integrator::integrate_indefinite(
        indefinite_conv_values, integrator.grid(), a, b);
    require_indefinite_close(indefinite, direct);
    require_indefinite_close(indefinite_conv, direct, "indefinite_conv");
    return {direct, indefinite, indefinite_conv};
}

template<class Real>
void check_real_type(int interval_index)
{
    const auto [a, b] = integration_interval<Real>(interval_index);
    CAPTURE(a, b);
    const auto polynomial = integrate_quantics<Real>(
        [](Real x) -> Real { return x * x; }, a, b);
    const Real expected_value = (b * b * b - a * a * a) / Real(3);
    const std::complex<Real> expected(expected_value);
    require_relative_close(polynomial.direct, expected);
    require_indefinite_analytic_close(polynomial.indefinite, expected);
    require_indefinite_analytic_close(polynomial.indefinite_conv, expected, "indefinite_conv");
}

template<class Real>
void check_smooth_non_polynomial_type(int interval_index)
{
    const auto [a, b] = integration_interval<Real>(interval_index);
    CAPTURE(a, b);
    const Real frequency = Real(17);
    const auto answer = integrate_quantics<Real>(
        [frequency](Real x) -> Real {
            return exp_value(x) * cos_value(frequency * x);
        },
        a, b);

    const auto primitive = [frequency](Real x) -> Real {
        const Real angle = frequency * x;
        return Real(exp_value(x) *
                    (cos_value(angle) + frequency * sin_value(angle)) /
                    (Real(1) + frequency * frequency));
    };
    const Real expected_value = primitive(b) - primitive(a);
    const std::complex<Real> expected(expected_value);
    require_relative_close(answer.direct, expected);
    require_indefinite_analytic_close(answer.indefinite, expected);
    require_indefinite_analytic_close(answer.indefinite_conv, expected, "indefinite_conv");
}

template<class Real>
void check_complex_type(int interval_index)
{
    using Complex = std::complex<Real>;
    const auto [a, b] = integration_interval<Real>(interval_index);
    CAPTURE(a, b);
    const auto answer = integrate_quantics<Real>(
        [](Real x) -> Complex {
            return Complex(cos_value(x), sin_value(x));
        },
        a, b);

    const Complex expected(sin_value(b) - sin_value(a),
                           cos_value(a) - cos_value(b));
    require_relative_close(answer.direct, expected);
    require_indefinite_analytic_close(answer.indefinite, expected);
    require_indefinite_analytic_close(answer.indefinite_conv, expected, "indefinite_conv");
}

} // namespace

TEST_CASE("direct and batched range sums match explicit MPS values",
          "[integrator][range]")
{
    using Complex = std::complex<double>;
    using Integrator = QuanticsIntegrator1D<Complex>;
    using Sint = util::i128;

    const auto interval_index = GENERATE(0, 1, 2);
    const auto [a, b] = integration_interval<double>(interval_index);
    CAPTURE(a, b);
    constexpr int nbits = 5;
    constexpr int point_count = 1 << nbits;
    const std::vector<Eigen::Index> bonds = {1, 2, 3, 2, 2, 1};
    std::vector<Tensor3D<Complex>> cores;
    cores.reserve(nbits);
    for (int site = 0; site < nbits; ++site) {
        Tensor3D<Complex> core(
            bonds[static_cast<std::size_t>(site)],
            2,
            bonds[static_cast<std::size_t>(site + 1)]);
        for (Eigen::Index left = 0; left < core.n_left; ++left) {
            for (Eigen::Index bit = 0; bit < core.n_phys; ++bit) {
                for (Eigen::Index right = 0; right < core.n_right; ++right) {
                    const double base =
                        0.03 * (site + 1) + 0.07 * (left + 1)
                        + 0.05 * (bit + 1) + 0.02 * (right + 1);
                    core(left, bit, right) =
                        Complex(base, 0.1 * base + 0.003 * (bit - left));
                }
            }
        }
        cores.push_back(std::move(core));
    }

    typename Integrator::Grid grid(a, b, nbits);
    typename Integrator::State values(
        std::move(cores), /*max_bond_dim=*/0, /*reltol=*/-1.0,
        /*working_site=*/-1);
    const Integrator integrator(std::move(grid), std::move(values));

    auto indefinite_values = make_indefinite_integral(
        integrator.values(), /*max_chi=*/0, /*reltol=*/-1.0, "zip-up");
    indefinite_values *= Complex(integrator.grid().delta_volume(), 0.0);
    TemporaryFitFiles files;
    const auto indefinite_conv_values = build_indefinite_conv<double>(
        integrator, files.prefix(), /*max_bond_dim=*/0, /*reltol=*/-1.0);
    double maximum_indefinite_error = 0.0;
    double maximum_indefinite_conv_error = 0.0;

    std::vector<Complex> prefix(point_count + 1, Complex(0.0));
    for (int index = 0; index < point_count; ++index) {
        std::vector<int> bits(nbits);
        for (int site = 0; site < nbits; ++site)
            bits[static_cast<std::size_t>(site)] = (index >> site) & 1;
        prefix[static_cast<std::size_t>(index + 1)] =
            prefix[static_cast<std::size_t>(index)]
            + integrator.values().eval(bits);
    }

    std::vector<typename Integrator::IndexRange> ranges;
    std::vector<Complex> expected;
    for (int begin = 0; begin <= point_count; ++begin) {
        for (int end = begin; end <= point_count; ++end) {
            const typename Integrator::IndexRange range{
                Sint(begin), Sint(end)};
            const Complex answer =
                prefix[static_cast<std::size_t>(end)]
                - prefix[static_cast<std::size_t>(begin)];
            ranges.push_back(range);
            expected.push_back(answer);

            const Complex direct = integrator.sum_v2(range);
            REQUIRE(std::abs(direct - answer)
                    < 1e-12 * (1.0 + std::abs(answer)));

            const Complex integrated = integrator.integrate(range);
            const Complex expected_integral =
                answer * integrator.grid().delta_volume();
            REQUIRE(std::abs(integrated - expected_integral)
                    < 1e-12 * (1.0 + std::abs(expected_integral)));

            // F_N is not represented by the current indefinite MPS, so only
            // compare intervals whose upper endpoint is a represented point.
            if (end < point_count) {
                const double x_min =
                    integrator.grid().get_a()
                    + double(begin) * integrator.grid().get_dx();
                const double x_max =
                    integrator.grid().get_a()
                    + double(end) * integrator.grid().get_dx();
                const Complex indefinite = Integrator::integrate_indefinite(
                    indefinite_values, integrator.grid(), x_min, x_max);
                const double error = std::abs(indefinite - integrated);
                maximum_indefinite_error =
                    std::max(maximum_indefinite_error, error);
                REQUIRE(error
                        < 1e-12 * (1.0 + std::abs(integrated)));

                const Complex indefinite_conv = Integrator::integrate_indefinite(
                    indefinite_conv_values, integrator.grid(), x_min, x_max);
                const double conv_error = std::abs(indefinite_conv - integrated);
                maximum_indefinite_conv_error =
                    std::max(maximum_indefinite_conv_error, conv_error);
                CAPTURE(begin, end);
                REQUIRE(conv_error < 1e-12 * (1.0 + std::abs(integrated)));
                REQUIRE(std::abs(indefinite_conv - expected_integral)
                        < 1e-12 * (1.0 + std::abs(expected_integral)));
            }
        }
    }

    std::cout << "maximum scalar indefinite/direct range error = "
              << maximum_indefinite_error << '\n'
              << "maximum scalar indefinite_conv/direct range error = "
              << maximum_indefinite_conv_error << '\n';

    const std::vector<Complex> batched = integrator.sum_batch_v2(ranges);
    REQUIRE(batched.size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        REQUIRE(std::abs(batched[index] - expected[index])
                < 1e-12 * (1.0 + std::abs(expected[index])));
    }

    const std::vector<Complex> integrated_batch =
        integrator.integrate_lrange(ranges);
    REQUIRE(integrated_batch.size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const Complex expected_integral =
            expected[index] * integrator.grid().delta_volume();
        REQUIRE(std::abs(integrated_batch[index] - expected_integral)
                < 1e-12 * (1.0 + std::abs(expected_integral)));
    }

    std::vector<double> x_min;
    std::vector<double> x_max;
    std::vector<Complex> represented_expected;
    std::vector<typename Integrator::IndexRange> represented_ranges;
    for (int begin = 0; begin < point_count; ++begin) {
        for (int end = begin; end < point_count; ++end) {
            x_min.push_back(
                integrator.grid().get_a()
                + double(begin) * integrator.grid().get_dx());
            x_max.push_back(
                integrator.grid().get_a()
                + double(end) * integrator.grid().get_dx());
            represented_expected.push_back(
                (prefix[static_cast<std::size_t>(end)]
                 - prefix[static_cast<std::size_t>(begin)])
                * integrator.grid().delta_volume());
            represented_ranges.push_back({Sint(begin), Sint(end)});
        }
    }

    const std::vector<Complex> indefinite_batch =
        Integrator::integrate_indefinite(
            indefinite_values, integrator.grid(), x_min, x_max);
    const std::vector<Complex> indefinite_conv_batch =
        Integrator::integrate_indefinite(
            indefinite_conv_values, integrator.grid(), x_min, x_max);
    const auto represented_direct = integrator.integrate_lrange(represented_ranges);
    REQUIRE(indefinite_batch.size() == represented_expected.size());
    REQUIRE(indefinite_conv_batch.size() == represented_expected.size());
    REQUIRE(represented_direct.size() == represented_expected.size());
    double maximum_batch_indefinite_error = 0.0;
    double maximum_batch_indefinite_conv_error = 0.0;
    for (std::size_t index = 0; index < indefinite_batch.size(); ++index) {
        const double error =
            std::abs(indefinite_batch[index] - represented_expected[index]);
        maximum_batch_indefinite_error =
            std::max(maximum_batch_indefinite_error, error);
        REQUIRE(error
                < 1e-12 * (1.0 + std::abs(represented_expected[index])));
        const double conv_error =
            std::abs(indefinite_conv_batch[index] - represented_direct[index]);
        maximum_batch_indefinite_conv_error =
            std::max(maximum_batch_indefinite_conv_error, conv_error);
        CAPTURE(index);
        REQUIRE(conv_error < 1e-12 * (1.0 + std::abs(represented_direct[index])));
        REQUIRE(std::abs(indefinite_conv_batch[index] - represented_expected[index])
                < 1e-12 * (1.0 + std::abs(represented_expected[index])));
    }
    std::cout << "maximum batched indefinite/direct range error = "
              << maximum_batch_indefinite_error << '\n'
              << "maximum batched indefinite_conv/direct range error = "
              << maximum_batch_indefinite_conv_error << '\n';
}

TEST_CASE("batched integration benchmark on a high-rank trigonometric fit",
          "[integrator][range][benchmark]")
{
    using Complex = std::complex<double>;
    using Integrator = QuanticsIntegrator1D<Complex>;
    using Clock = std::chrono::steady_clock;

    const auto interval_index = GENERATE(0, 1, 2);
    const auto [a, b] = integration_interval<double>(interval_index);
    CAPTURE(a, b);
    constexpr int nbits = 49;
    typename Integrator::Grid grid(a, b, nbits);
    TCI2_1D_runner_opts<Complex> runner_options;
    runner_options.reltol = 1e-12;
    runner_options.pivot1 = grid.coord_to_id(0.37123456789);
    runner_options.fullPiv = true;
    runner_options.cache = CacheLevel::runner;
    typename Integrator::RunnerParam parameters(
        nbits, /*nb_iter=*/60, /*bondDim=*/64, runner_options);

    typename Integrator::FitOptions fit_options;
    fit_options.additional_pivots = {0.11, 0.23, 0.37, 0.53, 0.71, 0.89};
    fit_options.verbose = false;
    fit_options.nb_point_res = 256;

    constexpr double pi = 3.141592653589793238462643383279502884;
    auto function = [pi](double x) -> Complex {
        double value = 0.0;
        for (int mode = 0; mode < 10; ++mode) {
            const double cosine_frequency = double(mode + 1);
            const double sine_frequency = double(1000 + 100 * mode);
            const double coefficient = 0.7 + 0.03 * mode;
            value += coefficient
                   * std::cos(2.0 * pi * cosine_frequency * x)
                   * std::sin(2.0 * pi * sine_frequency * x);
        }
        return Complex(value, 0.0);
    };
    const auto analytical_integral = [pi](double lower, double upper) -> Complex {
        double integral = 0.0;
        for (int mode = 0; mode < 10; ++mode) {
            const double cosine_frequency = double(mode + 1);
            const double sine_frequency = double(1000 + 100 * mode);
            const double coefficient = 0.7 + 0.03 * mode;
            const double plus = 2.0 * pi * (sine_frequency + cosine_frequency);
            const double minus = 2.0 * pi * (sine_frequency - cosine_frequency);
            integral += 0.5 * coefficient *
                ((std::cos(plus * lower) - std::cos(plus * upper)) / plus +
                 (std::cos(minus * lower) - std::cos(minus * upper)) / minus);
        }
        return Complex(integral, 0.0);
    };

    TemporaryFitFiles files;
    const Integrator integrator = Integrator::fit(
        std::move(grid), std::move(function), std::move(parameters),
        files.prefix(), std::move(fit_options), nullptr,
        std::string("zip-up"));
    typename Integrator::State indefinite_values(
        files.prefix() + "_indefinite_int.tt", /*max_bond_dim=*/64,
        /*reltol=*/1e-12, /*working_site=*/-1);

    const auto indefinite_conv_values = build_indefinite_conv<double>(
        integrator, files.prefix(), /*max_bond_dim=*/64, /*reltol=*/1e-12);

    const Eigen::Index fitted_rank = integrator.values().get_chi();
    std::cout << "trigonometric benchmark fitted rank = "
              << fitted_rank << '\n';
    REQUIRE(fitted_rank >= 20);

    const util::i128 point_count = integrator.grid().get_N();
    constexpr int range_count = 100;
    std::vector<typename Integrator::IndexRange> ranges;
    ranges.reserve(range_count);
    for (int index = 1; index <= range_count; ++index) {
        const util::i128 end =
            index == range_count
                ? point_count
                : (point_count * util::i128(index)) /
                      util::i128(range_count);
        ranges.push_back({util::i128(0), end});
    }

    std::vector<Complex> old_sum_results;
    old_sum_results.reserve(ranges.size());
    const Complex cell_volume(integrator.grid().delta_volume(), 0.0);
    const auto old_sum_start = Clock::now();
    for (const auto& range : ranges)
        old_sum_results.push_back(integrator.sum(range) * cell_volume);
    const auto old_sum_end = Clock::now();

    std::vector<Complex> individual_results;
    individual_results.reserve(ranges.size());
    const auto individual_start = Clock::now();
    for (const auto& range : ranges)
        individual_results.push_back(integrator.integrate(range));
    const auto individual_end = Clock::now();

    const auto batch_start = Clock::now();
    const std::vector<Complex> batch_results =
        integrator.integrate_lrange(ranges);
    const auto batch_end = Clock::now();

    const auto index_to_coordinate =
        [&integrator, point_count](util::i128 index) -> double {
        if (index == point_count)
            return integrator.grid().get_b();
        std::vector<int> bits(
            static_cast<std::size_t>(integrator.grid().get_nBits()));
        for (int site = 0; site < integrator.grid().get_nBits(); ++site) {
            bits[static_cast<std::size_t>(site)] =
                ((index >> site) & util::i128(1)) != util::i128(0) ? 1 : 0;
        }
        return integrator.grid().id_to_coord(bits);
    };

    std::vector<double> x_min;
    std::vector<double> x_max;
    x_min.reserve(ranges.size());
    x_max.reserve(ranges.size());
    for (const auto& range : ranges) {
        x_min.push_back(index_to_coordinate(range.begin));
        x_max.push_back(index_to_coordinate(range.end));
    }

    const auto indefinite_start = Clock::now();
    const std::vector<Complex> indefinite_results =
        Integrator::integrate_indefinite(
            indefinite_values, integrator.grid(), x_min, x_max);
    const auto indefinite_end = Clock::now();

    const auto indefinite_conv_start = Clock::now();
    const std::vector<Complex> indefinite_conv_results =
        Integrator::integrate_indefinite(
            indefinite_conv_values, integrator.grid(), x_min, x_max);
    const auto indefinite_conv_end = Clock::now();

    Complex full_integrate_checksum(0.0);
    const auto full_integrate_start = Clock::now();
    for (int repetition = 0; repetition < range_count; ++repetition)
        full_integrate_checksum += integrator.integrate();
    const auto full_integrate_end = Clock::now();

    Complex full_sum_checksum(0.0);
    const auto full_sum_start = Clock::now();
    for (int repetition = 0; repetition < range_count; ++repetition)
        full_sum_checksum += integrator.sum();
    const auto full_sum_end = Clock::now();

    Complex full_sum_v2_checksum(0.0);
    const auto full_sum_v2_start = Clock::now();
    for (int repetition = 0; repetition < range_count; ++repetition)
        full_sum_v2_checksum += integrator.sum_v2();
    const auto full_sum_v2_end = Clock::now();

    const double old_sum_ms =
        std::chrono::duration<double, std::milli>(
            old_sum_end - old_sum_start).count();
    const double individual_ms =
        std::chrono::duration<double, std::milli>(
            individual_end - individual_start).count();
    const double batch_ms =
        std::chrono::duration<double, std::milli>(
            batch_end - batch_start).count();
    const double indefinite_ms =
        std::chrono::duration<double, std::milli>(
            indefinite_end - indefinite_start).count();
    const double indefinite_conv_ms =
        std::chrono::duration<double, std::milli>(
            indefinite_conv_end - indefinite_conv_start).count();
    const double full_integrate_ms =
        std::chrono::duration<double, std::milli>(
            full_integrate_end - full_integrate_start).count();
    const double full_sum_ms =
        std::chrono::duration<double, std::milli>(
            full_sum_end - full_sum_start).count();
    const double full_sum_v2_ms =
        std::chrono::duration<double, std::milli>(
            full_sum_v2_end - full_sum_v2_start).count();
    std::cout << std::setprecision(8)
              << "100 old sum(range) integrations: " << old_sum_ms
              << " ms\n"
              << "100 individual integrate calls: " << individual_ms
              << " ms\n"
              << "one integrate_lrange call: " << batch_ms << " ms\n"
              << "one cached indefinite-integral call: " << indefinite_ms
              << " ms\n"
              << "one cached indefinite_conv call: " << indefinite_conv_ms
              << " ms\n"
              << "100 full-grid integrate() calls: " << full_integrate_ms
              << " ms\n"
              << "100 full-grid sum() calls: " << full_sum_ms
              << " ms\n"
              << "100 full-grid sum_v2() calls: " << full_sum_v2_ms
              << " ms\n";
    if (individual_ms > 0.0)
        std::cout << "sum_v2 speedup over old sum: "
                  << old_sum_ms / individual_ms << "x\n";
    if (batch_ms > 0.0)
        std::cout << "batch speedup over old sum: "
                  << old_sum_ms / batch_ms << "x\n";

    REQUIRE(std::abs(full_integrate_checksum
                     - full_sum_checksum * cell_volume)
            < 1e-10 * (1.0 + std::abs(full_integrate_checksum)));
    REQUIRE(std::abs(full_sum_v2_checksum - full_sum_checksum)
            < 1e-10 * (1.0 + std::abs(full_sum_checksum)));

    REQUIRE(batch_results.size() == individual_results.size());
    REQUIRE(old_sum_results.size() == individual_results.size());
    REQUIRE(indefinite_results.size() == individual_results.size());
    REQUIRE(indefinite_conv_results.size() == individual_results.size());
    double maximum_indefinite_error = 0.0;
    double maximum_indefinite_conv_error = 0.0;
    for (std::size_t index = 0; index < batch_results.size(); ++index) {
        REQUIRE(std::abs(individual_results[index] - old_sum_results[index])
                < 1e-10 * (1.0 + std::abs(old_sum_results[index])));
        REQUIRE(std::abs(batch_results[index] - old_sum_results[index])
                < 1e-10 * (1.0 + std::abs(old_sum_results[index])));
        const double indefinite_error =
            std::abs(indefinite_results[index] - old_sum_results[index]);
        maximum_indefinite_error =
            std::max(maximum_indefinite_error, indefinite_error);
        REQUIRE(indefinite_error
                < 1e-10 * (1.0 + std::abs(old_sum_results[index])));
        const double indefinite_conv_error =
            std::abs(indefinite_conv_results[index] - old_sum_results[index]);
        maximum_indefinite_conv_error =
            std::max(maximum_indefinite_conv_error, indefinite_conv_error);
        CAPTURE(index);
        REQUIRE(indefinite_conv_error
                < 1e-10 * (1.0 + std::abs(old_sum_results[index])));
        const Complex analytical = analytical_integral(x_min[index], x_max[index]);
        REQUIRE(std::abs(old_sum_results[index] - analytical)
                < 1e-10 * (1.0 + std::abs(analytical)));
        REQUIRE(std::abs(indefinite_conv_results[index] - analytical)
                < 1e-10 * (1.0 + std::abs(analytical)));
    }
    std::cout << "maximum benchmark indefinite/direct error = "
              << maximum_indefinite_error << '\n'
              << "maximum benchmark indefinite_conv/direct error = "
              << maximum_indefinite_conv_error << '\n';
}

TEST_CASE("quantics integrator forwards fit diagnostics to spdlog",
          "[integrator][logging]")
{
    using Complex = std::complex<double>;
    using Integrator = QuanticsIntegrator1D<Complex>;

    const auto interval_index = GENERATE(0, 1, 2);
    const auto [a, b] = integration_interval<double>(interval_index);
    CAPTURE(a, b);
    constexpr int nbits = 8;
    typename Integrator::Grid grid(a, b, nbits);
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

    const auto indefinite_conv_values = build_indefinite_conv<double>(
        integrator, files.prefix(), /*max_bond_dim=*/8, /*reltol=*/1e-12);
    const util::i128 end = integrator.grid().get_N() - util::i128(1);
    const double dx = integrator.grid().get_dx();
    const double upper = a + double(end) * dx;
    const Complex direct = integrator.integrate({util::i128(0), end});
    const Complex indefinite_conv = Integrator::integrate_indefinite(
        indefinite_conv_values, integrator.grid(), a, upper);
    // Exact left-rectangle sum for f(x)=x on this represented interval.
    const Complex analytical(
        double(end) * a * dx + 0.5 * double(end) * double(end - 1) * dx * dx, 0.0);
    require_indefinite_close(indefinite_conv, direct, "indefinite_conv");
    require_indefinite_analytic_close(indefinite_conv, analytical, "indefinite_conv");
}

TEST_CASE("quantics integrator supports all real scalar types", "[integrator]")
{
    const auto interval_index = GENERATE(0, 1, 2);
    check_real_type<double>(interval_index);
    check_real_type<dd_128>(interval_index);
    check_real_type<float128>(interval_index);
}

TEST_CASE("quantics integrator handles smooth non-polynomial functions",
          "[integrator]")
{
    const auto interval_index = GENERATE(0, 1, 2);
    check_smooth_non_polynomial_type<double>(interval_index);
    check_smooth_non_polynomial_type<dd_128>(interval_index);
    check_smooth_non_polynomial_type<float128>(interval_index);
}

TEST_CASE("quantics integrator supports all complex scalar types",
          "[integrator]")
{
    const auto interval_index = GENERATE(0, 1, 2);
    check_complex_type<double>(interval_index);
    check_complex_type<dd_128>(interval_index);
    check_complex_type<float128>(interval_index);
}

TEST_CASE("additional pivots resolve marked discontinuities", "[integrator]")
{
    const auto interval_index = GENERATE(0, 1, 2);
    const auto [a, b] = integration_interval<double>(interval_index);
    CAPTURE(a, b);
    QuanticsOptions<double> settings;
    settings.points = {0.75, 0.25, 0.25};
    settings.limit = 100;

    const auto answer = integrate_quantics<double>(
        [](double x) {
            return x < 0.25 ? 1.0 : (x < 0.75 ? 2.0 : 4.0);
        },
        a, b, settings);
    const auto primitive = [](double x) {
        return x + std::max(0.0, x - 0.25) + 2.0 * std::max(0.0, x - 0.75);
    };
    const std::complex<double> expected(primitive(b) - primitive(a));
    require_relative_close(answer.direct, expected);
    require_indefinite_analytic_close(answer.indefinite, expected);
    require_indefinite_analytic_close(answer.indefinite_conv, expected, "indefinite_conv");
}

TEST_CASE("quantics integrator resolves an unmarked cusp", "[integrator]")
{
    const auto interval_index = GENERATE(0, 1, 2);
    const auto [a, b] = integration_interval<double>(interval_index);
    CAPTURE(a, b);
    constexpr double cusp = 0.123456789;
    const auto answer = integrate_quantics<double>(
        [](double x) { return std::abs(x - cusp); }, a, b);
    const auto primitive = [](double x) {
        const double offset = x - cusp;
        return 0.5 * offset * std::abs(offset);
    };
    const double expected = primitive(b) - primitive(a);
    const std::complex<double> analytical(expected);
    require_relative_close(answer.direct, analytical);
    require_indefinite_analytic_close(answer.indefinite, analytical);
    require_indefinite_analytic_close(answer.indefinite_conv, analytical, "indefinite_conv");
}

TEST_CASE("quantics integrator resolves a narrow localized peak",
          "[integrator]")
{
    const auto interval_index = GENERATE(0, 1, 2);
    const auto [a, b] = integration_interval<double>(interval_index);
    CAPTURE(a, b);
    constexpr double centre = 0.3712345;
    constexpr double width = 0.01;
    QuanticsOptions<double> settings;
    settings.limit = 200;

    const auto answer = integrate_quantics<double>(
        [](double x) {
            const double offset = x - centre;
            return 1.0 / (offset * offset + width * width);
        },
        a, b, settings);
    const double expected =
        (std::atan((b - centre) / width) -
         std::atan((a - centre) / width)) /
        width;
    const std::complex<double> analytical(expected);
    require_relative_close(answer.direct, analytical);
    require_indefinite_analytic_close(answer.indefinite, analytical);
    require_indefinite_analytic_close(answer.indefinite_conv, analytical, "indefinite_conv");
}

TEST_CASE("oscillatory integrals converge", "[integrator]")
{
    const auto interval_index = GENERATE(0, 1, 2);
    const auto [a, b] = integration_interval<double>(interval_index);
    CAPTURE(a, b);
    QuanticsOptions<double> settings;
    settings.limit = 200;

    const auto oscillatory = integrate_quantics<double>(
        [](double x) { return std::cos(200.0 * x); },
        a, b, settings);
    const std::complex<double> expected(
        (std::sin(200.0 * b) - std::sin(200.0 * a)) / 200.0);
    require_relative_close(oscillatory.direct, expected);
    require_indefinite_analytic_close(oscillatory.indefinite, expected);
    require_indefinite_analytic_close(oscillatory.indefinite_conv, expected, "indefinite_conv");
}
