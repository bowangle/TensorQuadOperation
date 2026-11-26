#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <type_double_double.h>
#include <type_float128_boost.h>
#include "QuanticsIntegration/quantics_integrator_kD.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int bits = 4;
constexpr int samples = 1 << bits;
constexpr double h = 1.0 / samples;
constexpr int plot_dataset_count = 3 * 3; // Dimensions × precisions.
int plot_datasets_written = 0;

double polynomial(int dimension, double x, double y = 0, double z = 0)
{
    if (dimension == 1) return 1 + 2*x;
    if (dimension == 2) return 1 + 2*x + 3*y + 7*x*y;
    return 1 + 2*x + 3*y + 5*z + 7*x*y + 11*y*z + 13*x*z + 17*x*y*z;
}

template<typename Value>
TensorQuadOperation::TTN<Value> make_polynomial_ttn(int dimension, int bit_count = bits)
{
    const double spacing = std::ldexp(1.0, -bit_count);
    using Core = xfac_quad::Tensor3D<Value>;
    xfac_quad::TensorTree<Value> body(xfac_quad::makeTuckerTree(dimension, bit_count));
    const auto& branches = *body.tree.branches;
    for (int axis = 0; axis < dimension; ++axis) {
        const auto& branch = branches.at(axis);
        for (int bit = 0; bit < bit_count; ++bit) {
            const int node = branch.nodes.at(bit);
            const int next = bit + 1 < bit_count ? branch.nodes.at(bit + 1) : branch.attachment;
            const auto& neighbors = body.tree.neigh.at(node);
            const int right_axis = next >= 0 ? neighbors.pos(next)
                                             : 1 - neighbors.pos(branch.nodes.at(bit - 1));
            const double place_value = std::ldexp(spacing, bit);
            Core core(bit == 0 ? 1 : 2, 2, 2);
            for (int value = 0; value < 2; ++value) {
                // Start with [1, x] at the leaf. Reading each new bit adds
                // its place value to x, yielding the branch basis [1, v_i].
                core(0, 0, value) = 1;
                core(0, 1, value) = value * place_value;
                if (bit > 0) core(1, 1, value) = 1;
            }
            if (dimension < 3 && axis == 0 && bit == bit_count - 1) {
                // With no junction, fold the coefficients into the first
                // branch's endpoint: [1 + 2*x] or [1 + 2*x, 3 + 7*x].
                Core endpoint(core.n_rows, dimension, 2);
                for (int value = 0; value < 2; ++value) {
                    for (Eigen::Index left = 0; left < core.n_rows; ++left) {
                        const Value constant = core(left, 0, value);
                        const Value linear = core(left, 1, value);
                        endpoint(left, 0, value) = constant + Value(2) * linear;
                        if (dimension == 2)
                            endpoint(left, 1, value) = Value(3) * constant + Value(7) * linear;
                    }
                }
                core = std::move(endpoint);
            }
            if (right_axis == 1) {
                body.M.at(node) = std::move(core);
            } else {
                Core swapped(core.n_cols, core.n_rows, core.n_slices);
                for (int value = 0; value < 2; ++value)
                    swapped.slice(value) = core.slice_const(value).transpose();
                body.M.at(node) = std::move(swapped);
            }
        }
    }

    if (dimension == 3) {
        const int junction_id = branches.front().attachment;
        const auto& neighbors = body.tree.neigh.at(junction_id);
        const int x_axis = neighbors.pos(branches.at(0).nodes.back());
        const int y_axis = neighbors.pos(branches.at(1).nodes.back());
        const int z_axis = neighbors.pos(branches.at(2).nodes.back());
        // Mask bits select the linear basis function for x, y and z, respectively.
        constexpr std::array<double, 8> coefficients{1, 2, 3, 7, 5, 13, 11, 17};
        Core junction(2, 2, 2);
        for (int mask = 0; mask < 8; ++mask) {
            std::array<int, 3> index{};
            index[x_axis] = mask & 1;
            index[y_axis] = (mask >> 1) & 1;
            index[z_axis] = (mask >> 2) & 1;
            junction(index[0], index[1], index[2]) = coefficients.at(mask);
        }
        body.M.at(junction_id) = std::move(junction);
    }
    return TensorQuadOperation::TTN<Value>(std::move(body), /*max_bond_dim=*/0, /*reltol=*/-1, /*w=*/-1);
}

double discrete_reference(int dimension, int n)
{
    double sum = 0;
    for (int j1 = 0; j1 <= n; ++j1) {
        const int j2_max = dimension >= 2 ? n - j1 : 0;
        for (int j2 = 0; j2 <= j2_max; ++j2) {
            const int j3_max = dimension == 3 ? n - j1 - j2 : 0;
            for (int j3 = 0; j3 <= j3_max; ++j3)
                sum += polynomial(dimension, j1*h, j2*h, j3*h);
        }
    }
    for (int axis = 0; axis < dimension; ++axis)
        sum *= h;
    return sum;
}

// A degree-r square-free monomial contributes h^(dim+r) * choose(n+dim, dim+r).
// Each weighted index supplies z/(1-z)^2, each unweighted index supplies
// 1/(1-z), and the slack index supplies 1/(1-z). Take the coefficient of z^n.
template<typename Real>
Real closed_form_reference(int dimension, util::i128 n, const Real& spacing)
{
    constexpr int coefficients[3][4] = {{1, 2, 0, 0}, {1, 5, 7, 0}, {1, 10, 31, 17}};
    Real sum(0);
    for (int degree = 0; degree <= dimension && degree <= n; ++degree) {
        Real term(coefficients[dimension - 1][degree]);
        for (int i = 0; i < dimension + degree; ++i)
            term *= (Real(n + dimension - i) * spacing) / Real(i + 1);
        sum += term;
    }
    return sum;
}

template<typename Real>
Real continuous_reference(int dimension, const Real& t)
{
    const Real t2 = t * t;
    if (dimension == 1) return t + t2;
    if (dimension == 2)
        return t2 * (Real(1) / Real(2) + t * (Real(5) / Real(6) + t * Real(7) / Real(24)));
    return t2 * t * (Real(1) / Real(6) + t * (Real(10) / Real(24) +
        t * (Real(31) / Real(120) + t * Real(17) / Real(720))));
}

template<typename Real>
Real relative_tolerance()
{
    if constexpr (std::is_same_v<Real, double>)
        return 3e-12;
    else
        return Real(100000) * Eigen::NumTraits<Real>::epsilon();
}

} // namespace

TEMPLATE_TEST_CASE("Simplex integration matches the inclusive grid sum in 1D, 2D, and 3D",
                   "[simplex]", double, dd_128, float128)
{
    using Real = TestType;
    using Value = std::complex<Real>;
    using TestIntegrator = QuanticsIntegration::QuanticsIntegrator_kD<Value>;
    using std::abs;

    // 1D skips the merge, 2D closes a direct branch bond, and 3D absorbs a junction.
    const int dimension = GENERATE(1, 2, 3);
    CAPTURE(dimension);
    const typename TestIntegrator::mGrid grid(std::vector<Real>(dimension, Real(0)),
                                               std::vector<Real>(dimension, Real(1)),
                                               std::vector<int>(dimension, bits));
    const auto values = make_polynomial_ttn<Value>(dimension);
    const std::array<Real, 3> point{Real(2*h), Real(7*h), Real(11*h)};
    const std::vector<Real> gaps(point.begin(), point.begin() + dimension);
    REQUIRE(values.eval(grid.coord_to_id(gaps)) == Value(Real(polynomial(dimension, 2*h, 7*h, 11*h))));

    const auto [integral, time_grid] = TestIntegrator::make_integration_simplex(
        values, grid, /*kernels=*/{}, /*kernel_grid=*/grid, /*save_prefix=*/"",
        /*max_bond_dim=*/0, /*reltol=*/Real(-1), /*final_compress=*/false);
    REQUIRE(time_grid.get_a() == Real(0));
    REQUIRE(time_grid.get_b() == Real(1));
    REQUIRE(time_grid.get_nBits() == bits);
    REQUIRE(time_grid.get_dx() == Real(h));

    std::vector<Real> times;
    times.reserve(samples);
    for (int n = 0; n < samples; ++n)
        times.push_back(Real(n*h));
    const auto results = TestIntegrator::integrate_simplex(integral, time_grid, times);
    REQUIRE(results.size() == times.size());

    // The dyadic fixture and its small discrete sums are exact even in double.
    // Compare errors in the tested type. Allow for double roundoff at t=0.
    const Real tolerance = relative_tolerance<Real>();
    const auto check_real = [&](int n, const Real& expected) {
        const Real actual = results.at(n).real();
        const Real error = abs(actual - expected);
        const Real allowed = tolerance * abs(expected);
        CAPTURE(n, actual, expected, error, allowed);
        CHECK(error <= allowed);
    };
    for (int n = 0; n < samples; ++n) {
        check_real(n, Real(discrete_reference(dimension, n)));
        const Real imaginary_error = abs(results.at(n).imag());
        CAPTURE(n, imaginary_error);
        CHECK(imaginary_error <= tolerance);
    }
    constexpr std::array<double, 3> at_zero{h, h*h, h*h*h};
    constexpr std::array<double, 3> at_half{27.0 / 32.0, 22590.0 / 65536.0, 1757646.0 / 16777216.0};
    check_real(0, Real(at_zero.at(dimension - 1)));
    check_real(samples / 2, Real(at_half.at(dimension - 1)));
}

TEMPLATE_TEST_CASE("30-bit simplex integration matches sampled inclusive grid sums",
                   "[simplex][30bit]", double, dd_128, float128)
{
    using Real = TestType;
    using Value = std::complex<Real>;
    using Integrator = QuanticsIntegration::QuanticsIntegrator_kD<Value>;
    using std::abs;
    constexpr int bit_count = 30;
    constexpr std::int64_t count = std::int64_t(1) << bit_count;
    const Real spacing = Real(1) / Real(count);
    const int dimension = GENERATE(1, 2, 3);
    CAPTURE(dimension);

    // Cross-check the formula against the independent exhaustive four-bit sum.
    for (int n = 0; n < samples; ++n) {
        const double expected = discrete_reference(dimension, n);
        CHECK(abs(closed_form_reference(dimension, n, h) - expected) <= 1e-14 * expected);
    }

    const typename Integrator::mGrid grid(std::vector<Real>(dimension, Real(0)),
                                          std::vector<Real>(dimension, Real(1)),
                                          std::vector<int>(dimension, bit_count));
    const auto values = make_polynomial_ttn<Value>(dimension, bit_count);
    const auto [integral, time_grid] = Integrator::make_integration_simplex(
        values, grid, /*kernels=*/{}, /*kernel_grid=*/grid, /*save_prefix=*/"",
        /*max_bond_dim=*/16, /*reltol=*/Real(8) * Eigen::NumTraits<Real>::epsilon());
    REQUIRE(time_grid.get_nBits() == bit_count);
    REQUIRE(time_grid.get_dx() == spacing);

    // Include off-dyadic indices and the last stored sample, not just round times.
    const std::vector<std::int64_t> indices{0, 1, count / 4 + 1, count / 3,
                                            count / 2, 3 * count / 4 - 1, count - 1};
    std::vector<Real> times;
    for (const auto n : indices)
        times.push_back(Real(n) * spacing);
    const auto results = Integrator::integrate_simplex(integral, time_grid, times);
    REQUIRE(results.size() == indices.size());
    const Real tolerance = relative_tolerance<Real>();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto n = indices[i];
        const Real expected = closed_form_reference(dimension, n, spacing);
        const Real actual = results[i].real();
        const Real error = abs(actual - expected);
        // For the tiny near-origin values, use an absolute roundoff bound
        // referenced to the full curve; a relative bound isn't meaningful there.
        const Real scale = n <= 1 ? closed_form_reference(dimension, count - 1, spacing) : abs(expected);
        const Real allowed = tolerance * scale;
        CAPTURE(n, actual, expected, error, allowed);
        CHECK(error <= allowed);
        CHECK(abs(results[i].imag()) <= allowed);
    }
}

TEMPLATE_TEST_CASE("High-resolution simplex integration against discrete and continuous references",
                   "[simplex]", double, dd_128, float128)
{
    using Real = TestType;
    using Value = std::complex<Real>;
    using Integrator = QuanticsIntegration::QuanticsIntegrator_kD<Value>;
    using std::abs;
    using Index = util::i128;
    constexpr int bit_count = std::is_same_v<Real, double> ? 50 : 100;
    constexpr int time_count = 1000;
    constexpr Index count = Index(1) << bit_count;
    const Real spacing = Real(1) / Real(count);
    const int dimension = GENERATE(1, 2, 3);
    CAPTURE(dimension, bit_count);

    constexpr auto precision = std::is_same_v<Real, double> ? "double"
        : std::is_same_v<Real, dd_128> ? "dd_128" : "float128";
    const std::filesystem::path plot_dir("test/output/simplex_plot");
    std::filesystem::create_directories(plot_dir);
    const auto dataset = plot_dir / (std::string(precision) + "_" + std::to_string(dimension) + "D");

    const typename Integrator::mGrid grid(std::vector<Real>(dimension, Real(0)),
                                          std::vector<Real>(dimension, Real(1)),
                                          std::vector<int>(dimension, bit_count));
    const auto values = make_polynomial_ttn<Value>(dimension, bit_count);
    const auto [integral, time_grid] = Integrator::make_integration_simplex(
        values, grid, /*kernels=*/{}, /*kernel_grid=*/grid, /*save_prefix=*/"",
        /*max_bond_dim=*/16, /*reltol=*/Real(8) * Eigen::NumTraits<Real>::epsilon(),
        /*final_compress=*/true, /*path_logger=*/dataset.string() + "_");
    REQUIRE(time_grid.get_nBits() == bit_count);
    REQUIRE(time_grid.get_dx() == spacing);

    std::array<Index, time_count> indices{};
    std::vector<Real> times;
    times.reserve(time_count);
    for (int i = 0; i < time_count; ++i) {
        // Sample evenly in index space, including zero and the last stored time.
        indices[i] = (count - 1) * i / (time_count - 1);
        times.push_back(Real(indices[i]) * spacing);
    }
    const auto results = Integrator::integrate_simplex(integral, time_grid, times);
    REQUIRE(results.size() == times.size());

    const Real end_value = closed_form_reference(dimension, count - 1, spacing);
    const Real discrete_budget = relative_tolerance<Real>() * end_value;
    // Each degree-m term differs by replacing t^m with a product of m factors
    // t + (dimension - i)*h. Telescoping those products bounds the combined
    // quadrature error on [0,1) by 32*h for all three test polynomials.
    const Real quadrature_budget = Real(32) * spacing;
    const Real continuous_budget = discrete_budget + quadrature_budget;
    std::ofstream csv(dataset.string() + ".csv");
    REQUIRE(csv.is_open());
    csv << "t,mps,discrete,continuous,abs_discrete,abs_continuous,rel_discrete,rel_continuous,abs_quadrature\n"
        << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10);

    Real max_discrete(0), max_continuous(0), max_quadrature(0);
    Real max_discrete_relative(0), max_continuous_relative(0);
    for (int i = 0; i < time_count; ++i) {
        const Real& time = times[i];
        const Real discrete = closed_form_reference(dimension, indices[i], spacing);
        const Real continuous = continuous_reference(dimension, time);
        const Real discrete_error = abs(results[i] - Value(discrete));
        const Real continuous_error = abs(results[i] - Value(continuous));
        const Real quadrature_error = abs(discrete - continuous);
        max_discrete = std::max(max_discrete, discrete_error);
        max_continuous = std::max(max_continuous, continuous_error);
        max_quadrature = std::max(max_quadrature, quadrature_error);
        csv << time << ',' << results[i].real() << ',' << discrete << ',' << continuous << ','
            << discrete_error << ',' << continuous_error << ',';
        if (i > 0) {
            const Real discrete_relative = discrete_error / abs(discrete);
            const Real continuous_relative = continuous_error / abs(continuous);
            max_discrete_relative = std::max(max_discrete_relative, discrete_relative);
            max_continuous_relative = std::max(max_continuous_relative, continuous_relative);
            csv << discrete_relative << ',' << continuous_relative;
        } else {
            // The continuous relative error is undefined at zero; omit both there.
            csv << ',';
        }
        csv << ',' << quadrature_error << '\n';
        CAPTURE(i, time, discrete_error, continuous_error, quadrature_error);
        CHECK(discrete_error <= discrete_budget);
        CHECK(continuous_error <= continuous_budget);
    }

    csv.close();
    REQUIRE_FALSE(csv.fail());
    ++plot_datasets_written;
    std::ostringstream report;
    report << std::scientific << std::setprecision(8)
           << precision << ", " << dimension << "D, " << bit_count << " bits, " << time_count << " times\n"
           << "  MPS vs discrete:   max abs=" << max_discrete
           << ", max rel (t>0)=" << max_discrete_relative << '\n'
           << "  MPS vs continuous: max abs=" << max_continuous
           << ", max rel (t>0)=" << max_continuous_relative << '\n'
           << "  Discrete vs continuous: max abs=" << max_quadrature << '\n';
    std::cout << report.str();
}

int main(int argc, char* argv[])
{
    const int test_status = Catch::Session().run(argc, argv);
    // Do not plot stale CSVs after a filtered run or an early failure.
    if (plot_datasets_written != plot_dataset_count)
        return test_status;

    std::cout << std::flush;
    if (std::system("gnuplot test/simplex_plot/plot.gp") != 0) {
        std::cerr << "Could not generate simplex SVGs. Check that gnuplot is on PATH "
                     "and run from the TensorQuadOperation project root.\n";
        return test_status != 0 ? test_status : EXIT_FAILURE;
    }
    return test_status;
}
