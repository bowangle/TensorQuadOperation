#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "type_double_double.h"
#include "fft_tn.hpp"
#include "general_integrator.hpp"
#include "test_common.hpp"
#include "type_int128.h"

namespace {

using Real = dd_128;
using Complex = Cdd_128;
using Sint = util::i128;

constexpr int resolution_min = 10;
constexpr int resolution_max = 100;
constexpr int resolution_step = 10;
constexpr int padding_bits = 10;
constexpr int tci_iterations = 10;
constexpr int tci_max_bond_dimension = 200;
constexpr int tci_error_sample_count = 1024;
const Real tci_reltol = std::numeric_limits<Real>::epsilon() * Real(1000);
constexpr int fft_max_bond_dimension = 200;
const Real fft_reltol = std::numeric_limits<Real>::epsilon() * Real(1000);
constexpr int tci_random_pivot_count = 10;
constexpr int fourier_error_sample_count = 100;

const Real energy_min = Real(-18);
const Real energy_max = Real(22);
const Real d1 = Real(2);
// 7 is dyadically aligned on [-18, 22] at every tested resolution.
const Real d2 = Real(7);

const std::string output_root = "test/output_fft_conv";
const std::string output_path = output_root + "/dd128";
const std::string input_path = output_root + "/input";
const std::string qft_path = "mpo_data/data/";

Complex gaussian(Real x)
{
    using std::exp;
    return {Real(exp(-x * x / Real(2))), Real(0)};
}

Complex affine(Real x)
{
    return {Real(3) * x - Real(2), 0};
}

Complex discontinuous_affine(Real x)
{
    if (x < d1)
        return {-Real(2) * x + Real(5), 0};
    if (x < d2)
        return {Real(5) * x, 0};
    return {-Real(10) * x, 0};
}

struct TestFunction {
    std::string name;
    std::function<Complex(Real)> function;
    std::vector<Real> discontinuities;
    std::vector<Complex> left_limits;
};

const std::vector<TestFunction> functions = {
    {"gaussian", gaussian, {}, {}},
    {"affine", affine, {}, {}},
    {"discontinuous_affine",
     discontinuous_affine,
     {d1, d2},
     {{-Real(2) * d1 + Real(5), 0}, {Real(5) * d2, 0}}}
};

const std::vector<std::string> methods = {
    "vanilla", "trapezoid", "discontinuous_loop", "discontinuous_loopless"
};

std::string prefix(const TestFunction& test_function, int n_bit)
{
    return input_path + "/dd128/" + test_function.name
         + "_nB" + std::to_string(n_bit);
}

// Continuous Fourier transform with the normalization used by FFTmps.
Complex FT_quad(const TestFunction& test_function, Real t)
{
    general_integrator::options<Real> settings;
    // Keep the quadrature reference comfortably more accurate than the
    // 10000*epsilon FFT threshold. The convergence flag is deliberately not
    // asserted because its conservative round-off floor sees cancellation.
    settings.epsabs = Real(100) * std::numeric_limits<Real>::epsilon();
    settings.epsrel = settings.epsabs;
    settings.limit = 200;
    settings.points = test_function.discontinuities;

    const auto result = general_integrator::integrate<Real>(
        [&](Real x) {
            return test_function.function(x)
                 * std::exp(Complex(0, -t * x));
        },
        energy_min,
        energy_max,
        settings);
    return result.value / (Real(2) * magic_tensor_qft::pi<Real>());
}

void fit_tci(const TestFunction& test_function, int n_bit)
{
    QTGrid<Real, Sint> grid(energy_min, energy_max, n_bit);
    TCI2_1D_runner_opts<Complex> opts{
        .reltol = tci_reltol,
        .pivot1 = grid.coord_to_id(Real(0)),
        .fullPiv = true,
        .cache = CacheLevel::runner
    };
    TCI2_1D_runner_param<Complex> parameters(
        n_bit, tci_iterations, tci_max_bond_dimension, opts);
    TCI2_1D_Runner<Complex, Sint> runner(
        grid, parameters, test_function.function);

    std::vector<Real> pivots = {-5, 1.9, d1, 2.1, 5, 6.9, d2, 7.1, 9};
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> distribution(-18.0, 22.0);
    for (int i = 0; i < tci_random_pivot_count; ++i)
        pivots.push_back(Real(distribution(rng)));

    const std::string file_prefix = prefix(test_function, n_bit);
    std::filesystem::create_directories(
        std::filesystem::path(file_prefix).parent_path());
    runner.fit(pivots, false, true, file_prefix, tci_error_sample_count);
    runner.save_to_json(
        file_prefix + "_f_values.json",
        test_function.function(energy_max),
        test_function.discontinuities,
        test_function.left_limits);
}

Real fft_error(const TestFunction& test_function,
               const std::string& method,
               int n_bit,
               int padding)
{
    FFTmps<Complex, Sint> fft(
        prefix(test_function, n_bit), qft_path, padding,
        fft_max_bond_dimension, fft_reltol);

    const Complex f_b = test_function.function(energy_max);
    if (method == "vanilla") {
        fft.fft_vanilla(true, "zip-up");
    } else if (method == "trapezoid") {
        fft.fft_trapez(f_b, true, "zip-up");
    } else if (method == "discontinuous_loop") {
        fft.fft_discontinuous_with_loop(
            test_function.discontinuities, test_function.left_limits,
            f_b, true, "zip-up");
    } else {
        fft.fft_discontinuous_loopless(
            test_function.discontinuities, test_function.left_limits,
            f_b, true, "zip-up");
    }

    std::vector<std::vector<int>> ids;
    std::vector<Real> points;
    ids.reserve(fourier_error_sample_count);
    points.reserve(fourier_error_sample_count);
    for (int i = 0; i < fourier_error_sample_count; ++i) {
        const Real t = -Real(1) + Real(2 * i)
                     / Real(fourier_error_sample_count - 1);
        const auto id = fft.get_grid().coord_to_id(t);
        ids.push_back(id);
        points.push_back(fft.get_grid().id_to_coord(id));
    }

    const auto values = fft.get_mps().eval_list(ids);
    Real max_error = 0;
    for (std::size_t i = 0; i < values.size(); ++i)
        max_error = std::max(
            max_error, std::abs(values[i] - FT_quad(test_function, points[i])));
    return max_error;
}

// Apply the QFT MPO to an exact rank-one constant MPS and inspect only its
// zero-frequency output. This isolates the common QFT/contraction error from
// TCI and from the quadrature corrections.
Real qft_constant_dc_error(int n_bit, int padding)
{
    const int fft_n_bit = n_bit + padding;
    std::vector<Tensor3D<Complex>> cores;
    cores.reserve(fft_n_bit);

    for (int bit = 0; bit < n_bit; ++bit) {
        Tensor3D<Complex> core(1, 2, 1);
        core(0, 0, 0) = Complex(1, 0);
        core(0, 1, 0) = Complex(1, 0);
        cores.push_back(std::move(core));
    }
    for (int bit = 0; bit < padding; ++bit) {
        Tensor3D<Complex> core(1, 2, 1);
        core(0, 0, 0) = Complex(1, 0);
        cores.push_back(std::move(core));
    }

    MPS<Complex> constant_mps(
        FFTmps<Complex, Sint>::reverse_cores(cores),
        fft_max_bond_dimension,
        fft_reltol,
        -1);

    const Sint N_orig = Sint(1) << n_bit;
    const Sint N_fft = Sint(1) << fft_n_bit;
    const Real dx = (energy_max - energy_min) / Real(N_orig);
    constant_mps *= magic_tensor_qft::real_sqrt(Real(N_fft)) * dx
                  / (Real(2) * magic_tensor_qft::pi<Real>());

    auto qft = magic_tensor_qft::load_compressed_qft_mpo<Complex>(
        fft_n_bit, qft_path, fft_reltol, fft_max_bond_dimension);
    const MPS<Complex> transformed = qft._mul(constant_mps, "zip-up");
    const Complex actual = transformed.eval(std::vector<int>(fft_n_bit, 0));
    const Complex expected(
        (energy_max - energy_min)
            / (Real(2) * magic_tensor_qft::pi<Real>()),
        0);
    return std::abs(actual - expected);
}

bool should_reach_roundoff(const std::string& function,
                           const std::string& method)
{
    if (function == "gaussian")
        return true;
    if (function == "affine")
        return method != "vanilla";
    return method == "discontinuous_loop"
        || method == "discontinuous_loopless";
}

} // namespace

TEST_CASE("dd128 FFT methods converge to adaptive-quadrature references",
          "[fft][dd128][convergence]")
{
    std::filesystem::create_directories(output_path);
    std::ofstream output(output_path + "/fft_convergence.csv");
    REQUIRE(output.good());
    output << "function,method,padding_bits,n_bit,max_error\n"
           << std::scientific
           << std::setprecision(std::numeric_limits<Real>::max_digits10);

    for (int n_bit = resolution_min; n_bit <= resolution_max;
         n_bit += resolution_step) {
        for (const auto& test_function : functions)
            fit_tci(test_function, n_bit);

        const Real qft_dc_unpadded = qft_constant_dc_error(n_bit, 0);
        const Real qft_dc_padded = qft_constant_dc_error(n_bit, padding_bits);

        for (const auto& test_function : functions) {
            for (const int padding : {0, padding_bits}) {
                const Real qft_dc_error = padding == 0
                    ? qft_dc_unpadded
                    : qft_dc_padded;
                output << test_function.name << ",qft_constant_dc,"
                       << padding << ',' << n_bit << ',' << qft_dc_error << '\n';
                std::cout << test_function.name << ", qft_constant_dc"
                          << ", padding=" << padding
                          << ", nBit=" << n_bit
                          << ", error=" << qft_dc_error << '\n';

                for (const auto& method : methods) {
                    const Real error = fft_error(
                        test_function, method, n_bit, padding);
                    output << test_function.name << ',' << method << ','
                           << padding << ',' << n_bit << ',' << error << '\n';
                    std::cout << test_function.name << ", " << method
                              << ", padding=" << padding
                              << ", nBit=" << n_bit
                              << ", error=" << error << '\n';

                    if (n_bit == resolution_max
                        && should_reach_roundoff(test_function.name, method)) {
                        INFO("function=" << test_function.name
                             << ", method=" << method
                             << ", padding=" << padding
                             << ", error=" << error);
                        CHECK(error <= Real(10000)
                                    * std::numeric_limits<Real>::epsilon());
                    }
                }
            }
        }
    }
}
