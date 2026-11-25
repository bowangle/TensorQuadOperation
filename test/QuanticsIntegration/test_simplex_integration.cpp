#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <type_double_double.h>
#include <type_float128_boost.h>
#include "QuanticsIntegration/quantics_integrator_kD.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

namespace {

using Scalar = std::complex<double>;
using Integrator = QuanticsIntegration::QuanticsIntegrator_kD<Scalar>;
constexpr int bits = 4;
constexpr int samples = 1 << bits;
constexpr double h = 1.0 / samples;

double polynomial(double x, double y, double z)
{
    return 1 + 2*x + 3*y + 5*z + 7*x*y + 11*y*z + 13*x*z + 17*x*y*z;
}

template<typename Value>
TensorQuadOperation::TTN<Value> make_polynomial_ttn()
{
    using Core = xfac_quad::Tensor3D<Value>;
    xfac_quad::TensorTree<Value> body(xfac_quad::makeTuckerTree(3, bits));
    const auto& branches = *body.tree.branches;
    for (const auto& branch : branches) {
        for (int bit = 0; bit < bits; ++bit) {
            const int node = branch.nodes.at(bit);
            const int next = bit + 1 < bits ? branch.nodes.at(bit + 1) : branch.attachment;
            const int right_axis = body.tree.neigh.at(node).pos(next);
            const double place_value = h * (1 << bit);
            Core core(bit == 0 ? 1 : 2, 2, 2);
            for (int value = 0; value < 2; ++value) {
                // Start with [1, x] at the leaf. Reading each new bit adds
                // its place value to x, yielding the branch basis [1, v_i].
                core(0, 0, value) = 1;
                core(0, 1, value) = value * place_value;
                if (bit > 0) core(1, 1, value) = 1;
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
    return TensorQuadOperation::TTN<Value>(std::move(body), /*max_bond_dim=*/0, /*reltol=*/-1, /*w=*/-1);
}

double discrete_reference(int n)
{
    double sum = 0;
    for (int j1 = 0; j1 <= n; ++j1)
        for (int j2 = 0; j2 <= n - j1; ++j2)
            for (int j3 = 0; j3 <= n - j1 - j2; ++j3)
                sum += polynomial(j1*h, j2*h, j3*h);
    return h*h*h * sum;
}

} // namespace

TEST_CASE("Three-gap simplex integration matches the inclusive grid sum", "[simplex]")
{
    const Integrator::mGrid grid({0, 0, 0}, {1, 1, 1}, {bits, bits, bits});
    const auto values = make_polynomial_ttn<Scalar>();
    // Check that the fixture encodes the intended function before integrating.
    const std::vector<double> gaps{2*h, 7*h, 11*h};
    REQUIRE(values.eval(grid.coord_to_id(gaps)) == Scalar(polynomial(gaps[0], gaps[1], gaps[2])));

    const auto [integral, time_grid] = Integrator::make_integration_simplex(
        values, grid, /*kernels=*/{}, /*kernel_grid=*/grid, /*save_prefix=*/"",
        /*max_bond_dim=*/0, /*reltol=*/-1, /*final_compress=*/false);
    REQUIRE(time_grid.get_a() == 0);
    REQUIRE(time_grid.get_b() == 1);
    REQUIRE(time_grid.get_nBits() == bits);
    REQUIRE(time_grid.get_dx() == h);

    std::vector<double> times;
    times.reserve(samples);
    for (int n = 0; n < samples; ++n)
        times.push_back(n*h);
    const auto results = Integrator::integrate_simplex(integral, time_grid, times);
    REQUIRE(results.size() == times.size());
    for (int n = 0; n < samples; ++n) {
        CAPTURE(n, times.at(n));
        CHECK(results.at(n).real() == Approx(discrete_reference(n)).epsilon(1e-12));
        CHECK(results.at(n).imag() == Approx(0.0).margin(1e-12));
    }
    CHECK(results.front().real() == Approx(h*h*h).epsilon(1e-12));
    CHECK(results.at(samples / 2).real() == Approx(1757646.0 / 16777216.0).epsilon(1e-12));
}

TEMPLATE_TEST_CASE("Three-gap simplex integration matches the inclusive grid sum in extended precision",
                   "[simplex]", dd_128, float128)
{
    using Real = TestType;
    using Value = std::complex<Real>;
    using TestIntegrator = QuanticsIntegration::QuanticsIntegrator_kD<Value>;
    using std::abs;

    const typename TestIntegrator::mGrid grid({Real(0), Real(0), Real(0)},
                                               {Real(1), Real(1), Real(1)}, {bits, bits, bits});
    const auto values = make_polynomial_ttn<Value>();
    const std::vector<Real> gaps{Real(2*h), Real(7*h), Real(11*h)};
    REQUIRE(values.eval(grid.coord_to_id(gaps)) == Value(Real(polynomial(2*h, 7*h, 11*h))));

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
    // Compute errors and compare tolerances in the extended type, not via Approx.
    const Real tolerance = Real(100000) * Eigen::NumTraits<Real>::epsilon();
    const auto check_real = [&](int n, const Real& expected) {
        const Real actual = results.at(n).real();
        const Real error = abs(actual - expected);
        const Real allowed = tolerance * abs(expected);
        CAPTURE(n, actual, expected, error, allowed);
        CHECK(error <= allowed);
    };
    for (int n = 0; n < samples; ++n) {
        check_real(n, Real(discrete_reference(n)));
        const Real imaginary_error = abs(results.at(n).imag());
        CAPTURE(n, imaginary_error);
        CHECK(imaginary_error <= tolerance);
    }
    check_real(0, Real(h*h*h));
    check_real(samples / 2, Real(1757646.0 / 16777216.0));
}
