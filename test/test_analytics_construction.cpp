#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "generic_construction.hpp"
#include <type_double_double.h>
#include <type_float128_boost.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace analytic_mps_construction_tests {

// Construction checks only: no function evaluation or numerical accuracy tests.
template<typename Scalar>
void require_mps_structure(const MPS<Scalar>& mps, int nbit, int max_bond_dim)
{
    const auto& cores = mps.get_core();
    REQUIRE(cores.size() == static_cast<std::size_t>(nbit));
    REQUIRE_FALSE(cores.empty());
    REQUIRE(cores.front().n_left == 1);
    REQUIRE(cores.back().n_right == 1);

    for (std::size_t bit = 0; bit < cores.size(); ++bit) {
        CAPTURE(bit);
        const auto& core = cores[bit];
        REQUIRE(core.n_phys == 2);
        REQUIRE(core.n_left > 0);
        REQUIRE(core.n_right > 0);
        REQUIRE(core.n_left <= max_bond_dim);
        REQUIRE(core.n_right <= max_bond_dim);
        REQUIRE(core.data.size() == static_cast<std::size_t>(
            core.n_left * core.n_phys * core.n_right));
        if (bit > 0) {
            REQUIRE(cores[bit - 1].n_right == core.n_left);
        }
    }
}

template<typename Scalar, typename Grid>
void require_construction(
    const std::pair<MPS<Scalar>, Grid>& result,
    typename Grid::scalar_type a,
    typename Grid::scalar_type b,
    int nbit,
    int max_bond_dim)
{
    require_mps_structure(result.first, nbit, max_bond_dim);
    const auto& grid = result.second;
    REQUIRE(grid.get_a() == a);
    REQUIRE(grid.get_b() == b);
    REQUIRE(grid.get_nBits() == nbit);
    REQUIRE(grid.get_N() == (typename Grid::index_type(1) << nbit));
    REQUIRE(grid.get_dx() > typename Grid::scalar_type(0));
}

// Accuracy checks use exactly 2^4 samples and retain the tested precision.
// Smooth functions use |actual-reference|/max(1,|reference|) <= 100*epsilon.
// Discrete basis/step values retain the stronger exact-equality check.
template<typename Real, typename Function>
void require_all_grid_values(
    const char* name,
    const MPS<Real>& mps,
    const QTGrid<Real, util::i128>& grid,
    Function expected,
    bool exact = false)
{
    using std::abs;
    constexpr int nbit = 4;
    constexpr std::size_t count = std::size_t{1} << nbit;
    const Real tolerance = Real(100) * std::numeric_limits<Real>::epsilon();
    CAPTURE(name);
    REQUIRE(grid.get_nBits() == nbit);
    std::vector<int> bits(static_cast<std::size_t>(nbit));
    for (std::size_t index = 0; index < count; ++index) {
        for (int bit = 0; bit < nbit; ++bit) {
            bits[static_cast<std::size_t>(bit)] =
                static_cast<int>((index >> bit) & std::size_t{1});
        }
        const Real x = grid.id_to_coord(bits);
        const Real actual = mps.eval(bits);
        const Real reference = expected(x, index);
        const Real reference_size = abs(reference);
        const Real scale = reference_size > Real(1) ? reference_size : Real(1);
        const Real difference = actual - reference;
        const Real error = Real(abs(difference)) / scale;
        CAPTURE(index, x, actual, reference, error, tolerance);
        if (exact) {
            REQUIRE(actual == reference);
        } else {
            REQUIRE(error <= tolerance);
        }
    }

}

template<typename Real>
Real snapped_coordinate(Real x0, Real a, Real b, Real dx, bool fixed_to_left)
{
    using std::floor;
    using std::ceil;
    if (x0 < a || x0 >= b) return x0;
    const Real index = (x0 - a) / dx;
    const Real snapped = fixed_to_left ? Real(floor(index)) : Real(ceil(index));
    return a + snapped * dx;
}

template<typename Real>
Real jump_value(const std::string& mode)
{
    return mode == "one" ? Real(1) : mode == "half" ? Real(1) / Real(2) : Real(0);
}

template<typename Real>
Real heaviside_value(Real argument, Real at_jump)
{
    return argument > Real(0) ? Real(1) : argument < Real(0) ? Real(0) : at_jump;
}

} // namespace analytic_mps_construction_tests

TEMPLATE_TEST_CASE("Analytic MPS factories construct valid states and grids",
                   "[analytic-mps][construction]", double, dd_128, float128)
{
    using Real = TestType;
    using namespace analytic_mps_construction;
    using analytic_mps_construction_tests::require_construction;
    using analytic_mps_construction_tests::require_mps_structure;

    constexpr int nbit = 4;
    const Real a = Real(-2);
    const Real b = Real(2);

    SECTION("Basis MPS") {
        const std::vector<int> bits{0, 1, 0, 1};
        const auto mps = make_basis_mps<Real>(bits);
        require_mps_structure(mps, nbit, 1);
    }

    SECTION("Point defect MPS") {
        const auto result = make_point_defect_mps<Real>(a, b, nbit, a);
        require_construction(result, a, b, nbit, 2);
    }

    SECTION("Exclusive Heaviside MPS") {
        const Real length = b - a;
        const auto result = make_exclusive_heaviside_mps<Real>(length, nbit);
        require_construction(result, Real(0), length, nbit, 2);
    }

    SECTION("Heaviside MPS") {
        const auto result = make_heaviside_mps<Real>(
            a, b, nbit, Real(1) / Real(3),
            /*rev=*/false, /*fixed_to_left=*/false, "one");
        require_construction(result, a, b, nbit, 2);
    }

    SECTION("Step MPS") {
        const auto result = make_step_mps<Real>(
            a, b, nbit, -Real(1) / Real(2), Real(1) / Real(2),
            /*fixed_to_left=*/true, "half");
        require_construction(result, a, b, nbit, 3);
    }

    SECTION("Exponential MPS") {
        const auto result = make_exp_mps<Real>(
            a, b, nbit, Real(3) / Real(2), -Real(1) / Real(4));
        require_construction(result, a, b, nbit, 1);
    }

    SECTION("Cosine MPS") {
        const auto result = make_cos_mps<Real>(
            a, b, nbit, Real(2), Real(3), Real(1) / Real(4));
        require_construction(result, a, b, nbit, 2);
    }

    SECTION("Sine MPS") {
        const auto result = make_sin_mps<Real>(
            a, b, nbit, Real(2), Real(3), Real(1) / Real(4));
        require_construction(result, a, b, nbit, 2);
    }

    SECTION("Polynomial MPS") {
        const std::vector<Real> coefficients{
            Real(1), Real(-2), Real(3), Real(-4)};
        const auto result = make_polynomial_orderp_mps<Real>(
            a, b, nbit, coefficients);
        require_construction(result, a, b, nbit, 4);
    }
}

TEMPLATE_TEST_CASE("Analytic MPS values match their functions at every 4-bit grid point",
                   "[analytic-mps][accuracy]", double, dd_128)
{
    using Real = TestType;
    using namespace analytic_mps_construction;
    using namespace analytic_mps_construction_tests;

    constexpr int nbit = 4;
    const Real a = Real(-2);
    const Real b = Real(2);
    const std::vector<std::string> modes{"one", "half", "zero"};

    SECTION("Basis MPS") {
        const auto mps = make_basis_mps<Real>({0, 1, 0, 1});
        const QTGrid<Real, util::i128> grid(a, b, nbit);
        require_all_grid_values("Basis MPS", mps, grid,
            [](Real, std::size_t index) { return index == 10 ? Real(1) : Real(0); },
            /*exact=*/true);
    }

    SECTION("Point defect MPS defaults: zero at the first point and one elsewhere") {
        const auto result = make_point_defect_mps<Real>(a, b, nbit, a);
        require_all_grid_values("Point defect MPS defaults", result.first, result.second,
            [](Real, std::size_t index) { return index == 0 ? Real(0) : Real(1); },
            /*exact=*/true);
    }

    SECTION("Point defect MPS: every grid point and background/value combinations") {
        const QTGrid<Real, util::i128> defect_grid(a, b, nbit);
        constexpr std::size_t count = std::size_t{1} << nbit;
        const std::vector<std::pair<Real, Real>> choices{
            {Real(1), Real(0)}, {Real(-2), Real(3)},
            {Real(0), Real(5)}, {Real(2), Real(2)}};
        for (std::size_t defect_index = 0; defect_index < count; ++defect_index) {
            const Real x0 = a + Real(defect_index) * defect_grid.get_dx();
            for (const auto& choice : choices) {
                const Real background = choice.first;
                const Real point_value = choice.second;
                DYNAMIC_SECTION("defect_index=" << defect_index << ", background=" << background
                                << ", point_value=" << point_value) {
                    const auto result = make_point_defect_mps<Real>(
                        a, b, nbit, x0, background, point_value);
                    const int rank_bound = background == Real(0) || background == point_value ? 1 : 2;
                    require_construction(result, a, b, nbit, rank_bound);
                    require_all_grid_values("Point defect MPS", result.first, result.second,
                        [=](Real, std::size_t index) {
                            return index == defect_index ? point_value : background;
                        }, /*exact=*/true);
                }
            }
        }
    }

    SECTION("Exclusive Heaviside MPS: zero at the origin and one elsewhere") {
        const Real length = b - a;
        const auto result = make_exclusive_heaviside_mps<Real>(length, nbit);
        require_construction(result, Real(0), length, nbit, 2);
        require_all_grid_values("Exclusive Heaviside MPS", result.first, result.second,
            [](Real, std::size_t index) { return index == 0 ? Real(0) : Real(1); },
            /*exact=*/true);
    }

    SECTION("Heaviside MPS: every grid jump, both directions and all jump values") {
        const QTGrid<Real, util::i128> jump_grid(a, b, nbit);
        constexpr std::size_t jump_count = std::size_t{1} << nbit;
        for (std::size_t jump_index = 0; jump_index < jump_count; ++jump_index) {
            const Real x0 = a + Real(jump_index) * jump_grid.get_dx();
            for (bool rev : {false, true}) {
                for (bool fixed_to_left : {false, true}) {
                    for (const auto& mode : modes) {
                        DYNAMIC_SECTION("jump_index=" << jump_index << ", x0=" << x0
                                        << ", rev=" << rev << ", left=" << fixed_to_left
                                        << ", " << mode) {
                            const auto result = make_heaviside_mps<Real>(
                                a, b, nbit, x0, rev, fixed_to_left, mode);
                            const Real jump = snapped_coordinate<Real>(
                                x0, a, b, result.second.get_dx(), fixed_to_left);
                            const Real at_jump = jump_value<Real>(mode);
                            require_all_grid_values("Heaviside MPS", result.first, result.second,
                                [=](Real x, std::size_t) {
                                    return heaviside_value<Real>(rev ? jump - x : x - jump, at_jump);
                                }, /*exact=*/true);
                        }
                    }
                }
            }
        }
    }

    SECTION("Step MPS: every ordered pair of grid endpoints and all jump values") {
        const QTGrid<Real, util::i128> endpoint_grid(a, b, nbit);
        const Real dx = endpoint_grid.get_dx();
        constexpr std::size_t endpoint_count = std::size_t{1} << nbit;
        for (std::size_t begin_index = 0; begin_index < endpoint_count; ++begin_index) {
            const Real x0 = a + Real(begin_index) * dx;
            for (std::size_t end_index = begin_index + 1; end_index < endpoint_count; ++end_index) {
                const Real x1 = a + Real(end_index) * dx;
                for (bool fixed_to_left : {false, true}) {
                    for (const auto& mode : modes) {
                        DYNAMIC_SECTION("x0_index=" << begin_index << ", x1_index=" << end_index
                                        << ", left=" << fixed_to_left << ", " << mode) {
                            const auto result = make_step_mps<Real>(
                                a, b, nbit, x0, x1, fixed_to_left, mode);
                            const Real begin = snapped_coordinate<Real>(x0, a, b, dx, fixed_to_left);
                            const Real end = snapped_coordinate<Real>(x1, a, b, dx, fixed_to_left);
                            const Real at_jump = jump_value<Real>(mode);
                            require_all_grid_values("Step MPS", result.first, result.second,
                                [=](Real x, std::size_t) {
                                    return heaviside_value<Real>(x - begin, at_jump) *
                                           heaviside_value<Real>(end - x, at_jump);
                                }, /*exact=*/true);
                        }
                    }
                }
            }
        }
    }

    SECTION("Exponential MPS") {
        const Real rho = Real(3) / Real(2);
        const Real phi = -Real(1) / Real(4);
        const auto result = make_exp_mps<Real>(a, b, nbit, rho, phi);
        require_all_grid_values("Exponential MPS", result.first, result.second,
            [=](Real x, std::size_t) {
                using std::exp;
                const Real exponent = x * phi;
                return Real(rho * exp(exponent));
            });
    }

    SECTION("Cosine MPS") {
        const Real A = Real(2), B = Real(3), C = Real(1) / Real(4);
        const auto result = make_cos_mps<Real>(a, b, nbit, A, B, C);
        require_all_grid_values("Cosine MPS", result.first, result.second,
            [=](Real x, std::size_t) {
                using std::cos;
                const Real angle = B * x + C;
                return Real(A * cos(angle));
            });
    }

    SECTION("Sine MPS") {
        const Real A = Real(2), B = Real(3), C = Real(1) / Real(4);
        const auto result = make_sin_mps<Real>(a, b, nbit, A, B, C);
        require_all_grid_values("Sine MPS", result.first, result.second,
            [=](Real x, std::size_t) {
                using std::sin;
                const Real angle = B * x + C;
                return Real(A * sin(angle));
            });
    }

    SECTION("Polynomial MPS: constant, linear, cubic and degree six") {
        const std::vector<std::vector<Real>> polynomials{
            {Real(5) / Real(2)}, {Real(-1), Real(2)},
            {Real(1), Real(-2), Real(3), Real(-4)},
            {Real(1) / Real(2), -Real(1) / Real(4), Real(1) / Real(8),
             -Real(1) / Real(16), Real(1) / Real(32), -Real(1) / Real(64),
             Real(1) / Real(128)}};
        for (const auto& coefficients : polynomials) {
            DYNAMIC_SECTION("degree=" << coefficients.size() - 1) {
                const auto result = make_polynomial_orderp_mps<Real>(
                    a, b, nbit, coefficients);
                require_all_grid_values("Polynomial MPS", result.first, result.second,
                    [&](Real x, std::size_t) {
                        Real value = Real(0);
                        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
                            value = value * x + *it;
                        }
                        return value;
                    });
            }
        }
    }
}
