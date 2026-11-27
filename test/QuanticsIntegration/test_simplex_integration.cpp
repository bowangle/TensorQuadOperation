#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <type_double_double.h>
#include <type_float128_boost.h>
#include "QuanticsIntegration/quantics_integrator_kD.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
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
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int bits = 4;
constexpr int samples = 1 << bits;
constexpr double h = 1.0 / samples;
struct MergeCase {
    const char* method;
    bool merge_trucation;
};
constexpr std::array<MergeCase, 5> merge_cases{{
    {"zip-up", false}, {"zip-up", true},
    {"qrsvd", false}, {"qrsvd", true},
    {"optimize", true},
}};
constexpr auto plot_dataset_count = (3 * 3 + 2) * merge_cases.size() + 2; // Two extra dd_128 rank-30 datasets.
std::size_t plot_datasets_written = 0;

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

constexpr int rank30_degree = 29;
constexpr int rank30 = rank30_degree + 1;
constexpr int rank30_max_bond_dim = 100;
template<typename Real>
const Real rank30_reltol = Real(10000) * Eigen::NumTraits<Real>::epsilon();
// The polynomial has supremum 1e4 on the fitting cube [0,1)^3.
const double rank30_scale = std::expm1(std::log(1e4) / rank30_degree) / 3.0;

constexpr std::uint64_t choose_small(int n, int k)
{
    std::uint64_t value = 1;
    for (int i = 1; i <= k; ++i)
        value = value * (n - i + 1) / i;
    return value;
}

template<typename Value>
TensorQuadOperation::TTN<Value> make_rank30_polynomial_ttn(int bit_count)
{
    using Core = xfac_quad::Tensor3D<Value>;
    using Real = typename Eigen::NumTraits<Value>::Real;
    xfac_quad::TensorTree<Value> body(xfac_quad::makeTuckerTree(3, bit_count));
    const auto& branches = *body.tree.branches;
    for (const auto& branch : branches) {
        for (int bit = 0; bit < bit_count; ++bit) {
            const int node = branch.nodes.at(bit);
            const int next = bit + 1 < bit_count ? branch.nodes.at(bit + 1) : branch.attachment;
            const int right_axis = body.tree.neigh.at(node).pos(next);
            const Real place(std::ldexp(1.0, bit - bit_count));
            std::array<Real, rank30> powers;
            powers[0] = Real(1);
            for (int p = 1; p < rank30; ++p) powers[p] = powers[p - 1] * place;
            const int left_dim = bit == 0 ? 1 : rank30;
            Core core(right_axis == 1 ? left_dim : rank30,
                      right_axis == 1 ? rank30 : left_dim, 2);
            for (int p = 0; p < left_dim; ++p) {
                // Shift the power basis from x to x + place when the bit is set.
                for (int q = p; q < rank30; ++q) {
                    const int row = right_axis == 1 ? p : q;
                    const int col = right_axis == 1 ? q : p;
                    if (p == q) core(row, col, 0) = Value(1);
                    core(row, col, 1) = Value(Real(choose_small(q, p)) * powers[q - p]);
                }
            }
            body.M.at(node) = std::move(core);
        }
    }
    const int junction_id = branches.front().attachment;
    const auto& neighbors = body.tree.neigh.at(junction_id);
    const int x_axis = neighbors.pos(branches.at(0).nodes.back());
    const int y_axis = neighbors.pos(branches.at(1).nodes.back());
    const int z_axis = neighbors.pos(branches.at(2).nodes.back());
    Core junction(rank30, rank30, rank30);
    // Apply the scale to each total degree in the junction, keeping the branch bases unscaled.
    std::array<Real, rank30> scale_powers;
    scale_powers[0] = Real(1);
    for (int r = 1; r < rank30; ++r)
        scale_powers[r] = scale_powers[r - 1] * Real(rank30_scale);
    std::array<int, 3> power{};
    for (int a = 0; a <= rank30_degree; ++a) {
        power[x_axis] = a;
        for (int b = 0; b <= rank30_degree - a; ++b) {
            power[y_axis] = b;
            for (int c = 0; c <= rank30_degree - a - b; ++c) {
                power[z_axis] = c;
                const auto coefficient = choose_small(rank30_degree, a)
                    * choose_small(rank30_degree - a, b) * choose_small(rank30_degree - a - b, c);
                junction(power[0], power[1], power[2]) = Value(Real(coefficient) * scale_powers[a + b + c]);
            }
        }
    }
    body.M.at(junction_id) = std::move(junction);
    return TensorQuadOperation::TTN<Value>(std::move(body), 0, Real(-1), /*w=*/-1);
}

class Rank30Reference {
    using Integer = boost::multiprecision::cpp_int;
    static constexpr int integrated_degree = rank30_degree + 3;
    // The rounded double scale is in [2^-4, 2^-3), so it is exactly P / 2^56.
    static constexpr int scale_bits = std::numeric_limits<double>::digits + 3;
    static constexpr auto lcm = [] {
        std::uint64_t value = 1;
        for (int i = 3; i <= integrated_degree; ++i)
            value = std::lcm(value, static_cast<std::uint64_t>(i));
        return value;
    }();

    int bits_;
    std::array<Integer, integrated_degree> differences_;
    std::array<Integer, rank30> continuous_coefficients_;

    template<typename Real>
    static Real scaled_integer(Integer value, int denominator_bits)
    {
        if (value == 0) return Real(0);
        // Keep 160 leading bits before conversion (more than any tested type).
        const int shift = std::max(0, static_cast<int>(boost::multiprecision::msb(value)) + 1 - 160);
        value >>= shift;
        Real scaled(0);
        using std::ldexp;
        for (int exponent = shift - denominator_bits; value != 0; exponent += 32) {
            const std::uint32_t chunk = static_cast<std::uint32_t>(value & 0xffffffffu);
            scaled += ldexp(Real(chunk), exponent);
            value >>= 32;
        }
        return scaled;
    }

public:
    // Independent reference for the same affine kernel applied to every time.
    static auto weighted_discrete_reference(int bit_count)
    {
        // The summand has degree 29 + 3 in the grid indices. Three inclusive
        // sums raise the degree to at most 35, so 36 exact samples determine it.
        constexpr int terms = integrated_degree + 4;
        const Integer count = Integer(1) << bit_count;
        const Integer scale_numerator = static_cast<std::uint64_t>(std::ldexp(rank30_scale, scale_bits));
        const Integer base_denominator = count << scale_bits;
        std::array<Integer, terms> gap_powers, column, differences;
        for (int gap = 0; gap < terms; ++gap) {
            const Integer base = base_denominator + scale_numerator * gap;
            gap_powers[gap] = 1;
            for (int r = 0; r < rank30_degree; ++r) gap_powers[gap] *= base;
        }
        for (int n = 0; n < terms; ++n) {
            column[n] = 0;
            for (int m1 = 0; m1 <= n; ++m1)
                for (int m2 = 0; m2 <= m1; ++m2)
                    for (int m3 = 0; m3 <= m2; ++m3)
                        column[n] += gap_powers[n - m3] * (count + m1) * (count + m2) * (count + m3);
        }
        for (int r = 0; r < terms; ++r) {
            differences[r] = column[0];
            for (int n = 0; n + r + 1 < terms; ++n)
                column[n] = column[n + 1] - column[n];
        }
        // Denominator: one factor per weight, h^3, and the degree-29 gap power.
        const int denominator_bits = (rank30_degree + 6) * bit_count + rank30_degree * scale_bits;
        return [differences = std::move(differences), denominator_bits](util::i128 n) {
            const Integer index(n);
            Integer numerator = 0;
            Integer binomial = 1;
            for (int r = 0; r < terms && r <= n; ++r) {
                numerator += differences[r] * binomial;
                binomial *= index - r;
                binomial /= r + 1;
            }
            return scaled_integer<double>(std::move(numerator), denominator_bits);
        };
    }

    explicit Rank30Reference(int bit_count) : bits_(bit_count)
    {
        const Integer scale_numerator = static_cast<std::uint64_t>(std::ldexp(rank30_scale, scale_bits));
        const Integer base_denominator = Integer(1) << (bits_ + scale_bits);
        // At total gap m*h, the polynomial base is (D + P*m)/D, with D = 2^bits_ * 2^scale_bits.
        // Sum the Newton expansion of choose(m+2,2) * (D + P*m)^29, then scale by h^3 / D^29.
        std::array<Integer, integrated_degree> column;
        for (int m = 0; m < integrated_degree; ++m) {
            const Integer base_numerator = base_denominator + scale_numerator * m;
            column[m] = (m + 1) * (m + 2) / 2;
            for (int r = 0; r < rank30_degree; ++r) column[m] *= base_numerator;
        }
        for (int r = 0; r < integrated_degree; ++r) {
            differences_[r] = column[0];
            for (int m = 0; m + r + 1 < integrated_degree; ++m)
                column[m] = column[m + 1] - column[m];
        }
        // Common denominator for 1/2 * sum_r choose(29,r) * c^r * t^(r+3)/(r+3).
        Integer scale_power = 1;
        for (int r = 0; r <= rank30_degree; ++r) {
            continuous_coefficients_[r] = scale_power * choose_small(rank30_degree, r) * (lcm / (r + 3));
            continuous_coefficients_[r] <<= (rank30_degree - r) * (bits_ + scale_bits);
            scale_power *= scale_numerator;
        }
    }

    template<typename Real>
    Real discrete(util::i128 n) const
    {
        const Integer index(n);
        Integer numerator = 0;
        Integer binomial = index + 1;
        for (int r = 0; r < integrated_degree && r <= n; ++r) {
            numerator += differences_[r] * binomial;
            binomial *= index - r;
            binomial /= r + 2;
        }
        return scaled_integer<Real>(std::move(numerator), integrated_degree * bits_ + rank30_degree * scale_bits);
    }

    template<typename Real>
    Real continuous(util::i128 n) const
    {
        const Integer index(n);
        Integer numerator = 0;
        for (int r = rank30_degree; r >= 0; --r)
            numerator = numerator * index + continuous_coefficients_[r];
        numerator *= index * index * index;
        return scaled_integer<Real>(std::move(numerator), integrated_degree * bits_ + rank30_degree * scale_bits + 1)
            / Real(lcm);
    }
};

template<typename Real>
Real relative_tolerance()
{
    if constexpr (std::is_same_v<Real, double>)
        return 3e-12;
    else
        return Real(100000) * Eigen::NumTraits<Real>::epsilon();
}

} // namespace

TEST_CASE("Simplex kernel composition weights the input time", "[simplex][kernel]")
{
    using Value = std::complex<double>;
    using TT = TensorQuadOperation::TT<Value>;
    using MPS = TensorQuadOperation::MPS<Value>;
    using MPO = TensorQuadOperation::MPO<Value>;
    using Core = xfac_quad::Tensor3D<Value>;

    // Ordering 2, two bits, and a three-dimensional open MPO attachment.
    Core b0(1, 2, 4), b1(2, 3, 4);
    for (int slot = 0; slot < 4; ++slot) {
        for (int bond = 0; bond < 2; ++bond) {
            b0(0, bond, slot) = Value(1 + slot + 2 * bond, slot - bond);
            for (int attachment = 0; attachment < 3; ++attachment)
                b1(bond, attachment, slot) = Value(2 + 2 * slot + bond + attachment, attachment - slot);
        }
    }
    // A nonconstant rank-two kernel, also stored LSB first.
    Core k0(1, 2, 2), k1(2, 1, 2);
    for (int bit = 0; bit < 2; ++bit) {
        k0(0, 0, bit) = Value(1 + bit, bit);
        k0(0, 1, bit) = Value(bit, 1);
        k1(0, 0, bit) = Value(2 + bit, 0);
        k1(1, 0, bit) = Value(0, 1 + bit);
    }

    const int mpo_ordering = GENERATE(0, 1, 2);
    const int kernel_ordering = GENERATE(0, 1, 2);
    CAPTURE(mpo_ordering, kernel_ordering);
    const MPO branch(TT::change_ordering(TT({b0, b1}, 0, -1, -1, 2), mpo_ordering));
    const MPS kernel(TT::change_ordering(TT({k0, k1}, 0, -1, -1, 2), kernel_ordering));
    const auto weighted = QuanticsIntegration::simplex_helper::add_kernel_k_step(branch, kernel);
    REQUIRE(weighted.get_left_phys() == 2);
    REQUIRE(weighted.get_right_phys() == 2);
    REQUIRE(weighted.get_ordering() == mpo_ordering);
    const auto ordered = TT::change_ordering(weighted, 2);
    const auto& cores = ordered.get_core();
    REQUIRE(cores.size() == 2);
    REQUIRE(cores.front().n_rows == 1);
    REQUIRE(cores.front().n_cols == 4);
    REQUIRE(cores.back().n_rows == 4);
    REQUIRE(cores.back().n_cols == 3);

    for (int input = 0; input < 4; ++input) {
        const int x0 = input & 1, x1 = input >> 1;
        const Value weight = k0(0, 0, x0) * k1(0, 0, x1) + k0(0, 1, x0) * k1(1, 0, x1);
        for (int output = 0; output < 4; ++output) {
            const int slot0 = 2 * (output & 1) + x0, slot1 = 2 * (output >> 1) + x1;
            for (int attachment = 0; attachment < 3; ++attachment) {
                CAPTURE(input, output, attachment);
                const Value original = b0(0, 0, slot0) * b1(0, attachment, slot1)
                    + b0(0, 1, slot0) * b1(1, attachment, slot1);
                Value actual{};
                for (int bond = 0; bond < 4; ++bond)
                    actual += cores[0](0, bond, slot0) * cores[1](bond, attachment, slot1);
                // Small integer-valued complex entries give exact products and sums.
                CHECK(actual == original * weight);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Simplex integration matches the inclusive grid sum in 1D, 2D, and 3D",
                   "[simplex]", double, dd_128, float128)
{
    using Real = TestType;
    using Value = std::complex<Real>;
    using TestIntegrator = QuanticsIntegration::QuanticsIntegrator_kD<Value>;
    using std::abs;

    // 1D skips the merge, 2D closes a direct branch bond, and 3D absorbs a junction.
    const int dimension = GENERATE(1, 2, 3);
    const auto [method, merge_trucation] = GENERATE(Catch::Generators::from_range(merge_cases));
    CAPTURE(dimension, method, merge_trucation);
    const typename TestIntegrator::mGrid grid(std::vector<Real>(dimension, Real(0)),
                                               std::vector<Real>(dimension, Real(1)),
                                               std::vector<int>(dimension, bits));
    const auto values = make_polynomial_ttn<Value>(dimension);
    const std::array<Real, 3> point{Real(2*h), Real(7*h), Real(11*h)};
    const std::vector<Real> gaps(point.begin(), point.begin() + dimension);
    REQUIRE(values.eval(grid.coord_to_id(gaps)) == Value(Real(polynomial(dimension, 2*h, 7*h, 11*h))));

    const auto [integral, time_grid] = TestIntegrator::make_integration_simplex(
        values, grid, /*kernel=*/nullptr, /*kernel_grid=*/{Real(0), Real(1), bits}, /*save_prefix=*/"",
        /*max_bond_dim=*/merge_trucation ? 16 : 0, /*reltol=*/Real(-1),
        /*final_compress=*/false, /*path_logger=*/{}, method, merge_trucation);
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
    const auto [method, merge_trucation] = GENERATE(Catch::Generators::from_range(merge_cases));
    CAPTURE(dimension, method, merge_trucation);

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
        values, grid, /*kernel=*/nullptr, /*kernel_grid=*/{Real(0), Real(1), bit_count}, /*save_prefix=*/"",
        /*max_bond_dim=*/16, /*reltol=*/Real(8) * Eigen::NumTraits<Real>::epsilon(),
        /*final_compress=*/true, /*path_logger=*/{}, method, merge_trucation);
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

namespace {

template<typename Real, bool Rank30, bool FitWithTCI = false>
void check_high_resolution_simplex(int dimension, const std::string& method, bool merge_trucation)
{
    using Value = std::complex<Real>;
    using Integrator = QuanticsIntegration::QuanticsIntegrator_kD<Value>;
    using std::abs;
    using Index = util::i128;
    constexpr int bit_count = std::is_same_v<Real, double> ? 50 : 100;
    constexpr int time_count = 1000;
    constexpr Index count = Index(1) << bit_count;
    const Real spacing = Real(1) / Real(count);
    CAPTURE(dimension, bit_count, method, merge_trucation);
    const auto reference_at = [&](Index n, const Real& time) -> std::pair<Real, Real> {
        if constexpr (Rank30) {
            static const Rank30Reference reference(bit_count);
            return {reference.template discrete<Real>(n), reference.template continuous<Real>(n)};
        } else {
            return {closed_form_reference(dimension, n, spacing), continuous_reference(dimension, time)};
        }
    };

    constexpr auto precision = std::is_same_v<Real, double> ? "double"
        : std::is_same_v<Real, dd_128> ? "dd_128" : "float128";
    const std::string dimension_label = std::to_string(dimension) + "D";
    const auto plot_dir = std::filesystem::path("test/output/simplex_plot")
        / (dimension_label + (Rank30 ? "rank30" : "") + (FitWithTCI ? "_tci" : ""));
    std::filesystem::create_directories(plot_dir);
    const auto dataset = plot_dir / (std::string(precision) + "_" + dimension_label
                                     + (Rank30 ? "_rank30" : "") + (FitWithTCI ? "_tci" : "")
                                     + (method == "zip-up" ? "" : "_" + method)
                                     + (merge_trucation ? "_trunc" : ""));

    const typename Integrator::mGrid grid(std::vector<Real>(dimension, Real(0)),
                                          std::vector<Real>(dimension, Real(1)),
                                          std::vector<int>(dimension, bit_count));
    const auto rank30_function = [](const std::vector<Real>& gaps) {
        const Real base = Real(1) + Real(rank30_scale) * (gaps.at(0) + gaps.at(1) + gaps.at(2));
        Real result(1);
        for (int r = 0; r < rank30_degree; ++r) result *= base;
        return Value(result);
    };
    const auto values = [&] {
        if constexpr (FitWithTCI) {
            static_assert(Rank30);
            xfac_quad_runner::ttci_runner_opts<Value> options;
            options.reltol = rank30_reltol<Real>;
            options.pivot1 = grid.coord_to_id({Real(0.5), Real(0.5), Real(0.5)});
            options.fullPiv = true;
            const xfac_quad_runner::ttci_runner_param<Value> parameters(/*nb_iter=*/8, rank30_max_bond_dim, options);

            // Sample grid points directly from their bits with a reproducible seed.
            constexpr std::uint64_t pivot_seed = 42;
            constexpr int additional_pivot_count = 10;
            std::mt19937_64 pivot_rng(pivot_seed);
            typename Integrator::FitOptions fit_options;
            fit_options.additional_pivots.reserve(additional_pivot_count);
            std::vector<int> pivot_bits(grid.get_tensorLen());
            for (int i = 0; i < additional_pivot_count; ++i) {
                for (int& bit : pivot_bits)
                    bit = static_cast<int>(pivot_rng() >> 63);
                fit_options.additional_pivots.push_back(grid.id_to_coord(pivot_bits));
            }
            const auto fit_prefix = dataset.string() + "_fit";
            const auto fit_log_path = fit_prefix + ".log";
            auto fit_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fit_log_path, /*truncate=*/true);
            auto fit_logger = std::make_shared<spdlog::logger>(fit_log_path, fit_sink);
            fit_logger->set_level(spdlog::level::info);
            fit_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
            fit_logger->flush_on(spdlog::level::info);
            fit_logger->info("Additional random pivots: {} (seed: {})", additional_pivot_count, pivot_seed);
            (void)Integrator::fit(grid, rank30_function, parameters, fit_prefix, fit_options, fit_logger);
            // fit() saves the TTN; the integration helper accepts the tree directly.
            return typename Integrator::mState(fit_prefix + ".tt", rank30_max_bond_dim, options.reltol, /*w=*/-1);
        } else if constexpr (Rank30) {
            return make_rank30_polynomial_ttn<Value>(bit_count);
        } else {
            return make_polynomial_ttn<Value>(dimension, bit_count);
        }
    }();
    if constexpr (Rank30) {
        if constexpr (FitWithTCI) {
            std::cout << "TCI-fitted degree-29 input: max_chi=" << values.get_chi() << '\n';
        } else {
            REQUIRE(values.get_chi() == rank30);
        }
        const std::vector<Real> gaps{Real(1) / Real(2), Real(1) / Real(4), Real(1) / Real(8)};
        const Value expected = rank30_function(gaps);
        const Real error = abs(values.eval(grid.coord_to_id(gaps)) - expected);
        CAPTURE(error);
        if constexpr (FitWithTCI) {
            // Check the input fit separately from the integration error budget.
            const Real fit_rtol = std::is_same_v<Real, double> ? Real(5e-10)
                : Real(1000000) * Eigen::NumTraits<Real>::epsilon();
            const Real fit_budget = fit_rtol * abs(expected);
            CAPTURE(fit_budget);
            CHECK(error <= fit_budget);
        } else {
            REQUIRE(error <= relative_tolerance<Real>() * abs(expected));
        }
    }
    const auto [integral, time_grid] = Integrator::make_integration_simplex(
        values, grid, /*kernel=*/nullptr, /*kernel_grid=*/{Real(0), Real(1), bit_count}, /*save_prefix=*/"",
        /*max_bond_dim=*/Rank30 ? rank30_max_bond_dim : 16,
        /*reltol=*/Rank30 ? rank30_reltol<Real> : Real(8) * Eigen::NumTraits<Real>::epsilon(),
        /*final_compress=*/true, /*path_logger=*/dataset.string() + "_", method, merge_trucation);
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

    const Real end_value = reference_at(count - 1, times.back()).first;
    // Degree-29 double needs more room for contraction error; the fitted cases
    // also inherit interpolation error from the input TTN.
    const Real integration_rtol = [] {
        if constexpr (Rank30 && std::is_same_v<Real, double>)
            return FitWithTCI ? Real(1.5e-9) : Real(2e-11);
        else if constexpr (FitWithTCI)
            return Real(4000000) * Eigen::NumTraits<Real>::epsilon();
        else
            return relative_tolerance<Real>();
    }();
    const Real discrete_budget = integration_rtol * end_value;
    CAPTURE(discrete_budget);
    // Each degree-m term differs by replacing t^m with a product of m factors
    // t + (dimension - i)*h. Telescoping those products bounds the combined
    // quadrature error on [0,1) by 32*h for all three test polynomials.
    const Real low_degree_quadrature_budget = Real(32) * spacing;
    std::ofstream csv(dataset.string() + ".csv");
    REQUIRE(csv.is_open());
    csv << "t,mps,discrete,continuous,abs_discrete,abs_continuous,rel_discrete,rel_continuous,abs_quadrature\n"
        << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10);

    Real max_discrete(0), max_continuous(0), max_quadrature(0);
    Real max_discrete_relative(0), max_continuous_relative(0);
    for (int i = 0; i < time_count; ++i) {
        const Real& time = times[i];
        const auto [discrete, continuous] = reference_at(indices[i], time);
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
        // For degree 29, account for the exact pointwise quadrature bias.
        const Real quadrature_budget = Rank30 ? quadrature_error : low_degree_quadrature_budget;
        CHECK(continuous_error <= Real(discrete_budget + quadrature_budget));
    }

    csv.close();
    REQUIRE_FALSE(csv.fail());
    ++plot_datasets_written;
    std::ostringstream report;
    report << std::scientific << std::setprecision(8)
           << precision << ", " << method << ", merge truncation " << (merge_trucation ? "on" : "off")
           << ", " << dimension << "D, " << bit_count << " bits, " << time_count << " times"
           << (Rank30 ? (FitWithTCI ? " (degree 29, TCI fit)" : " (degree 29)") : "") << '\n'
           << "  MPS vs discrete:   max abs=" << max_discrete
           << ", max rel (t>0)=" << max_discrete_relative << '\n'
           << "  MPS vs continuous: max abs=" << max_continuous
           << ", max rel (t>0)=" << max_continuous_relative << '\n'
           << "  Discrete vs continuous: max abs=" << max_quadrature << '\n';
    std::cout << report.str();
}

} // namespace

TEMPLATE_TEST_CASE("High-resolution simplex integration against discrete and continuous references",
                   "[simplex]", double, dd_128, float128)
{
    const int dimension = GENERATE(1, 2, 3);
    const auto [method, merge_trucation] = GENERATE(Catch::Generators::from_range(merge_cases));
    check_high_resolution_simplex<TestType, false>(dimension, method, merge_trucation);
}

TEMPLATE_TEST_CASE("High-resolution simplex integration of a rank-30 polynomial",
                   "[simplex]", double)
{
    const auto [method, merge_trucation] = GENERATE(Catch::Generators::from_range(merge_cases));
    check_high_resolution_simplex<TestType, true>(3, method, merge_trucation);
}

TEST_CASE("High-resolution simplex integration of a TCI-fitted rank-30 polynomial",
          "[simplex][fit]")
{
    const auto [method, merge_trucation] = GENERATE(Catch::Generators::from_range(merge_cases));
    check_high_resolution_simplex<double, true, true>(3, method, merge_trucation);
}

TEST_CASE("100-bit dd_128 simplex integration of a rank-30 polynomial",
          "[simplex][rank30]")
{
    const bool fit_with_tci = GENERATE(false, true);
    if (fit_with_tci)
        check_high_resolution_simplex<dd_128, true, true>(3, "zip-up", /*merge_trucation=*/true);
    else
        check_high_resolution_simplex<dd_128, true>(3, "zip-up", /*merge_trucation=*/true);
}

TEST_CASE("Weighted rank-30 simplex integration matches the ordered grid sum",
          "[simplex][kernel][rank30]")
{
    using Value = std::complex<double>;
    using Integrator = QuanticsIntegration::QuanticsIntegrator_kD<Value>;
    using Index = util::i128;
    const int bit_count = GENERATE(4, 50);
    const bool merge_trucation = GENERATE(false, true);
    CAPTURE(bit_count, merge_trucation);
    const Index count = Index(1) << bit_count;
    const Integrator::mGrid grid({0, 0, 0}, {1, 1, 1}, {bit_count, bit_count, bit_count});
    const auto [kernel, kernel_grid] = TensorQuadOperation::analytic_mps_construction::
        make_polynomial_orderp_mps<Value>(0, 1, bit_count, {Value(1), Value(1)}, /*ordering=*/2);
    const auto values = make_rank30_polynomial_ttn<Value>(bit_count);
    REQUIRE(values.get_chi() == rank30);
    const auto [integral, time_grid] = Integrator::make_integration_simplex(
        values, grid, &kernel, kernel_grid, /*save_prefix=*/"", rank30_max_bond_dim,
        rank30_reltol<double>, /*final_compress=*/true, /*path_logger=*/{}, "zip-up", merge_trucation);
    REQUIRE(time_grid.get_nBits() == bit_count);
    const double h = time_grid.get_dx();
    const auto reference = Rank30Reference::weighted_discrete_reference(bit_count);
    // Allow for double roundoff from the extra kernel contractions on the full-curve scale.
    const double budget = 3e-11 * reference(count - 1);
    const int time_count = bit_count == 4 ? 16 : 1000;
    double max_error = 0;
    for (int i = 0; i < time_count; ++i) {
        const Index n = (count - 1) * i / (time_count - 1);
        const double t = static_cast<double>(n) * h;
        const double expected = reference(n);
        CAPTURE(i, t, expected, budget);
        if (bit_count == 4) {
            double direct = 0;
            for (int m1 = 0; m1 <= n; ++m1)
                for (int m2 = 0; m2 <= m1; ++m2)
                    for (int m3 = 0; m3 <= m2; ++m3) {
                        const double u1 = m1 * h, u2 = m2 * h, u3 = m3 * h;
                        const double gaps = (t - u1) + (u1 - u2) + (u2 - u3);
                        direct += std::pow(1 + rank30_scale * gaps, rank30_degree)
                            * (1 + u1) * (1 + u2) * (1 + u3);
                    }
            CHECK(std::abs(direct * h * h * h - expected) <= 1e-13 * expected);
        }
        const double error = std::abs(Integrator::integrate_simplex(integral, time_grid, t) - Value(expected));
        max_error = std::max(max_error, error);
        CHECK(error <= budget);
    }
    std::cout << "Weighted rank-30 simplex: " << bit_count << " bits, merge truncation "
              << (merge_trucation ? "on" : "off") << ", max absolute error=" << max_error
              << ", budget=" << budget << '\n';
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
