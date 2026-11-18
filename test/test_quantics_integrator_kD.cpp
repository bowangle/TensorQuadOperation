#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "quantics_integrator_kD.hpp"

namespace {

class TemporaryTreeFitFiles {
public:
    TemporaryTreeFitFiles()
    {
        static std::atomic<std::size_t> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        do {
            directory_ = std::filesystem::temp_directory_path() /
                ("tq_integrator_kD_" + std::to_string(stamp) + "_" +
                 std::to_string(counter.fetch_add(1)));
        } while (!std::filesystem::create_directory(directory_));
    }

    ~TemporaryTreeFitFiles()
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    std::string prefix() const { return (directory_ / "fit").string(); }

private:
    std::filesystem::path directory_;
};

template<class Real>
Real reference_polynomial(Real x, Real y, Real z)
{
    return Real(1) + x + Real(2) * y + Real(3) * z
        + x * y + Real(2) * x * z + Real(3) * y * z + x * y * z;
}

template<class Scalar>
void report_integration_error(const std::string& label, const Scalar& actual, const Scalar& expected)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using std::abs;
    const Real absolute_error = Real(abs(actual - expected));
    const Real reference_magnitude = Real(abs(expected));
    std::ostringstream out;
    out << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10)
        << '[' << Catch::getResultCapture().getCurrentTestName() << "] " << label << '\n'
        << "  actual = " << actual << ", expected = " << expected << '\n'
        << "  absolute error = " << absolute_error << ", relative error = ";
    if (reference_magnitude == Real(0))
        out << "undefined (zero reference)";
    else
        out << Real(absolute_error / reference_magnitude);
    std::cout << out.str() << '\n';
}

template<class Scalar>
TTN<Scalar> make_rectangular_integration_tree(int dimensions, int bits, bool reverse_axes)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    auto tree = xfac_quad::makeTuckerTree(dimensions, bits);
    if (reverse_axes) {
        for (auto& [node, neighbors] : tree.neigh) {
            auto order = neighbors.from_int();
            std::reverse(order.begin(), order.end());
            neighbors = xfac_quad::IndexSet<int>(order);
        }
        tree.root = dimensions > 2 ? dimensions * bits : bits - 1;
    }
    xfac_quad::TensorTree<Scalar> body(tree);
    for (const auto& [node, neighbors] : tree.neigh) {
        std::array<Eigen::Index, 3> shape{1, 1, 1};
        for (std::size_t axis = 0; axis < neighbors.size(); ++axis) shape[axis] = 2;
        if (tree.nodes.contains(node)) shape[2] = 2;
        auto& core = body.M.at(node);
        core = xfac_quad::Tensor3D<Scalar>(shape[0], shape[1], shape[2]);
        for (std::size_t entry = 0; entry < core.data.size(); ++entry) {
            const Real re = Real(1 + (entry + node) % 5) / Real(8);
            if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
                const Real im = Real(static_cast<int>((2 * entry + node) % 5) - 2) / Real(16);
                core.data[entry] = Scalar(re, im);
            } else {
                core.data[entry] = re;
            }
        }
    }
    return TTN<Scalar>(std::move(body), 0, Real(-1), -1);
}

} // namespace

TEMPLATE_TEST_CASE("fitted polynomial TTN integrates the full grid and rectangular ranges",
                   "[integrator][tree][fit][range]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using Scalar = TestType;
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Sint = util::i128;
    using Integrator = QuanticsIntegrator_kD<Scalar, Sint>;
    const int bits = 30;
    CAPTURE(bits);
    typename Integrator::mGrid grid({Real(0), Real(-1), Real(2)},
                                    {Real(1), Real(2), Real(4)}, {bits, bits, bits});
    const Sint n = grid.get_N().front();
    const std::function<Scalar(const std::vector<Real>&)> function =
        [](const std::vector<Real>& point) -> Scalar {
            return Scalar(reference_polynomial(point.at(0), point.at(1), point.at(2)));
        };
    xfac_quad_runner::ttci_runner_opts<Scalar> runner_options;
    runner_options.reltol = Real(100) * Eigen::NumTraits<Real>::epsilon();
    runner_options.pivot1 = grid.coord_to_id({Real(1) / Real(2), Real(1) / Real(2), Real(3)});
    runner_options.fullPiv = true;
    xfac_quad_runner::ttci_runner_param<Scalar> parameters(8, 16, runner_options);
    typename Integrator::FitOptions fit_options;
    fit_options.nb_point_res = 64;
    fit_options.additional_pivots = {
        {Real(0), Real(-1), Real(2)},
        {Real(3) / Real(4), Real(5) / Real(4), Real(7) / Real(2)}};
    TemporaryTreeFitFiles files;
    const auto integrator = Integrator::fit(grid, function, parameters,
                                           files.prefix(), fit_options);
    REQUIRE(std::filesystem::is_regular_file(files.prefix() + ".tt"));
    REQUIRE(std::filesystem::is_regular_file(files.prefix() + "_multgrid_E.json"));

    const Real tolerance = Real(50000) * Eigen::NumTraits<Real>::epsilon();
    const auto require_close = [&](const std::string& label, Scalar actual, Real expected) {
        using std::abs;
        report_integration_error(label, actual, Scalar(expected));
        CAPTURE(actual, expected);
        const Real scale = expected == Real(0) ? Real(1) : Real(abs(expected));
        REQUIRE(abs(actual - Scalar(expected)) <= tolerance * scale);
    };

    // The continuous integral is 120. For N left-endpoint samples per axis,
    // the exact discrete integral is 120 - 357/(2N) + 57/N^2 - 9/(2N^3).
    const Real inverse_n = Real(1) / Real(n);
    const Real expected_full = Real(120) - Real(357) / Real(2) * inverse_n
        + Real(57) * inverse_n * inverse_n
        - Real(9) / Real(2) * inverse_n * inverse_n * inverse_n;
    require_close("full grid (exact discrete reference)", integrator.integrate(), expected_full);
    require_close("full grid via IndexRange", integrator.integrate({{0, 0, 0}, {n, n, n}}), expected_full);

    const std::vector<typename Integrator::IndexRange> ranges{
        {{n / 4, n / 8, n / 2}, {3 * n / 4, 7 * n / 8, n - 1}}, // interior
        {{0, 0, 0}, {n / 2, 3 * n / 4, n / 4}},                // prefixes
        {{n / 2, n / 4, 3 * n / 4}, {n, n, n}},               // suffixes
        {{1, 2, 3}, {2, 3, 4}},                              // one cell
        {{n / 3, 0, n / 2}, {n / 3 + 1, n, n / 2 + 1}},       // thin box
        {{0, n / 2, 0}, {n, n / 2, n}}};                      // empty
    const std::array<const char*, 6> range_labels{
        "interior", "prefixes", "suffixes", "one cell", "thin box", "empty"};
    for (std::size_t box = 0; box < ranges.size(); ++box) {
        CAPTURE(box);
        const auto& range = ranges[box];
        Real volume = Real(1);
        std::array<Real, 3> mean;
        for (std::size_t i = 0; i < mean.size(); ++i) {
            volume *= Real(range.end[i] - range.begin[i]) * grid.get_dx()[i];
            mean[i] = grid.get_a()[i]
                + Real(range.begin[i] + range.end[i] - Sint(1)) * grid.get_dx()[i] / Real(2);
        }
        // Multilinearity makes the discrete average exactly f(mean_x, mean_y, mean_z).
        const Real expected = volume * reference_polynomial(mean[0], mean[1], mean[2]);
        require_close(range_labels[box], integrator.integrate(range), expected);
    }
}

TEMPLATE_TEST_CASE("rectangular TTN integrals match exhaustive grid sums",
                   "[integrator][tree][range]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using Scalar = TestType;
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Sint = util::i128;
    using Integrator = QuanticsIntegrator_kD<Scalar, Sint>;
    const int dimensions = GENERATE(1, 2, 3);
    const int bits = GENERATE(1, 2);
    const bool reverse_axes = GENERATE(false, true);
    CAPTURE(dimensions, bits, reverse_axes);
    const auto values = make_rectangular_integration_tree<Scalar>(dimensions, bits, reverse_axes);
    std::vector<Real> a(dimensions), b(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        a[i] = Real(-i);
        b[i] = a[i] + Real(i + 2);
    }
    typename Integrator::mGrid grid(a, b, std::vector<int>(dimensions, bits));
    const Integrator integrator(grid, values);
    const int n = 1 << bits;
    const std::array<std::pair<int, int>, 5> intervals{{{0, n}, {0, 1}, {1, n}, {1, n - 1}, {n, n}}};
    int box_count = 1;
    for (int i = 0; i < dimensions; ++i) box_count *= static_cast<int>(intervals.size());
    Real worst_absolute_error = Real(-1);
    Scalar worst_actual = Scalar(0), worst_expected = Scalar(0);
    int worst_box = 0;
    for (int box = 0; box < box_count; ++box) {
        CAPTURE(box);
        typename Integrator::IndexRange range{
            std::vector<Sint>(dimensions), std::vector<Sint>(dimensions)};
        int selection = box;
        for (int i = 0; i < dimensions; ++i) {
            const auto interval = intervals[selection % intervals.size()];
            selection /= static_cast<int>(intervals.size());
            range.begin[i] = interval.first;
            range.end[i] = interval.second;
        }
        Scalar expected = Scalar(0);
        for (int code = 0; code < (1 << (dimensions * bits)); ++code) {
            bool inside = true;
            for (int i = 0; i < dimensions; ++i) {
                const int index = (code >> (i * bits)) & (n - 1);
                inside = inside && index >= range.begin[i] && index < range.end[i];
            }
            if (!inside) continue;
            std::vector<int> id(dimensions * bits);
            for (std::size_t site = 0; site < id.size(); ++site) id[site] = (code >> site) & 1;
            expected += values.eval(id);
        }
        expected *= Scalar(grid.delta_volume());
        using std::abs;
        const Scalar actual = integrator.integrate(range);
        const Real absolute_error = Real(abs(actual - expected));
        if (absolute_error > worst_absolute_error) {
            worst_absolute_error = absolute_error;
            worst_actual = actual;
            worst_expected = expected;
            worst_box = box;
        }
        const Real scale = std::max(Real(1), Real(abs(expected)));
        REQUIRE(absolute_error <= Real(10000) * Eigen::NumTraits<Scalar>::epsilon() * scale);
        if (box == 0)
            REQUIRE(abs(integrator.integrate() - expected) <= Real(10000) * Eigen::NumTraits<Scalar>::epsilon() * scale);
    }
    report_integration_error("dense reference: dimensions=" + std::to_string(dimensions)
        + ", bits=" + std::to_string(bits) + ", reversed axes=" + std::to_string(reverse_axes)
        + ", worst absolute-error box=" + std::to_string(worst_box), worst_actual, worst_expected);
}

TEST_CASE("rectangular TTN integrals support narrow 128-bit ranges and twenty variables",
          "[integrator][tree][range]")
{
    using Scalar = std::complex<double>;
    using Sint = util::i128;
    using Integrator = QuanticsIntegrator_kD<Scalar, Sint>;
    const bool many_variables = GENERATE(false, true);
    const int dimensions = many_variables ? 20 : 3;
    const int bits = many_variables ? 3 : 70;
    xfac_quad::TensorTree<Scalar> body(xfac_quad::makeTuckerTree(dimensions, bits));
    for (const auto& [node, neighbors] : body.tree.neigh) {
        auto& core = body.M.at(node);
        core = xfac_quad::Tensor3D<Scalar>(1, 1, body.tree.nodes.contains(node) ? 2 : 1);
        std::fill(core.data.begin(), core.data.end(), Scalar(1));
    }
    typename Integrator::mGrid grid(std::vector<double>(dimensions, 0.0),
                                    std::vector<double>(dimensions, 1.0),
                                    std::vector<int>(dimensions, bits));
    const Integrator integrator(grid, TTN<Scalar>(std::move(body), 0, -1.0, -1));
    Integrator::IndexRange range{std::vector<Sint>(dimensions), std::vector<Sint>(dimensions)};
    double count = 1;
    for (int i = 0; i < dimensions; ++i) {
        const int width = many_variables ? 3 : i + 1;
        range.begin[i] = many_variables ? Sint(2) : (Sint(1) << 65) + Sint(i + 3);
        range.end[i] = range.begin[i] + Sint(width);
        count *= width;
    }
    const Scalar actual = integrator.integrate(range);
    const Scalar expected(count * grid.delta_volume());
    report_integration_error(many_variables ? "twenty variables" : "narrow 128-bit range", actual, expected);
    REQUIRE(actual == expected);
}

TEST_CASE("rectangular TTN index ranges are validated in every variable",
          "[integrator][tree][range]")
{
    using Integrator = QuanticsIntegrator_kD<double>;
    const auto values = make_rectangular_integration_tree<double>(2, 2, false);
    const Integrator integrator(Integrator::mGrid({0.0, 0.0}, {1.0, 1.0}, {2, 2}), values);
    REQUIRE_THROWS_AS(integrator.integrate({{0}, {4}}), std::invalid_argument);
    REQUIRE_THROWS_AS(integrator.integrate({{0, 0}, {4}}), std::invalid_argument);
    REQUIRE_THROWS_AS(integrator.integrate({{-1, 0}, {4, 4}}), std::out_of_range);
    REQUIRE_THROWS_AS(integrator.integrate({{0, 3}, {4, 2}}), std::out_of_range);
    REQUIRE_THROWS_AS(integrator.integrate({{0, 0}, {4, 5}}), std::out_of_range);
    // Empty intervals do not bypass validation of the other variables.
    REQUIRE_THROWS_AS(integrator.integrate({{0, 0}, {0, 5}}), std::out_of_range);
    REQUIRE(integrator.integrate({{0, 4}, {4, 4}}) == 0.0);
}
