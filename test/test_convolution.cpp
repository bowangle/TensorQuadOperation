#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "generic_construction.hpp"
#include <type_double_double.h>
#include <type_float128_boost.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace convolution_tests {

inline std::vector<int> index_bits(std::size_t index, int nbit)
{
    std::vector<int> bits(static_cast<std::size_t>(nbit));
    for (int bit = 0; bit < nbit; ++bit) {
        bits[static_cast<std::size_t>(bit)] =
            static_cast<int>((index >> bit) & std::size_t{1});
    }
    return bits;
}

template<typename Real>
struct FunctionCase {
    using Grid = QTGrid<Real, util::i128>;
    std::string name;
    MPS<Real> mps;
    Grid grid;
    std::vector<Real> samples;
};

// One shared function list for signals, kernels, and translation tests.
// Reference samples come from the analytical functions, not MPS evaluation.
template<typename Real>
std::vector<FunctionCase<Real>> make_function_cases(int nbit)
{
    using namespace analytic_mps_construction;
    using Grid = typename FunctionCase<Real>::Grid;
    const Real a = Real(0);
    const Real b = Real(1);
    const Real amplitude = Real(1);
    const Real frequency = Real(2);
    const Real phase = Real(1) / Real(4);
    const Real offset = Real(1);
    const Real slope = Real(1) / Real(2);
    const Real rho = Real(3) / Real(2);
    const Real phi = -Real(1) / Real(4);
    const Real quadratic = Real(1) / Real(4);
    const std::size_t count = std::size_t{1} << nbit;
    std::vector<FunctionCase<Real>> functions;
    functions.reserve(4);

    const auto add_function = [&](const char* name,
                                  std::pair<MPS<Real>, Grid> construction,
                                  auto evaluate) {
        std::vector<Real> samples(count);
        for (std::size_t index = 0; index < count; ++index) {
            const Real x = construction.second.id_to_coord(index_bits(index, nbit));
            samples[index] = evaluate(x);
        }
        functions.push_back({name, std::move(construction.first),
                             std::move(construction.second), std::move(samples)});
    };

    add_function("cos", make_cos_mps<Real>(a, b, nbit, amplitude, frequency, phase),
        [=](Real x) -> Real {
            using std::cos;
            const Real angle = frequency * x + phase;
            return Real(amplitude * cos(angle));
        });
    add_function("affine", make_polynomial_orderp_mps<Real>(
        a, b, nbit, std::vector<Real>{offset, slope}),
        [=](Real x) -> Real { return slope * x + offset; });
    add_function("exp", make_exp_mps<Real>(a, b, nbit, rho, phi),
        [=](Real x) -> Real {
            using std::exp;
            const Real exponent = x * phi;
            return Real(rho * exp(exponent));
        });
    add_function("polynomial order 2", make_polynomial_orderp_mps<Real>(
        a, b, nbit, std::vector<Real>{offset, slope, quadratic}),
        [=](Real x) -> Real { return offset + slope * x + quadratic * x * x; });
    return functions;
}

template<typename Real>
void require_structure(const MPS<Real>& mps, int nbit)
{
    const auto& cores = mps.get_core();
    REQUIRE(cores.size() == static_cast<std::size_t>(nbit));
    REQUIRE(cores.front().n_left == 1);
    REQUIRE(cores.back().n_right == 1);
    for (std::size_t bit = 0; bit < cores.size(); ++bit) {
        CAPTURE(bit);
        REQUIRE(cores[bit].n_phys == 2);
        if (bit > 0) {
            REQUIRE(cores[bit - 1].n_right == cores[bit].n_left);
        }
    }
}

template<typename Real>
void require_close(Real actual, Real reference, std::size_t index, Real x,
                   Real tolerance)
{
    using std::abs;
    const Real reference_size = abs(reference);
    const Real scale = reference_size > Real(1) ? reference_size : Real(1);
    const Real difference = actual - reference;
    const Real error = Real(abs(difference)) / scale;
    CAPTURE(index, x, actual, reference, error, tolerance);
    REQUIRE(error <= tolerance);
}

} // namespace convolution_tests

TEMPLATE_TEST_CASE("All function pairs convolve correctly on an 8-bit grid",
                   "[convolution][construction][accuracy]", double, dd_128, float128)
{
    using Real = TestType;
    using namespace convolution_tests;
    constexpr int nbit = 8;
    constexpr std::size_t count = std::size_t{1} << nbit;
    const Real tolerance = Real(1000) * std::numeric_limits<Real>::epsilon();
    auto functions = make_function_cases<Real>(nbit);

    // All 4 x 4 ordered pairs, including equal signal and kernel functions.
    for (auto& signal : functions) {
        for (const auto& kernel : functions) {
            DYNAMIC_SECTION("signal=" << signal.name << ", kernel=" << kernel.name) {
                REQUIRE(signal.grid.get_a() == kernel.grid.get_a());
                REQUIRE(signal.grid.get_b() == kernel.grid.get_b());
                REQUIRE(signal.grid.get_nBits() == kernel.grid.get_nBits());
                REQUIRE(signal.grid.get_dx() == kernel.grid.get_dx());
                const auto result = signal.mps.conv(kernel.mps);
                require_structure(result, nbit);

                // Inclusive, nonperiodic discrete convolution, without dx.
                for (std::size_t index = 0; index < count; ++index) {
                    Real reference = Real(0);
                    Real compensation = Real(0);
                    for (std::size_t j = 0; j <= index; ++j) {
                        const Real term = signal.samples[j] * kernel.samples[index - j];
                        const Real adjusted = term - compensation;
                        const Real next = reference + adjusted;
                        compensation = (next - reference) - adjusted;
                        reference = next;
                    }
                    const auto bits = index_bits(index, nbit);
                    const Real x = signal.grid.id_to_coord(bits);
                    require_close(result.eval(bits), reference, index, x, tolerance);
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("All functions translate with zero padding on an 8-bit grid",
                   "[convolution][translation][accuracy]", double, dd_128, float128)
{
    using Real = TestType;
    using namespace convolution_tests;
    constexpr int nbit = 8;
    constexpr std::size_t count = std::size_t{1} << nbit;
    const Real tolerance = Real(1000) * std::numeric_limits<Real>::epsilon();
    auto functions = make_function_cases<Real>(nbit);

    SECTION("Nonnegative shifts") {
        const std::vector<util::i128> shifts{
            0, 1, 2, 3, 17, 127, 128,
            util::i128(count - 1), util::i128(count), util::i128(count + 1),
            util::i128(1) << 100};
        for (auto& signal : functions) {
            for (const util::i128 shift : shifts) {
                DYNAMIC_SECTION("signal=" << signal.name << ", shift=" << util::to_string(shift)) {
                    const int original_center = signal.mps.get_w();
                    const auto result = signal.mps.neg_translation(shift);
                    REQUIRE(signal.mps.get_w() == original_center);
                    require_structure(result, nbit);

                    // T(-shift): sample at index-shift, or zero outside the grid.
                    for (std::size_t index = 0; index < count; ++index) {
                        Real reference = Real(0);
                        if (shift < util::i128(count) && util::i128(index) >= shift) {
                            const auto source = index - static_cast<std::size_t>(shift);
                            reference = signal.samples[source];
                        }
                        const auto bits = index_bits(index, nbit);
                        const Real x = signal.grid.id_to_coord(bits);
                        require_close(result.eval(bits), reference, index, x, tolerance);
                    }
                }
            }
        }
    }

    SECTION("Negative shifts are rejected") {
        for (auto& signal : functions) {
            CAPTURE(signal.name);
            REQUIRE_THROWS_AS(signal.mps.neg_translation(util::i128(-1)),
                              std::invalid_argument);
        }
    }
}
