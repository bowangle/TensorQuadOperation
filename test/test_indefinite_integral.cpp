#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "indefinite_integral.hpp"

#include <cstddef>
#include <vector>

TEST_CASE("4-bit Heaviside MPO represents Theta(n-m) on the full grid",
          "[indefinite-integral][mpo]")
{
    constexpr int nbits = 4;
    constexpr std::size_t size = std::size_t{1} << nbits;

    // Theta(0) = 0: the resulting matrix is strictly lower triangular.
    auto heaviside = make_heaviside_mpo<double>(nbits, 0.0);

    REQUIRE(heaviside.get_size() == nbits);
    REQUIRE(heaviside.get_chi() == 2);

    std::vector<int> physical_indices(static_cast<std::size_t>(nbits));
    bool valid = true;
    std::size_t failing_n = 0;
    std::size_t failing_m = 0;
    double actual = 0.0;
    double expected = 0.0;

    for (std::size_t n = 0; n < size && valid; ++n) {
        for (std::size_t m = 0; m < size; ++m) {
            for (int site = 0; site < nbits; ++site) {
                const int n_bit = static_cast<int>((n >> site) & 1U);
                const int m_bit = static_cast<int>((m >> site) & 1U);
                physical_indices[static_cast<std::size_t>(site)] =
                    2 * n_bit + m_bit;
            }

            actual = heaviside.eval(physical_indices);
            expected = n > m ? 1.0 : 0.0;
            if (actual != expected) {
                valid = false;
                failing_n = n;
                failing_m = m;
                break;
            }
        }
    }

    CAPTURE(failing_n, failing_m, actual, expected);
    REQUIRE(valid);
}
