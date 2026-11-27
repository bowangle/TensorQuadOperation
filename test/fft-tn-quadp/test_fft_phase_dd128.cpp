#include <algorithm>
#include <array>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "fft_tn.hpp"
#include "magic_tensor_qft.hpp"
#include "type_double_double.h"
#include "type_int128.h"

namespace {

using Real = dd_128;
using Complex = Cdd_128;
using Sint = util::i128;

constexpr int max_bond_dimension = 200;
const Real relative_tolerance =
    std::numeric_limits<Real>::epsilon() * Real(1000);

MPS<Complex> delta_zero_mps(int n_bit)
{
    std::vector<Tensor3D<Complex>> cores;
    cores.reserve(n_bit);
    for (int bit = 0; bit < n_bit; ++bit) {
        Tensor3D<Complex> core(1, 2, 1);
        core(0, 0, 0) = Complex(1, 0);
        cores.push_back(std::move(core));
    }
    return MPS<Complex>(
        std::move(cores), max_bond_dimension, relative_tolerance, -1);
}

std::vector<int> bits_of(Sint index, int n_bit)
{
    std::vector<int> bits(n_bit, 0);
    for (int bit = 0; bit < n_bit; ++bit)
        bits[bit] = int((index >> bit) & Sint(1));
    return bits;
}

struct PhaseErrors {
    Real zero_origin = 0;
    Real translated_positive = 0;
    Real translated_negative = 0;
    Real direct_positive = 0;
    Real direct_negative = 0;
};

PhaseErrors translated_grid_phase_errors(int n_bit, int padding_bit)
{
    const Real original_a = Real(-18);
    const Real original_b = Real(22);
    const Real shifted_a = Real(0);
    const Real shifted_b = Real(40);

    const auto input = delta_zero_mps(n_bit);
    const QTGrid<Real, Sint> original_grid(original_a, original_b, n_bit);
    const QTGrid<Real, Sint> shifted_grid(shifted_a, shifted_b, n_bit);
    const Real output_scale = original_grid.get_dx()
                            / (Real(2) * magic_tensor_qft::pi<Real>());

    const int fft_n_bit = n_bit + padding_bit;
    const auto qft = magic_tensor_qft::load_compressed_qft_mpo<Complex>(
        fft_n_bit,
        "mpo_data/data/",
        relative_tolerance,
        max_bond_dimension);

    FFTmps<Complex, Sint> original_fft(
        input, original_grid, qft, padding_bit,
        max_bond_dimension, relative_tolerance);
    FFTmps<Complex, Sint> shifted_fft(
        input, shifted_grid, qft, padding_bit,
        max_bond_dimension, relative_tolerance);

    original_fft.fft_vanilla(/*do_shift=*/true, "zip-up");
    shifted_fft.fft_vanilla(/*do_shift=*/true, "zip-up");

    PhaseErrors errors;
    const Sint N_fft = Sint(1) << fft_n_bit;
    const Sint centre = N_fft / Sint(2);

    // These offsets produce the same physical times for both padding modes:
    // q * dt = base * 2^padding * 2*pi/(40*2^padding).
    for (const int base_offset : std::array<int, 3>{1, 3, 6}) {
        const Sint offset = Sint(base_offset) << padding_bit;

        for (const int sign : {-1, 1}) {
            const Sint index = centre + Sint(sign) * offset;
            const auto bits = bits_of(index, fft_n_bit);
            const Real time = shifted_fft.get_grid().id_to_coord(bits);

            const Complex zero_origin_value = shifted_fft.eval(bits);
            const Complex original_value = original_fft.eval(bits);
            const Complex expected_phase = std::exp(
                Complex(Real(0), -original_a * time));

            const Real zero_error = std::abs(
                zero_origin_value / Complex(output_scale, 0)
                - Complex(1, 0));
            errors.zero_origin = std::max(errors.zero_origin, zero_error);

            // Comparing the two transforms cancels their common input and QFT
            // errors. Only the grid-origin phase (and its MPO application)
            // differs between them.
            const Real relation_error = std::abs(
                original_value - expected_phase * zero_origin_value)
                / std::abs(zero_origin_value);
            const Real direct_error = std::abs(
                original_value / Complex(output_scale, 0) - expected_phase);

            if (sign > 0) {
                errors.translated_positive = std::max(
                    errors.translated_positive, relation_error);
                errors.direct_positive = std::max(
                    errors.direct_positive, direct_error);
            } else {
                errors.translated_negative = std::max(
                    errors.translated_negative, relation_error);
                errors.direct_negative = std::max(
                    errors.direct_negative, direct_error);
            }
        }
    }

    return errors;
}

} // namespace

TEST_CASE("dd128 translated grids isolate the post-QFT origin phase",
          "[fft][dd128][phase]")
{
    // This is intentionally much looser than the QFT/identity-phase errors
    // observed in the existing data (~1e-28), while still tight enough to
    // catch the former large-angle phase regression at high resolution.
    const Real roundoff_tolerance = Real(1e-24);

    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<Real>::max_digits10)
              << "n_bit,padding,a0_error,positive_relation_error,"
                 "negative_relation_error,positive_direct_error,"
                 "negative_direct_error\n";

    for (const int n_bit : {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}) {
        for (const int padding_bit : {0, 10}) {
            const PhaseErrors errors =
                translated_grid_phase_errors(n_bit, padding_bit);

            std::cout << n_bit << ',' << padding_bit << ','
                      << errors.zero_origin << ','
                      << errors.translated_positive << ','
                      << errors.translated_negative << ','
                      << errors.direct_positive << ','
                      << errors.direct_negative << '\n';

            CAPTURE(n_bit, padding_bit,
                    errors.zero_origin,
                    errors.translated_positive,
                    errors.translated_negative,
                    errors.direct_positive,
                    errors.direct_negative);

            CHECK(errors.zero_origin <= roundoff_tolerance);
            CHECK(errors.translated_positive <= roundoff_tolerance);

            // Guard the negative-time two's-complement cancellation that the
            // former independent large-angle phase construction destroyed.
            CHECK(errors.translated_negative <= roundoff_tolerance);
        }
    }
}
