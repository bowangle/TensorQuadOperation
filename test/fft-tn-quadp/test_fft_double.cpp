#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "fft_tn.hpp"
#include "type_int128.h"
#include "test_common.hpp"

namespace {

std::vector<std::size_t> make_sampled_fft_indices(
    std::size_t N,
    std::size_t sample_count)
{
    std::vector<std::size_t> indices{
        0, 1, N / 2 - 1, N / 2, N / 2 + 1, N - 2, N - 1
    };

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, N - 1);
    while (indices.size() < sample_count) {
        const std::size_t index = dist(rng);
        if (std::find(indices.begin(), indices.end(), index) == indices.end())
            indices.push_back(index);
    }

    std::sort(indices.begin(), indices.end());
    return indices;
}

} // namespace

TEST_CASE("double FFT test data can be saved and reloaded", "[fft][double]")
{
    using Sint = util::i128;
    using RealT = double;
    using ComplexT = std::complex<RealT>;

    constexpr int n_bit = 14;
    constexpr int n_iter = 10;
    constexpr int padding_bit = 10;
    constexpr std::size_t N = std::size_t(1) << n_bit;
    constexpr std::size_t padded_N = N << padding_bit;
    constexpr std::size_t padded_sample_count = 128;
    const std::string output_path = "test/output_fft_test";

    std::filesystem::create_directories(output_path);

    const auto& functions =
        fft_tn_test::fft_test_functions<ComplexT>();

    // Catch2 re-enters the test case for every SECTION. Generate the shared
    // TCI data only once per test process.
    static bool data_initialized = false;
    if (!data_initialized) {
        for (const auto& function : functions) {
            fft_tn_test::tci_function_for_test<ComplexT, Sint>(
                output_path, "double", function, n_bit, n_iter);
        }
        data_initialized = true;
    }

    for (const auto& function : functions) {
        DYNAMIC_SECTION("function: " << function.name)
        {
            auto data = fft_tn_test::load_mps_and_grid<ComplexT, Sint>(
                output_path, "double", std::string(function.name), n_bit);

            REQUIRE_NOTHROW(
                fft_tn_test::verify_loaded_data(data, function, n_bit));

            const std::string prefix = output_path
                                     + "/double/"
                                     + std::string(function.name)
                                     + "_nB"
                                     + std::to_string(n_bit);

            const auto& discontinuities = data.discontinuities;
            const auto& left_values = data.f_discontinuities;

            const RealT tolerance =
                std::numeric_limits<RealT>::epsilon() * RealT(170);
            const RealT loop_tolerance =
                std::numeric_limits<RealT>::epsilon() * RealT(200);
            const RealT loopless_tolerance =
                std::numeric_limits<RealT>::epsilon() * RealT(132);
            const RealT padded_tolerance =
                std::numeric_limits<RealT>::epsilon() * RealT(1150);

            const std::vector<std::string> contraction_methods{
                "zip-up", "qrsvd", "optimize"
            };
            const std::size_t total_fft_cases =
                functions.size() * contraction_methods.size() * 8;
            for (const auto& contraction_method : contraction_methods) {
                DYNAMIC_SECTION("contraction method: " << contraction_method)
                {
                    auto check_error = [&](const char* fft_method,
                                           const auto& error,
                                           RealT tol) {
                        const std::size_t case_index =
                            fft_tn_test::next_fft_case_index();
                        std::cout << "\n[FFT case " << case_index << '/'
                                  << total_fft_cases << "; "
                                  << total_fft_cases - case_index
                                  << " remaining] " << function.name << " / "
                                  << contraction_method << " / " << fft_method
                                  << '\n';
                        std::cout << fft_method << " [" << contraction_method
                                  << "] reference error for " << function.name
                                  << ":\n" << error;
                        INFO(error);
                        std::ostringstream recap;
                        recap << std::scientific << std::setprecision(6)
                              << error.abs_abs_max << " <= " << tol;
                        INFO("FFT_RECAP " << recap.str());
                        REQUIRE(error.abs_abs_max <= tol);
                    };

                    SECTION("FFT without padding")
            {
                SECTION("Vanilla")
                {
                    const auto reference =
                        fft_tn_test::compute_reference_fft(function, N);
                    FFTmps<ComplexT, Sint> fft(prefix, "mpo_data/data/");
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_vanilla(/*do_shift=*/true, contraction_method);
                    });
                    const auto error = fft_tn_test::error_reference_fft_mps(
                        reference, fft.get_mps());
                    check_error("Vanilla FFT", error, tolerance);
                }

                SECTION("Trapezoid")
                {
                    const auto reference =
                        fft_tn_test::compute_trapezoid_reference_fft(function, N);
                    FFTmps<ComplexT, Sint> fft(prefix, "mpo_data/data/");
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_trapez(
                            data.f_b, /*do_shift=*/true, contraction_method);
                    });
                    const auto error = fft_tn_test::error_reference_fft_mps(
                        reference, fft.get_mps());
                    check_error("Trapezoid FFT", error, tolerance);
                }

                SECTION("Discontinuous with loop")
                {
                    const auto reference =
                        fft_tn_test::compute_discontinuous_reference_fft(
                            function, N, discontinuities, left_values);
                    FFTmps<ComplexT, Sint> fft(prefix, "mpo_data/data/");
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_discontinuous_with_loop(
                            discontinuities, left_values, data.f_b,
                            /*do_shift=*/true, contraction_method);
                    });
                    const auto error = fft_tn_test::error_reference_fft_mps(
                        reference, fft.get_mps());
                    check_error(
                        "Discontinuous FFT with loop", error, loop_tolerance);
                }

                SECTION("Discontinuous loopless")
                {
                    const auto reference =
                        fft_tn_test::compute_discontinuous_reference_fft(
                            function, N, discontinuities, left_values);
                    FFTmps<ComplexT, Sint> fft(prefix, "mpo_data/data/");
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_discontinuous_loopless(
                            discontinuities, left_values, data.f_b,
                            /*do_shift=*/true, contraction_method);
                    });
                    const auto error = fft_tn_test::error_reference_fft_mps(
                        reference, fft.get_mps());
                    check_error(
                        "Discontinuous loopless FFT", error, loopless_tolerance);
                }
            }

            SECTION("FFT with 10 padding bits")
            {
                const auto sampled_indices = make_sampled_fft_indices(
                    padded_N, padded_sample_count);

                SECTION("Vanilla")
                {
                    const auto reference =
                        fft_tn_test::compute_sampled_padded_reference_fft(
                            function, N, padding_bit, sampled_indices);
                    FFTmps<ComplexT, Sint> fft(
                        prefix, "mpo_data/data/", padding_bit);
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_vanilla(/*do_shift=*/true, contraction_method);
                    });
                    const auto error =
                        fft_tn_test::error_sampled_reference_fft_mps(
                            reference, fft.get_mps());
                    check_error("Padded vanilla FFT", error, padded_tolerance);
                }

                SECTION("Trapezoid")
                {
                    const auto reference =
                        fft_tn_test::compute_sampled_padded_reference_fft(
                            function, N, padding_bit, sampled_indices,
                            /*do_shift=*/true, /*use_trapezoid=*/true);
                    FFTmps<ComplexT, Sint> fft(
                        prefix, "mpo_data/data/", padding_bit);
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_trapez(
                            data.f_b, /*do_shift=*/true, contraction_method);
                    });
                    const auto error =
                        fft_tn_test::error_sampled_reference_fft_mps(
                            reference, fft.get_mps());
                    check_error("Padded trapezoid FFT", error, padded_tolerance);
                }

                SECTION("Discontinuous with loop")
                {
                    const auto reference =
                        fft_tn_test::compute_sampled_padded_reference_fft(
                            function, N, padding_bit, sampled_indices,
                            /*do_shift=*/true, /*use_trapezoid=*/true,
                            discontinuities, left_values);
                    FFTmps<ComplexT, Sint> fft(
                        prefix, "mpo_data/data/", padding_bit);
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_discontinuous_with_loop(
                            discontinuities, left_values, data.f_b,
                            /*do_shift=*/true, contraction_method);
                    });
                    const auto error =
                        fft_tn_test::error_sampled_reference_fft_mps(
                            reference, fft.get_mps());
                    check_error(
                        "Padded discontinuous FFT with loop",
                        error, padded_tolerance);
                }

                SECTION("Discontinuous loopless")
                {
                    const auto reference =
                        fft_tn_test::compute_sampled_padded_reference_fft(
                            function, N, padding_bit, sampled_indices,
                            /*do_shift=*/true, /*use_trapezoid=*/true,
                            discontinuities, left_values);
                    FFTmps<ComplexT, Sint> fft(
                        prefix, "mpo_data/data/", padding_bit);
                    fft_tn_test::time_mpo_mps_case(contraction_method, [&] {
                        fft.fft_discontinuous_loopless(
                            discontinuities, left_values, data.f_b,
                            /*do_shift=*/true, contraction_method);
                    });
                    const auto error =
                        fft_tn_test::error_sampled_reference_fft_mps(
                            reference, fft.get_mps());
                    check_error(
                        "Padded discontinuous loopless FFT",
                        error, padded_tolerance);
                }
            }
                }
            }
        }
    }
}
