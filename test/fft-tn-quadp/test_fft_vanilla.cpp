#include "fft_tn.hpp"
#include "type_int128.h"
#include "type_double_double.h"
#include "type_float128_boost.h"
#include <iostream>
#include <chrono>
#include <random>

template<typename Cscalar, typename Sint>
void test_fft_construction(const std::string& type_name, const Sint padding_bit_)
{
    std::cout << "Testing FFTmps<" << type_name << ">...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        FFTmps<Cscalar, Sint> fft(
            "test/gE<_site[[1, 0], [0, 0]]",    // prefix_mps
            "mpo_data/data/",                   // folder_qft_data
            padding_bit_                        // padding bit
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  Constructed successfully.\n";
        std::cout << "  Time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << "\n";
    }
}

template<typename Cscalar, typename Sint>
void test_fft_fft_vanilla_execution(const std::string& type_name, const Sint padding_bit_)
{
    std::cout << "Testing FFTmps fft_vanilla run<" << type_name << ">...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        FFTmps<Cscalar, Sint> fft(
            "test/gE<_site[[1, 0], [0, 0]]",    // prefix_mps
            "mpo_data/data/",                   // folder_qft_data
            padding_bit_                        // padding bit
        );
        fft.fft_vanilla();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  fft_vanilla run successfully.\n";
        std::cout << "  Time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
        std::cout << "  Max bond dim: " << fft.get_chi() << "\n";
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << "\n";
    }
}

template<typename Cscalar, typename Sint>
void test_fft_ifft_roundtrip(const std::string& type_name, const Sint padding_bit_, int n_samples = 5)
{
    std::cout << "Testing FFT → IFFT roundtrip<" << type_name << ">...\n";
    try {
        using Real = typename FFTmps<Cscalar, Sint>::RealScalar;

        FFTmps<Cscalar, Sint> fft(
            "test/gE<_site[[1, 0], [0, 0]]",
            "mpo_data/data/",
            padding_bit_
        );

        int nBits = static_cast<int>(fft.get_mps().get_core().size());
        std::cout << "  nBits: " << nBits << ", chi_initial: " << fft.get_chi() << "\n";

        // Sample a few random bitstrings
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> bit(0, 1);
        std::vector<std::vector<int>> samples(n_samples);
        for (int s = 0; s < n_samples; ++s) {
            samples[s].resize(nBits);
            for (int i = 0; i < nBits; ++i)
                samples[s][i] = bit(rng);
        }

        // Evaluate original MPS
        std::vector<Cscalar> orig_vals(n_samples);
        for (int s = 0; s < n_samples; ++s)
            orig_vals[s] = fft.eval(samples[s]);

        // Save primal a before forward FFT
        Real primal_a = fft.get_a();

        // Forward FFT
        fft.fft_vanilla();
        std::cout << "  chi_fft: " << fft.get_chi() << "\n";

        // Inverse FFT (pass original a so the phase is correctly undone)
        fft.ifft_vanilla(/*do_shift=*/true, primal_a);
        std::cout << "  chi_ifft: " << fft.get_chi() << "\n";

        // Compare
        Real max_abs_err = 0, max_rel_err = 0;
        for (int s = 0; s < n_samples; ++s) {
            Cscalar val = fft.eval(samples[s]);
            Cscalar diff = val - orig_vals[s];
            Real abs_err  = std::abs(diff);
            Real abs_orig = std::abs(orig_vals[s]);
            Real rel_err  = (abs_orig > Real(0)) ? abs_err / abs_orig : abs_err;
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        std::cout << "  Max absolute error: " << max_abs_err << "\n";
        std::cout << "  Max relative error: " << max_rel_err << "\n";
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << "\n";
    }
}

int main()
{
    // test_fft_ifft_roundtrip<std::complex<double>, util::i128>("complex<double>", util::i128(0));
    // test_fft_ifft_roundtrip<std::complex<double>, util::i128>("complex<double>", util::i128(10));

    // test_fft_ifft_roundtrip<Cdd_128, util::i128>("Cdd_128", util::i128(0));
    // test_fft_ifft_roundtrip<Cdd_128, util::i128>("Cdd_128", util::i128(10));

    test_fft_construction<std::complex<double>, util::i128>("complex<double>", util::i128(0));
    test_fft_construction<Cdd_128, util::i128>("Cdd_128", util::i128(0));
    test_fft_construction<Cfloat128, util::i128>("Cfloat128", util::i128(0));

    test_fft_construction<std::complex<double>, util::i128>("complex<double>", util::i128(10));
    test_fft_construction<Cdd_128, util::i128>("Cdd_128", util::i128(10));
    test_fft_construction<Cfloat128, util::i128>("Cfloat128", util::i128(10));

    test_fft_fft_vanilla_execution<std::complex<double>, util::i128>("complex<double>", util::i128(0));
    test_fft_fft_vanilla_execution<Cdd_128, util::i128>("Cdd_128>", util::i128(0));
    //test_fft_fft_vanilla_execution<Cfloat128, util::i128>("Cfloat128", util::i128(0));      // too long

    test_fft_fft_vanilla_execution<std::complex<double>, util::i128>("complex<double>", util::i128(10));
    test_fft_fft_vanilla_execution<Cdd_128, util::i128>("Cdd_128>", util::i128(10));
    //test_fft_fft_vanilla_execution<Cfloat128, util::i128>("Cfloat128", util::i128(0));    // too long

    

    return 0;
}
