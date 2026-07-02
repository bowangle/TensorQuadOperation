#include "type_double_double.h"
#include "type_float128_boost.h"
#include "tt_base.h"
#include <iostream>
#include <string>
#include <chrono>

// Timer helper
using Clock = std::chrono::high_resolution_clock;

auto now() { return Clock::now(); }

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Helper to print shape
template<typename T>
void print_shape(const TT<T>& tt, const std::string& name)
{
    std::cout << name << " shape: ";
    for (auto const& [l, p, r] : tt.get_shape())
        std::cout << "(" << l << "," << p << "," << r << ") ";
    std::cout << "\n";
    std::cout << name << " chi: " << tt.get_chi() << "\n";
}

template<typename T>
void test_tt(const std::string& name, const std::string& filename)
{
    std::cout << "\n========================================\n";
    std::cout << "Testing " << name << "\n";
    std::cout << "========================================\n";

    auto t_type = now();

    // -------------------------------------------------------
    // 1. Load
    // -------------------------------------------------------
    {
        auto t0 = now();
        TT<T> tt(filename);
        std::cout << "[LOAD] OK (" << elapsed_ms(t0) << " ms)\n";
        print_shape(tt, "original");
    }

    // -------------------------------------------------------
    // 2. _initialize_w at center site
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        int center = tt2.get_size() / 2;
        auto t0 = now();
        tt2._initialize_w(center);
        std::cout << "[_initialize_w(" << center << ")] OK (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
    }

    // -------------------------------------------------------
    // 3. _initialize_w(0) — fully right canonical
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        auto t0 = now();
        tt2._initialize_w(0);
        std::cout << "[_initialize_w(0)] OK (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
    }

    // -------------------------------------------------------
    // 4. _initialize_w(nBit-1) — fully left canonical
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        int last = tt2.get_size() - 1;
        auto t0 = now();
        tt2._initialize_w(last);
        std::cout << "[_initialize_w(" << last << ")] OK (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
    }

    // -------------------------------------------------------
    // 5. shift_w: left to right sweep
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        tt2._initialize_w(0);
        auto t0 = now();
        std::cout << "[shift_w left->right]: ";
        for (int i = 1; i < tt2.get_size(); i++)
        {
            tt2.shift_w(i);
            tt2.check_canonical();
            std::cout << i << " ";
        }
        std::cout << "OK (" << elapsed_ms(t0) << " ms)\n";
    }

    // -------------------------------------------------------
    // 6. shift_w: right to left sweep
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        int last = tt2.get_size() - 1;
        tt2._initialize_w(last);
        auto t0 = now();
        std::cout << "[shift_w right->left]: ";
        for (int i = last - 1; i >= 0; i--)
        {
            tt2.shift_w(i);
            tt2.check_canonical();
            std::cout << i << " ";
        }
        std::cout << "OK (" << elapsed_ms(t0) << " ms)\n";
    }

    // -------------------------------------------------------
    // 7. compress_svd — no working site
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        Eigen::Index chi_before = tt2.get_chi();
        auto t0 = now();
        tt2.compress_svd(1e-12, -1);
        Eigen::Index chi_after = tt2.get_chi();
        std::cout << "[compress_svd no w] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
    }

    // -------------------------------------------------------
    // 8. compress_svd — with working site at center
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        int center = tt2.get_size() / 2;
        tt2._initialize_w(center);
        Eigen::Index chi_before = tt2.get_chi();
        auto t0 = now();
        tt2.compress_svd(1e-12, -1);
        Eigen::Index chi_after = tt2.get_chi();
        std::cout << "[compress_svd w=" << center << "] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
    }

    // -------------------------------------------------------
    // 9. compress_svd with bond dim truncation
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        Eigen::Index chi_before = tt2.get_chi();
        int max_bond = std::max((Eigen::Index)1, chi_before / 2);
        auto t0 = now();
        tt2.compress_svd(1e-12, max_bond);
        Eigen::Index chi_after = tt2.get_chi();
        std::cout << "[compress_svd max_bond=" << max_bond << "] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
    }

    std::cout << "----------------------------------------\n";
    std::cout << "Total time for " << name << ": " << elapsed_ms(t_type) << " ms\n";
    std::cout << "All tests passed for " << name << "\n";
}

int main()
{
    const std::string file = "gE<_site[[1, 0], [0, 0]].tt";

    auto t_total = now();

    test_tt<std::complex<double>>("complex<double>", file);
    test_tt<Cdd_128>             ("Cdd_128",         file);
    test_tt<Cfloat128>           ("Cfloat128",       file);

    std::cout << "\n========================================\n";
    std::cout << "Total time (all types): " << elapsed_ms(t_total) << " ms\n";

    return 0;
}