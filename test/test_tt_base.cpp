#include "type_double_double.h"
#include "type_float128_boost.h"
#include "tt_base.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cmath>

// ============================================================
// Timer
// ============================================================
using Clock = std::chrono::high_resolution_clock;
auto now() { return Clock::now(); }
double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ============================================================
// Type helpers
// ============================================================
template<typename T>
typename Eigen::NumTraits<T>::Real get_real(T const& v) { return std::real(v); }

template<typename T>
typename Eigen::NumTraits<T>::Real get_imag(T const& v) { return std::imag(v); }

template<typename RealT>
double to_double(RealT const& v) { return static_cast<double>(v); }

template<>
double to_double<dd_real>(dd_real const& v) { return v.x[0]; }

template<>
double to_double<float128>(float128 const& v) { return v.convert_to<double>(); }

template<typename T>
typename Eigen::NumTraits<T>::Real abs2(T const& v)
{
    auto re = get_real<T>(v);
    auto im = get_imag<T>(v);
    return re*re + im*im;
}

template<typename T>
typename Eigen::NumTraits<T>::Real abs2_diff(T const& a, T const& b)
{
    return abs2<T>(a - b);
}

template<typename T>
double epsilon_as_double()
{
    return to_double<typename Eigen::NumTraits<T>::Real>(
        Eigen::NumTraits<typename Eigen::NumTraits<T>::Real>::epsilon()
    );
}

template<typename T>
typename Eigen::NumTraits<T>::Real default_tol()
{
    return Eigen::NumTraits<typename Eigen::NumTraits<T>::Real>::epsilon() * 10;
}

// ============================================================
// Print shape
// ============================================================
template<typename T>
void print_shape(const TT<T>& tt, const std::string& name)
{
    std::cout << name << " shape: ";
    for (auto const& [l, p, r] : tt.get_shape())
        std::cout << "(" << l << "," << p << "," << r << ") ";
    std::cout << "\n";
    std::cout << name << " chi: " << tt.get_chi() << "\n";
}

// ============================================================
// Main test function
// ============================================================
template<typename T>
void test_tt(const std::string& name, const std::string& filename, int n_points = 100)
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
    // Generate shared points + reference values
    // -------------------------------------------------------
    std::vector<std::vector<int>> points;
    std::vector<T> values_before;
    {
        TT<T> tt(filename);
        points = tt.generate_points(n_points);
        auto t0 = now();
        values_before = tt.eval_list(points);
        std::cout << "[eval " << n_points << " points original] (" << elapsed_ms(t0) << " ms)\n";
    }

    // -------------------------------------------------------
    // check_values lambda — reports BOTH absolute and relative error
    // tol_factor is applied to the RELATIVE error vs eps
    // -------------------------------------------------------
    auto check_values = [&](const std::string& label,
                             std::vector<T> const& values_after,
                             double tol_factor)
    {
        using Real = typename Eigen::NumTraits<T>::Real;
        Real max_abs2 = Real(0);   // max |after - before|^2
        Real max_rel2 = Real(0);   // max (|after-before| / |before|)^2
        for (size_t i = 0; i < points.size(); i++)
        {
            Real num2 = abs2_diff<T>(values_after[i], values_before[i]);
            Real den2 = abs2<T>(values_before[i]);
            if (num2 > max_abs2) max_abs2 = num2;
            Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
            if (rel2 > max_rel2) max_rel2 = rel2;
        }
        double max_abs = std::sqrt(to_double<Real>(max_abs2));
        double max_rel = std::sqrt(to_double<Real>(max_rel2));
        double eps_d   = epsilon_as_double<T>();

        std::cout << "  abs err = " << max_abs
                  << "   rel err = " << max_rel
                  << "   (rel ratio = " << max_rel / eps_d << " * eps)\n";
        if (max_rel > tol_factor * eps_d)
            std::cerr << "  WARNING [" << label << "]: relative error exceeds "
                      << tol_factor << " * eps!\n";
        else
            std::cout << "  [" << label << "] OK\n";
    };

    const double canon_tol = 1e4;

    // -------------------------------------------------------
    // 2. _initialize_w at center
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        int center = tt2.get_size() / 2;
        auto t0 = now();
        tt2._initialize_w(center);
        std::cout << "[_initialize_w(" << center << ")] (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";
        check_values("_initialize_w(center)", vals, canon_tol);
    }

    // -------------------------------------------------------
    // 3. _initialize_w(0)
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        auto t0 = now();
        tt2._initialize_w(0);
        std::cout << "[_initialize_w(0)] (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";
        check_values("_initialize_w(0)", vals, canon_tol);
    }

    // -------------------------------------------------------
    // 4. _initialize_w(last)
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        int last = tt2.get_size() - 1;
        auto t0 = now();
        tt2._initialize_w(last);
        std::cout << "[_initialize_w(" << last << ")] (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";
        check_values("_initialize_w(last)", vals, canon_tol);
    }

    // -------------------------------------------------------
    // 5. shift_w left->right
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
        std::cout << "(" << elapsed_ms(t0) << " ms)\n";
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";
        check_values("shift_w left->right", vals, canon_tol);
    }

    // -------------------------------------------------------
    // 6. shift_w right->left
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
        std::cout << "(" << elapsed_ms(t0) << " ms)\n";
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";
        check_values("shift_w right->left", vals, canon_tol);
    }

    // -------------------------------------------------------
    // 7. compress_svd — no working site
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        Eigen::Index chi_before = tt2.get_chi();
        auto t0 = now();
        tt2.compress_svd(default_tol<T>(), -1);
        Eigen::Index chi_after = tt2.get_chi();
        std::cout << "[compress_svd no w] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";
        check_values("compress_svd no w", vals, canon_tol);
    }

    // -------------------------------------------------------
    // 8. compress_svd — working site at center
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        int center = tt2.get_size() / 2;
        tt2._initialize_w(center);
        Eigen::Index chi_before = tt2.get_chi();
        auto t0 = now();
        tt2.compress_svd(default_tol<T>(), -1);
        Eigen::Index chi_after = tt2.get_chi();
        std::cout << "[compress_svd w=" << center << "] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";
        check_values("compress_svd w=center", vals, canon_tol);
    }

    // -------------------------------------------------------
    // 9. compress_svd — bond dim truncation (error expected)
    // -------------------------------------------------------
    {
        TT<T> tt2(filename);
        Eigen::Index chi_before = tt2.get_chi();
        int max_bond = std::max((Eigen::Index)1, chi_before / 2);
        auto t0 = now();
        tt2.compress_svd(default_tol<T>(), max_bond);
        Eigen::Index chi_after = tt2.get_chi();
        std::cout << "[compress_svd max_bond=" << max_bond << "] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        tt2.check_canonical();
        auto t1 = now();
        auto vals = tt2.eval_list(points);
        std::cout << "  [eval " << n_points << " points] (" << elapsed_ms(t1) << " ms)\n";

        // Truncation is expected to change values — report both errors, no warning
        using Real = typename Eigen::NumTraits<T>::Real;
        Real max_abs2 = Real(0);
        Real max_rel2 = Real(0);
        for (size_t i = 0; i < points.size(); i++)
        {
            Real num2 = abs2_diff<T>(vals[i], values_before[i]);
            Real den2 = abs2<T>(values_before[i]);
            if (num2 > max_abs2) max_abs2 = num2;
            Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
            if (rel2 > max_rel2) max_rel2 = rel2;
        }
        double max_abs = std::sqrt(to_double<Real>(max_abs2));
        double max_rel = std::sqrt(to_double<Real>(max_rel2));
        std::cout << "  [truncation] abs err = " << max_abs
                  << "   rel err = " << max_rel
                  << "  (expected non-zero)\n";
    }

    std::cout << "----------------------------------------\n";
    std::cout << "Total time for " << name << ": " << elapsed_ms(t_type) << " ms\n";
}

// ============================================================
// Main
// ============================================================
int main()
{
    const std::string file = "gE<_site[[1, 0], [0, 0]].tt";
    const int n_points = 1000;  // change here to control

    auto t_total = now();

    test_tt<std::complex<double>>("complex<double>", file, n_points);
    test_tt<Cfloat128>           ("Cfloat128",       file, n_points);
    test_tt<Cdd_128>             ("Cdd_128",         file, n_points);
    

    std::cout << "\n========================================\n";
    std::cout << "Total time (all types): " << elapsed_ms(t_total) << " ms\n";

    return 0;
}