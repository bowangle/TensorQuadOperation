#include "type_double_double.h"
#include "type_float128_boost.h"
#include "type_int128.h"
#include "mps_base.h"

#include "runner.h"
#include "grid.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <functional>

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

// We TCI f(x) and g(x) in quad precision and save them to an armadille format.

// cos(x)*cos**2(3x)+icos(2x)
template<typename cT> // T is std::complex<Scalar>
cT function_1(typename Eigen::NumTraits<cT>::Real const& v){
    return cos(v) * cos(3*v) * cos(3*v) + cT(0.0,1.0) * cos(2*v);
}

// x-x**2-ix
template<typename cT> // T is std::complex<Scalar>
cT function_2(typename Eigen::NumTraits<cT>::Real const& v){
    return cos(v) - cT(0.0,1.0) * v;
}

template<typename Scalar>  // T is a scalar
void do_save_TCI(
    Scalar a_1,
    Scalar b_1,
    Scalar c0,
    int nBit,
    std::function<std::complex<Scalar>(Scalar)> function_to_tci,
    int n_iter,
    bool do_cache,
    const std::string& filename,
    int nb_point_out){
    QTGrid<Scalar, long long> grid(a_1, b_1, nBit);
    TCI_param tci_param = TCI_param(grid.nBits, n_iter, do_cache);

    auto logger = spdlog::get("test_file_logger");
    if (!logger)
    {
        logger = spdlog::basic_logger_mt("test_file_logger", "test/output/file.log");
    }

    TCI_Runner<Scalar, long long> tci_runner(grid, tci_param, function_to_tci, logger);

    std::vector<Scalar> v = {};
    std::vector<Scalar> E_discontinuity = {};

    tci_runner.fit(
        c0,                 // E_init
        v,                  // additional pivot, not supported
        true,               // verbose
        true,               // save
        filename,           // file_prefix (path+prefix) to save
        nb_point_out,       // nb_point to test error
        E_discontinuity,    // point discontinuity to store
        grid.a,             // grid minimal point
        grid.b              // grid maximal point
    );
}

// ============================================================
// Numeric helpers (mirror test_tt_base.cpp)
// ============================================================
template<typename T>
typename Eigen::NumTraits<T>::Real get_real(T const& v) { return std::real(v); }

template<typename T>
typename Eigen::NumTraits<T>::Real get_imag(T const& v) { return std::imag(v); }

template<typename RealT>
double to_double(RealT const& v) { return static_cast<double>(v); }

template<>
double to_double<dd_128>(dd_128 const& v) { return v.x[0]; }

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
    return Eigen::NumTraits<typename Eigen::NumTraits<T>::Real>::epsilon() * 100;
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
template<typename cScalar, typename Sint>
void test_mps(const std::string& name, const std::string& filename_1, const std::string& filename_2, int n_points = 100)
{
    using Real = typename Eigen::NumTraits<cScalar>::Real;

    std::cout << "\n========================================\n";
    std::cout << "Testing " << name << "\n";
    std::cout << "========================================\n";

    auto t_type = now();

    // -------------------------------------------------------
    // 1. Load
    // -------------------------------------------------------
    {
        auto t0 = now();
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        MPS<cScalar> mps_f2(filename_2 + ".tt");
        std::cout << "[LOAD] OK (" << elapsed_ms(t0) << " ms)\n";
        print_shape(mps_f1, "original f1");
        print_shape(mps_f2, "original f2");
    }

    // -------------------------------------------------------
    // Generate shared points + reference values
    // -------------------------------------------------------
    std::vector<std::vector<int>> points;
    std::vector<cScalar> values_before_f1;
    std::vector<cScalar> values_before_f2;
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        MPS<cScalar> mps_f2(filename_2 + ".tt");

        points = mps_f1.generate_points(n_points);
        auto t0 = now();
        values_before_f1 = mps_f1.eval_list(points);
        values_before_f2 = mps_f2.eval_list(points);
        std::cout << "[eval " << n_points << " points original] (" << elapsed_ms(t0) << " ms)\n";
    }

    // -------------------------------------------------------
    // check_values lambda
    // -------------------------------------------------------
    auto check_values = [&](const std::string& label,
                             const std::vector<cScalar>& values_after,
                             const std::vector<cScalar>& reference,
                             double tol_factor)
    {
        Real max_abs2 = Real(0);
        Real max_rel2 = Real(0);
        for (size_t i = 0; i < points.size(); i++)
        {
            Real num2 = abs2_diff<cScalar>(values_after[i], reference[i]);
            Real den2 = abs2<cScalar>(reference[i]);
            if (num2 > max_abs2) max_abs2 = num2;
            Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
            if (rel2 > max_rel2) max_rel2 = rel2;
        }
        double max_abs = std::sqrt(to_double<Real>(max_abs2));
        double max_rel = std::sqrt(to_double<Real>(max_rel2));
        double eps_d   = epsilon_as_double<cScalar>();

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
    // 2. _initialize_w at center (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        int center = mps_f1.get_size() / 2;
        auto t0 = now();
        mps_f1._initialize_w(center);
        std::cout << "[_initialize_w(" << center << ")] (" << elapsed_ms(t0) << " ms)\n";
        mps_f1.check_canonical();
        auto vals = mps_f1.eval_list(points);
        check_values("_initialize_w(center)", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 3. _initialize_w(0) (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        auto t0 = now();
        mps_f1._initialize_w(0);
        std::cout << "[_initialize_w(0)] (" << elapsed_ms(t0) << " ms)\n";
        mps_f1.check_canonical();
        auto vals = mps_f1.eval_list(points);
        check_values("_initialize_w(0)", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 4. _initialize_w(last) (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        int last = mps_f1.get_size() - 1;
        auto t0 = now();
        mps_f1._initialize_w(last);
        std::cout << "[_initialize_w(" << last << ")] (" << elapsed_ms(t0) << " ms)\n";
        mps_f1.check_canonical();
        auto vals = mps_f1.eval_list(points);
        check_values("_initialize_w(last)", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 5. shift_w left->right (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        mps_f1._initialize_w(0);
        auto t0 = now();
        std::cout << "[shift_w left->right]: ";
        for (int i = 1; i < mps_f1.get_size(); i++)
        {
            mps_f1.shift_w(i);
            mps_f1.check_canonical();
            std::cout << i << " ";
        }
        std::cout << "(" << elapsed_ms(t0) << " ms)\n";
        auto vals = mps_f1.eval_list(points);
        check_values("shift_w left->right", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 6. shift_w right->left (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        int last = mps_f1.get_size() - 1;
        mps_f1._initialize_w(last);
        auto t0 = now();
        std::cout << "[shift_w right->left]: ";
        for (int i = last - 1; i >= 0; i--)
        {
            mps_f1.shift_w(i);
            mps_f1.check_canonical();
            std::cout << i << " ";
        }
        std::cout << "(" << elapsed_ms(t0) << " ms)\n";
        auto vals = mps_f1.eval_list(points);
        check_values("shift_w right->left", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 7. compress_svd — no working site (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        Eigen::Index chi_before = mps_f1.get_chi();
        auto t0 = now();
        mps_f1.compress_svd(default_tol<cScalar>(), -1);
        Eigen::Index chi_after = mps_f1.get_chi();
        std::cout << "[compress_svd no w] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        mps_f1.check_canonical();
        auto vals = mps_f1.eval_list(points);
        check_values("compress_svd no w", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 8. compress_svd — working site at center (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        int center = mps_f1.get_size() / 2;
        mps_f1._initialize_w(center);
        Eigen::Index chi_before = mps_f1.get_chi();
        auto t0 = now();
        mps_f1.compress_svd(default_tol<cScalar>(), -1);
        Eigen::Index chi_after = mps_f1.get_chi();
        std::cout << "[compress_svd w=" << center << "] chi: " << chi_before << " -> " << chi_after
                  << " (" << elapsed_ms(t0) << " ms)\n";
        mps_f1.check_canonical();
        auto vals = mps_f1.eval_list(points);
        check_values("compress_svd w=center", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 9. compress_svd — bond dim truncation sweep (f1)
    //     Truncate from chi_before down to chi_before-5
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");
        Eigen::Index chi_before = mps_f1.get_chi();
        Eigen::Index chi_min = std::max((Eigen::Index)1, chi_before - 5);
        std::cout << "[compress_svd truncation sweep] chi_before = " << chi_before
                  << ", sweeping down to " << chi_min << "\n";
        for (Eigen::Index max_bond = chi_before; max_bond >= chi_min; max_bond--)
        {
            MPS<cScalar> tmp(filename_1 + ".tt");
            auto t0 = now();
            tmp.compress_svd(default_tol<cScalar>(), max_bond);
            Eigen::Index chi_after = tmp.get_chi();
            double t_ms = elapsed_ms(t0);
            auto vals = tmp.eval_list(points);

            Real max_abs2 = Real(0);
            Real max_rel2 = Real(0);
            for (size_t i = 0; i < points.size(); i++)
            {
                Real num2 = abs2_diff<cScalar>(vals[i], values_before_f1[i]);
                Real den2 = abs2<cScalar>(values_before_f1[i]);
                if (num2 > max_abs2) max_abs2 = num2;
                Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
                if (rel2 > max_rel2) max_rel2 = rel2;
            }
            double max_abs = std::sqrt(to_double<Real>(max_abs2));
            double max_rel = std::sqrt(to_double<Real>(max_rel2));
            std::cout << "  max_bond=" << max_bond
                      << "  chi: " << chi_before << " -> " << chi_after
                      << "  abs err = " << max_abs
                      << "  rel err = " << max_rel
                      << "  (" << t_ms << " ms)\n";
        }
    }

    // -------------------------------------------------------
    // 10. Scalar multiplication / division (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");

        auto t0 = now();
        TT<cScalar> tt3 = mps_f1 * cScalar(3);
        std::cout << "[scalar multiply *3] (" << elapsed_ms(t0) << " ms)\n";

        auto vals = tt3.eval_list(points);

        std::vector<cScalar> expected;
        expected.reserve(values_before_f1.size());
        for (auto const& x : values_before_f1)
            expected.push_back(x * cScalar(3));

        Real max_err2 = Real(0);
        for (size_t i = 0; i < vals.size(); i++)
        {
            Real e = abs2_diff<cScalar>(vals[i], expected[i]);
            if (e > max_err2)
                max_err2 = e;
        }
        std::cout << "  scalar multiply error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";

        tt3 = tt3 / cScalar(3);
        auto vals_back = tt3.eval_list(points);
        check_values("scalar multiply/divide", vals_back, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 11. Zero scalar multiplication (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> mps_f1(filename_1 + ".tt");

        auto t0 = now();
        TT<cScalar> zero = mps_f1 * cScalar(0);
        std::cout << "[scalar multiply *0] (" << elapsed_ms(t0) << " ms)\n";

        auto vals = zero.eval_list(points);

        Real max_val = Real(0);
        for (auto const& x : vals)
        {
            auto a = abs2<cScalar>(x);
            if (a > max_val)
                max_val = a;
        }
        std::cout << "  max |zero(x)| = "
                  << std::sqrt(to_double(max_val))
                  << "\n";

        if (max_val != Real(0))
            std::cerr << "WARNING: zero TT is not zero\n";
    }

    // -------------------------------------------------------
    // 12. Addition and subtraction (f1 + f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");
        MPS<cScalar> b(filename_1 + ".tt");

        auto t0 = now();
        TT<cScalar> c = a + b;
        std::cout << "[TT addition f1+f1] (" << elapsed_ms(t0) << " ms)\n";
        print_shape(c, "f1+f1");

        auto vals = c.eval_list(points);

        std::vector<cScalar> expected;
        expected.reserve(values_before_f1.size());
        for (auto const& x : values_before_f1)
            expected.push_back(x + x);

        Real max_err2 = Real(0);
        for (size_t i = 0; i < vals.size(); i++)
        {
            Real e = abs2_diff<cScalar>(vals[i], expected[i]);
            if (e > max_err2)
                max_err2 = e;
        }
        std::cout << "  addition error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";

        TT<cScalar> d = c - a;
        auto vals2 = d.eval_list(points);
        check_values("addition then subtraction", vals2, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 13. Addition (f1 + f2)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");
        MPS<cScalar> b(filename_2 + ".tt");

        auto t0 = now();
        TT<cScalar> c = a + b;
        std::cout << "[TT addition f1+f2] (" << elapsed_ms(t0) << " ms)\n";
        print_shape(c, "f1+f2");

        auto vals = c.eval_list(points);

        std::vector<cScalar> expected;
        expected.reserve(values_before_f1.size());
        for (size_t i = 0; i < values_before_f1.size(); i++)
            expected.push_back(values_before_f1[i] + values_before_f2[i]);

        Real max_err2 = Real(0);
        for (size_t i = 0; i < vals.size(); i++)
        {
            Real e = abs2_diff<cScalar>(vals[i], expected[i]);
            if (e > max_err2)
                max_err2 = e;
        }
        std::cout << "  f1+f2 error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";
    }

    // -------------------------------------------------------
    // 14. Subtraction (f1 - f2)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");
        MPS<cScalar> b(filename_2 + ".tt");

        auto t0 = now();
        TT<cScalar> c = a - b;
        std::cout << "[TT subtraction f1-f2] (" << elapsed_ms(t0) << " ms)\n";

        auto vals = c.eval_list(points);

        std::vector<cScalar> expected;
        expected.reserve(values_before_f1.size());
        for (size_t i = 0; i < values_before_f1.size(); i++)
            expected.push_back(values_before_f1[i] - values_before_f2[i]);

        Real max_err2 = Real(0);
        for (size_t i = 0; i < vals.size(); i++)
        {
            Real e = abs2_diff<cScalar>(vals[i], expected[i]);
            if (e > max_err2)
                max_err2 = e;
        }
        std::cout << "  f1-f2 error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";
    }

    // -------------------------------------------------------
    // 15. Unary minus (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");

        auto t0 = now();
        TT<cScalar> neg = -a;
        std::cout << "[unary minus] (" << elapsed_ms(t0) << " ms)\n";

        auto vals = neg.eval_list(points);

        Real max_err2 = Real(0);
        for (size_t i = 0; i < vals.size(); i++)
        {
            Real e = abs2_diff<cScalar>(vals[i], -values_before_f1[i]);
            if (e > max_err2)
                max_err2 = e;
        }
        std::cout << "  unary minus error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";
    }

    // -------------------------------------------------------
    // 16. In-place operators (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");

        a += a;
        a -= a;

        auto vals = a.eval_list(points);

        Real max_val = Real(0);
        for (auto const& x : vals)
        {
            Real e = abs2<cScalar>(x);
            if (e > max_val)
                max_val = e;
        }
        std::cout << "[in-place += -=] residual = "
                  << std::sqrt(to_double<Real>(max_val))
                  << "\n";
    }

    // -------------------------------------------------------
    // 17. In-place scalar operators (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");

        a *= cScalar(2);
        a /= cScalar(2);

        auto vals = a.eval_list(points);
        check_values("in-place *= /=", vals, values_before_f1, canon_tol);
    }

    // -------------------------------------------------------
    // 18. Conjugation (f1)
    // -------------------------------------------------------
    {
        MPS<cScalar> tt2(filename_1 + ".tt");

        auto t0 = now();
        TT<cScalar> conj_tt = tt2.conj();
        std::cout << "[conjugation] (" << elapsed_ms(t0) << " ms)\n";

        auto vals = conj_tt.eval_list(points);

        Real max_err2 = Real(0);
        for (size_t i = 0; i < vals.size(); i++)
        {
            cScalar expected = Eigen::numext::conj(values_before_f1[i]);
            Real e = abs2_diff<cScalar>(vals[i], expected);
            if (e > max_err2)
                max_err2 = e;
        }
        std::cout << "  conjugation error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";
    }

    // -------------------------------------------------------
    // 19. Copy constructor independence
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");
        MPS<cScalar> b = a;

        b *= cScalar(5);

        auto va = a.eval_list(points);
        auto vb = b.eval_list(points);

        bool identical = true;
        for(size_t i = 0; i < va.size(); i++)
        {
            if(abs2_diff<cScalar>(va[i], vb[i]) == Real(0))
            {
                identical = false;
                break;
            }
        }
        std::cout << "[copy independence] "
                  << (identical ? "OK" : "FAILED")
                  << "\n";
    }

    // -------------------------------------------------------
    // 20. get_core consistency
    // -------------------------------------------------------
    {
        MPS<cScalar> tt2(filename_1 + ".tt");

        auto const& cref = tt2.get_core();
        bool ok = (cref.size() == static_cast<size_t>(tt2.get_size()));
        std::cout << "[get_core const access] "
                  << (ok ? "OK" : "FAILED")
                  << "\n";
    }

    // -------------------------------------------------------
    // 21. generate_points range check
    // -------------------------------------------------------
    {
        MPS<cScalar> tt2(filename_1 + ".tt");

        auto pts = tt2.generate_points(1000);

        bool ok = true;
        for(auto const& p : pts)
        {
            if((int)p.size() != tt2.get_size())
                ok = false;
            for(int i = 0; i < tt2.get_size(); i++)
            {
                auto shape = tt2.get_shape()[i];
                int nphys = std::get<1>(shape);
                if(p[i] < 0 || p[i] >= nphys)
                    ok = false;
            }
        }
        std::cout << "[generate_points range] "
                  << (ok ? "OK" : "FAILED")
                  << "\n";
    }

    // -------------------------------------------------------
    // 22. Evaluation linearity: (a+b)(x) == a(x)+b(x)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");
        MPS<cScalar> b(filename_1 + ".tt");

        TT<cScalar> c = a + b;

        auto va = a.eval_list(points);
        auto vb = b.eval_list(points);
        auto vc = c.eval_list(points);

        Real max_err2 = Real(0);
        for(size_t i = 0; i < points.size(); i++)
        {
            Real e = abs2_diff<cScalar>(
                vc[i],
                va[i] + vb[i]
            );
            if(e > max_err2)
                max_err2 = e;
        }
        std::cout << "[linearity check] error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";
    }

    // -------------------------------------------------------
    // 23. MPS dot product: <f1|f1>
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");

        auto t0 = now();
        cScalar dot_val = a.dot(a);
        auto n2 = a.norm2();
        std::cout << "[MPS dot f1.f1] (" << elapsed_ms(t0) << " ms)\n";
        std::cout << "  <f1|f1> = " << dot_val << "\n";
        std::cout << "  norm2   = " << n2 << "\n";

        Real diff2 = abs2_diff<cScalar>(dot_val, cScalar(n2));
        std::cout << "  |dot - norm2| = "
                  << std::sqrt(to_double<Real>(diff2))
                  << "\n";
    }

    // -------------------------------------------------------
    // 24. MPS dot product: <f1|f2>
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");
        MPS<cScalar> b(filename_2 + ".tt");

        auto t0 = now();
        cScalar dot_val = a.dot(b);
        std::cout << "[MPS dot f1.f2] (" << elapsed_ms(t0) << " ms)\n";
        std::cout << "  <f1|f2> = " << dot_val << "\n";
    }

    // -------------------------------------------------------
    // 25. MPS dot product: <f2|f2>
    // -------------------------------------------------------
    {
        MPS<cScalar> b(filename_2 + ".tt");

        auto t0 = now();
        cScalar dot_val = b.dot(b);
        auto n2 = b.norm2();
        std::cout << "[MPS dot f2.f2] (" << elapsed_ms(t0) << " ms)\n";
        std::cout << "  <f2|f2> = " << dot_val << "\n";
        std::cout << "  norm2   = " << n2 << "\n";

        Real diff2 = abs2_diff<cScalar>(dot_val, cScalar(n2));
        std::cout << "  |dot - norm2| = "
                  << std::sqrt(to_double<Real>(diff2))
                  << "\n";
    }

    // -------------------------------------------------------
    // 26. MPS promotion from TT (construct MPS from TT result)
    // -------------------------------------------------------
    {
        MPS<cScalar> a(filename_1 + ".tt");
        MPS<cScalar> b(filename_1 + ".tt");

        TT<cScalar> sum_tt = a + b;
        MPS<cScalar> sum_mps(std::move(sum_tt));

        auto vals = sum_mps.eval_list(points);

        std::vector<cScalar> expected;
        expected.reserve(values_before_f1.size());
        for (auto const& x : values_before_f1)
            expected.push_back(x + x);

        Real max_err2 = Real(0);
        for (size_t i = 0; i < vals.size(); i++)
        {
            Real e = abs2_diff<cScalar>(vals[i], expected[i]);
            if (e > max_err2)
                max_err2 = e;
        }
        std::cout << "[MPS from TT] eval error = "
                  << std::sqrt(to_double<Real>(max_err2))
                  << "\n";
    }

    std::cout << "----------------------------------------\n";
    std::cout << "Total time for " << name << ": " << elapsed_ms(t_type) << " ms\n";
}

int main()
{
    // TCI the two function in float128
    {
        using scalar_type = float128;
        using complex_type = Cfloat128;

        scalar_type a_1 = -2.0;
        scalar_type b_1 = 10.0;
        scalar_type c_0 = 0.0;

        int nBit = 30;
        int n_iter = 15;
        int nb_point_out = 1000;

        std::string file_prefix_1 = "test/output/test_out_f1";
        std::string file_prefix_2 = "test/output/test_out_f2";

        do_save_TCI<scalar_type>(
            a_1,
            b_1,
            c_0,
            nBit,
            function_1<complex_type>,
            n_iter,
            true,
            file_prefix_1,
            nb_point_out);
        
        do_save_TCI<scalar_type>(
            a_1,
            b_1,
            c_0,
            nBit,
            function_2<complex_type>,
            n_iter,
            true,
            file_prefix_2,
            nb_point_out);
    }

    const int n_points = 1000;
    // now we have the following file:
    std::string file_tt_f1 = "test/output/test_out_f1";
    std::string file_tt_f2 = "test/output/test_out_f2";

    test_mps<std::complex<double>, long long>("complex<double> long long", file_tt_f1, file_tt_f2, n_points);
    test_mps<std::complex<double>, util::i128>("complex<double> i128", file_tt_f1, file_tt_f2, n_points);

    test_mps<std::complex<float128>, long long>("complex<float128> long long", file_tt_f1, file_tt_f2, n_points);
    test_mps<std::complex<float128>, util::i128>("complex<float128> i128", file_tt_f1, file_tt_f2, n_points);

    test_mps<std::complex<dd_128>, long long>("complex<dd_128>long long ", file_tt_f1, file_tt_f2, n_points);
    test_mps<std::complex<dd_128>, util::i128>("complex<dd_128> i128", file_tt_f1, file_tt_f2, n_points);

    return 0;
}
