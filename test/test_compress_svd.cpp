#include "type_double_double.h"
#include "type_float128_boost.h"
#include "type_int128.h"
#include "tt_base.h"

#include "xf-qd-runner.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <iomanip>

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
// Numeric helpers
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

// ============================================================
// Print helpers
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

template<typename Scalar>
void do_save_TCI(
    typename Eigen::NumTraits<Scalar>::Real a_1,
    typename Eigen::NumTraits<Scalar>::Real b_1,
    typename Eigen::NumTraits<Scalar>::Real c0,
    int nBit,
    std::function<Scalar(typename Eigen::NumTraits<Scalar>::Real)> function_to_tci,
    int n_iter,
    const std::string& log_filename,
    const std::string& filename,
    int nb_point_out){
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    QTGrid<Real, long long> grid(a_1, b_1, nBit);

    std::vector<int> pivot1 = grid.coord_to_id(c0);

    TCI2_1D_runner_param<Scalar> tci_param(nBit, n_iter, 100,
        {.pivot1 = pivot1, .fullPiv = true, .cache = CacheLevel::runner});

    std::string logger_name = "logger_" + log_filename;
    auto logger = spdlog::get(logger_name);
    if (!logger)
    {
        logger = spdlog::basic_logger_mt(logger_name, "test/output/" + log_filename);
    }

    TCI2_1D_Runner<Scalar, long long> tci_runner(grid, tci_param, function_to_tci, logger);

    std::vector<Real> additional_pivot = {};

    tci_runner.fit(
        additional_pivot,
        true,               // verbose
        true,               // save
        filename,           // file_prefix (path+prefix) to save
        nb_point_out        // nb_point to test error
    );
}

template<typename cScalar, typename Sint>
void test_function(const std::string& name, const std::string& filename_1, const std::string& filename_2, int n_points = 100)
{
    using Real = typename Eigen::NumTraits<cScalar>::Real;

    std::cout << "\n========================================\n";
    std::cout << "Testing " << name << "\n";
    std::cout << "========================================\n";

    // -------------------------------------------------------
    // 1. Load
    // -------------------------------------------------------
    {
        auto t0 = now();
        TT<cScalar> mps_f1(filename_1 + ".tt");
        TT<cScalar> mps_f2(filename_2 + ".tt");
        std::cout << "[LOAD] OK (" << elapsed_ms(t0) << " ms)\n";
        print_shape(mps_f1, "original f1");
        print_shape(mps_f2, "original f2");
    }

    // -------------------------------------------------------
    // Generate shared points + reference values
    // -------------------------------------------------------
    // points is the list of point in id. [[0, 1, ...], [0, 0, ...], ...]
    std::vector<std::vector<int>> points;

    std::vector<cScalar> tt_value_f1;
    std::vector<cScalar> tt_value_f2;

    // point ref is the list of point in coordinate no in id. Mps expect id. [1.5, 2.3, -2.5, ...]
    std::vector<Real> point_ref;
    std::vector<cScalar> value_ref_f1;
    std::vector<cScalar> value_ref_f2;

    // fill all the data define just before
    {
        TT<cScalar> tt_f1(filename_1 + ".tt");
        TT<cScalar> tt_f2(filename_2 + ".tt");

        QTGrid<Real, Sint> grid_1(filename_1 + "_grid_E.json");

        points = tt_f1.generate_points(n_points);

        for (int i=0; i< n_points; i++){
            point_ref.push_back(grid_1.id_to_coord(points[i]));
            value_ref_f1.push_back(function_1<cScalar>(point_ref[i]));
            value_ref_f2.push_back(function_2<cScalar>(point_ref[i]));
        }

        auto t0 = now();
        tt_value_f1 = tt_f1.eval_list(points);
        tt_value_f2 = tt_f2.eval_list(points);
        std::cout << "[eval " << n_points << " points original] (" << elapsed_ms(t0) << " ms)\n";
    }

    // ================================================================
    // test 1: compare TT evaluation to direct function evaluation
    //         report abs/rel error for f1 and f2
    // ================================================================
    {
        std::cout << "\n[test 1] TT vs exact function evaluation\n";

        auto err = [&](const std::string& label,
                        const std::vector<cScalar>& tt_vals,
                        const std::vector<cScalar>& ref_vals)
        {
            Real max_abs2 = Real(0), max_rel2 = Real(0);
            for (size_t i = 0; i < points.size(); i++)
            {
                Real num2 = abs2_diff<cScalar>(tt_vals[i], ref_vals[i]);
                Real den2 = abs2<cScalar>(ref_vals[i]);
                if (num2 > max_abs2) max_abs2 = num2;
                Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
                if (rel2 > max_rel2) max_rel2 = rel2;
            }
            double max_abs = std::sqrt(to_double<Real>(max_abs2));
            double max_rel = std::sqrt(to_double<Real>(max_rel2));
            std::cout << "  " << label
                      << "  abs_err=" << std::scientific << std::setprecision(4) << max_abs
                      << "  rel_err=" << max_rel << "\n";
        };

        err("f1", tt_value_f1, value_ref_f1);
        err("f2", tt_value_f2, value_ref_f2);
    }

    // ================================================================
    // test 2: compress_svd sweep from chi down to chi-5
    //         compare compressed TT against:
    //           (a) the exact function (value_ref)
    //           (b) the original TT evaluation (tt_value before SVD)
    // ================================================================
    {
        std::cout << "\n[test 2] compress_svd truncation sweep\n";

        auto sweep = [&](const std::string& label,
                          const std::string& filename,
                          const std::vector<cScalar>& ref_exact,
                          const std::vector<cScalar>& ref_tt)
        {
            TT<cScalar> tt_orig(filename + ".tt");
            Eigen::Index chi_before = tt_orig.get_chi();
            Eigen::Index chi_min = 1;

            std::cout << "  [" << label << "] chi_before=" << chi_before
                      << "  sweeping down to " << chi_min << "\n";

            for (Eigen::Index max_bond = chi_before; max_bond >= chi_min; max_bond--)
            {
                TT<cScalar> tmp(filename + ".tt");
                tmp.compress_svd(Real(0), max_bond);
                Eigen::Index chi_after = tmp.get_chi();
                auto vals = tmp.eval_list(points);

                // error vs exact function
                Real abs2_exact = Real(0), rel2_exact = Real(0);
                for (size_t i = 0; i < points.size(); i++)
                {
                    Real num2 = abs2_diff<cScalar>(vals[i], ref_exact[i]);
                    Real den2 = abs2<cScalar>(ref_exact[i]);
                    if (num2 > abs2_exact) abs2_exact = num2;
                    Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
                    if (rel2 > rel2_exact) rel2_exact = rel2;
                }

                // error vs original TT
                Real abs2_tt = Real(0), rel2_tt = Real(0);
                for (size_t i = 0; i < points.size(); i++)
                {
                    Real num2 = abs2_diff<cScalar>(vals[i], ref_tt[i]);
                    Real den2 = abs2<cScalar>(ref_tt[i]);
                    if (num2 > abs2_tt) abs2_tt = num2;
                    Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
                    if (rel2 > rel2_tt) rel2_tt = rel2;
                }

                std::cout << "    max_bond=" << std::setw(2) << max_bond
                          << "  chi: " << chi_before << " -> " << std::setw(2) << chi_after
                          << "  vs_exact: abs=" << std::scientific << std::setprecision(4) << std::sqrt(to_double<Real>(abs2_exact))
                          << "  rel=" << std::sqrt(to_double<Real>(rel2_exact))
                          << "  |  vs_tt: abs=" << std::sqrt(to_double<Real>(abs2_tt))
                          << "  rel=" << std::sqrt(to_double<Real>(rel2_tt))
                          << "\n";
            }
        };

        sweep("f1", filename_1, value_ref_f1, tt_value_f1);
        sweep("f2", filename_2, value_ref_f2, tt_value_f2);
    }
}



int main()
{
    std::string file_tt_f1 = "test/output/test2_out_f1";
    std::string file_tt_f2 = "test/output/test2_out_f2";
    // TCI the two function in float128
    {
        using scalar_type = Cfloat128;
        using real_type = float128;

        real_type a_1 = -2.0;
        real_type b_1 = 10.0;
        real_type c_0 = 0.0;

        int nBit = 30;
        int n_iter = 15;
        int nb_point_out = 1000;

        do_save_TCI<scalar_type>(
            a_1,
            b_1,
            c_0,
            nBit,
            function_1<scalar_type>,
            n_iter,
            "compress_svd_f1.log",
            file_tt_f1,
            nb_point_out);
        
        do_save_TCI<scalar_type>(
            a_1,
            b_1,
            c_0,
            nBit,
            function_2<scalar_type>,
            n_iter,
            "compress_svd_f2.log",
            file_tt_f2,
            nb_point_out);
    }
    // now the tt and the grid_E are costruct

    const int n_points = 1000;
    
    test_function<std::complex<double>, long long>("complex<double> long long", file_tt_f1, file_tt_f2, n_points);
    test_function<std::complex<double>, util::i128>("complex<double> i128", file_tt_f1, file_tt_f2, n_points);

    test_function<std::complex<dd_128>, long long>("complex<dd_128>long long ", file_tt_f1, file_tt_f2, n_points);
    test_function<std::complex<dd_128>, util::i128>("complex<dd_128> i128", file_tt_f1, file_tt_f2, n_points);

    test_function<std::complex<float128>, long long>("complex<float128> long long", file_tt_f1, file_tt_f2, n_points);
    test_function<std::complex<float128>, util::i128>("complex<float128> i128", file_tt_f1, file_tt_f2, n_points);

    return 0;
}