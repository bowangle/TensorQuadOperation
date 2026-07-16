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

template<typename T>
void print_shape(const MPS<T>& tt, const std::string& name)
{
    std::cout << name << " shape: ";
    for (auto const& [l, p, r] : tt.get_shape())
        std::cout << "(" << l << "," << p << "," << r << ") ";
    std::cout << "\n";
    std::cout << name << " chi: " << tt.get_chi() << "\n";
}

template<typename cScalar, typename Sint>
void test_mps(const std::string& name, const std::string& filename_1, const std::string& filename_2, int n_points = 100)
{
    using real_Scalar = typename Eigen::NumTraits<cScalar>::Real;
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
    std::vector<cScalar> values_before_mps_f1;
    std::vector<cScalar> values_before_mps_f2;

    std::vector<cScalar> value_f1;
    std::vector<cScalar> value_f2;

    std::vector<cScalar> value_ref_f1;
    std::vector<cScalar> value_ref_f2;
    {

        QTGrid<real_Scalar, Sint> grid(filename_1 + "_grid_E.json");

        MPS<cScalar> mps_f1(filename_1 + ".tt");
        MPS<cScalar> mps_f2(filename_2 + ".tt");

        points = mps_f1.generate_points(n_points);
        auto t0 = now();
        values_before_mps_f1 = mps_f1.eval_list(points);
        values_before_mps_f2 = mps_f2.eval_list(points);
        std::cout << "[eval " << n_points << " points original] (" << elapsed_ms(t0) << " ms)\n";

        for (int i=0; i< n_points; i++){
            const real_Scalar x = grid.id_to_coord(MultiIndex(points[i]));
            value_ref_f1.push_back(function_1<cScalar>(x));
            value_ref_f2.push_back(function_2<cScalar>(x));
        }
    }
}

int main()
{
    // TCI the two function
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

    test_mps<std::complex<double>, long long>("complex<double>", file_tt_f1, file_tt_f2, n_points);
    test_mps<std::complex<float128>, util::i128>("complex<float128>", file_tt_f1, file_tt_f2, n_points);
    //test_mps<std::complex<dd_128>, util::i128>("complex<dd_128>", file_tt_f1, file_tt_f2, n_points);

    return 0;
}