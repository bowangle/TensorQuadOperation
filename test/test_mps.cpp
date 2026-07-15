#include "type_double_double.h"
#include "type_float128_boost.h"
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
    return v - v*v - cT(0.0,1.0) * v;
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
        logger = spdlog::basic_logger_mt("test_file_logger", "file.log");
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

int main()
{
    float128 a_1 = -2.0;
    float128 b_1 = 10.0;
    float128 c_0 = 0.0;

    int nBit = 20;
    int n_iter = 20;
    int nb_point_out = 1000;

    std::string file_prefix_1 = "test/test_out_f1";
    std::string file_prefix_2 = "test/test_out_f2";

    do_save_TCI<float128>(
        a_1,
        b_1,
        c_0,
        nBit,
        function_1<Cfloat128>,
        n_iter,
        true,
        file_prefix_1,
        nb_point_out);
    
    do_save_TCI<float128>(
        a_1,
        b_1,
        c_0,
        nBit,
        function_2<Cfloat128>,
        n_iter,
        true,
        file_prefix_2,
        nb_point_out);

    return 0;
}