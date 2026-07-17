#include "type_double_double.h"
#include "type_float128_boost.h"
#include "type_int128.h"
#include "tt_base.h"

#include "runner.h"
#include "grid.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

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
        logger = spdlog::basic_logger_mt("test_file_logger", "test/output/file2.log");
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

template<typename cScalar, typename Sint>
void test_function(const std::string& name, const std::string& filename_1, const std::string& filename_2, int n_points = 100)
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
            point_ref.push_back(grid_1.id_to_coord(MultiIndex(points[i])));
            value_ref_f1.push_back(function_1<cScalar>(point_ref[i]));
            value_ref_f2.push_back(function_2<cScalar>(point_ref[i]));
        }

        auto t0 = now();
        tt_value_f1 = tt_f1.eval_list(points);
        value_ref_f2 = tt_f2.eval_list(points);
        std::cout << "[eval " << n_points << " points original] (" << elapsed_ms(t0) << " ms)\n";
    }

    // everytime do it for f1 and f2.

    // test 1: compare it to direct function evaluation value_ref_f -tt_value_f
    {
        // TODO
    }

    // test 2: compress_svd from max_bond_dim to max_bond_dim - 5, compare against with value_ref, and against tt_value_f1 which are the mps value before svd
    {
        // TODO
    }
}



int main()
{
    std::string file_tt_f1 = "test/output/test2_out_f1";
    std::string file_tt_f2 = "test/output/test2_out_f2";
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

        do_save_TCI<scalar_type>(
            a_1,
            b_1,
            c_0,
            nBit,
            function_1<complex_type>,
            n_iter,
            true,
            file_tt_f1,
            nb_point_out);
        
        do_save_TCI<scalar_type>(
            a_1,
            b_1,
            c_0,
            nBit,
            function_2<complex_type>,
            n_iter,
            true,
            file_tt_f2,
            nb_point_out);
    }
    // now the tt and the grid_E are costruct

    const int n_points = 1000;
    
    

    test_function<std::complex<double>, long long>("complex<double> long long", file_tt_f1, file_tt_f2, n_points);
    test_function<std::complex<double>, util::i128>("complex<double> i128", file_tt_f1, file_tt_f2, n_points);

    // test_function<std::complex<float128>, long long>("complex<float128> long long", file_tt_f1, file_tt_f2, n_points);
    // test_function<std::complex<float128>, util::i128>("complex<float128> i128", file_tt_f1, file_tt_f2, n_points);

    test_function<std::complex<dd_128>, long long>("complex<dd_128>long long ", file_tt_f1, file_tt_f2, n_points);
    test_function<std::complex<dd_128>, util::i128>("complex<dd_128> i128", file_tt_f1, file_tt_f2, n_points);

    return 0;
}