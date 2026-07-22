#include "type_double_double.h"
#include "type_float128_boost.h"
#include "type_int128.h"
#include "mps_base.h"
#include "mpo_base.h"

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

// ============================================================
// Numeric helpers (abs2 / to_double)
// ============================================================
template<typename RealT>
double to_double(RealT const& v) { return static_cast<double>(v); }

template<>
double to_double<dd_128>(dd_128 const& v) { return v.x[0]; }

template<>
double to_double<float128>(float128 const& v) { return v.convert_to<double>(); }

template<typename T>
typename Eigen::NumTraits<T>::Real abs2(T const& v)
{
    return std::norm(v);
}

template<typename T>
typename Eigen::NumTraits<T>::Real abs2_diff(T const& a, T const& b)
{
    return abs2<T>(a - b);
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
    TCI_param tci_param = TCI_param(grid.get_nBits(), n_iter, do_cache);

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
        grid.get_a(),       // grid minimal point
        grid.get_b()        // grid maximal point
    );
}

template<typename cScalar, typename Sint>
void test_mps(const std::string& name, const std::string& filename_1, const std::string& filename_2, int n_points = 100)
{
    using Real = typename Eigen::NumTraits<cScalar>::Real;

    std::cout << "\n========================================\n";
    std::cout << "Testing " << name << "\n";
    std::cout << "========================================\n";

    auto t_type = now();

    // -------------------------------------------------------
    // 1. Load and compute reference value
    // -------------------------------------------------------

    auto tload = now();
    MPS<cScalar> mps_f1(filename_1 + ".tt");
    MPS<cScalar> mps_f2(filename_2 + ".tt");
    std::cout << "[LOAD] OK (" << elapsed_ms(tload) << " ms)\n";
    print_shape(mps_f1, "original f1");
    print_shape(mps_f2, "original f2");

    MPO<cScalar> mpo_f1 = MPO<cScalar>::from_mps(mps_f1);
    MPO<cScalar> mpo_f2 = MPO<cScalar>::from_mps(mps_f2);

    std::vector<std::vector<int>> points;   // points is the list of point in id. [[0, 1, ...], [0, 0, ...], ...] 

    std::vector<cScalar> mps_value_f1;      // value of the mps right after loading for function 1
    std::vector<cScalar> mps_value_f2;      // value of the mps right after loading for function 2

    // point ref is the list of point in coordinate no in id. Mps expect id. [1.5, 2.3, -2.5, ...]
    std::vector<Real> point_ref;            // point_ref is the list of point in coordinate: [2.3, -0.5, 0.3, ...]
    std::vector<cScalar> value_ref_f1;      // reference value right out of the function 1 (without passing through TCI and MPS)
    std::vector<cScalar> value_ref_f2;      // reference value right out of the function 2 (without passing through TCI and MPS)

    // fill all the data define just before
    {
        QTGrid<Real, Sint> grid_1(filename_1 + "_grid_E.json");

        points = mps_f1.generate_points(n_points);

        for (int i=0; i< n_points; i++){
            point_ref.push_back(grid_1.id_to_coord(MultiIndex(points[i])));
            value_ref_f1.push_back(function_1<cScalar>(point_ref[i]));
            value_ref_f2.push_back(function_2<cScalar>(point_ref[i]));
        }

        auto t0 = now();
        mps_value_f1 = mps_f1.eval_list(points);
        mps_value_f2 = mps_f2.eval_list(points);
        std::cout << "[eval " << n_points << " points original] (" << elapsed_ms(t0) << " ms)\n";
    }

    // Test 2: test the from_mps method:
    {
        auto t0 = now();

        // 2a. Shape: every core should have n_phys == 4
        print_shape(mpo_f1, "mpo_f1");
        print_shape(mpo_f2, "mpo_f2");

        // 2b. Diagonal structure — check element-wise:
        //     MPO.phys(0) == MPS.phys(0),  MPO.phys(3) == MPS.phys(1),
        //     MPO.phys(1) == 0,            MPO.phys(2) == 0
        auto const& mps1_cores = mps_f1.get_core();
        auto const& mpo1_cores = mpo_f1.get_core();
        Real max_err = Real(0);
        for (size_t k = 0; k < mps1_cores.size(); k++) {
            // phys(0) should match MPS phys(0)
            max_err = std::max(max_err,
                (mpo1_cores[k].phys_const(0) - mps1_cores[k].phys_const(0)).norm());
            // phys(3) should match MPS phys(1)
            max_err = std::max(max_err,
                (mpo1_cores[k].phys_const(3) - mps1_cores[k].phys_const(1)).norm());
            // phys(1) and phys(2) should be zero
            max_err = std::max(max_err, mpo1_cores[k].phys_const(1).norm());
            max_err = std::max(max_err, mpo1_cores[k].phys_const(2).norm());
        }
        std::cout << "  mpo_f1 diagonal structure: max element-wise diff = "
                  << to_double<Real>(max_err) << "\n";

        // 2c. w, max_bond_dim should be preserved from the MPS
        std::cout << "  mpo_f1 w = " << mpo_f1.get_w()
                  << " (MPS: " << mps_f1.get_w() << ")" << "\n";
        std::cout << "  mpo_f1 max_bond_dim = " << mpo_f1.get_max_bond_dim()
                  << " (MPS: " << mps_f1.get_max_bond_dim() << ")" << "\n";

        std::cout << "[from_mps test] (" << elapsed_ms(t0) << " ms)\n";
    }

    // Test 3.1: test mpo*mps (_mul) for method = "zip-up"
    {
        // reference: pointwise product of the two functions
        std::vector<cScalar> ref_f1f2(n_points);   // f1(x) * f2(x)
        std::vector<cScalar> ref_f1f1(n_points);   // f1(x) * f1(x)
        std::vector<cScalar> ref_f2f2(n_points);   // f2(x) * f2(x)
        for (int i = 0; i < n_points; i++) {
            ref_f1f2[i] = value_ref_f1[i] * value_ref_f2[i];
            ref_f1f1[i] = value_ref_f1[i] * value_ref_f1[i];
            ref_f2f2[i] = value_ref_f2[i] * value_ref_f2[i];
        }

        auto t0 = now();
        MPS<cScalar> mpo_f1_mps_f2 = mpo_f1._mul(mps_f2, "zip-up");
        MPS<cScalar> mpo_f2_mps_f1 = mpo_f2._mul(mps_f1, "zip-up");
        MPS<cScalar> mpo_f1_mps_f1 = mpo_f1._mul(mps_f1, "zip-up");
        MPS<cScalar> mpo_f2_mps_f2 = mpo_f2._mul(mps_f2, "zip-up");
        std::cout << "[zip-up _mul x4] (" << elapsed_ms(t0) << " ms)\n";

        // evaluate and compute errors
        auto check = [&](const std::string& label,
                         const MPS<cScalar>& result,
                         const std::vector<cScalar>& ref) {
            auto t1 = now();
            std::vector<cScalar> vals = result.eval_list(points);
            std::cout << "  [eval " << label << "] (" << elapsed_ms(t1) << " ms)\n";

            Real max_abs2 = Real(0);
            Real max_rel2 = Real(0);
            for (int i = 0; i < n_points; i++) {
                Real num2 = abs2_diff<cScalar>(vals[i], ref[i]);
                Real den2 = abs2<cScalar>(ref[i]);
                if (num2 > max_abs2) max_abs2 = num2;
                Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
                if (rel2 > max_rel2) max_rel2 = rel2;
            }
            double max_abs = std::sqrt(to_double<Real>(max_abs2));
            double max_rel = std::sqrt(to_double<Real>(max_rel2));
            std::cout << "  " << label << ": abs err = " << max_abs
                      << "   rel err = " << max_rel << "\n";
        };

        check("mpo_f1 @ mps_f2", mpo_f1_mps_f2, ref_f1f2);
        check("mpo_f2 @ mps_f1", mpo_f2_mps_f1, ref_f1f2);
        check("mpo_f1 @ mps_f1", mpo_f1_mps_f1, ref_f1f1);
        check("mpo_f2 @ mps_f2", mpo_f2_mps_f2, ref_f2f2);
    }

    // Test 3.2: test mpo*mps (_mul) for method = "qrsvd"
    {
        // reference: pointwise product of the two functions
        std::vector<cScalar> ref_f1f2(n_points);   // f1(x) * f2(x)
        std::vector<cScalar> ref_f1f1(n_points);   // f1(x) * f1(x)
        std::vector<cScalar> ref_f2f2(n_points);   // f2(x) * f2(x)
        for (int i = 0; i < n_points; i++) {
            ref_f1f2[i] = value_ref_f1[i] * value_ref_f2[i];
            ref_f1f1[i] = value_ref_f1[i] * value_ref_f1[i];
            ref_f2f2[i] = value_ref_f2[i] * value_ref_f2[i];
        }

        auto t0 = now();
        MPS<cScalar> mpo_f1_mps_f2 = mpo_f1._mul(mps_f2, "qrsvd");
        MPS<cScalar> mpo_f2_mps_f1 = mpo_f2._mul(mps_f1, "qrsvd");
        MPS<cScalar> mpo_f1_mps_f1 = mpo_f1._mul(mps_f1, "qrsvd");
        MPS<cScalar> mpo_f2_mps_f2 = mpo_f2._mul(mps_f2, "qrsvd");
        std::cout << "[qrsvd _mul x4] (" << elapsed_ms(t0) << " ms)\n";

        // evaluate and compute errors
        auto check = [&](const std::string& label,
                         const MPS<cScalar>& result,
                         const std::vector<cScalar>& ref) {
            auto t1 = now();
            std::vector<cScalar> vals = result.eval_list(points);
            std::cout << "  [eval " << label << "] (" << elapsed_ms(t1) << " ms)\n";

            Real max_abs2 = Real(0);
            Real max_rel2 = Real(0);
            for (int i = 0; i < n_points; i++) {
                Real num2 = abs2_diff<cScalar>(vals[i], ref[i]);
                Real den2 = abs2<cScalar>(ref[i]);
                if (num2 > max_abs2) max_abs2 = num2;
                Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
                if (rel2 > max_rel2) max_rel2 = rel2;
            }
            double max_abs = std::sqrt(to_double<Real>(max_abs2));
            double max_rel = std::sqrt(to_double<Real>(max_rel2));
            std::cout << "  " << label << ": abs err = " << max_abs
                      << "   rel err = " << max_rel << "\n";
        };

        check("mpo_f1 @ mps_f2", mpo_f1_mps_f2, ref_f1f2);
        check("mpo_f2 @ mps_f1", mpo_f2_mps_f1, ref_f1f2);
        check("mpo_f1 @ mps_f1", mpo_f1_mps_f1, ref_f1f1);
        check("mpo_f2 @ mps_f2", mpo_f2_mps_f2, ref_f2f2);
    }

    // Test 3.3: test mpo*mps (_mul) for method = "optimize"
    {
        // reference: pointwise product of the two functions
        std::vector<cScalar> ref_f1f2(n_points);   // f1(x) * f2(x)
        std::vector<cScalar> ref_f1f1(n_points);   // f1(x) * f1(x)
        std::vector<cScalar> ref_f2f2(n_points);   // f2(x) * f2(x)
        for (int i = 0; i < n_points; i++) {
            ref_f1f2[i] = value_ref_f1[i] * value_ref_f2[i];
            ref_f1f1[i] = value_ref_f1[i] * value_ref_f1[i];
            ref_f2f2[i] = value_ref_f2[i] * value_ref_f2[i];
        }

        auto t0 = now();
        MPS<cScalar> mpo_f1_mps_f2 = mpo_f1._mul(mps_f2, "optimize");
        MPS<cScalar> mpo_f2_mps_f1 = mpo_f2._mul(mps_f1, "optimize");
        MPS<cScalar> mpo_f1_mps_f1 = mpo_f1._mul(mps_f1, "optimize");
        MPS<cScalar> mpo_f2_mps_f2 = mpo_f2._mul(mps_f2, "optimize");
        std::cout << "[optimize _mul x4] (" << elapsed_ms(t0) << " ms)\n";

        // evaluate and compute errors
        auto check = [&](const std::string& label,
                         const MPS<cScalar>& result,
                         const std::vector<cScalar>& ref) {
            auto t1 = now();
            std::vector<cScalar> vals = result.eval_list(points);
            std::cout << "  [eval " << label << "] (" << elapsed_ms(t1) << " ms)\n";

            Real max_abs2 = Real(0);
            Real max_rel2 = Real(0);
            for (int i = 0; i < n_points; i++) {
                Real num2 = abs2_diff<cScalar>(vals[i], ref[i]);
                Real den2 = abs2<cScalar>(ref[i]);
                if (num2 > max_abs2) max_abs2 = num2;
                Real rel2 = (den2 > Real(0)) ? num2 / den2 : num2;
                if (rel2 > max_rel2) max_rel2 = rel2;
            }
            double max_abs = std::sqrt(to_double<Real>(max_abs2));
            double max_rel = std::sqrt(to_double<Real>(max_rel2));
            std::cout << "  " << label << ": abs err = " << max_abs
                      << "   rel err = " << max_rel << "\n";
        };

        check("mpo_f1 @ mps_f2", mpo_f1_mps_f2, ref_f1f2);
        check("mpo_f2 @ mps_f1", mpo_f2_mps_f1, ref_f1f2);
        check("mpo_f1 @ mps_f1", mpo_f1_mps_f1, ref_f1f1);
        check("mpo_f2 @ mps_f2", mpo_f2_mps_f2, ref_f2f2);
    }

    // -------------------------------------------------------
    // Test 4: Save / reload roundtrip (mpo_f1)
    // -------------------------------------------------------
    {
        // use w_=-1 to avoid canonicalisation and numerical drift from it
        MPO<cScalar> mpo_tmp = MPO<cScalar>::from_mps(
            MPS<cScalar>(filename_1 + ".tt", /*max_bond_dim_=*/0, /*reltol_=*/-1, /*w_=*/-1));

        auto t0 = now();
        std::string tmp_prefix = filename_1 + "_mpo_saved";
        mpo_tmp.save(tmp_prefix);
        MPO<cScalar> reloaded(tmp_prefix + ".tt", /*max_bond_dim_=*/0, /*reltol_=*/-1, /*w_=*/-1);
        std::cout << "[save+reload MPO] (" << elapsed_ms(t0) << " ms)\n";

        // Compare cores element-by-element
        auto const& orig = mpo_tmp.get_core();
        auto const& reload = reloaded.get_core();
        if (orig.size() != reload.size())
        {
            std::cerr << "  FAILED [save+reload MPO]: core count mismatch "
                      << orig.size() << " vs " << reload.size() << "\n";
        }
        else
        {
            Real max_diff = Real(0);
            for (size_t k = 0; k < orig.size(); k++)
            {
                Real d = (orig[k].flatten_as_matrix2_const()
                        - reload[k].flatten_as_matrix2_const()).norm();
                if (d > max_diff) max_diff = d;
            }
            std::cout << "  max core diff = " << to_double<Real>(max_diff) << "\n";
            if (max_diff > Real(0))
                std::cerr << "  WARNING [save+reload MPO]: core mismatch (diff > 0)!\n";
            else
                std::cout << "  [save+reload MPO] cores preserved OK\n";
        }
    }

    std::cout << name << " done in " << elapsed_ms(t_type) << " ms)\n";
}

int main()
{
    std::string file_mps_f1 = "test/output/test_mpo_out_f1";
    std::string file_mps_f2 = "test/output/test_mpo_out_f2";

    const int n_points = 1000;

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
            file_mps_f1,
            nb_point_out);
        
        do_save_TCI<scalar_type>(
            a_1,
            b_1,
            c_0,
            nBit,
            function_2<complex_type>,
            n_iter,
            true,
            file_mps_f2,
            nb_point_out);
    }

    // now we have the following file:
    
    // some case are commented because already tested somewhere else
    // float128 is commented because too long

    test_mps<std::complex<double>, long long>("complex<double> long long", file_mps_f1, file_mps_f2, n_points);
    //test_mps<std::complex<double>, util::i128>("complex<double> i128", file_mps_f1, file_mps_f2, n_points);

    test_mps<std::complex<dd_128>, long long>("complex<dd_128>long long ", file_mps_f1, file_mps_f2, n_points);
    //test_mps<std::complex<dd_128>, util::i128>("complex<dd_128> i128", file_mps_f1, file_mps_f2, n_points);

    //test_mps<std::complex<float128>, long long>("complex<float128> long long", file_mps_f1, file_mps_f2, n_points);
    //test_mps<std::complex<float128>, util::i128>("complex<float128> i128", file_mps_f1, file_mps_f2, n_points);

    return 0;
}