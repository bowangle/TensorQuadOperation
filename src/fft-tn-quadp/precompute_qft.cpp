// Here we will precompute the MPO of the QFT for multiple number of bit(10->100) and type of variable
// From previous calculation we saw that for our type, calculation at chi=35 is enough. 
// Also, the compression only work if we did the construction with a bit more chi thant necessary.

#include "parser_precompute_mpo_qft.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <type_double_double.h>
#include <type_float128_boost.h>
#include <mpo_base.h>
#include "magic_tensor_qft.hpp"
 
template <typename T>
T pow10(int n) {          // n >= 0
    T r{1};
    while (n--) r *= T(10);
    return r;
}

template<typename ComplexT>
void save_info_on_calc(
    const std::string& filename,
    const ComplexT& error_before,
    const ComplexT& error_after,
    int chi_before,
    int chi_after,
    double t_construction,
    double t_all)
{
    std::ofstream file(filename);
    if (!file)
        throw std::runtime_error("Cannot open file");
    file << "# relative_rms_error_before  chi_before  relative_rms_error_after  chi_after  t_construction  t_all\n";
    file << error_before << "  " << chi_before << "  " << error_after << "  " << chi_after
         << "  " << t_construction << "  " << t_all << "\n";
}

template<typename ComplexT>
void compute_and_save(BenchArgs args_, const std::string& type_label, const std::string& path){
    using RealScalar = typename Eigen::NumTraits<ComplexT>::Real;
    RealScalar reltol = Eigen::NumTraits<RealScalar>::epsilon() * pow10<int>(args_.eps_factor);

    for (int i = args_.min_a; i < args_.max_a+1; i++){
        std::string file_prefix = path + (args_.compress ? "compress_" : "") 
                               + std::string("QFT_nBit=") + std::to_string(i) + "_" + type_label;

        auto t0 = std::chrono::high_resolution_clock::now();
        MPO<ComplexT> qft_mpo = magic_tensor_qft::build_qft_mpo_magic<ComplexT>(i, args_.chi);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_construction = std::chrono::duration<double>(t1 - t0).count();
        double t_all = t_construction;

        // Compute error & chi before compression
        int n_points = 100;
        auto err_before = magic_tensor_qft::compute_qft_mpo_error<ComplexT>(qft_mpo, n_points);
        auto list_chi_before = qft_mpo.compute_list_chi();
        int chi_before = *std::max_element(list_chi_before.begin(), list_chi_before.end());
        ComplexT error_before(err_before[1] / err_before[2], 0);

        ComplexT error_after = error_before;
        int chi_after = chi_before;

        if (args_.compress){
            auto t2 = std::chrono::high_resolution_clock::now();
            qft_mpo.compress_svd(reltol, -1);
            auto t3 = std::chrono::high_resolution_clock::now();
            t_all += std::chrono::duration<double>(t3 - t2).count();

            auto err_after = magic_tensor_qft::compute_qft_mpo_error<ComplexT>(qft_mpo, n_points);
            auto list_chi_after = qft_mpo.compute_list_chi();
            chi_after = *std::max_element(list_chi_after.begin(), list_chi_after.end());
            error_after = ComplexT(err_after[1] / err_after[2], 0);
        }

        qft_mpo.save(file_prefix);

        std::string file_info = file_prefix + "_info.txt";
        save_info_on_calc(file_info, error_before, error_after, chi_before, chi_after, t_construction, t_all);
    }
}

int main(int argc, char* argv[]) {
    BenchArgs args;
    try {
        args = parse_bench_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_bench_usage(argv[0]);
        return 1;
    }
 
    std::cout << std::boolalpha
              << "double   = " << args.use_double   << "\n"
              << "float128 = " << args.use_float128 << "\n"
              << "dd128    = " << args.use_dd128    << "\n"
              << "compress = " << args.compress     << "\n"
              << "min_a    = " << args.min_a        << "\n"
              << "max_a    = " << args.max_a        << "\n"
              << "chi      = " << args.chi          << "\n"
              << "eps_fac  = " << args.eps_factor   << "\n";
 
    if (!args.any_type()) {
        std::cout << "(note: no types enabled — nothing to benchmark)\n";
        return 0;
    }

    const std::string path = "mpo_data/data/";

    if (args.use_double){
        compute_and_save<std::complex<double>>(args, "Cdouble", path);
    }
    if (args.use_dd128)
    {
        compute_and_save<Cdd_128>(args, "Cdd128", path);
    }
    if (args.use_float128)
    {
        compute_and_save<Cfloat128>(args, "Cfloat128", path);
    }
    return 0;
}
 
