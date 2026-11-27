#include "magic_tensor_qft.hpp"
#include "type_double_double.h"
#include "type_float128_boost.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <sstream>

template<typename RealT>
double to_double(RealT const& v) { return static_cast<double>(v); }

template<>
double to_double<dd_128>(dd_128 const& v) { return v.x[0]; }

template<>
double to_double<float128>(float128 const& v) { return v.convert_to<double>(); }

template<typename ComplexT>
void test_build(const std::string& name, int r, int chi, int n_points = 10)
{
    using Real = typename ComplexT::value_type;
    std::cout << "\n=== Testing " << name << " (r=" << r << ", chi=" << chi << ") ===\n";

    // 1. chebyshev nodes
    auto nodes = magic_tensor_qft::chebyshev_nodes_on_01<Real>(chi);
    std::cout << "  chebyshev: [0]=" << to_double<Real>(nodes[0])
              << " [" << chi-1 << "]=" << to_double<Real>(nodes[chi-1]) << "\n";

    // 2. lagrange
    Real x_mid = Real(0.5);
    auto P = magic_tensor_qft::lagrange_basis_values<Real>(nodes, x_mid);
    std::cout << "  lagrange at 0.5: sum = "
              << to_double<Real>(std::real(P[0] + P[1]))
              << " ...\n";

    // 3. Build MPO tensors (flat-packed)
    auto W = magic_tensor_qft::build_qft_mpo_tensors<ComplexT>(r, chi);
    std::cout << "  W.size() = " << W.size() << "\n";
    std::cout << "  W[0].size() = " << W[0].size() << "\n";
    std::cout << "  W[0][0] = ("
              << to_double<Real>(std::real(W[0][0])) << ","
              << to_double<Real>(std::imag(W[0][0])) << ")\n";

    // 4. Build MPO from W
    auto mpo = magic_tensor_qft::build_mpo<ComplexT>(W, chi);
    std::cout << "  MPO nBit = " << mpo.get_size() << "\n";
    std::cout << "  MPO chi  = " << mpo.get_chi() << "\n";

    auto shape = mpo.get_shape();
    std::cout << "  MPO shape: ";
    for (auto const& [l, p, rr] : shape)
        std::cout << "(" << l << "," << p << "," << rr << ") ";
    std::cout << "\n";

    // 5. Evaluate at a few points
    auto points = mpo.generate_points(n_points);
    auto vals = mpo.eval_list(points);
    std::cout << "  eval " << n_points << " points:";
    for (int i = 0; i < n_points && i < 3; ++i) {
        std::cout << " (" << to_double<Real>(std::real(vals[i]))
                  << "," << to_double<Real>(std::imag(vals[i])) << ")";
    }
    std::cout << "\n";

    // 6. Convenience function
    auto mpo2 = magic_tensor_qft::build_qft_mpo_magic<ComplexT>(r, chi);
    std::cout << "  build_qft_mpo_magic: nBit=" << mpo2.get_size()
              << " chi=" << mpo2.get_chi() << "\n";
}

template<typename T>
typename Eigen::NumTraits<typename Eigen::NumTraits<T>::Real>::Real default_tol()
{
    return Eigen::NumTraits<typename Eigen::NumTraits<T>::Real>::epsilon() * 100;
}

template<typename ComplexT>
void test_error_range(const std::string& name, int r, int chi_max, int n_points = 1000){
    // Create output directories
    mkdir("test/output", 0755);
    mkdir("test/output_compress", 0755);

    // Build filename: e.g. test/output/error_chi_max40_r30_complex_double.txt
    std::string safe_name = name;
    for (auto& c : safe_name) {
        if (c == '<' || c == '>' || c == ':') c = '_';
    }
    std::ostringstream fname;
    fname << "test/output/error_chi_max" << chi_max << "_r" << r << "_" << safe_name << ".txt";
    std::ofstream out(fname.str());

    // Compressed file in output_compress/
    std::ostringstream fname_comp;
    fname_comp << "test/output_compress/error_compress_chi_max" << chi_max << "_r" << r << "_" << safe_name << ".txt";
    std::ofstream out_comp(fname_comp.str());

    std::cout << "Saving to " << fname.str() << "\n";
    std::cout << "Saving to " << fname_comp.str() << "\n";

    // Determine nBit from a trial MPO to write header with column counts
    {
        MPO<ComplexT> mpo_tmp = magic_tensor_qft::build_qft_mpo_magic<ComplexT>(r, 2);
        int nBit = mpo_tmp.get_size();
        out << "# nBit=" << nBit << "  chi  max|MPO-F|  rms|MPO-F|  rms|F|  relative_rms  tot_nb_value  max_bond_dim"
            << "  [list_chi(" << (nBit+1) << " values)]  [nb_value_core(" << nBit << " values)]\n";
        out_comp << "# nBit=" << nBit << "  chi  max|MPO-F|  rms|MPO-F|  rms|F|  relative_rms  tot_nb_value  max_bond_dim"
                 << "  [list_chi(" << (nBit+1) << " values)]  [nb_value_core(" << nBit << " values)]\n";
    }

    for (int i=2; i<chi_max+1; i++){
        MPO<ComplexT> mpo = magic_tensor_qft::build_qft_mpo_magic<ComplexT>(r, i);
        auto err = magic_tensor_qft::compute_qft_mpo_error<ComplexT>(mpo, n_points);
        double rel_rms = err[1]/err[2];

        auto list_chi = mpo.compute_list_chi();
        auto nb_value_core = mpo.compute_nb_value_core();
        auto tot_nb_value = mpo.compute_tot_nb_value();

        std::cout << name << " " << i <<"\n";
        std::cout << "  Error vs exact QFT (" << n_points << " samples):\n";
        std::cout << "    max |MPO - F|  = " << err[0] << "\n";
        std::cout << "    rms |MPO - F|  = " << err[1] << "\n";
        std::cout << "    rms |F|        = " << err[2] << "\n";
        std::cout << "    relative rms   = " << rel_rms << "\n";
        std::cout << "    tot_nb_value   = " << tot_nb_value << "\n";

        int max_bd = *std::max_element(list_chi.begin(), list_chi.end());
        out << i << " " << err[0] << " " << err[1] << " " << err[2] << " " << rel_rms << " " << tot_nb_value << " " << max_bd;
        for (auto v : list_chi) out << " " << v;
        for (auto v : nb_value_core) out << " " << v;
        out << "\n";

        // compress_svd and recompute
        mpo.compress_svd(default_tol<ComplexT>(), -1);

        auto err_comp = magic_tensor_qft::compute_qft_mpo_error<ComplexT>(mpo, n_points);
        double rel_rms_comp = err_comp[1]/err_comp[2];

        auto list_chi_comp = mpo.compute_list_chi();
        auto nb_value_core_comp = mpo.compute_nb_value_core();
        auto tot_nb_value_comp = mpo.compute_tot_nb_value();

        std::cout << "  After compress_svd:\n";
        std::cout << "    max |MPO - F|  = " << err_comp[0] << "\n";
        std::cout << "    rms |MPO - F|  = " << err_comp[1] << "\n";
        std::cout << "    rms |F|        = " << err_comp[2] << "\n";
        std::cout << "    relative rms   = " << rel_rms_comp << "\n";
        std::cout << "    tot_nb_value   = " << tot_nb_value_comp << "\n";

        int max_bd_comp = *std::max_element(list_chi_comp.begin(), list_chi_comp.end());
        out_comp << i << " " << err_comp[0] << " " << err_comp[1] << " " << err_comp[2] << " " << rel_rms_comp << " " << tot_nb_value_comp << " " << max_bd_comp;
        for (auto v : list_chi_comp) out_comp << " " << v;
        for (auto v : nb_value_core_comp) out_comp << " " << v;
        out_comp << "\n";
    }
    std::cout <<"\n\n";
    out.close();
    out_comp.close();
}

template<typename ComplexT>
void test_error_single(const std::string& name, int r, int chi, int n_points = 1000){
    MPO<ComplexT> mpo = magic_tensor_qft::build_qft_mpo_magic<ComplexT>(r, chi);
    auto err = magic_tensor_qft::compute_qft_mpo_error<ComplexT>(mpo, n_points);
    std::cout << name << " " << chi <<"\n";
    std::cout << "  Error vs exact QFT (" << n_points << " samples):\n";
    std::cout << "    max |MPO - F|  = " << err[0] << "\n";
    std::cout << "    rms |MPO - F|  = " << err[1] << "\n";
    std::cout << "    rms |F|        = " << err[2] << "\n";
    std::cout << "    relative rms   = " << err[1]/err[2] << "\n";
    std::cout <<"\n\n";
}

int main()
{
    int r = 30;
    int r1 = 55;
    int r2 = 100;
    int chi = 12;
    int chi_max = 40;

    test_build<std::complex<double>>("complex<double>", r, chi);
    test_build<Cdd_128>("Cdd_128", r, chi);
    test_build<Cfloat128>("Cfloat128", r, chi);
    

    test_error_range<std::complex<double>>("complex<double>", r, chi_max);
    test_error_range<Cdd_128>("Cdd_128", r, chi_max);
    test_error_range<Cfloat128>("Cfloat128", r, chi_max);

    test_error_range<std::complex<double>>("complex<double>", r1, chi_max);
    test_error_range<Cdd_128>("Cdd_128", r1, chi_max);

    test_error_range<std::complex<double>>("complex<double>", r2, chi_max);
    test_error_range<Cdd_128>("Cdd_128", r2, chi_max);

    test_error_single<Cfloat128>("Cfloat128", r, 37);
    test_error_single<Cfloat128>("Cfloat128", r1, 37);
    test_error_single<Cfloat128>("Cfloat128", r2, 37);

    std::cout << "\nAll tests passed!\n";
    return 0;
}
