#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

#include "mpo_base.h"
#include "type_double_double.h"
#include "type_float128_boost.h"
#include <boost/math/constants/constants.hpp>

namespace magic_tensor_qft {

// ============================================================
// Math helpers — ADL-dispatched so they work for double,
// dd_128 and float128 alike.
// ============================================================
template<typename Real>
Real real_cos(Real x) { using std::cos; return cos(x); }

template<typename Real>
Real real_sin(Real x) { using std::sin; return sin(x); }

template<typename Real>
Real real_sqrt(Real x) { using std::sqrt; return sqrt(x); }

// Primary template — explodes with a clear message for unknown types
template<typename Real>
Real pi() {
    static_assert(sizeof(Real) == 0,
        "pi() not specialised for this type — add a specialisation or include the right type header");
    return Real(0);
}

template<> inline double     pi<double>()     { return M_PI; }
template<> inline dd_128     pi<dd_128>()     { return dd_128::_pi; }
template<> inline float128   pi<float128>()   { return boost::math::constants::pi<float128>(); }

// ============================================================
// chebyshev_nodes_on_01
//   c_a = [1 - cos(pi a / (chi-1))] / 2,   a = 0..chi-1
//   For chi <= 1 returns a single node at 0.5.
// ============================================================
template<typename Real>
std::vector<Real> chebyshev_nodes_on_01(int chi) {
    std::vector<Real> nodes(chi);
    if (chi <= 1) {
        if (chi == 1) nodes[0] = Real(0.5);
        return nodes;
    }
    Real pi_val = pi<Real>();
    Real denom = Real(chi - 1);
    for (int a = 0; a < chi; ++a) {
        nodes[a] = Real(0.5) * (Real(1) - real_cos(pi_val * Real(a) / denom));
    }
    return nodes;
}

// ============================================================
// lagrange_basis_values
//   Evaluate all Lagrange basis polynomials P_beta(x) for
//   given nodes.  P_beta(x) = Π_{γ≠β} (x - c_γ)/(c_β - c_γ)
// ============================================================
template<typename Real>
std::vector<std::complex<Real>> lagrange_basis_values(
    const std::vector<Real>& nodes, Real x)
{
    using Complex = std::complex<Real>;
    int chi = static_cast<int>(nodes.size());
    std::vector<Complex> P(chi, Complex(1, 0));
    for (int beta = 0; beta < chi; ++beta) {
        Complex denom(1, 0);
        Complex numer(1, 0);
        Real c_beta = nodes[beta];
        for (int gamma = 0; gamma < chi; ++gamma) {
            if (gamma == beta) continue;
            denom *= Complex(c_beta - nodes[gamma]);
            numer *= Complex(x - nodes[gamma]);
        }
        P[beta] = numer / denom;
    }
    return P;
}

// ============================================================
// build_qft_mpo_tensors
//   Construct local MPO tensors W[n] with shape (2,2,chi,chi)
//   and return them as a vector of 4D arrays (flat-packed).
//
//   M^n_{αβ} = exp(-iπ (t_n ω_n + c_α t_n)) * P_β((c_α+ω_n)/2)
// ============================================================
// Returns flat-packed W tensors.  For each site n:
//   M^n_{αβ} = exp(-iπ (t_n ω_n + c_α t_n)) · P_β((c_α + ω_n)/2)
// where P_β are the Lagrange basis on Chebyshev nodes.
template<typename ComplexT>
std::vector<std::vector<ComplexT>>
build_qft_mpo_tensors(int r, int chi)
{
    using Real = typename ComplexT::value_type;
    auto nodes = chebyshev_nodes_on_01<Real>(chi);
    Real pi_val = pi<Real>();

    // Each W[n] stored as a flat vector of length 2*2*chi*chi,
    // with index = ((tbit*2 + wbit)*chi + alpha)*chi + beta
    std::vector<std::vector<ComplexT>> W(r);

    for (int n = 0; n < r; ++n) {
        auto& Wn = W[n];
        Wn.resize(2 * 2 * chi * chi);

        for (int tbit = 0; tbit < 2; ++tbit) {
            for (int wbit = 0; wbit < 2; ++wbit) {
                Real wbit_r = Real(wbit);
                Real tbit_r = Real(tbit);

                for (int alpha = 0; alpha < chi; ++alpha) {
                    Real c_alpha = nodes[alpha];
                    Real x = Real(0.5) * (c_alpha + wbit_r);
                    auto P = lagrange_basis_values<Real>(nodes, x);
                    Real theta = pi_val * (tbit_r * wbit_r + c_alpha * tbit_r);
                    ComplexT phase(real_cos(theta), -real_sin(theta));

                    for (int beta = 0; beta < chi; ++beta) {
                        std::size_t idx =
                            ((static_cast<std::size_t>(tbit) * 2 + static_cast<std::size_t>(wbit)) * chi
                             + static_cast<std::size_t>(alpha)) * chi
                            + static_cast<std::size_t>(beta);
                        Wn[idx] = phase * P[beta];
                    }
                }
            }
        }
    }
    return W;
}

// ============================================================
// build_mpo
//   Assemble the MPO cores from the pre-computed W tensors
//   into an MPO object.
//
//   Core layout (before boundary contraction):
//       core[alpha, phys, beta]  with phys = wbit*2 + tbit
//   After left/right boundary:
//       site 0:     (1, 4, chi)
//       site r-1:   (chi, 4, 1)
//       otherwise:  (chi, 4, chi)
//
//   Overall normalisation 1 / 2^{r/2} is folded into core[0].
// ============================================================
template<typename ComplexT>
MPO<ComplexT> build_mpo(
    const std::vector<std::vector<ComplexT>>& W,
    int chi)
{
    using Real = typename ComplexT::value_type;
    int r = static_cast<int>(W.size());

    using VecTT = std::vector<Tensor3D<ComplexT>>;
    VecTT cores;
    cores.reserve(r);

    for (int n = 0; n < r; ++n) {
        const auto& Wn = W[n];

        Eigen::Index left_dim  = (n == 0)     ? 1 : chi;
        Eigen::Index right_dim = (n == r - 1) ? 1 : chi;

        Tensor3D<ComplexT> core(left_dim, 4, right_dim);

        for (int tbit = 0; tbit < 2; ++tbit) {
            for (int wbit = 0; wbit < 2; ++wbit) {
                int phys = wbit * 2 + tbit;

                for (int alpha = 0; alpha < chi; ++alpha) {
                    for (int beta = 0; beta < chi; ++beta) {
                        std::size_t idx =
                            ((static_cast<std::size_t>(tbit) * 2 + static_cast<std::size_t>(wbit)) * chi
                             + static_cast<std::size_t>(alpha)) * chi
                            + static_cast<std::size_t>(beta);
                        ComplexT val = Wn[idx];

                        if (n == 0) {
                            if (alpha == 0)
                                core(0, phys, beta) = val;
                        } else if (n == r - 1) {
                            core(alpha, phys, 0) += val;
                        } else {
                            core(alpha, phys, beta) = val;
                        }
                    }
                }
            }
        }

        cores.push_back(std::move(core));
    }

    // Normalisation:  divide core[0] by 2^{r/2} = (√2)^r
    Real sqrt2 = real_sqrt(Real(2));
    Real norm = Real(1);
    for (int i = 0; i < r; ++i)
        norm *= sqrt2;

    for (auto& x : cores[0].data)
        x /= norm;

    return MPO<ComplexT>(std::move(cores));
}

// ============================================================
// build_qft_mpo_magic
//   Convenience:  build W tensors, then assemble the MPO.
// ============================================================
template<typename ComplexT>
MPO<ComplexT> build_qft_mpo_magic(int r, int chi)
{
    auto W = build_qft_mpo_tensors<ComplexT>(r, chi);
    return build_mpo<ComplexT>(W, chi);
}

// ============================================================
// Helpers: real_pow (integer-exponent power), to_double
// ============================================================
inline double real_pow(double base, int exp) {
    double r = 1.0;
    for (int i = 0; i < exp; ++i) r *= base;
    for (int i = 0; i > exp; --i) r /= base;
    return r;
}
inline dd_128 real_pow(dd_128 base, int exp) {
    dd_128 r(1.0);
    for (int i = 0; i < exp; ++i) r *= base;
    for (int i = 0; i > exp; --i) r /= base;
    return r;
}
inline float128 real_pow(float128 base, int exp) {
    float128 r(1.0);
    for (int i = 0; i < exp; ++i) r *= base;
    for (int i = 0; i > exp; --i) r /= base;
    return r;
}

inline double to_double(double v)           { return v; }
inline double to_double(dd_128 const& v)    { return v.x[0]; }
inline double to_double(float128 const& v)  { return v.convert_to<double>(); }

template<typename T>
double to_double(std::complex<T> const& v) {
    return std::abs(to_double(std::real(v)) + to_double(std::imag(v)));
}

// ============================================================
// real_from_i128 — convert __int128 directly to Real, avoiding
// the 64-bit truncation of long long.  Needed for r > 63.
// ============================================================
template<typename Real>
Real real_from_i128(util::i128 x);

template<> inline double   real_from_i128<double>(util::i128 x)   { return static_cast<double>(x); }
template<> inline dd_128   real_from_i128<dd_128>(util::i128 x)   { return dd_128(x); }
template<> inline float128 real_from_i128<float128>(util::i128 x) { return float128(x); }

// ============================================================
// exact_qft_value
//   Compute the exact QFT matrix element for a given pair of
//   bit strings t_bits[0..r-1], w_bits[0..r-1].
//
//   F = 1/√(2^r) exp(-i 2π ω t / 2^r)
//   with  t = Σ_{j=1..r} 2^{j-1} t_j    (t_1 = LSB)
//         ω = Σ_{j=1..r} 2^{r-j} ω_j    (ω_1 = MSB)
//
//   To avoid large arguments in sin/cos (which the dd_real
//   library cannot reduce modulo π/2), the integer product
//   ω·t is computed exactly and reduced modulo 2^r *before*
//   converting to floating point.
// ============================================================
template<typename ComplexT>
ComplexT exact_qft_value(const std::vector<int>& t_bits,
                         const std::vector<int>& w_bits)
{
    using Real = typename ComplexT::value_type;
    int r = static_cast<int>(t_bits.size());
    Real two = Real(2);

    // Build integers t and ω as util::i128 (at r=30, each < 2^30)
    util::i128 t_val = 0;
    {
        util::i128 pow2 = 1;
        for (int j = 0; j < r; ++j) {
            t_val += pow2 * static_cast<util::i128>(t_bits[j]);
            pow2 *= 2;
        }
    }

    util::i128 w_val = 0;
    {
        util::i128 pow2 = util::i128(1) << (r - 1);
        for (int j = 0; j < r; ++j) {
            w_val += pow2 * static_cast<util::i128>(w_bits[j]);
            pow2 /= 2;
        }
    }

    // Reduce modulo 2^r:  exp(-i·2π·k) = 1 for integer k
    // Use unsigned i128 so multiplication wraps (mod 2^128),
    // giving correct low bits for r ≤ 100.  Signed overflow
    // would be UB for r > 63.
    using u128 = unsigned __int128;
    u128       two_pow_r_u = u128(1) << r;
    u128       reduced_u   = (u128(w_val) * u128(t_val)) & (two_pow_r_u - 1);
    util::i128 reduced     = static_cast<util::i128>(reduced_u);

    Real two_pow_r_r = real_pow(two, r);
    Real reduced_r   = real_from_i128<Real>(reduced);
    Real phase = -Real(2) * pi<Real>() * reduced_r / two_pow_r_r;
    Real norm  = Real(1) / real_sqrt(two_pow_r_r);

    return ComplexT(norm * real_cos(phase), norm * real_sin(phase));
}

// ============================================================
// compute_qft_mpo_error
//   Sample n_samples random (t, ω) bit strings, compare the
//   magic MPO against the exact QFT, and return {max_abs_err,
//   l2_err, l2_exact_norm}.
// ============================================================
template<typename ComplexT>
std::vector<double> compute_qft_mpo_error(
    MPO<ComplexT>& mpo, int n_samples = 1000)
{
    int r = mpo.get_size();

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> bit_dist(0, 1);

    double max_abs_err = 0.0;
    double sum_abs2_err = 0.0;
    double sum_abs2_exact = 0.0;

    for (int s = 0; s < n_samples; ++s) {
        std::vector<int> t_bits(r), w_bits(r);
        std::vector<int> phys_ids(r);
        for (int k = 0; k < r; ++k) {
            t_bits[k] = bit_dist(rng);
            w_bits[k] = bit_dist(rng);
            phys_ids[k] = w_bits[k] * 2 + t_bits[k];
        }

        ComplexT exact = exact_qft_value<ComplexT>(w_bits, t_bits);
        ComplexT approx = mpo.eval(phys_ids);
        ComplexT diff = approx - exact;

        double abs_err = std::abs(std::complex<double>(
            to_double(std::real(diff)), to_double(std::imag(diff))));
        double abs_exact = std::abs(std::complex<double>(
            to_double(std::real(exact)), to_double(std::imag(exact))));

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        sum_abs2_err   += abs_err * abs_err;
        sum_abs2_exact += abs_exact * abs_exact;
    }

    double l2_err   = std::sqrt(sum_abs2_err / n_samples);
    double l2_exact = std::sqrt(sum_abs2_exact / n_samples);
    return {max_abs_err, l2_err, l2_exact};
}

// ============================================================
// qft_type_label
//   Maps ComplexT to the label string used in file names.
// ============================================================
template<typename> constexpr bool always_false = false;

template<typename ComplexT>
const char* qft_type_label() {
    if constexpr (std::is_same_v<ComplexT, std::complex<double>>)  return "Cdouble";
    else if constexpr (std::is_same_v<ComplexT, Cdd_128>)          return "Cdd128";
    else if constexpr (std::is_same_v<ComplexT, Cfloat128>)        return "Cfloat128";
    else static_assert(always_false<ComplexT>, "qft_type_label: unsupported ComplexT type — see template argument above");
}

// ============================================================
// load_compressed_qft_mpo
//   Load a precomputed compressed QFT MPO from disk.
//   Files are expected at path/compress_QFT_nBit={nBit}_{type_label}.tt.
//   If the requested file is absent, build and compress it using the same
//   settings as precompute_qft, save it in path, and then load it normally.
//   The type_label is deduced from ComplexT via qft_type_label.
// ============================================================
template<typename ComplexT>
MPO<ComplexT> load_compressed_qft_mpo(
    int nBit,
    const std::string& path = "mpo_data/data/",
    typename Eigen::NumTraits<ComplexT>::Real reltol_ = -1,
    int max_bond_dim_ = 0
)
{
    using Real = typename Eigen::NumTraits<ComplexT>::Real;

    const std::filesystem::path directory(path);
    const std::string basename =
        "compress_QFT_nBit=" + std::to_string(nBit) + "_"
        + qft_type_label<ComplexT>();
    const std::filesystem::path file_prefix = directory / basename;
    const std::filesystem::path filename = file_prefix.string() + ".tt";

    if (!std::filesystem::exists(filename)) {
        std::cerr
            << "Warning: compressed QFT MPO '" << filename.string()
            << "' was not found. Building, compressing, and saving it now; "
               "this may take some time.\n";

        if (!directory.empty())
            std::filesystem::create_directories(directory);

        constexpr int construction_chi = 35;
        const Real compression_reltol =
            Eigen::NumTraits<Real>::epsilon() * Real(100);

        auto qft_mpo = build_qft_mpo_magic<ComplexT>(
            nBit, construction_chi);
        qft_mpo.compress_svd(compression_reltol, -1);
        qft_mpo.save(file_prefix.string());
    }

    // w_=-1 skips the initial QR because the saved MPO was already compressed.
    return MPO<ComplexT>(filename.string(),
                         /*max_bond_dim_=*/max_bond_dim_,
                         /*reltol_=*/reltol_,
                         /*w_=*/-1);
}

} // namespace magic_tensor_qft
