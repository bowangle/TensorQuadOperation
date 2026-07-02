#include "type_double_double.h"
#include "type_float128_boost.h"
#include "mat_decomp.h"
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <string>
#include <cassert>
#include <chrono>
#include <limits>
#include <complex>
#include <vector>
#include <cmath>
#include <boost/multiprecision/float128.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>


using float128  = boost::multiprecision::float128;
using Cfloat128 = std::complex<float128>;

template<class Scalar>
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;   // col-major, like SVDDecomp::Mat

// ---------------------------------------------------------------------------
// Generic random matrix (real or complex, chosen at compile time)
// ---------------------------------------------------------------------------
template<class Scalar>
static MatX<Scalar> randomMatrix(int rows, int cols, unsigned seed)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    MatX<Scalar> A(rows, cols);
    boost::random::mt19937_64 gen(seed);

    if constexpr (std::is_same_v<Real, dd_real>)
    {
        // ============================
        // dd_real SAFE PATH (NO BOOST)
        // ============================
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
                if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
                {
                    A(i, j) = Scalar(
                        dd_real(dist(gen)),
                        dd_real(dist(gen))
                    );
                }
                else
                {
                    A(i, j) = Scalar(dd_real(dist(gen)));
                }
            }
    }
    else
    {
        // ============================
        // DEFAULT PATH (your original)
        // ============================
        boost::random::uniform_real_distribution<Real> dist(-1, 1);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
                if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
                    A(i, j) = Scalar(dist(gen), dist(gen));
                else
                    A(i, j) = Scalar(dist(gen));
            }
    }

    return A;
}

// Deterministic rank-2 matrix: A(i,j) = a_i + b_j  (real) or a_i + i*b_j (complex)
template<class Scalar>
static MatX<Scalar> deterministicMatrix(int rows, int cols)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    MatX<Scalar> A(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
                A(i, j) = Scalar(Real(i + 1), Real(j + 1));
            else
                A(i, j) = Scalar(Real(i + 1) + Real(j + 1));
        }
    return A;
}

// ---------------------------------------------------------------------------
// Reconstruction + orthogonality test for one matrix, both orthogonal modes.
// tol = 0 keeps every non-zero singular value, so a full-rank input must
// reconstruct exactly (to round-off).
// ---------------------------------------------------------------------------
template<class Scalar>
static void runReconstruction(MatX<Scalar> const& A, std::string const& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    const Real eps       = Eigen::NumTraits<Real>::epsilon();
    const Real threshold = eps * Real(1e6);   // generous, type-adaptive; tighten if you like

    MatSVDFixedTol<Scalar> svd(Real(0), 0);   // tol = 0  -> no truncation

    // ---- LEFT ORTHOGONAL: left() = U (orthonormal cols), right() = diag(s) V^H ----
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [L, R] = svd(A, true);
        auto t1 = std::chrono::steady_clock::now();

        MatX<Scalar> A_rec = L * R;
        Real err2 = (A - A_rec).norm() / A.norm();

        // U^H U == I  (L holds the orthonormal factor in this mode)
        MatX<Scalar> I = MatX<Scalar>::Identity(L.cols(), L.cols());
        Real orth = (L.adjoint() * L - I).norm();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[" << label << "][LEFT ] rel err = " << err2
                  << "  orthogonality = " << orth
                  << "  rank = " << L.cols()
                  << "  time = " << ms << " ms\n";
        assert(err2 < threshold);
        assert(orth < threshold);
    }

    // ---- RIGHT ORTHOGONAL: left() = U diag(s), right() = V^H (orthonormal rows) ----
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [L, R] = svd(A, false);
        auto t1 = std::chrono::steady_clock::now();

        MatX<Scalar> A_rec = L * R;
        Real err2 = (A - A_rec).norm() / A.norm();

        // V^H V == I  (R holds the orthonormal factor in this mode; rows orthonormal)
        MatX<Scalar> I = MatX<Scalar>::Identity(R.rows(), R.rows());
        Real orth = (R * R.adjoint() - I).norm();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[" << label << "][RIGHT] rel err = " << err2
                  << "  orthogonality = " << orth
                  << "  rank = " << R.rows()
                  << "  time = " << ms << " ms\n";
        assert(err2 < threshold);
        assert(orth < threshold);
    }
}

// ---------------------------------------------------------------------------
// Fixed-rank test: build a known rank-r matrix, check rank detection and the
// effect of a hard rankMax cap.
// ---------------------------------------------------------------------------
template<class Scalar>
static void runRankTest(int M, int N, std::string const& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    const int r = 3;

    // A = B * C  has rank <= r
    MatX<Scalar> B = randomMatrix<Scalar>(M, r, 101);
    MatX<Scalar> C = randomMatrix<Scalar>(r, N, 202);
    MatX<Scalar> A = B * C;

    // reltol = 1e-10 sits well below the O(1) signal singular values and well
    // above the numerical noise floor for BOTH double (~1e-16) and float128 (~1e-34),
    // so it must recover exactly rank r.
    {
        MatSVDFixedTol<Scalar> svd(Real(1e-10), 0);
        auto [L, R] = svd(A, true);
        Eigen::Index detected = L.cols();
        Real err2 = (A - L * R).norm() / A.norm();
        std::cout << "[" << label << "][RANK ] detected rank = " << detected
                  << " (expected " << r << ")  rel err = " << err2 << "\n";
        assert(detected == r);
        assert(err2 < Eigen::NumTraits<Real>::epsilon() * Real(1e6));
    }

    // Hard cap below the true rank: must return exactly rankMax columns,
    // and reconstruction is deliberately lossy (drops the 3rd component).
    {
        const int cap = 2;
        MatSVDFixedTol<Scalar> svd(Real(1e-10), cap);
        auto [L, R] = svd(A, true);
        Real err2 = (A - L * R).norm() / A.norm();
        std::cout << "[" << label << "][RANK ] capped rank = " << L.cols()
                  << " (rankMax = " << cap << ")  truncation rel err = " << err2 << "\n";
        assert(L.cols() == cap);
        assert(err2 > Real(0));   // strictly lossy: the dropped singular value shows up
    }
}

// ---------------------------------------------------------------------------
// Build a matrix with a PRESCRIBED singular-value spectrum:
//   A = U diag(sigma) V^H,  U (M x r) and V (N x r) with orthonormal columns.
// This lets us compare truncation error against the exact Eckart-Young value.
// ---------------------------------------------------------------------------
template<class Scalar>
static MatX<Scalar> matrixWithSpectrum(
    int M, int N,
    std::vector<typename Eigen::NumTraits<Scalar>::Real> const& sigma,
    unsigned seed)
{
    const int r = static_cast<int>(sigma.size());
    MatX<Scalar> U = Eigen::HouseholderQR<MatX<Scalar>>(randomMatrix<Scalar>(M, r, seed))
                         .householderQ() * MatX<Scalar>::Identity(M, r);
    MatX<Scalar> V = Eigen::HouseholderQR<MatX<Scalar>>(randomMatrix<Scalar>(N, r, seed + 1))
                         .householderQ() * MatX<Scalar>::Identity(N, r);
    MatX<Scalar> S = MatX<Scalar>::Zero(r, r);
    for (int i = 0; i < r; ++i) S(i, i) = Scalar(sigma[i]);
    return U * S * V.adjoint();
}

// Exact relative Frobenius error of truncating to the top-k singular values.
template<class Real>
static Real eckartYoung(std::vector<Real> const& sigma, int k)
{
    using std::sqrt;
    Real dropped = 0, total = 0;
    for (int i = 0; i < static_cast<int>(sigma.size()); ++i) {
        total += sigma[i] * sigma[i];
        if (i >= k) dropped += sigma[i] * sigma[i];
    }
    return sqrt(dropped / total);
}

// ---------------------------------------------------------------------------
// Sweep 1: increase reltol, watch detected rank fall and error rise.
//          Asserts the tolerance CONTRACT: relative error <= reltol.
// Sweep 2: decrease rankMax, watch error rise, and confirm it matches the
//          exact Eckart-Young truncation error for the known spectrum.
// ---------------------------------------------------------------------------
template<class Scalar>
static void sweepReltolAndRank(int M, int N, std::string const& type)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using std::abs;
    const Real eps = Eigen::NumTraits<Real>::epsilon();

    // Well-conditioned geometric spectrum sigma_i = 0.5^i (cond ~ 2^(r-1)),
    // so the SVD resolves every singular value accurately and measured error
    // can be checked tightly against theory.
    const int r = std::min(M, N);
    std::vector<Real> sigma(r);
    { Real v = Real(1); for (int i = 0; i < r; ++i) { sigma[i] = v; v *= Real(0.5); } }

    MatX<Scalar> A = matrixWithSpectrum<Scalar>(M, N, sigma, 555);

    std::cout << "\n----- [" << type << "] reltol sweep (M=" << M << " N=" << N
              << ", spectrum 0.5^i) -----\n";
    {
        const Real reltols[] = { Real(1e-3), Real(3e-3), Real(1e-2),
                                 Real(3e-2), Real(1e-1), Real(3e-1) };
        Real prevErr  = -1;
        Eigen::Index prevRank = r + 1;
        for (Real rt : reltols) {
            MatSVDFixedTol<Scalar> svd(rt, 0);
            auto [L, R] = svd(A, true);
            Eigen::Index rank = L.cols();
            Real err = (A - L * R).norm() / A.norm();
            std::cout << "  reltol = " << rt << "  -> rank = " << rank
                      << "  rel err = " << err
                      << "  (predicted " << eckartYoung(sigma, (int)rank) << ")\n";

            // Contract: the relative reconstruction error stays under reltol.
            assert(err <= rt + Real(1e3) * eps);
            // Monotonic: larger reltol never keeps more rank or lowers error.
            assert(rank <= prevRank);
            assert(err  >= prevErr - Real(1e3) * eps);
            prevErr = err; prevRank = rank;
        }
    }

    std::cout << "----- [" << type << "] rankMax sweep (reltol = 0) -----\n";
    {
        Real prevErr = -1;
        for (int k = r; k >= 1; --k) {
            MatSVDFixedTol<Scalar> svd(Real(0), k);   // only the hard cap acts
            auto [L, R] = svd(A, true);
            Real err      = (A - L * R).norm() / A.norm();
            Real expected = eckartYoung(sigma, k);
            std::cout << "  rankMax = " << k << "  -> rel err = " << err
                      << "  (exact " << expected << ")\n";

            assert(L.cols() == k);
            // Measured truncation error matches Eckart-Young for the known spectrum.
            assert(abs(err - expected) <= Real(1e-6) * expected + Real(1e3) * eps);
            // Monotonic: fewer components never lowers the error.
            assert(err >= prevErr - Real(1e3) * eps);
            prevErr = err;
        }
    }
}

// ---------------------------------------------------------------------------
// All tests for one scalar type.
// ---------------------------------------------------------------------------
template<class Scalar>
static void runAll(std::string const& type)
{
    std::cout << "\n===== SVD tests for " << type << " =====\n";

    constexpr int M = 20, N = 5;

    runReconstruction<Scalar>(deterministicMatrix<Scalar>(M, N), type + " DETERMINISTIC");
    runReconstruction<Scalar>(randomMatrix<Scalar>(M, N, 42),    type + " RANDOM tall");
    runReconstruction<Scalar>(randomMatrix<Scalar>(N, M, 1337),  type + " RANDOM wide");
    runReconstruction<Scalar>(randomMatrix<Scalar>(200, 80, 7),  type + " RANDOM big");

    runRankTest<Scalar>(40, 12, type);
    sweepReltolAndRank<Scalar>(30, 10, type);
}

// ================================= MAIN ====================================
int main()
{
    std::cout << "float128 std epsilon:   " << std::numeric_limits<float128>::epsilon() << "\n";
    std::cout << "float128 Eigen epsilon: " << Eigen::NumTraits<float128>::epsilon() << "\n";

    // Each type is instantiated independently. If a given scalar fails to
    // COMPILE (e.g. JacobiSVD for a Boost type on your Eigen/Boost combo),
    // comment out just that line to isolate it, and report the error.
    runAll<double>("double");
    runAll<std::complex<double>>("complex double");
    runAll<dd_128>("double double");
    runAll<Cdd_128>("complex double double");
    runAll<float128>("float128");
    runAll<Cfloat128>("Cfloat128");

    std::cout << "\nMatSVD TEST PASSED\n";
    return 0;
}