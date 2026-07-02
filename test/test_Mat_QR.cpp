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

template<class Scalar>
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;   // col-major

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
        // DEFAULT PATH
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

// Deterministic matrix: A(i,j) = (i+1) + (j+1) (real) or (i+1) + i*(j+1) (complex)
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
// LEFT : A = Q R, Q has orthonormal columns  -> check Q^H Q == I
// RIGHT: A = R Q, Q has orthonormal rows     -> check Q Q^H == I
// ---------------------------------------------------------------------------
template<class Scalar>
static void runReconstruction(MatX<Scalar> const& A, std::string const& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    const Real eps       = Eigen::NumTraits<Real>::epsilon();
    const Real threshold = eps * Real(1e6);   // generous, type-adaptive

    MatQR<Scalar> qr;

    // ---- LEFT ORTHOGONAL: A = Q R, Q orthonormal columns ----
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [Q, R] = qr(A, true);
        auto t1 = std::chrono::steady_clock::now();

        MatX<Scalar> A_rec = Q * R;
        Real err2 = (A - A_rec).norm() / A.norm();

        // Q^H Q == I  (Q holds the orthonormal factor in this mode)
        MatX<Scalar> I = MatX<Scalar>::Identity(Q.cols(), Q.cols());
        Real orth = (Q.adjoint() * Q - I).norm();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[" << label << "][LEFT ] rel err = " << err2
                  << "  orthogonality = " << orth
                  << "  Qcols = " << Q.cols()
                  << "  time = " << ms << " ms\n";
        assert(err2 < threshold);
        assert(orth < threshold);
    }

    // ---- RIGHT ORTHOGONAL: A = R Q, Q orthonormal rows ----
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [R, Q] = qr(A, false);
        auto t1 = std::chrono::steady_clock::now();

        MatX<Scalar> A_rec = R * Q;
        Real err2 = (A - A_rec).norm() / A.norm();

        // Q Q^H == I  (Q holds the orthonormal factor in this mode; rows orthonormal)
        MatX<Scalar> I = MatX<Scalar>::Identity(Q.rows(), Q.rows());
        Real orth = (Q * Q.adjoint() - I).norm();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[" << label << "][RIGHT] rel err = " << err2
                  << "  orthogonality = " << orth
                  << "  Qrows = " << Q.rows()
                  << "  time = " << ms << " ms\n";
        assert(err2 < threshold);
        assert(orth < threshold);
    }
}

// ---------------------------------------------------------------------------
// All tests for one scalar type.
// ---------------------------------------------------------------------------
template<class Scalar>
static void runAll(std::string const& type)
{
    std::cout << "\n===== QR tests for " << type << " =====\n";

    constexpr int M = 20, N = 5;

    runReconstruction<Scalar>(deterministicMatrix<Scalar>(M, N), type + " DETERMINISTIC");
    runReconstruction<Scalar>(randomMatrix<Scalar>(M, N, 42),    type + " RANDOM tall");
    runReconstruction<Scalar>(randomMatrix<Scalar>(N, M, 1337),  type + " RANDOM wide");
    runReconstruction<Scalar>(randomMatrix<Scalar>(200, 80, 7),  type + " RANDOM big");
}

// ================================= MAIN ====================================
int main()
{
    std::cout << "float128 std epsilon:   " << std::numeric_limits<float128>::epsilon() << "\n";
    std::cout << "float128 Eigen epsilon: " << Eigen::NumTraits<float128>::epsilon() << "\n";

    // Each type is instantiated independently. If a given scalar fails to
    // COMPILE on your Eigen/Boost combo, comment out just that line to
    // isolate it, and report the error.
    runAll<double>("double");
    runAll<std::complex<double>>("complex double");
    runAll<dd_128>("double double");
    runAll<Cdd_128>("complex double double");
    runAll<float128>("float128");
    runAll<Cfloat128>("Cfloat128");

    std::cout << "\nMatQR TEST PASSED\n";
    return 0;
}