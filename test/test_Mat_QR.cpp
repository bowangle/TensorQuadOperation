#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <complex>
#include <cassert>
#include <random>
#include <chrono>
#include "mat_decomp.h"
#include <boost/multiprecision/float128.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <iostream>
#include <limits>

using float128 = boost::multiprecision::float128;
using Cfloat128 = std::complex<float128>;
using Real = float128;
using Matrix =
    Eigen::Matrix<Cfloat128, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

// Fill a matrix with random complex values
static Matrix randomMatrix(int rows, int cols, unsigned seed)
{
    Matrix A(rows, cols);
    boost::random::mt19937_64 gen(seed);
    boost::random::uniform_real_distribution<float128> dist(-1, 1);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A(i, j) = Cfloat128(dist(gen), dist(gen));
    return A;
}

// Run both LEFT and RIGHT orthogonal tests on a given matrix
static void runTest(Matrix const& A, std::string const& label)
{
    int M = A.rows();
    int N = A.cols();
    MatQR<Cfloat128> qr;

    // TEST: LEFT ORTHOGONAL (A = Q R)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [Q, R] = qr(A, true);
        auto t1 = std::chrono::steady_clock::now();

        Matrix A_rec = Q * R;
        Real err = 0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
            {
                auto d = A(i, j) - A_rec(i, j);
                err += std::abs(d);
            }
        Real err2 = (A - A_rec).norm() / A.norm();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[" << label << "][LEFT] reconstruction error = " << err
                   << " relative error = " << err2
                   << " time = " << ms << " ms\n";
        assert(err2 < Real("1e-30"));
    }

    // TEST: RIGHT ORTHOGONAL (A = R Q)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto [R, Q] = qr(A, false);
        auto t1 = std::chrono::steady_clock::now();

        Matrix A_rec = R * Q;
        Real err = 0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
            {
                auto d = A(i, j) - A_rec(i, j);
                err += std::abs(d);
            }
        Real err2 = (A - A_rec).norm() / A.norm();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[" << label << "][RIGHT] reconstruction abs error = " << err
                   << " relative error = " << err2
                   << " time = " << ms << " ms\n";
        assert(err2 < Real("1e-30"));
    }
}

// ================= TEST =================
int main()
{
    std::cout << "std epsilon:   " << std::numeric_limits<float128>::epsilon() << "\n";
    std::cout << "Eigen epsilon: " << Eigen::NumTraits<float128>::epsilon() << "\n";

    constexpr int M = 20;
    constexpr int N = 5;

    // Deterministic matrix
    Matrix A(M, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            A(i, j) = Cfloat128(Real(i + 1), Real(j + 1));

    runTest(A, "DETERMINISTIC");

    // Randomly generated matrix (fixed seed for reproducibility)
    Matrix R1 = randomMatrix(M, N, 42);
    runTest(R1, "RANDOM (tall)");

    // Random wide matrix too, to exercise the rows < cols path directly
    Matrix R2 = randomMatrix(N, M, 1337);
    runTest(R2, "RANDOM (wide)");

    // Big random matrix, to stress accumulation/timing at scale
    constexpr int BIG_M = 200;
    constexpr int BIG_N = 80;
    Matrix R3 = randomMatrix(BIG_M, BIG_N, 7);
    runTest(R3, "RANDOM (big)");

    std::cout << "MatQR TEST PASSED\n";
    return 0;
}