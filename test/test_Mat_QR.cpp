#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <complex>
#include <cassert>
#include <random>
#include "mat_decomp.h"
#include <boost/multiprecision/float128.hpp>

using float128 = boost::multiprecision::float128;
using Cfloat128 = std::complex<float128>;
using Real = float128;
using Matrix =
    Eigen::Matrix<Cfloat128, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

// Fill a matrix with random complex values
static Matrix randomMatrix(int rows, int cols, unsigned seed)
{
    Matrix A(rows, cols);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A(i, j) = Cfloat128(Real(dist(gen)), Real(dist(gen)));

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
        auto [Q, R] = qr(A, true);
        Matrix A_rec = Q * R;
        Real err = 0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
            {
                auto d = A(i, j) - A_rec(i, j);
                err += std::norm(d);
            }
        std::cout << "[" << label << "][LEFT] reconstruction error = " << err << "\n";
        assert(err < Real("1e-30"));
    }

    // TEST: RIGHT ORTHOGONAL (A = R Q)
    {
        auto [R, Q] = qr(A, false);
        Matrix A_rec = R * Q;
        Real err = 0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
            {
                auto d = A(i, j) - A_rec(i, j);
                err += std::norm(d);
            }
        std::cout << "[" << label << "][RIGHT] reconstruction error = " << err << "\n";
        assert(err < Real("1e-30"));
    }
}

// ================= TEST =================
int main()
{
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

    std::cout << "MatQR TEST PASSED\n";
    return 0;
}