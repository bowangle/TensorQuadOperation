#pragma once
#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <utility>
#include <boost/multiprecision/float128.hpp>
#include <complex>

template<class T>
struct MatQR {
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    std::array<Mat, 2> operator()(Mat const& A, bool leftOrthogonal) {
        return leftOrthogonal ? mat_qr(A) : mat_qr_t(A);
    }

private:
    static std::array<Mat, 2> mat_qr(Mat const& A)
    {
        Eigen::ColPivHouseholderQR<Mat> qr(A);

        Eigen::Index rows = A.rows();
        Eigen::Index cols = A.cols();
        Eigen::Index k = std::min(rows, cols);

        Mat Q = qr.householderQ() * Mat::Identity(rows, k);
        Mat R = qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();

        // Undo column pivoting so that Q * R == A (not A * P)
        R.applyOnTheRight(qr.colsPermutation().inverse());

        return {std::move(Q), std::move(R)};
    }

    static std::array<Mat, 2> mat_qr_t(Mat const& A)
    {
        auto [Q, R] = mat_qr(Mat(A.transpose()));
        return {Mat(R.transpose()), Mat(Q.transpose())};
    }
};

template<class T>
struct SVDDecomp {
    using value_type = T;
    using RealScalar = typename Eigen::NumTraits<T>::Real;              // float128 for Cfloat128
    using Mat        = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec        = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    Mat U, V;
    Vec s;
    bool leftOrthogonal = true;

    SVDDecomp(Mat const& M, bool leftOrthogonal_ = true,
              RealScalar reltol = RealScalar(1e-12), int rankMax = 0)
        : leftOrthogonal(leftOrthogonal_)
    {
        // JacobiSVD works with boost::multiprecision scalars; BDCSVD does not reliably.
        //Eigen::BDCSVD<Mat, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(M);
        Eigen::JacobiSVD<Mat> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        U = svd.matrixU();
        V = svd.matrixV();          // M = U * diag(s) * V^H
        s = svd.singularValues();

        Eigen::Index n = findnValues(s, reltol);
        if (rankMax > 0 && rankMax < n) n = rankMax;

        s.conservativeResize(n);            // conservativeResize preserves the leading block
        U.conservativeResize(U.rows(), n);
        V.conservativeResize(V.rows(), n);
    }

    Mat left()  const { if (leftOrthogonal) return U; return U * s.template cast<T>().asDiagonal(); }
    Mat right() const { if (leftOrthogonal) return s.template cast<T>().asDiagonal() * V.adjoint(); return V.adjoint(); }

private:
    static Eigen::Index findnValues(Vec const& s, RealScalar reltol)
    {
        // Sentinel: reltol < 0 means "no tolerance truncation" -> keep every value
        if (reltol < RealScalar(0))
            return s.size();

        RealScalar tol2  = reltol * reltol;
        RealScalar norm2 = s.squaredNorm();                 // s is real, stays in RealScalar
        if (norm2 == RealScalar(0)) return 1;               // corner case
        RealScalar sum = 0;
        Eigen::Index n = s.size();
        for (Eigen::Index i = s.size() - 1; i >= 0; --i) {
            sum += s(i) * s(i);
            if (sum > tol2 * norm2) { n = i + 1; break; }
        }
        return n;
    }
};

template<class Decomp>
struct MatDecompFixedTol
{
    using T          = typename Decomp::value_type;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using Mat        = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    RealScalar tol;
    int rankMax;

    MatDecompFixedTol(RealScalar tol_ = RealScalar(1e-14), int rankMax_ = 0)
        : tol(tol_), rankMax(rankMax_) {}

    std::array<Mat, 2> operator()(Mat const& M, bool leftOrthogonal)
    {
        Decomp s{M, leftOrthogonal, tol, rankMax};
        return { s.left(), s.right() };
    }
};

template<class T>
struct MatSVDFixedTol : public MatDecompFixedTol<SVDDecomp<T>> {
    using MatDecompFixedTol<SVDDecomp<T>>::MatDecompFixedTol;
};
