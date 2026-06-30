#include <Eigen/Dense>
#include <array>

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