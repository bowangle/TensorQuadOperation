#pragma once
#include <boost/multiprecision/float128.hpp>
#include <Eigen/Dense>

using float128 = boost::multiprecision::float128;
using Cfloat128   = std::complex<float128>;
using MatrixX128 = Eigen::Matrix<Cfloat128, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline std::ostream& operator<<(std::ostream& os,
                                const Cfloat128& z)
{
    os << "(" << z.real() << "," << z.imag() << ")";
    return os;
}