#pragma once

#include "matrix.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <complex>
#include <fstream>
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

// Class for the Tensor object
template<typename T>
struct Tensor3D {
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using Stride_dim = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
    public:
        size_t n_left, n_phys, n_right;    // 99% of the time n_cols=2
        std::vector<T> data;                // in this format, we want the ordering [left, physical, right]

        [[nodiscard]] inline size_t idx(size_t id_left, size_t id_phys, size_t id_right) const {
            return id_right*n_left*n_phys + id_phys*n_left + id_left;
        }

        T& operator()(size_t id_left, size_t id_phys, size_t id_right) {
            return data[idx(id_left,id_phys,id_right)];
        }

        const T& operator()(size_t id_left, size_t id_phys, size_t id_right) const {
            return data[idx(id_left,id_phys,id_right)];
        }

        Tensor3D(size_t r, size_t c, size_t s) : n_left(r), n_phys(c), n_right(s), data(r * c * s) {}

        // ===========================================================
        //                  TENSOR MANIPULATION: SLICE VIEW
        // ===========================================================

        // =================VIEW OF TENSOR============================
        // right view mutable
        Eigen::Map<MatrixX> right(size_t id_right) {
            return Eigen::Map<MatrixX>(
                data.data() + id_right * n_left * n_phys,
                n_left,
                n_phys
            );
        }
        
        // right view const
        Eigen::Map<const MatrixX> right_const(size_t id_right) const{
            return {
                data.data() + id_right * n_left * n_phys,
                n_left,
                n_phys
            };
        }

        // phys view mutable
        Eigen::Map<MatrixX,0,Stride_dim> phys(size_t id_phys)
        {
            return {
                data.data() + id_phys*n_left,
                n_left,
                n_right,
                Stride_dim(n_left*n_phys,1)
            };
        }

        // phys view const
        Eigen::Map<const MatrixX,0,Stride_dim> phys_const(size_t id_phys) const{
            return {
                data.data() + id_phys * n_left,
                n_left,
                n_right,
                Stride_dim(n_left * n_phys, 1)
            };
        }

        // left view mutable
        Eigen::Map<MatrixX,0,Stride_dim> left(size_t id_left){
            return {
                data.data() + id_left,
                n_phys,
                n_right,
                Stride_dim(n_left*n_phys,n_left)
            };
        }

        // left view const
        Eigen::Map<const MatrixX,0,Stride_dim> left_const(size_t id_left) const{
            return {
                data.data() + id_left,
                n_phys,
                n_right,
                Stride_dim(n_left*n_phys, n_left)
            };
        }

        // =================COPY OF TENSOR============================
        // make copy of slice of the tensor

        MatrixX phys_copy(size_t id_phys) {
            MatrixX out(n_left, n_right);
            for (size_t id_left = 0; id_left < n_left; ++id_left)
                for (size_t id_right = 0; id_right < n_right; ++id_right)
                    out(id_left, id_right) = data[idx(id_left, id_phys, id_right)];
            return out;
        }

        MatrixX right_copy(size_t id_right) {
            MatrixX out(n_left, n_phys);
            for (size_t id_left = 0; id_left < n_left; ++id_left)
                for (size_t id_phys = 0; id_phys < n_phys; ++id_phys)
                    out(id_left, id_phys) = data[idx(id_left, id_phys, id_right)];
            return out;
        }

        MatrixX left_copy(size_t id_left) {
            MatrixX out(n_phys, n_right);
            for (size_t id_phys = 0; id_phys < n_phys; ++id_phys)
                for (size_t id_right = 0; id_right < n_right; ++id_right)
                    out(id_phys, id_right) = data[idx(id_left, id_phys, id_right)];
            return out;
        }

        // =================phys flattening===========================
        // flatten phys into either left or right dimension
        // return either left * phys, rigth or left, phys * right

        Eigen::Map<MatrixX> flatten_phys(bool phys_on_left){
            if (phys_on_left) {
                // (left*phys, right)
                return {
                    data.data(),
                    n_left * n_phys,
                    n_right
                };
            } else {
                // (left, right*phys)
                return {
                    data.data(),
                    n_left,
                    n_right * n_phys
                };
            }
        }

        Eigen::Map<const MatrixX> flatten_phys_const(bool phys_on_left) const{
            if (phys_on_left) {
                // (left*phys, right)
                return Eigen::Map<const MatrixX>(
                    data.data(),
                    n_left * n_phys,
                    n_right
                );
            } else {
                // (left, right*phys)
                return Eigen::Map<const MatrixX>(
                    data.data(),
                    n_left,
                    n_right * n_phys
                );
            }
        }
        
        MatrixX flatten_phys_copy(bool phys_on_left) const{
            if (phys_on_left) {
                // (left*phys, right)
                MatrixX out(n_left * n_phys, n_right);

                for (size_t r = 0; r < n_right; ++r)
                    for (size_t p = 0; p < n_phys; ++p)
                        for (size_t l = 0; l < n_left; ++l)
                            out(l + p * n_left, r) = (*this)(l, p, r);

                return out;
            } else {
                // (left, right*phys)
                MatrixX out(n_left, n_right * n_phys);

                for (Eigen::Index r = 0; r < n_right; ++r)
                    for (Eigen::Index p = 0; p < n_phys; ++p)
                        for (Eigen::Index l = 0; l < n_left; ++l)
                            out(l, r * n_phys + p) = (*this)(l, p, r);

                return out;
            }
        }
};

// ================================================
//              Save load Tensor3D
// =================================================

template<typename T>
std::vector<Tensor3D<T>> load_vector_tensor(std::istream& in)
{
    using RealT = typename T::value_type;
    size_t nBit;
    in >> nBit;

    std::vector<Tensor3D<T>> Xs;
    Xs.reserve(nBit);

    for (size_t t = 0; t < nBit; ++t)
    {
        std::string header;
        in >> header;

        if (header != "ARMA_CUB_TXT_FC016")
            throw std::runtime_error("Bad header");

        size_t r, c, s;
        in >> r >> c >> s;
        Tensor3D<T> X(r, c, s);

        for (size_t id_left = 0; id_left < X.n_left; ++id_left)
        {
            for (size_t id_phys = 0; id_phys < X.n_phys; ++id_phys)
            {
                for (size_t id_right = 0; id_right < X.n_right; ++id_right)
                {
                    std::string token;
                    in >> token; // "(re,im)"

                    if (token.size() < 5)
                        throw std::runtime_error("Bad complex token");

                    // strip '(' and ')'
                    if (token.front() == '(') token.erase(token.begin());
                    if (token.back()  == ')') token.pop_back();

                    auto comma = token.find(',');
                    if (comma == std::string::npos)
                        throw std::runtime_error("Missing comma in complex");

                    RealT re(token.substr(0, comma));
                    RealT im(token.substr(comma + 1));

                    X(id_left, id_phys, id_right) = T(re, im);
                }
            }
        }

        Xs.push_back(std::move(X));
    }

    return Xs;
}

template<typename T>
std::vector<Tensor3D<T>> load_vector_tensor(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error("Cannot open file: " + filename);

    return load_vector_tensor<T>(file);
}

// TODO
template<class T>
void save_Tensor3D_to_arma(
    std::ostream& out,
    const std::vector<Tensor3D<T>>& Xs)
{
    using RealT = typename T::value_type;
    out << Xs.size() << "\n";

    for (size_t t = 0; t < Xs.size(); ++t)
    {
        out << "ARMA_CUB_TXT_FC016\n";

        const Tensor3D<T>& X = Xs[t];

        out << X.n_left << " "
            << X.n_phys << " "
            << X.n_rigth << "\n";

        out << std::scientific;
        out << std::setprecision(std::numeric_limits<RealT>::digits10 + 5);

        for (size_t id_left = 0; id_left < X.n_left; ++id_left)
        {
            for (size_t id_phys = 0; id_phys < X.n_phys; ++id_phys)
            {
                for (size_t id_right = 0; id_right < X.n_right; ++id_right)
                {
                    const auto& z = X(id_left, id_phys, id_right);

                    out << "("
                        << z.real() << ","
                        << z.imag()
                        << ") ";
                }
            }
            out << "\n";
        }
        out << "\n";
    }
}

// TODO
template<typename T>
void save_Tensor3D_to_arma(std::vector<Tensor3D<T>> cores, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file)
        throw std::runtime_error("Cannot open file");
    save_Tensor3D_to_arma(file, cores);
}

// ================================================
//              printing Tensor3D
// =================================================

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor3D<T>& X)
{
    os << "[" << X.n_left << " " << X.n_phys << " " << X.n_right << "]\n\n";

    for (size_t id_left = 0; id_left < X.n_left; ++id_left)
    {
        os << "[ ";   // slice start

        for (size_t id_phys = 0; id_phys < X.n_phys; ++id_phys)
        {
            os << "[ ";

            for (size_t id_right = 0; id_right < X.n_right; ++id_right)
                os << "(" << X(id_left,id_phys,id_right).real() << "," << X(id_left,id_phys,id_right).imag() << ") ";

            os << "]\n";
        }

        os << "]\n\n"; // slice end
    }

    return os;
}

// TODO
template<typename T>
std::ostream& operator<<(std::ostream& out,
                         const std::vector<Tensor3D<T>>& Xs)
{
    out << Xs.size() << "\n";

    for (const auto& X : Xs)
    {
        out << "ARMA_CUB_TXT_FC016\n";

        out << X.n_rows << " "
            << X.n_cols << " "
            << X.n_slices << "\n";

        using RealT = typename T::value_type;

        out << std::scientific;
        out << std::setprecision(std::numeric_limits<RealT>::digits10 + 5);

        for (size_t id_left = 0; id_left < X.n_left; ++id_left)
        {
            for (size_t id_phys = 0; id_phys < X.n_phys; ++id_phys)
            {
                for (size_t id_right = 0; id_right < X.n_right; ++id_right)
                {
                    const auto& z = X(id_left, id_phys, id_right);

                    out << "("
                        << z.real() << ","
                        << z.imag()
                        << ") ";
                }
            }
            out << "\n";
        }
        out << "\n";
    }

    return out;
}







