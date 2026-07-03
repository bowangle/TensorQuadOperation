#pragma once

#include <vector>
#include <iostream>
#include <functional>  // for std::function
#include <array>       // for std::array
#include <vector>      // for std::vector (VecTT)
#include <algorithm>   // for std::copy
#include "tensor.h"
#include "mat_decomp.h"

template<typename T>
class TT{
    using VecTT = std::vector<Tensor3D<T>>;
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

    private:
    int nBit;                   // number of tensor
    VecTT core;                 // list of tensor
    int w;                      // working tensor, -1 for no defined (None)
    int max_bond_dim_param;
    float a;

    public:
    // =============================================================
    //                      TT constructor:
    // =============================================================
    // from core
    TT(VecTT core_, int max_bond_dim_=0)
    :
        core(std::move(core_)),
        nBit(core.size()),
        w(-1),
        max_bond_dim_param(max_bond_dim_),
        a(-1)
    {}

    // from file
    TT(const std::string& filename, int max_bond_dim_=0)
    :
        core(std::move(load_vector_tensor<T>(filename))),
        nBit(core.size()),
        w(-1),
        max_bond_dim_param(max_bond_dim_),
        a(-1)
    {}

    // =============================================================
    //                      TT access:
    // =============================================================

    int get_size(){
        return nBit;
    }

    VecTT get_core(){
        return core;
    }

    // Return the shape of each tensor as a vector of tuples (n_left, n_phys, n_right)
    std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> get_shape() const
    {
        std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> s;
        s.reserve(core.size());
        for (auto const& t : core)
            s.emplace_back(t.n_left, t.n_phys, t.n_right);
        return s;
    }

    // Return the maximum bond dimension over all internal bonds
    Eigen::Index get_chi() const
    {
        Eigen::Index max_chi = 0;
        for (int i = 0; i < static_cast<int>(core.size()) - 1; i++)
            max_chi = std::max(max_chi, core[i].n_right);
        return max_chi;
    }

    T eval(std::vector<int> const& id) const
    {
        if (static_cast<int>(id.size()) != nBit)
            throw std::invalid_argument("TT::eval: id.size() != nBit");

        // start with 1x1 identity
        MatrixX prod = MatrixX::Identity(1, 1);

        for (int k = 0; k < nBit; k++)
            prod = prod * core[k].phys_const(id[k]);  // (1 x n_left) * (n_left x n_right)

        return prod(0, 0);
    }

    // =============================================================
    //                      TT Canonicalisation:
    // =============================================================

    void _left_to_right(std::function<std::array<MatrixX, 2>(MatrixX, bool)> mat_decomp)
    {
        for (auto i = 0u; i + 1 < core.size(); i++) {
            auto ab = mat_decomp(core[i].flatten_as_matrix2(), true);
            MatrixX& M1 = ab[0];
            MatrixX  M2 = ab[1] * core[i+1].flatten_as_matrix1();

            Eigen::Index new_right = M1.cols();
            core[i] = Tensor3D<T>(core[i].n_left, core[i].n_phys, new_right);
            std::copy(M1.data(), M1.data() + M1.size(), core[i].data.begin());

            Eigen::Index new_left = M2.rows();
            core[i+1] = Tensor3D<T>(new_left, core[i+1].n_phys, core[i+1].n_right);
            std::copy(M2.data(), M2.data() + M2.size(), core[i+1].data.begin());
        }
    }

    void _right_to_left(std::function<std::array<MatrixX, 2>(MatrixX, bool)> mat_decomp)
    {
        for (int i = core.size()-1; i > 0; i--) {
            // Decompose M[i]: n_left × (n_phys*n_right)
            auto ab = mat_decomp(core[i].flatten_as_matrix1(), false);
            MatrixX& M2 = ab[1];

            // Contract remainder into M[i-1]: (n_left*n_phys) × n_right
            MatrixX M1 = core[i-1].flatten_as_matrix2() * ab[0];

            // Write M1 back into VecTT[i-1]
            Eigen::Index new_right = M1.cols();
            core[i-1] = Tensor3D<T>(core[i-1].n_left, core[i-1].n_phys, new_right);
            std::copy(M1.data(), M1.data() + M1.size(), core[i-1].data.begin());

            // Write M2 back into VecTT[i]
            Eigen::Index new_left = M2.rows();
            core[i] = Tensor3D<T>(new_left, core[i].n_phys, core[i].n_right);
            std::copy(M2.data(), M2.data() + M2.size(), core[i].data.begin());
        }
    }

    void _sweep(std::function<std::array<MatrixX, 2>(MatrixX, bool)> mat_decomp)
    {
        _left_to_right(mat_decomp);
        _right_to_left(mat_decomp);
    }

    void _compressSVD(typename Eigen::NumTraits<T>::Real reltol=1e-12, int maxBondDim=0) 
    { 
        _right_to_left(MatQR<T>{}); 
        _sweep(MatSVDFixedTol<T>{reltol, maxBondDim}); 
    }

    void compress_svd(typename Eigen::NumTraits<T>::Real reltol = 1e-12, int max_bond_dim = -1)
        {
            int old_w = (w == -1) ? 0 : w;  // capture before any shift
            shift_w(0);
            _compressSVD(reltol, max_bond_dim);
            w = 0;
            shift_w(old_w);
        }

    void _left_canonify_k(int k)
    {
        // flatten M[k] as (n_left*n_phys) × n_right and decompose
        auto ab = MatQR<T>{}(core[k].flatten_as_matrix2(), true);
        MatrixX& Q = ab[0];
        MatrixX& R = ab[1];

        // Store Q back into site k
        core[k] = Tensor3D<T>(core[k].n_left, core[k].n_phys, Q.cols());
        std::copy(Q.data(), Q.data() + Q.size(), core[k].data.begin());

        // Contract R into site k+1
        MatrixX M2 = R * core[k+1].flatten_as_matrix1();
        core[k+1] = Tensor3D<T>(M2.rows(), core[k+1].n_phys, core[k+1].n_right);
        std::copy(M2.data(), M2.data() + M2.size(), core[k+1].data.begin());
    }

    void _right_canonify_k(int k)
    {
        // flatten M[k] as n_left × (n_phys*n_right) and decompose
        auto ab = MatQR<T>{}(core[k].flatten_as_matrix1(), false);
        MatrixX& L = ab[0];
        MatrixX& Q = ab[1];

        // Store Q back into site k
        core[k] = Tensor3D<T>(Q.rows(), core[k].n_phys, core[k].n_right);
        std::copy(Q.data(), Q.data() + Q.size(), core[k].data.begin());

        // Contract L into site k-1
        MatrixX M1 = core[k-1].flatten_as_matrix2() * L;
        core[k-1] = Tensor3D<T>(core[k-1].n_left, core[k-1].n_phys, M1.cols());
        std::copy(M1.data(), M1.data() + M1.size(), core[k-1].data.begin());
    }

    void increase_w()
    {
        // Move the working site to the left
        _left_canonify_k(w);
        w += 1;
    }

    void decrease_w()
    {
        //Move the working site to the right
        _right_canonify_k(w);
        w -= 1;
    }

    void _initialize_w(int new_w)
    {
        if (new_w == -1)
            return;
        for (int i = 0; i < new_w; i++)
            _left_canonify_k(i);
        for (int i = static_cast<int>(core.size()) - 1; i > new_w; i--)
            _right_canonify_k(i);
        this->w = new_w;
    }

    void shift_w(int new_w)
    {
        if (w == -1)
        {
            _initialize_w(new_w);
            return;
        }

        int steps = new_w - w;
        if (steps == 0)
            return;

        if (steps > 0)
            for (int i = 0; i < steps; i++)
                increase_w();
        else
            for (int i = 0; i < -steps; i++)
                decrease_w();
    }

    void check_canonical(typename Eigen::NumTraits<T>::Real tol = 1e-8) const
    {
        if (w == -1)
            return;

        // Check left-canonical sites (i < w)
        for (int i = 0; i < w; i++)
        {
            Eigen::Map<const MatrixX> m = core[i].flatten_as_matrix2_const(); // (n_left*n_phys) x n_right
            MatrixX should_be_eye = m.adjoint() * m;                           // n_right x n_right
            MatrixX eye = MatrixX::Identity(should_be_eye.rows(), should_be_eye.cols());
            typename Eigen::NumTraits<T>::Real gap = (eye - should_be_eye).norm();
            if (gap > tol)
                std::cerr << "Warning: tensor at site " << i
                        << " is not left canonical with an error of " << gap << "\n";
        }

        // Check right-canonical sites (i > w)
        for (int i = w + 1; i < static_cast<int>(core.size()); i++)
        {
            Eigen::Map<const MatrixX> m = core[i].flatten_as_matrix1_const(); // n_left x (n_phys*n_right)
            MatrixX should_be_eye = m * m.adjoint();                            // n_left x n_left
            MatrixX eye = MatrixX::Identity(should_be_eye.rows(), should_be_eye.cols());
            typename Eigen::NumTraits<T>::Real gap = (eye - should_be_eye).norm();
            if (gap > tol)
                std::cerr << "Warning: tensor at site " << i
                        << " is not right canonical with an error of " << gap << "\n";
        }
    }

    // =============================================================
    //                      TT misc:
    // =============================================================

    // Generate n_point random evaluation points, each of size nBit
    // with each index in [0, n_phys-1]
    std::vector<std::vector<int>> generate_points(int n_point) const
    {
        std::mt19937 rng(std::random_device{}());

        std::vector<std::vector<int>> points(n_point, std::vector<int>(nBit));
        for (auto& pt : points)
            for (int k = 0; k < nBit; k++)
            {
                std::uniform_int_distribution<int> dist(0, core[k].n_phys - 1);
                pt[k] = dist(rng);
            }
        return points;
    }

    // Evaluate the TT at a list of points
    std::vector<T> eval_list(std::vector<std::vector<int>> const& points) const
    {
        std::vector<T> results;
        results.reserve(points.size());
        for (auto const& pt : points)
            results.push_back(eval(pt));
        return results;
    } 
};