#pragma once

#include <random>
#include <stdexcept>
#include <tuple>
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

    protected:
    VecTT core;                 // list of tensor
    int nBit;                   // number of tensor
    int w;                      // working tensor, -1 for no defined (None)
    int max_bond_dim_param;
    RealScalar reltol_param;    // -1 for no trucation on reltol

    public:
    // =============================================================
    //                      TT constructor:
    // =============================================================
    // from core
    TT(VecTT core_, int max_bond_dim_=0, RealScalar reltol_=-1, int w_=0)
    :
        core(std::move(core_)),
        nBit(core.size()),
        w(-1),
        max_bond_dim_param(max_bond_dim_),
        reltol_param(reltol_)
    {
        _initialize_w(w_);
    }

    // from file
    TT(const std::string& filename, int max_bond_dim_=0, RealScalar reltol_=-1, int w_=0)
    :
        core(std::move(load_vector_tensor<T>(filename))),
        nBit(core.size()),
        w(-1),
        max_bond_dim_param(max_bond_dim_),
        reltol_param(reltol_)
    {
        _initialize_w(w_);
    }

    // =============================================================
    //                      TT access:
    // =============================================================

    const VecTT& get_core() const { return core; }
    int get_w() const { return w; }
    int get_max_bond_dim() const { return max_bond_dim_param; }
    RealScalar get_reltol() const { return reltol_param; }

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
        // we have to start at the canonical center: w = 0 then propagate right->left->right
        _left_to_right(mat_decomp);
        _right_to_left(mat_decomp);
        // At the end the canonical center is still w = 0
    }

    void compress_svd(RealScalar reltol = -1, int max_bond_dim = -1)
    {
        int old_w = (w == -1) ? 0 : w;  // capture before any shift
        shift_w(0); // (does the QR sweep to set canonical center at 0. After it, the SVD trucation is correct.)
        _sweep(MatSVDFixedTol<T>{reltol, max_bond_dim}); // truncation is valid.
        w = 0;
        shift_w(old_w);
    }

    void _left_canonify_k(int k, bool compress = false)
    {
        // M[k] flattened as (n_left*n_phys) × n_right,  M = Q * R
        auto ab = MatQR<T>{}(core[k].flatten_as_matrix2(), true);
        MatrixX& Q = ab[0];
        MatrixX& R = ab[1];

        if (!compress) {
            core[k] = Tensor3D<T>(core[k].n_left, core[k].n_phys, Q.cols());
            std::copy(Q.data(), Q.data() + Q.size(), core[k].data.begin());

            MatrixX M2 = R * core[k+1].flatten_as_matrix1();
            core[k+1] = Tensor3D<T>(M2.rows(), core[k+1].n_phys, core[k+1].n_right);
            std::copy(M2.data(), M2.data() + M2.size(), core[k+1].data.begin());
        }
        else {
            // R = U * diag(s) * V^H, truncated to mu inside SVDDecomp
            SVDDecomp<T> svd(R, /*leftOrthogonal=*/true, reltol_param, max_bond_dim_param);
            Eigen::Index mu = svd.s.size();

            MatrixX Qnew = Q * svd.left();     // Q * U          : (n_left*n_phys) × mu
            MatrixX Rnew = svd.right();        // diag(s) * V^H  : mu × n_right_old

            core[k] = Tensor3D<T>(core[k].n_left, core[k].n_phys, mu);
            std::copy(Qnew.data(), Qnew.data() + Qnew.size(), core[k].data.begin());

            MatrixX M2 = Rnew * core[k+1].flatten_as_matrix1();
            core[k+1] = Tensor3D<T>(mu, core[k+1].n_phys, core[k+1].n_right);
            std::copy(M2.data(), M2.data() + M2.size(), core[k+1].data.begin());
        }
    }

    void _right_canonify_k(int k, bool compress = false)
    {
        // M[k] flattened as n_left × (n_phys*n_right),  M = L * Q
        auto ab = MatQR<T>{}(core[k].flatten_as_matrix1(), false);
        MatrixX& L = ab[0];   // n_left × K   (this is rtr.T in Python terms)
        MatrixX& Q = ab[1];   // K × (n_phys*n_right)

        if (!compress) {
            core[k] = Tensor3D<T>(Q.rows(), core[k].n_phys, core[k].n_right);
            std::copy(Q.data(), Q.data() + Q.size(), core[k].data.begin());

            MatrixX M1 = core[k-1].flatten_as_matrix2() * L;
            core[k-1] = Tensor3D<T>(core[k-1].n_left, core[k-1].n_phys, M1.cols());
            std::copy(M1.data(), M1.data() + M1.size(), core[k-1].data.begin());
        }
        else {
            // L = U * diag(s) * V^H, truncated to mu inside SVDDecomp
            SVDDecomp<T> svd(L, /*leftOrthogonal=*/false, reltol_param, max_bond_dim_param);
            Eigen::Index mu = svd.s.size();

            MatrixX Qnew = svd.right() * Q;    // V^H * Q        : mu × (n_phys*n_right)
            MatrixX Lnew = svd.left();         // U * diag(s)    : n_left × mu

            core[k] = Tensor3D<T>(mu, core[k].n_phys, core[k].n_right);
            std::copy(Qnew.data(), Qnew.data() + Qnew.size(), core[k].data.begin());

            MatrixX M1 = core[k-1].flatten_as_matrix2() * Lnew;
            core[k-1] = Tensor3D<T>(core[k-1].n_left, core[k-1].n_phys, mu);
            std::copy(M1.data(), M1.data() + M1.size(), core[k-1].data.begin());
        }
    }

    void increase_w(bool compress=false)
    {
        // Move the working site to the left
        _left_canonify_k(w, compress);
        w += 1;
    }

    void decrease_w(bool compress=false)
    {
        //Move the working site to the right
        _right_canonify_k(w, compress);
        w -= 1;
    }

    void _initialize_w(int new_w, bool compress=false)
    {
        if (new_w == -1)
            return;
        for (int i = 0; i < new_w; i++)
            _left_canonify_k(i, compress);
        for (int i = static_cast<int>(core.size()) - 1; i > new_w; i--)
            _right_canonify_k(i, compress);
        this->w = new_w;
    }

    void shift_w(int new_w, bool compress=false)
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
                increase_w(compress);
        else
            for (int i = 0; i < -steps; i++)
                decrease_w(compress);
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
    //                      TT operator:
    // =============================================================

    TT operator+(TT const& other) const
    {
        if (nBit != other.nBit)
            throw std::invalid_argument("TT::operator+: the lengths of the two TTs are different");

        // --- single-site case: plain element-wise sum ---
        if (nBit == 1) {
            auto const& a = core[0];
            auto const& b = other.core[0];
            if (a.n_left != b.n_left || a.n_phys != b.n_phys || a.n_right != b.n_right)
                throw std::invalid_argument("TT::operator+: incompatible single-site shapes");

            Tensor3D<T> t(a.n_left, a.n_phys, a.n_right);
            for (std::size_t j = 0; j < t.data.size(); j++)
                t.data[j] = a.data[j] + b.data[j];

            VecTT res;
            res.push_back(std::move(t));
            return TT(std::move(res), max_bond_dim_param, reltol_param, /*w_=*/-1);
        }

        VecTT res;
        res.reserve(nBit);

        // --- first core: shape (n_left, n_phys, r1 + r2) ---
        {
            auto const& a = core[0];
            auto const& b = other.core[0];
            if (a.n_left != b.n_left || a.n_phys != b.n_phys)
                throw std::invalid_argument("TT::operator+: incompatible first cores");

            Tensor3D<T> t(a.n_left, a.n_phys, a.n_right + b.n_right);  // zero-initialized
            for (Eigen::Index p = 0; p < a.n_phys; p++) {
                t.phys(p).leftCols(a.n_right)  = a.phys_const(p);
                t.phys(p).rightCols(b.n_right) = b.phys_const(p);
            }
            res.push_back(std::move(t));
        }

        // --- middle cores: shape (l1 + l2, n_phys, r1 + r2), block diagonal ---
        for (int i = 1; i < nBit - 1; i++) {
            auto const& a = core[i];
            auto const& b = other.core[i];
            if (a.n_phys != b.n_phys)
                throw std::invalid_argument("TT::operator+: the physical dimensions of the two TTs are different");

            Tensor3D<T> t(a.n_left + b.n_left, a.n_phys, a.n_right + b.n_right);  // zero-initialized
            for (Eigen::Index p = 0; p < a.n_phys; p++) {
                t.phys(p).topLeftCorner(a.n_left, a.n_right)     = a.phys_const(p);
                t.phys(p).bottomRightCorner(b.n_left, b.n_right) = b.phys_const(p);
            }
            res.push_back(std::move(t));
        }

        // --- last core: shape (l1 + l2, n_phys, n_right) ---
        {
            auto const& a = core[nBit - 1];
            auto const& b = other.core[nBit - 1];
            if (a.n_phys != b.n_phys || a.n_right != b.n_right)
                throw std::invalid_argument("TT::operator+: incompatible last cores");

            Tensor3D<T> t(a.n_left + b.n_left, a.n_phys, a.n_right);
            for (Eigen::Index p = 0; p < a.n_phys; p++) {
                t.phys(p).topRows(a.n_left)    = a.phys_const(p);
                t.phys(p).bottomRows(b.n_left) = b.phys_const(p);
            }
            res.push_back(std::move(t));
        }

        // w = -1: the sum does not preserve the canonical properties of the side tensors.
        // Call shift_w(...) or compress_svd(...) on the result if you need canonical form.
        return TT(std::move(res), max_bond_dim_param, reltol_param, /*w_=*/-1);
    }

    // in-place variant
    TT& operator+=(TT const& other)
    {
        *this = *this + other;
        return *this;
    }

    TT operator*(T scalar) const
    {
        TT res(*this);

        if (scalar == T(0)) {
            for (int i = 0; i < nBit; i++)
                res.core[i] = Tensor3D<T>(1, core[i].n_phys, 1);  // zero-initialized
            res.w = -1;
        }
        else {
            int k = (res.w == -1) ? 0 : res.w;
            for (auto& x : res.core[k].data)
                x *= scalar;
        }
        return res;
    }

    // scalar * TT (so both  tt * s  and  s * tt  work)
    friend TT operator*(T scalar, TT const& tt)
    {
        return tt * scalar;
    }

    // Unary minus: -tt
    TT operator-() const
    {
        return (*this) * T(-1);
    }

    // Subtraction, mirroring the Python:  self + (-1) * other
    TT operator-(TT const& other) const
    {
        return *this + other * T(-1);
    }

    // Division by scalar, mirroring the Python:  self * (1 / other)
    TT operator/(T scalar) const
    {
        return (*this) * (T(1) / scalar);
    }

    // Optional in-place variants
    TT& operator-=(TT const& other) { *this = *this - other; return *this; }
    TT& operator/=(T scalar) { return (*this) *= (T(1) / scalar); }
    TT& operator*=(T scalar)
    {
        if (scalar == T(0)) {
            for (int i = 0; i < nBit; i++)
                core[i] = Tensor3D<T>(1, core[i].n_phys, 1);
            w = -1;
        }
        else {
            int k = (w == -1) ? 0 : w;
            for (auto& x : core[k].data)
                x *= scalar;
        }
        return *this;
    }

    // Complex conjugate of the TT (element-wise conj of every core).
    // Preserves the working site: conjugation keeps orthogonality,
    // since Q†Q = I  ⇒  conj(Q)† conj(Q) = conj(Q†Q) = I.
    TT conj() const
    {
        TT res(*this);
        for (auto& t : res.core)
            for (auto& x : t.data)
                x = Eigen::numext::conj(x);
        return res;
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

    // Compute the list of bond dimension: (padded 1-chi1-chi2-1) for a tt of len 3)
    std::vector<int> compute_list_chi() const {
        std::vector<int> chi;
        chi.reserve(nBit + 1);
        chi.push_back(1);  // left boundary
        for (int i = 0; i < nBit; i++)
            chi.push_back(static_cast<int>(core[i].n_right));
        return chi;
    }

    // Number of values in each core: n_left * n_phys * n_right per core
    std::vector<Eigen::Index> compute_nb_value_core() const {
        std::vector<Eigen::Index> nb;
        nb.reserve(nBit);
        for (int i = 0; i < nBit; i++)
            nb.push_back(core[i].n_left * core[i].n_phys * core[i].n_right);
        return nb;
    }

    // Total number of values across all cores
    Eigen::Index compute_tot_nb_value() const {
        Eigen::Index tot = 0;
        for (int i = 0; i < nBit; i++)
            tot += core[i].n_left * core[i].n_phys * core[i].n_right;
        return tot;
    }

};

