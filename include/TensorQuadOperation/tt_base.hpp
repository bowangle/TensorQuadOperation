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
#include <xfac_quad/cubemat_helper.h>
#include <xfac_quad/matrix/mat_decomp.h>

namespace TensorQuadOperation {

/// The first core's left bond and the last core's right bond may be dangling.
/// Either or both boundary bonds may have dimension greater than one.
/// Scalar evaluation requires both boundary bonds to have dimension one.
template<typename T>
class TT{
    using VecTT = std::vector<xfac_quad::Tensor3D<T>>;
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

    protected:
    VecTT core;                 // list of tensor
    int nBit;                   // number of tensor
    int w;                      // working tensor, -1 for no defined (None)
    int max_bond_dim_param;
    int ordering;               // tell which dimension old the leg (0 for (leg, left, right), 1 for (left, leg, right), 2 for (left, right, leg))
    RealScalar reltol_param;    // -1 for no trucation on reltol

    private:
    void check_eval_boundaries() const
    {
        if (core.empty()) return;
        const auto& first = core.front();
        const auto& last = core.back();
        const Eigen::Index left_boundary = ordering == 0 ? first.n_cols : first.n_rows;
        const Eigen::Index right_boundary = ordering == 2 ? last.n_cols : last.n_slices;
        if (left_boundary != 1 || right_boundary != 1)
            throw std::invalid_argument("TT evaluation requires left and right boundary bonds of dimension one");
    }

    public:
    // =============================================================
    //                      TT constructor:
    // =============================================================
    // from core
    TT(VecTT core_, int max_bond_dim_=0, RealScalar reltol_=-1, int w_=0, int ordering_=1)
    :
        core(std::move(core_)),
        nBit(core.size()),
        w(-1),
        max_bond_dim_param(max_bond_dim_),
        ordering(ordering_),
        reltol_param(reltol_)
    {
        _initialize_w(w_);
    }

    // Files store cores in ordering 1. Convert to the requested in-memory ordering.
    TT(const std::string& filename, int max_bond_dim_=0, RealScalar reltol_=-1, int w_=0, int ordering_=1)
    :
        core(std::move(xfac_quad::load_vector_tensor<T>(filename))),
        nBit(core.size()),
        w(-1),
        max_bond_dim_param(max_bond_dim_),
        ordering(1),
        reltol_param(reltol_)
    {
        if (ordering_ != 1)
            *this = change_ordering(std::move(*this), ordering_);
        // initialise w correctly
        _initialize_w(w_);
    }

    

    // =============================================================
    //                      TT access:
    // =============================================================

    /// Move the physical axis of every core to new_ordering (0, 1 or 2).
    /// The left and right bonds keep their relative order, and the orthogonality
    /// center stays at the same site in the returned train.
    static TT change_ordering(TT train, int new_ordering)
    {
        if (train.ordering < 0 || train.ordering > 2 || new_ordering < 0 || new_ordering > 2)
            throw std::invalid_argument("TT::change_ordering: ordering must be 0, 1 or 2");
        if (train.ordering == new_ordering)
            return train;

        for (auto& tensor : train.core) {
            std::vector<Eigen::Index> shape{tensor.n_rows, tensor.n_cols, tensor.n_slices};
            const Eigen::Index physical = shape[train.ordering];
            shape.erase(shape.begin() + train.ordering);
            shape.insert(shape.begin() + new_ordering, physical);

            // Unfold along the physical axis and fold it back at its new position.
            const auto physical_columns = xfac_quad::cubeToMat_R(tensor, train.ordering);
            tensor = xfac_quad::matToCube_R(physical_columns, shape, new_ordering);
        }
        train.ordering = new_ordering;
        return train;
    }

    const VecTT& get_core() const { return core; }
    int get_w() const { return w; }
    int get_ordering() const { return ordering; }
    int get_max_bond_dim() const { return max_bond_dim_param; }
    RealScalar get_reltol() const { return reltol_param; }

    int get_size(){
        return nBit;
    }

    VecTT get_core(){
        return core;
    }

    void save(const std::string& file_prefix) const
    {
        auto file_train = change_ordering(*this, 1);
        xfac_quad::save_Tensor3D_to_arma(std::move(file_train.core), file_prefix + ".tt");
    }

    // Return the shape of each tensor as a vector of tuples
    std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> get_shape() const
    {
        std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> s;
        s.reserve(core.size());
        for (auto const& t : core)
            s.emplace_back(t.n_rows, t.n_cols, t.n_slices);
        return s;
    }

    // Return the maximum bond dimension over all internal bonds
    Eigen::Index get_chi() const
    {
        Eigen::Index max_chi = 0;
        for (int i = 0; i < static_cast<int>(core.size()) - 1; i++) {
            const Eigen::Index right_bond = ordering == 2 ? core[i].n_cols : core[i].n_slices;
            max_chi = std::max(max_chi, right_bond);
        }
        return max_chi;
    }

    // Skip the boundary check when the caller has already validated both bonds.
    T eval(std::vector<int> const& id, bool check_boundaries = true) const
    {
        if (int(id.size()) != nBit) throw std::invalid_argument("TT::eval: id.size() != nBit");
        if (check_boundaries) check_eval_boundaries();

        MatrixX a = MatrixX::Identity(1, 1), b;
        for (int k = 0; k < nBit; k++) {
            switch (ordering) {
                case 0:
                    b.noalias() = a * core[k].row_as_mat_const(id[k]);
                    break;
                case 1:
                    b.noalias() = a * core[k].col_as_mat_const(id[k]);
                    break;
                case 2:
                    b.noalias() = a * core[k].slice_const(id[k]);
                    break;
                default:
                    throw std::invalid_argument("TT::eval: ordering must be 0, 1 or 2");
            }
            a.swap(b);
        }
        return a(0, 0);
    }

    // =============================================================
    //                      TT Canonicalisation:
    // =============================================================

    void _left_to_right(std::function<std::array<MatrixX, 2>(MatrixX, bool)> mat_decomp)
    {
        if (ordering < 0 || ordering > 2)
            throw std::invalid_argument("TT::_left_to_right: ordering must be 0, 1 or 2");

        using Core = xfac_quad::Tensor3D<T>;
        for (auto i = 0u; i + 1 < core.size(); i++) {
            const auto& current = core[i];
            const auto& next = core[i + 1];
            // Decompose core[i]: (n_left*n_phys) × n_right.
            auto [basis, transfer] = ordering == 2
                ? mat_decomp(xfac_quad::cubeToMat_R(current, 1), true)
                : mat_decomp(current.flatten_as_matrix2_const(), true);

            // Rebuild core[i] in its stored order with the new right bond.
            Core updated_current;
            if (ordering == 2) {
                updated_current = xfac_quad::matToCube_R(
                    basis, {current.n_rows, basis.cols(), current.n_slices}, 1);
            } else {
                updated_current = Core(current.n_rows, current.n_cols, basis.cols());
                std::copy_n(basis.data(), basis.size(), updated_current.data.begin());
            }

            // Contract the transfer matrix into the left bond of core[i+1].
            Core updated_next;
            if (ordering == 0) {
                // Each stored slice has axes (physical, left).
                updated_next = Core(next.n_rows, transfer.rows(), next.n_slices);
                for (Eigen::Index slice = 0; slice < next.n_slices; ++slice)
                    updated_next.slice(slice).noalias() = next.slice_const(slice) * transfer.transpose();
            } else {
                updated_next = Core(transfer.rows(), next.n_cols, next.n_slices);
                updated_next.flatten_as_matrix1().noalias() = transfer * next.flatten_as_matrix1_const();
            }

            core[i] = std::move(updated_current);
            core[i + 1] = std::move(updated_next);
        }
    }

    void _right_to_left(std::function<std::array<MatrixX, 2>(MatrixX, bool)> mat_decomp)
    {
        if (ordering < 0 || ordering > 2)
            throw std::invalid_argument("TT::_right_to_left: ordering must be 0, 1 or 2");

        using Core = xfac_quad::Tensor3D<T>;
        for (int i = static_cast<int>(core.size()) - 1; i > 0; --i) {
            const auto& current = core[i];
            const auto& previous = core[i - 1];
            // Decompose M[i]: n_left × (n_phys*n_right)
            auto [transfer, basis] = ordering == 0
                ? mat_decomp(xfac_quad::cubeToMat_L(current, 1), false)
                : mat_decomp(current.flatten_as_matrix1_const(), false);

            // Contract remainder into M[i-1]: (n_left*n_phys) × n_right
            Core updated_previous;
            if (ordering == 2) {
                // Each stored slice has axes (left, right).
                updated_previous = Core(previous.n_rows, transfer.cols(), previous.n_slices);
                for (Eigen::Index slice = 0; slice < previous.n_slices; ++slice)
                    updated_previous.slice(slice).noalias() = previous.slice_const(slice) * transfer;
            } else {
                updated_previous = Core(previous.n_rows, previous.n_cols, transfer.cols());
                updated_previous.flatten_as_matrix2().noalias() = previous.flatten_as_matrix2_const() * transfer;
            }

            // Rebuild core[i] in its stored order with the new left bond.
            Core updated_current;
            if (ordering == 0) {
                updated_current = xfac_quad::matToCube_L(
                    basis, {current.n_rows, basis.rows(), current.n_slices}, 1);
            } else {
                updated_current = Core(basis.rows(), current.n_cols, current.n_slices);
                std::copy_n(basis.data(), basis.size(), updated_current.data.begin());
            }

            core[i - 1] = std::move(updated_previous);
            core[i] = std::move(updated_current);
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
        _sweep(xfac_quad::MatSVDFixedTol<T>{reltol, max_bond_dim}); // truncation is valid.
        w = 0;
        shift_w(old_w);
    }

    void _left_canonify_k(int k, bool compress = false)
    {
        if (ordering < 0 || ordering > 2)
            throw std::invalid_argument("TT::_left_canonify_k: ordering must be 0, 1 or 2");

        using Core = xfac_quad::Tensor3D<T>;
        const auto& current = core[k];
        const auto& next = core[k + 1];
        // M[k] flattened as (n_left*n_phys) × n_right,  M = Q * R
        const xfac_quad::MatQR<T> qr;
        auto [Q, R] = ordering == 2
            ? qr(xfac_quad::cubeToMat_R(current, 1), true)
            : qr(current.flatten_as_matrix2_const(), true);

        if (compress) {
            // R = U * diag(s) * V^H, truncated to mu inside SVDDecomp
            xfac_quad::SVDDecomp<T> svd(R, /*leftOrthogonal=*/true, reltol_param, max_bond_dim_param);
            Q = (Q * svd.left()).eval();       // Q * U          : (n_left*n_phys) × mu
            R = svd.right();                   // diag(s) * V^H  : mu × n_right_old
        }

        // Rebuild core[k] in its stored order with the new right bond.
        Core updated_current;
        if (ordering == 2) {
            updated_current = xfac_quad::matToCube_R(
                Q, {current.n_rows, Q.cols(), current.n_slices}, 1);
        } else {
            updated_current = Core(current.n_rows, current.n_cols, Q.cols());
            std::copy_n(Q.data(), Q.size(), updated_current.data.begin());
        }

        // Contract R into the left bond of core[k+1].
        Core updated_next;
        if (ordering == 0) {
            // Each stored slice has axes (physical, left).
            updated_next = Core(next.n_rows, R.rows(), next.n_slices);
            for (Eigen::Index slice = 0; slice < next.n_slices; ++slice)
                updated_next.slice(slice).noalias() = next.slice_const(slice) * R.transpose();
        } else {
            updated_next = Core(R.rows(), next.n_cols, next.n_slices);
            updated_next.flatten_as_matrix1().noalias() = R * next.flatten_as_matrix1_const();
        }

        core[k] = std::move(updated_current);
        core[k + 1] = std::move(updated_next);
    }

    void _right_canonify_k(int k, bool compress = false)
    {
        if (ordering < 0 || ordering > 2)
            throw std::invalid_argument("TT::_right_canonify_k: ordering must be 0, 1 or 2");

        using Core = xfac_quad::Tensor3D<T>;
        const auto& current = core[k];
        const auto& previous = core[k - 1];
        // M[k] flattened as n_left × (n_phys*n_right),  M = L * Q
        // L has shape n_left × K (rtr.T in Python terms), Q has shape K × (n_phys*n_right).
        const xfac_quad::MatQR<T> qr;
        auto [L, Q] = ordering == 0
            ? qr(xfac_quad::cubeToMat_L(current, 1), false)
            : qr(current.flatten_as_matrix1_const(), false);

        if (compress) {
            // L = U * diag(s) * V^H, truncated to mu inside SVDDecomp
            xfac_quad::SVDDecomp<T> svd(L, /*leftOrthogonal=*/false, reltol_param, max_bond_dim_param);
            Q = (svd.right() * Q).eval();      // V^H * Q        : mu × (n_phys*n_right)
            L = svd.left();                    // U * diag(s)    : n_left × mu
        }

        // Rebuild core[k] in its stored order with the new left bond.
        Core updated_current;
        if (ordering == 0) {
            updated_current = xfac_quad::matToCube_L(
                Q, {current.n_rows, Q.rows(), current.n_slices}, 1);
        } else {
            updated_current = Core(Q.rows(), current.n_cols, current.n_slices);
            std::copy_n(Q.data(), Q.size(), updated_current.data.begin());
        }

        // Contract L into the right bond of core[k-1].
        Core updated_previous;
        if (ordering == 2) {
            // Each stored slice has axes (left, right).
            updated_previous = Core(previous.n_rows, L.cols(), previous.n_slices);
            for (Eigen::Index slice = 0; slice < previous.n_slices; ++slice)
                updated_previous.slice(slice).noalias() = previous.slice_const(slice) * L;
        } else {
            updated_previous = Core(previous.n_rows, previous.n_cols, L.cols());
            updated_previous.flatten_as_matrix2().noalias() = previous.flatten_as_matrix2_const() * L;
        }

        core[k] = std::move(updated_current);
        core[k - 1] = std::move(updated_previous);
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
        if (ordering < 0 || ordering > 2)
            throw std::invalid_argument("TT::check_canonical: ordering must be 0, 1 or 2");

        // Check left-canonical sites (i < w)
        for (int i = 0; i < w; i++)
        {
            // Isolate the right bond as the columns of (n_left*n_phys) × n_right.
            MatrixX should_be_eye;
            if (ordering == 2) {
                const auto m = xfac_quad::cubeToMat_R(core[i], 1);
                should_be_eye = m.adjoint() * m;
            } else {
                const auto m = core[i].flatten_as_matrix2_const();
                should_be_eye = m.adjoint() * m;
            }
            MatrixX eye = MatrixX::Identity(should_be_eye.rows(), should_be_eye.cols());
            typename Eigen::NumTraits<T>::Real gap = (eye - should_be_eye).norm();
            if (gap > tol)
                std::cerr << "Warning: tensor at site " << i
                        << " is not left canonical with an error of " << gap << "\n";
        }

        // Check right-canonical sites (i > w)
        for (int i = w + 1; i < static_cast<int>(core.size()); i++)
        {
            // Isolate the left bond as the rows of n_left × (n_phys*n_right).
            MatrixX should_be_eye;
            if (ordering == 0) {
                const auto m = xfac_quad::cubeToMat_L(core[i], 1);
                should_be_eye = m * m.adjoint();
            } else {
                const auto m = core[i].flatten_as_matrix1_const();
                should_be_eye = m * m.adjoint();
            }
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
        if (ordering < 0 || ordering > 2 || other.ordering < 0 || other.ordering > 2)
            throw std::invalid_argument("TT::operator+: ordering must be 0, 1 or 2");

        const int left = ordering == 0 ? 1 : 0;
        const int right = ordering == 2 ? 1 : 2;
        const int other_left = other.ordering == 0 ? 1 : 0;
        const int other_right = other.ordering == 2 ? 1 : 2;

        // Read the input in its stored order and place it in the output's bond block.
        using Core = xfac_quad::Tensor3D<T>;
        const auto add_block = [&](Core& target, const Core& source, int physical_axis,
                                   Eigen::Index left_offset, Eigen::Index right_offset) {
            const int source_left = physical_axis == 0 ? 1 : 0;
            const int source_right = physical_axis == 2 ? 1 : 2;
            for (Eigen::Index slice = 0; slice < source.n_slices; ++slice)
                for (Eigen::Index column = 0; column < source.n_cols; ++column)
                    for (Eigen::Index row = 0; row < source.n_rows; ++row) {
                        const std::array<Eigen::Index, 3> from{row, column, slice};
                        std::array<Eigen::Index, 3> to;
                        to[ordering] = from[physical_axis];
                        to[left] = from[source_left] + left_offset;
                        to[right] = from[source_right] + right_offset;
                        target(to[0], to[1], to[2]) += source(row, column, slice);
                    }
        };

        VecTT res;
        res.reserve(nBit);
        for (int i = 0; i < nBit; ++i) {
            const auto& a = core[i];
            const auto& b = other.core[i];
            const std::array<Eigen::Index, 3> a_shape{a.n_rows, a.n_cols, a.n_slices};
            const std::array<Eigen::Index, 3> b_shape{b.n_rows, b.n_cols, b.n_slices};
            const bool first = i == 0;
            const bool last = i == nBit - 1;
            if (a_shape[ordering] != b_shape[other.ordering] ||
                (first && a_shape[left] != b_shape[other_left]) ||
                (last && a_shape[right] != b_shape[other_right]))
                throw std::invalid_argument("TT::operator+: incompatible physical or boundary dimensions");

            // Form a direct sum on each internal bond. A single core adds element-wise.
            auto shape = a_shape;
            if (!first) shape[left] += b_shape[other_left];
            if (!last) shape[right] += b_shape[other_right];
            Core tensor(shape[0], shape[1], shape[2]);
            add_block(tensor, a, ordering, 0, 0);
            add_block(tensor, b, other.ordering,
                      first ? 0 : a_shape[left], last ? 0 : a_shape[right]);
            res.push_back(std::move(tensor));
        }

        // w = -1: the sum does not preserve the canonical properties of the side tensors.
        // Call shift_w(...) or compress_svd(...) on the result if you need canonical form.
        return TT(std::move(res), max_bond_dim_param, reltol_param, /*w_=*/-1, ordering);
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
        res *= scalar;
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
            if (ordering < 0 || ordering > 2)
                throw std::invalid_argument("TT::operator*=: ordering must be 0, 1 or 2");
            for (auto& tensor : core) {
                const std::array<Eigen::Index, 3> stored_shape{tensor.n_rows, tensor.n_cols, tensor.n_slices};
                std::array<Eigen::Index, 3> zero_shape{1, 1, 1};
                zero_shape[ordering] = stored_shape[ordering];
                tensor = xfac_quad::Tensor3D<T>(zero_shape[0], zero_shape[1], zero_shape[2]);
            }
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
        if (ordering < 0 || ordering > 2)
            throw std::invalid_argument("TT::generate_points: ordering must be 0, 1 or 2");
        std::mt19937 rng(std::random_device{}());

        std::vector<std::vector<int>> points(n_point, std::vector<int>(nBit));
        for (auto& pt : points)
            for (int k = 0; k < nBit; k++)
            {
                const auto& tensor = core[k];
                const std::array<Eigen::Index, 3> shape{tensor.n_rows, tensor.n_cols, tensor.n_slices};
                std::uniform_int_distribution<int> dist(0, shape[ordering] - 1);
                pt[k] = dist(rng);
            }
        return points;
    }

    // Evaluate the TT at a list of points
    std::vector<T> eval_list(std::vector<std::vector<int>> const& points) const
    {
        check_eval_boundaries();
        std::vector<T> results(points.size());          // pre-size, no push_back
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(points.size()); ++i)
            results[i] = eval(points[i], /*check_boundaries=*/false);
        return results;
    }

    // List the bond dimensions from the left boundary through the right boundary.
    std::vector<int> compute_list_chi() const {
        if (ordering < 0 || ordering > 2)
            throw std::invalid_argument("TT::compute_list_chi: ordering must be 0, 1 or 2");
        std::vector<int> chi;
        chi.reserve(nBit + 1);
        if (core.empty()) {
            chi.push_back(1);
            return chi;
        }
        const auto& first = core.front();
        chi.push_back(static_cast<int>(ordering == 0 ? first.n_cols : first.n_rows));
        for (const auto& tensor : core)
            chi.push_back(static_cast<int>(ordering == 2 ? tensor.n_cols : tensor.n_slices));
        return chi;
    }

    // Number of values in each core: n_left * n_phys * n_right per core
    std::vector<Eigen::Index> compute_nb_value_core() const {
        std::vector<Eigen::Index> nb;
        nb.reserve(nBit);
        for (int i = 0; i < nBit; i++)
            nb.push_back(core[i].n_rows * core[i].n_cols * core[i].n_slices);
        return nb;
    }

    // Total number of values across all cores
    Eigen::Index compute_tot_nb_value() const {
        Eigen::Index tot = 0;
        for (int i = 0; i < nBit; i++)
            tot += core[i].n_rows * core[i].n_cols * core[i].n_slices;
        return tot;
    }

};

} // namespace TensorQuadOperation

