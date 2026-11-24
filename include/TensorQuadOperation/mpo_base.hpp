#pragma once

#include "TensorQuadOperation/tt_base.hpp"
#include "TensorQuadOperation/mps_base.hpp"
#include "TensorQuadOperation/utc.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <type_traits>

namespace TensorQuadOperation {

// Matrix Product Operator with a fused physical leg of size left_phys * right_phys.
// The pair (row, column) occupies slot row * right_phys + column.
// The TT ordering determines where the physical axis is stored in each core.
// The first core's left bond and the last core's right bond may both be dangling.
template<typename T>
class MPO : public TT<T> {
    using Base       = TT<T>;
    using VecTT      = std::vector<xfac_quad::Tensor3D<T>>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

    int left_phys;
    int right_phys;

public:
    int get_left_phys() const { return left_phys; }
    int get_right_phys() const { return right_phys; }

    void _check_mpo_phys() const {
        if (left_phys <= 0 || right_phys <= 0)
            throw std::invalid_argument("MPO: left_phys and right_phys must be positive");
        const int physical_axis = this->ordering;
        if (physical_axis < 0 || physical_axis > 2)
            throw std::invalid_argument("MPO: ordering must be 0, 1 or 2");
        const auto fused_phys = static_cast<Eigen::Index>(left_phys) * right_phys;
        for (const auto& x : this->core) {
            const std::array<Eigen::Index, 3> shape{x.n_rows, x.n_cols, x.n_slices};
            if (shape[physical_axis] != fused_phys)
                throw std::invalid_argument(
                    "MPO: expected physical dimension left_phys * right_phys on every core");
        }
    }

public:
    // Forwarding constructors (instead of `using Base::Base;`) so that
    // EVERY construction path validates the fused physical dimension.
    MPO(VecTT core_, int max_bond_dim_ = 0, RealScalar reltol_ = -1, int w_ = 0,
        int left_phys_ = 2, int right_phys_ = 2, int ordering_ = 1)
        : Base(std::move(core_), max_bond_dim_, reltol_, w_, ordering_),
          left_phys(left_phys_), right_phys(right_phys_)
    { _check_mpo_phys(); }

    MPO(const std::string& filename, int max_bond_dim_ = 0,
        RealScalar reltol_ = -1, int w_ = 0, int left_phys_ = 2, int right_phys_ = 2,
        int ordering_ = 1)
        : Base(filename, max_bond_dim_, reltol_, w_, ordering_),
          left_phys(left_phys_), right_phys(right_phys_)
    { _check_mpo_phys(); }

    // Promote a TT (e.g. the result of mpo1 + mpo2) back into an MPO.
    MPO(Base base, int left_phys_ = 2, int right_phys_ = 2)
        : Base(std::move(base)), left_phys(left_phys_), right_phys(right_phys_)
    { _check_mpo_phys(); }

    // Reinterpret an MPS as an MPO in its existing storage ordering.
    MPO(MPS<T> mps, int left_phys_ = 2, int right_phys_ = 2)
        : MPO(Base(std::move(mps)), left_phys_, right_phys_)
    {}

    // =============================================================
    //                 MPO from MPS (diagonal operator)
    // =============================================================
    // An MPS with physical dimension d becomes a diagonal d × d MPO.
    // Physical entry p is stored in fused slot p*d + p. Off-diagonal slots are zero.
    // The physical axis stays in the same storage position.
    //
    // Copying w is legitimate: padding zeros along the physical leg
    // preserves both canonical conditions, so the result is canonical
    // at the same site.
    // See an MPS as a diagonal MPO (put a delta on the leg)
    static MPO from_mps(MPS<T> const& mps)
    {
        auto const& src = mps.get_core();
        const int physical_axis = mps.get_ordering();
        if (physical_axis < 0 || physical_axis > 2)
            throw std::invalid_argument("MPO::from_mps: ordering must be 0, 1 or 2");

        Eigen::Index physical_dim = 2;
        if (!src.empty()) {
            const auto& first = src.front();
            const std::array<Eigen::Index, 3> shape{first.n_rows, first.n_cols, first.n_slices};
            physical_dim = shape[physical_axis];
        }
        if (physical_dim <= 0 || physical_dim > std::numeric_limits<int>::max() ||
            physical_dim > std::numeric_limits<Eigen::Index>::max() / physical_dim)
            throw std::invalid_argument("MPO::from_mps: invalid physical dimension for a diagonal MPO");
        const Eigen::Index fused_dim = physical_dim * physical_dim;

        VecTT core;
        core.reserve(src.size());

        for (auto const& x : src) {
            std::array<Eigen::Index, 3> shape{x.n_rows, x.n_cols, x.n_slices};
            if (shape[physical_axis] != physical_dim)
                throw std::invalid_argument("MPO::from_mps: expected the same physical dimension on every MPS core");

            shape[physical_axis] = fused_dim;
            xfac_quad::Tensor3D<T> t(shape[0], shape[1], shape[2]); // zero-initialized
            for (Eigen::Index p = 0; p < physical_dim; ++p) {
                const Eigen::Index slot = p * physical_dim + p;
                if (physical_axis == 0)
                    t.row_as_mat(slot) = x.row_as_mat_const(p);
                else if (physical_axis == 1)
                    t.col_as_mat(slot) = x.col_as_mat_const(p);
                else
                    t.slice(slot) = x.slice_const(p);
            }
            core.push_back(std::move(t));
        }

        // Build with w_ = -1 so the constructor does NOT re-canonicalize,
        // then set w directly: the cores are already canonical (see above).
        const int dim = static_cast<int>(physical_dim);
        MPO res(std::move(core), mps.get_max_bond_dim(), mps.get_reltol(), /*w_=*/-1,
                dim, dim, physical_axis);
        res.w = mps.get_w();
        return res;
    }

    // =============================================================
    //                 MPO transpose  (Python .T)
    // =============================================================
    // Move entry (row, col) from row*right_phys + col to col*left_phys + row.
    // The physical-axis permutation preserves both canonical conditions, so
    // the transposed MPO can keep the same orthogonality center.
    MPO transpose() const
    {
        VecTT c;
        c.reserve(this->core.size());
        const int physical_axis = this->ordering;

        for (auto const& x : this->core) {
            xfac_quad::Tensor3D<T> t(x.n_rows, x.n_cols, x.n_slices);
            for (Eigen::Index row = 0; row < left_phys; ++row) {
                for (Eigen::Index col = 0; col < right_phys; ++col) {
                    const Eigen::Index source = row * right_phys + col;
                    const Eigen::Index target = col * left_phys + row;
                    if (physical_axis == 0)
                        t.row_as_mat(target) = x.row_as_mat_const(source);
                    else if (physical_axis == 1)
                        t.col_as_mat(target) = x.col_as_mat_const(source);
                    else
                        t.slice(target) = x.slice_const(source);
                }
            }
            c.push_back(std::move(t));
        }

        MPO res(std::move(c), this->get_max_bond_dim(), this->get_reltol(), /*w_=*/-1,
                right_phys, left_phys, physical_axis);
        res.w = this->get_w();
        return res;
    }

    // =============================================================
    //                 MPO diagonal  (Python .diag)
    // =============================================================
    // Keep fused slots p*right_phys + p for p < min(left_phys, right_phys).
    // Selecting the diagonal can change canonical form. The TT constructor
    // uses QR sweeps to restore the same center when w is defined.
    MPS<T> diag() const
    {
        VecTT c;
        c.reserve(this->core.size());
        const int physical_axis = this->ordering;
        const int diagonal_size = std::min(left_phys, right_phys);

        for (auto const& x : this->core) {
            std::array<Eigen::Index, 3> shape{x.n_rows, x.n_cols, x.n_slices};
            shape[physical_axis] = diagonal_size;
            xfac_quad::Tensor3D<T> t(shape[0], shape[1], shape[2]);
            for (Eigen::Index p = 0; p < diagonal_size; ++p) {
                const Eigen::Index slot = p * right_phys + p;
                if (physical_axis == 0)
                    t.row_as_mat(p) = x.row_as_mat_const(slot);
                else if (physical_axis == 1)
                    t.col_as_mat(p) = x.col_as_mat_const(slot);
                else
                    t.slice(p) = x.slice_const(slot);
            }
            c.push_back(std::move(t));
        }

        return MPS<T>(std::move(c), this->get_max_bond_dim(), this->get_reltol(),
                      this->get_w(), physical_axis);
    }

    // =============================================================
    //              MPO application  (Python ._mul)
    // =============================================================
    // max_bond_dim merging: the tighter of the two caps; a
    // non-positive value means "no cap" and defers to the other
    // operand.
private:
    int _merged_max_bond_dim(Base const& other) const
    {
        int a = this->get_max_bond_dim();
        int b = other.get_max_bond_dim();
        if (a <= 0) return b;
        if (b <= 0) return a;
        return std::min(a, b);
    }

public:
    // ---------------- MPO @ MPS -> MPS ----------------
    template<typename U>
    MPS<T> _mul(MPS<U> const& other,
                std::string const& method = "optimize",
                int order = 2,
                int n_sweeps = 2,
                MPS<U> const* previous = nullptr)
    {
        // Scalar types must match
        static_assert(std::is_same_v<T, U>,
                      "MPO::_mul: the scalar type T of the MPO and the MPS must be identical");

        // keep in mind the old position of the canonical center
        int w0_mps = other.get_w();

        RealScalar reltol   = std::max(this->get_reltol(), other.get_reltol());
        int max_bond_dim    = _merged_max_bond_dim(other);
        VecTT core_out;
        {
            // Work on a copy of `other` because we move the canonical center to 0.
            MPS<T> other_mps(other);
            this->shift_w(0);
            other_mps.shift_w(0);

            // Looser of the two tolerances
            
            if (method == "zip-up")
                core_out = utc::zip_up::zip_up_mpo_mps<T>(*this, other_mps, reltol, max_bond_dim);
            else if (method == "qrsvd")
                core_out = utc::qrsvd::qrsvd_contract_mpo_mps<T>(*this, other_mps);
            else if (method == "optimize") {
                if (previous != nullptr)
                    core_out = utc::optimized::optimize_dm_generic<T>(
                        *previous, *this, other_mps, reltol, max_bond_dim, order, n_sweeps);
                else
                    core_out = utc::optimized::optimize_dm_generic<T>(
                        *this, other_mps, reltol, max_bond_dim, order, n_sweeps);
            } else
                throw std::invalid_argument("MPO::_mul: method not recognized");
        }
        // At this point we have what we wanted core_out and we can discard all the rest
        int nBit_ = static_cast<int>(core_out.size());
        int w_ = (method != "optimize") ? nBit_ - 1 : 0;
        MPS<T> res(std::move(core_out), max_bond_dim, reltol, w_, this->get_ordering());

        // qrsvd's forward contraction is exact QR. Sweep all the way
        // back from the last site to site 0 with SVD compression so
        // reltol / max_bond_dim are applied to every internal bond,
        // including when the requested center is already the last site.
        // Restore the center afterward with exact QR moves to avoid a
        // second truncation of any bond.
        if (method == "qrsvd") {
            res.shift_w(0, /*compress=*/true);
            int target_w = (w0_mps == -1) ? nBit_ - 1 : w0_mps;
            if (target_w != 0)
                res.shift_w(target_w, /*compress=*/false);
        }
        else if (w0_mps != -1) {
            res.shift_w(w0_mps, /*compress=*/true);
        }

        return res;
    }

    // ---------------- MPO @ MPO -> MPO ----------------
    //
    // Always zip-up
    // as above: *this is left canonicalized at site 0.
    // template<typename U>
    // MPO _mul(MPO<U> const& other)
    // {
    //     static_assert(std::is_same_v<T, U>,
    //                   "MPO::_mul: the scalar type T of the two MPOs must be identical");

    //     int w0 = other.get_w();

    //     MPO res(other);
    //     this->shift_w(0);
    //     res.shift_w(0);

    //     RealScalar reltol   = std::max(this->get_reltol(), other.get_reltol());
    //     int max_bond_dim    = _merged_max_bond_dim(other);

    //     res.core = utc::zip_up::zip_up_mpo_mpo<T>(this->core, res.core,
    //                                       reltol, max_bond_dim);
    //     res.nBit = static_cast<int>(res.core.size());
    //     res.w    = res.nBit - 1;

    //     if (w0 != -1)
    //         res.shift_w(w0);
    //     return res;
    // }
};

} // namespace TensorQuadOperation
