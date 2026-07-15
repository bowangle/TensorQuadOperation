#pragma once

#include "tt_base.h"
#include "mps_base.h"
#include "utc.h"

#include <algorithm>
#include <string>
#include <type_traits>

// Matrix Product Operator: a TT whose physical leg is the fused
// (row x column) operator index — n_phys = 4 for qubits, i.e. cores
// of shape (chi, 4, chi'), with slot ordering
//   0 : (0,0)   1 : (0,1)   2 : (1,0)   3 : (1,1)
template<typename T>
class MPO : public TT<T> {
    using Base       = TT<T>;
    using VecTT      = std::vector<Tensor3D<T>>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

    void _check_mpo_phys() const {
        for (auto const& x : this->core)
            if (x.n_phys != 4)
                throw std::invalid_argument(
                    "MPO: expected physical dimension 4 on every core");
    }

public:
    // Forwarding constructors (instead of `using Base::Base;`) so that
    // EVERY construction path runs the n_phys == 4 validation.
    MPO(VecTT core_, int max_bond_dim_ = 0, RealScalar reltol_ = -1, int w_ = 0)
        : Base(std::move(core_), max_bond_dim_, reltol_, w_)
    { _check_mpo_phys(); }

    MPO(const std::string& filename, int max_bond_dim_ = 0,
        RealScalar reltol_ = -1, int w_ = 0)
        : Base(filename, max_bond_dim_, reltol_, w_)
    { _check_mpo_phys(); }

    // Promote a TT (e.g. the result of mpo1 + mpo2) back into an MPO.
    MPO(Base base) : Base(std::move(base))
    { _check_mpo_phys(); }

    // =============================================================
    //                 MPO from MPS (diagonal operator)
    // =============================================================
    // Each MPS core x of shape (chi, 2, chi') becomes an MPO core of
    // shape (chi, 4, chi') with
    //     t[:, 0, :] = x[:, 0, :]      // (0,0) diagonal entry
    //     t[:, 3, :] = x[:, 1, :]      // (1,1) diagonal entry
    // and the off-diagonal slots 1, 2 left at zero, i.e. the MPO
    // representing diag(psi) in the computational basis.
    //
    // Copying w is legitimate: padding zeros along the physical leg
    // preserves both canonical conditions, so the result is canonical
    // at the same site.
    static MPO from_mps(MPS<T> const& mps)
    {
        auto const& src = mps.get_core();

        VecTT core;
        core.reserve(src.size());

        for (auto const& x : src) {
            if (x.n_phys != 2)
                throw std::invalid_argument("MPO::from_mps: expected MPS physical dimension 2");

            Tensor3D<T> t(x.n_left, 4, x.n_right);   // zero-initialized
            t.phys(0) = x.phys_const(0);
            t.phys(3) = x.phys_const(1);
            core.push_back(std::move(t));
        }

        // Build with w_ = -1 so the constructor does NOT re-canonicalize,
        // then set w directly: the cores are already canonical (see above).
        MPO res(Base(std::move(core),
                     mps.get_max_bond_dim(),
                     mps.get_reltol(),
                     /*w_=*/-1));
        res.w = mps.get_w();
        return res;
    }

    // =============================================================
    //                 MPO transpose  (Python .T)
    // =============================================================
    // With slot ordering p = 2*row + col, transposition is the
    // permutation 1 <-> 2 of the physical leg. Copying w is safe:
    // permuting physical slices only permutes rows/columns of the
    // flattened cores, which preserves both canonical conditions.
    MPO transpose() const
    {
        VecTT c;
        c.reserve(this->core.size());

        for (auto const& x : this->core) {
            Tensor3D<T> t(x.n_left, 4, x.n_right);
            t.phys(0) = x.phys_const(0);
            t.phys(1) = x.phys_const(2);   // new (0,1)  <-  old (1,0)
            t.phys(2) = x.phys_const(1);   // new (1,0)  <-  old (0,1)
            t.phys(3) = x.phys_const(3);
            c.push_back(std::move(t));
        }

        MPO res(Base(std::move(c),
                     this->get_max_bond_dim(),
                     this->get_reltol(),
                     /*w_=*/-1));
        res.w = this->get_w();
        return res;
    }

    // =============================================================
    //                 MPO diagonal  (Python .diag)
    // =============================================================
    // Keep slots 0 and 3 of the physical leg. Slicing does NOT
    // preserve canonicity, so w is passed through the TT constructor,
    // which re-canonicalizes at that site via QR sweeps.
    MPS<T> diag() const
    {
        VecTT c;
        c.reserve(this->core.size());

        for (auto const& x : this->core) {
            Tensor3D<T> t(x.n_left, 2, x.n_right);
            t.phys(0) = x.phys_const(0);   // (0,0) diagonal entry
            t.phys(1) = x.phys_const(3);   // (1,1) diagonal entry
            c.push_back(std::move(t));
        }

        return MPS<T>(Base(std::move(c),
                           this->get_max_bond_dim(),
                           this->get_reltol(),
                           /*w_=*/this->get_w()));
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

        // Initial guess for "optimize", captured BEFORE any shifting
        MPS<T> prev = (previous != nullptr) ? *previous : other;

        int w0_mps = other.get_w();

        // Work on a copy of `other`, move canonical center
        MPS<T> res(other);
        this->shift_w(0);
        res.shift_w(0);

        // Looser of the two tolerances
        RealScalar reltol   = std::max(this->get_reltol(), other.get_reltol());
        int max_bond_dim    = _merged_max_bond_dim(other);

        VecTT core;
        if (method == "zip-up")
            core = utc::zip_up_mpo_mps<T>(this->core, res.core,
                                          reltol, max_bond_dim);
        else if (method == "qrsvd")
            core = utc::qrsvd_contract_mpo_mps<T>(this->core, res.core);
        else if (method == "optimize")
            core = utc::optimize_dm_generic<T>(prev.get_core(),
                                               this->core, res.core,
                                               reltol, max_bond_dim,
                                               order, n_sweeps);
        else
            throw std::invalid_argument("MPO::_mul: method not recognized");

        // res
        res.core = std::move(core);
        res.nBit = static_cast<int>(res.core.size());
        res.w    = (method != "optimize") ? res.nBit - 1 : 0;

        if (w0_mps != -1)
            res.shift_w(w0_mps, /*compress=*/true);
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

    //     res.core = utc::zip_up_mpo_mpo<T>(this->core, res.core,
    //                                       reltol, max_bond_dim);
    //     res.nBit = static_cast<int>(res.core.size());
    //     res.w    = res.nBit - 1;

    //     if (w0 != -1)
    //         res.shift_w(w0);
    //     return res;
    // }
};