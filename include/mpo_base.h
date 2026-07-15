#pragma once

#include "tt_base.h"
#include "mps_base.h"

// Matrix Product Operator: a TT whose physical leg is the fused
// (row x column) operator index — n_phys = 4 for qubits, i.e. cores
// of shape (chi, 4, chi'), with slot ordering
//   0 : (0,0)   1 : (0,1)   2 : (1,0)   3 : (1,1)

template<typename T>
class MPO : public TT<T> {
    using Base       = TT<T>;
    using VecTT      = std::vector<Tensor3D<T>>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

public:
    using Base::Base;                    // inherit both TT constructors

private:

    void _check_mpo_phys() const {
        for (auto const& x : this->core)
            if (x.n_phys != 4)
                throw std::invalid_argument(
                    "MPO: expected physical dimension 4 on every core");
    }

public:

    MPO(VecTT core_, int max_bond_dim_ = 0, RealScalar reltol_ = -1, int w_ = 0)
        : Base(std::move(core_), max_bond_dim_, reltol_, w_)
    { _check_mpo_phys(); }

    MPO(const std::string& filename, int max_bond_dim_ = 0,
        RealScalar reltol_ = -1, int w_ = 0)
        : Base(filename, max_bond_dim_, reltol_, w_)
    { _check_mpo_phys(); }

    // Promote a TT (e.g. the result of mpo1 + mpo2) back into an MPO.
    MPO(Base base) : Base(std::move(base)) {_check_mpo_phys();}

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
    // The working site w and the compression parameters are carried
    // over from the MPS. Copying w is legitimate: padding zeros along
    // the physical leg preserves both canonical conditions
    // (Q^H Q = I gains only zero rows; M M^H = I gains only zero
    // columns), so the result is canonical at the same site.
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
 
        // w_ = -1: skip the constructor's QR sweeps, then set w
        // directly — the permuted cores are already canonical.
        MPO res(Base(std::move(c),
                     this->get_max_bond_dim(),
                     this->get_reltol(),
                     /*w_=*/-1));
        res.w = this->get_w();
        return res;
    }

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

};