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

    // Promote a TT (e.g. the result of mpo1 + mpo2) back into an MPO.
    MPO(Base base) : Base(std::move(base)) {}

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
};