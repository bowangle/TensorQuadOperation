#pragma once

#include "tt_base.h"

template<typename T>
class MPS : public TT<T> {
    using Base       = TT<T>;
    using VecTT      = std::vector<Tensor3D<T>>;
    using MatrixX    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

public:
    using Base::Base;                       // inherit both TT constructors

    // Allow promoting a TT (e.g. the result of mps1 + mps2, which
    // returns TT) back into an MPS.
    MPS(Base base) : Base(std::move(base)) {}

    // =============================================================
    //            MPS-MPS contraction  (Python __matmul__)
    // =============================================================
    // dot(other) computes  <this|other> :
    // this's cores are conjugated (matching jnp.conjugate(mps_a[k])
    // in the Python), other's are not.
    //
    // Instead of the Python's growing tensordot chain, we carry a
    // chi_a x chi_b "environment" matrix E across the train:
    //
    //   E_0(ra, rb)  = sum_{l,p} conj(A_0(l,p,ra)) B_0(l,p,rb)
    //   E_k          = sum_p  A_k.phys(p)^H * E_{k-1} * B_k.phys(p)
    //
    // which is the same contraction in a different order, at
    // O(chi^3) per site.

    // Full environment: returns the (ra_last x rb_last) matrix.
    // With trivial boundaries this is 1x1; if you later add extra
    // legs on the right boundary, this is the object you want.
    MatrixX contract_env(MPS const& other) const
    {
        auto const& a = this->core;
        auto const& b = other.core;

        if (a.size() != b.size())
            throw std::invalid_argument("MPS::contract_env: trying to contract MPSs of different lengths");
        if (a[0].n_left != 1 || b[0].n_left != 1)
            throw std::invalid_argument("MPS::contract_env: non-trivial left boundary legs");

        // First site: E = A^H * B over the merged (left*phys) index.
        if (a[0].n_phys != b[0].n_phys)
            throw std::invalid_argument("MPS::contract_env: physical dimensions differ at site 0");
        MatrixX E = a[0].flatten_as_matrix2_const().adjoint()
                  * b[0].flatten_as_matrix2_const();          // (ra_0 x rb_0)

        // Sweep: E <- sum_p A_k.phys(p)^H * E * B_k.phys(p)
        for (std::size_t k = 1; k < a.size(); k++) {
            if (a[k].n_phys != b[k].n_phys)
                throw std::invalid_argument("MPS::contract_env: physical dimensions differ");

            MatrixX Enew = MatrixX::Zero(a[k].n_right, b[k].n_right);
            for (Eigen::Index p = 0; p < a[k].n_phys; p++)
                Enew.noalias() += a[k].phys_const(p).adjoint()
                                * (E * b[k].phys_const(p));
            E = std::move(Enew);
        }
        return E;
    }

    // MPS MPS dot product. Connect all the leg and contract
    T dot(MPS const& other) const
    {
        MatrixX E = contract_env(other);
        if (E.rows() != 1 || E.cols() != 1)
            throw std::invalid_argument("MPS::dot: non-trivial right boundary legs, use contract_env instead");
        return E(0, 0);
    }

    // Convenience: norm^2 = <this|this> (real and >= 0 up to rounding)
    RealScalar norm2() const
    {
        return Eigen::numext::real(dot(*this));
    }
};