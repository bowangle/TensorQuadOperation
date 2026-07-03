#pragma once

#include "tt_base.h"

// Matrix Product Operator: a TT whose physical leg is the fused
// (row ⊗ column) operator index — conventionally n_phys = 4 for
// qubits (2 x 2), i.e. cores of shape (chi, 4, chi').
template<typename T>
class MPO : public TT<T> {
    using Base = TT<T>;

public:
    using Base::Base;                    // inherit both TT constructors

    // Promote a TT (e.g. the result of mpo1 + mpo2) back into an MPO.
    MPO(Base base) : Base(std::move(base)) {}
};