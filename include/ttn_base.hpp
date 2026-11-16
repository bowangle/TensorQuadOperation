#pragma once

#include <xfac_quad/cubemat_helper.h>
#include <xfac_quad/tree/tree.hpp>

#include <vector>

template<typename T>
class TTN {
public:
    using VecTT = std::vector<xfac_quad::Tensor3D<T>>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

protected:
    TensorTree ttn_body;
    int w = -1;                         // No defined working tensor.
    int max_bond_dim = 0;                // No bond-dimension cap.
    RealScalar reltol = RealScalar(-1);  // No relative-tolerance truncation.
};
