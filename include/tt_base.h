#pragma once

#include <vector>
#include <iostream>
#include "tensor.h"

template<typename T>
class TT{
    using VecTT = std::vector<Tensor3D<T>>;

    private:
    int nBit;                   // number of tensor
    VecTT core;  // list of tensor
    int w;                      // working tensor, -1 for no defined (None)

    public:
    TT(VecTT core_){
        core = core_;
        nBit = core_.size();
        w = -1;
    }

    TT(int nBit_){

    }

    int get_size(){
        return nBit;
    }

    VecTT get_core(){
        return core;
    }
};