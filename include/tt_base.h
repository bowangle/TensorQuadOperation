#pragma once

#include <vector>
#include <iostream>
#include "tensor.h"
#include "mat_decomp.h"

template<typename T>
class TT{
    using VecTT = std::vector<Tensor3D<T>>;

    private:
    int nBit;                   // number of tensor
    VecTT core;  // list of tensor
    int w;                      // working tensor, -1 for no defined (None)

    public:
    // =============================================================
    //                      TT constructor:
    // =============================================================
    // from core
    TT(VecTT core_){
        core = core_;
        nBit = core_.size();
        w = -1;
    }

    // from file
    TT(const std::string& filename)
    {
        std::ifstream in(filename);

        if (!in)
            throw std::runtime_error("Cannot open file: " + filename);

        using RealT = typename T::value_type;
        Eigen::Index nBit;
        in >> nBit;

        std::vector<Tensor3D<T>> Xs;
        Xs.reserve(nBit);

        for (Eigen::Index t = 0; t < nBit; ++t)
        {
            std::string header;
            in >> header;

            if (header != "ARMA_CUB_TXT_FC016")
                throw std::runtime_error("Bad header");

            Eigen::Index r, c, s;
            in >> r >> c >> s;
            Tensor3D<T> X(r, c, s);

            for (Eigen::Index id_left = 0; id_left < X.n_left; ++id_left)
            {
                for (Eigen::Index id_phys = 0; id_phys < X.n_phys; ++id_phys)
                {
                    for (Eigen::Index id_right = 0; id_right < X.n_right; ++id_right)
                    {
                        std::string token;
                        in >> token; // "(re,im)"

                        if (token.size() < 5)
                            throw std::runtime_error("Bad complex token");

                        // strip '(' and ')'
                        if (token.front() == '(') token.erase(token.begin());
                        if (token.back()  == ')') token.pop_back();

                        auto comma = token.find(',');
                        if (comma == std::string::npos)
                            throw std::runtime_error("Missing comma in complex");

                        RealT re(token.substr(0, comma));
                        RealT im(token.substr(comma + 1));

                        X(id_left, id_phys, id_right) = T(re, im);
                    }
                }
            }

            Xs.push_back(std::move(X));
        }
        core = Xs;
        nBit = Xs.size();
        w = -1;
    }

    // =============================================================
    //                      TT access:
    // =============================================================

    int get_size(){
        return nBit;
    }

    VecTT get_core(){
        return core;
    }

    // =============================================================
    //                      TT Canonicalisation:
    // =============================================================

    void left_to_right(){
        
    }

    // =============================================================
    //                      TT method:
    // =============================================================

    void compressSVD(T reltol, int max_bond_dim){
        // TODO use QR to make the whole compress SVD
    }

};