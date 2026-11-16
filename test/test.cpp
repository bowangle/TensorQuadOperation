#include "type_double_double.h"
#include "type_float128_boost.h"
#include <xfac_quad/cubemat_helper.h>
#include <iostream>
#include <complex>

int main()
{
    {
        std::cout << "\nCase Cfloat128: \n\n";
        auto tt = xfac_quad::load_vector_tensor<Cfloat128>("gE<_site[[1, 0], [0, 0]].tt");
        std::cout << std::scientific;
        std::cout << std::setprecision(std::numeric_limits<float128>::digits10 + 5);
        std::cout << tt[1];

        std::cout <<"1.0:   " << tt[1].slice(0) << "\n";

        std::cout <<"1.1:   " << tt[1].slice(1);
    }

    {
        std::cout << "\nCase Cdd_128: \n\n";
        auto tt = xfac_quad::load_vector_tensor<Cdd_128>("gE<_site[[1, 0], [0, 0]].tt");
        std::cout << std::scientific;
        std::cout << std::setprecision(std::numeric_limits<dd_128>::digits10 + 5);
        std::cout << tt[1];

        std::cout <<"1.0:   " << tt[1].slice(0) << "\n";

        std::cout <<"1.1:   " << tt[1].slice(1);
    }

    {
        std::cout << "\nCase Cdouble: \n\n";
        auto tt = xfac_quad::load_vector_tensor<std::complex<double>>("gE<_site[[1, 0], [0, 0]].tt");
        std::cout << std::scientific;
        std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 5);
        std::cout << tt[1];

        std::cout <<"1.0:   " << tt[1].slice(0) << "\n";

        std::cout <<"1.1:   " << tt[1].slice(1);
    }

    return 0;
}