#include "tensor.h"
#include <iostream>
#include <complex>



int main()
{
    auto tt = load_vector_tensor<Cfloat128>("gE<_site[[1, 0], [0, 0]].tt");
    std::cout << std::scientific;
    std::cout << std::setprecision(std::numeric_limits<float128>::digits10 + 5);
    std::cout << tt[1];

    std::cout <<"1.0:   " << tt[1].right(0) << "\n";

    std::cout <<"1.1:   " << tt[1].right(1);

    return 0;
}