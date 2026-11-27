#include "fft_param.hpp"
#include "type_float128_boost.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>

namespace {

template <typename Real>
bool exactly_equal(const Real& lhs, const Real& rhs)
{
    if constexpr (std::is_same_v<Real, dd_128>)
        return lhs.x[0] == rhs.x[0] && lhs.x[1] == rhs.x[1];
    else
        return lhs == rhs;
}

template <typename Real>
bool test_roundtrip(const std::string& type_name)
{
    const std::filesystem::path path =
        "fft_param_roundtrip_" + type_name + ".json";
    const Real reltol = Eigen::NumTraits<Real>::epsilon() * Real(137);
    const fft_param<Real> saved(9, 321, reltol, "fit", false);

    saved.save(path.string());
    const fft_param<Real> loaded(path.string());
    const fft_param<Real> statically_loaded =
        fft_param<Real>::load(path.string());
    std::filesystem::remove(path);

    const auto matches = [&](const fft_param<Real>& value) {
        return value.padding_bit == saved.padding_bit
            && value.max_chi == saved.max_chi
            && exactly_equal(value.reltol, saved.reltol)
            && value.method == saved.method
            && value.do_shift == saved.do_shift;
    };

    if (!matches(loaded) || !matches(statically_loaded)) {
        std::cerr << "fft_param<" << type_name
                  << "> did not roundtrip exactly\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    const bool passed = test_roundtrip<double>("double")
                     && test_roundtrip<dd_128>("dd_128")
                     && test_roundtrip<float128>("float128");
    return passed ? 0 : 1;
}
