#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <type_double_double.h>

namespace fft_param_detail {

template <typename Real>
std::string real_to_string(const Real& value)
{
    std::ostringstream output;
    output << std::scientific
           << std::setprecision(std::numeric_limits<Real>::max_digits10)
           << value;
    return output.str();
}

template <typename Real>
Real real_from_json(const nlohmann::json& value)
{
    if (!value.is_string())
        return Real(value.get<double>());

    Real result;
    std::istringstream input(value.get<std::string>());
    input >> result;
    if (!input)
        throw std::runtime_error("fft_param: invalid real value in JSON");
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error(
            "fft_param: trailing characters in real JSON value");
    return result;
}

inline nlohmann::json encode_dd128_exact(const dd_128& value)
{
    return {
        {"hi", real_to_string(value._hi())},
        {"lo", real_to_string(value._lo())}
    };
}

inline dd_128 decode_dd128_exact(const nlohmann::json& value)
{
    const double hi = real_from_json<double>(value.at("hi"));
    const double lo = real_from_json<double>(value.at("lo"));
    return dd_128(dd_real(hi, lo));
}

} // namespace fft_param_detail

// Parameters needed to configure FFTmps and select how an FFT is applied.
// Scalar may be either the FFT's complex scalar type or its real component type.
template <typename Scalar>
struct fft_param {
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    int padding_bit = 0;
    int max_chi = 200;
    Real reltol = Eigen::NumTraits<Real>::epsilon() * Real(100);
    std::string method = "zip-up";
    bool do_shift = true;

    fft_param() = default;

    fft_param(int padding_bit_, int max_chi_, Real reltol_,
              std::string method_ = "zip-up", bool do_shift_ = true)
        : padding_bit(padding_bit_),
          max_chi(max_chi_),
          reltol(std::move(reltol_)),
          method(std::move(method_)),
          do_shift(do_shift_)
    {}

    // Load parameters previously written by save().
    explicit fft_param(const std::string& path)
        : fft_param(load_json(path))
    {}

    // Save to exactly path. reltol is represented as a decimal string for all
    // real types; dd_128 additionally stores its two components exactly.
    void save(const std::string& path) const
    {
        nlohmann::json document = {
            {"padding_bit", padding_bit},
            {"max_chi", max_chi},
            {"reltol", fft_param_detail::real_to_string(reltol)},
            {"method", method},
            {"do_shift", do_shift}
        };

        if constexpr (std::is_same_v<Real, dd_128>) {
            document["dd128_exact"] = {
                {"reltol", fft_param_detail::encode_dd128_exact(reltol)}
            };
        }

        std::ofstream file(path);
        if (!file.is_open())
            throw std::runtime_error(
                "fft_param: cannot open '" + path + "' for writing");
        file << document.dump(4);
        if (!file)
            throw std::runtime_error(
                "fft_param: failed to write '" + path + "'");
    }

    static fft_param load(const std::string& path)
    {
        return fft_param(load_json(path));
    }

private:
    struct loaded_param {
        int padding_bit;
        int max_chi;
        Real reltol;
        std::string method;
        bool do_shift;
    };

    explicit fft_param(loaded_param loaded)
        : padding_bit(loaded.padding_bit),
          max_chi(loaded.max_chi),
          reltol(std::move(loaded.reltol)),
          method(std::move(loaded.method)),
          do_shift(loaded.do_shift)
    {}

    static loaded_param load_json(const std::string& path)
    {
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error(
                "fft_param: cannot open '" + path + "' for reading");

        nlohmann::json document;
        file >> document;

        Real loaded_reltol;
        if constexpr (std::is_same_v<Real, dd_128>) {
            if (document.contains("dd128_exact")) {
                loaded_reltol = fft_param_detail::decode_dd128_exact(
                    document.at("dd128_exact").at("reltol"));
            } else {
                loaded_reltol = fft_param_detail::real_from_json<Real>(
                    document.at("reltol"));
            }
        } else {
            loaded_reltol = fft_param_detail::real_from_json<Real>(
                document.at("reltol"));
        }

        return {
            document.at("padding_bit").get<int>(),
            document.at("max_chi").get<int>(),
            std::move(loaded_reltol),
            document.at("method").get<std::string>(),
            document.at("do_shift").get<bool>()
        };
    }
};
