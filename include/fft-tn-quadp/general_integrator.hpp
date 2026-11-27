#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "type_double_double.h"
#include "type_float128_boost.h"

// Globally adaptive, finite-interval Gauss-Kronrod integration.  The public
// interface is intentionally similar to scipy.integrate.quad: `limit` bounds
// the number of active subintervals and `points` supplies known breakpoints.
namespace general_integrator {

namespace detail {

template<class>
inline constexpr bool dependent_false = false;

template<class T>
struct is_std_complex : std::false_type {
    using real_type = void;
};

template<class T>
struct is_std_complex<std::complex<T>> : std::true_type {
    using real_type = T;
};

template<class T>
inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

template<class Real>
inline constexpr bool is_supported_real_v =
    std::is_same_v<Real, double> ||
    std::is_same_v<Real, dd_128> ||
    std::is_same_v<Real, float128>;

template<class Real>
Real abs_value(const Real& value)
{
    using std::abs;
    return Real(abs(value));
}

template<class Real>
Real sqrt_value(const Real& value)
{
    using std::sqrt;
    return Real(sqrt(value));
}

// Construct quadrature constants without first rounding them to double.
template<class Real>
Real decimal(const char* text, double double_value)
{
    if constexpr (std::is_same_v<Real, double>) {
        return double_value;
    } else if constexpr (std::is_same_v<Real, dd_128>) {
        return dd_128(dd_real(text));
    } else if constexpr (std::is_same_v<Real, float128>) {
        return float128(text);
    } else {
        static_assert(dependent_false<Real>,
                      "general_integrator supports double, dd_128, and float128");
    }
}

template<class Real>
const std::array<Real, 11>& kronrod_abscissas()
{
    // Positive abscissas for the 21-point Kronrod rule.  Entry zero is the
    // centre.  Odd entries are the embedded 10-point Gauss abscissas.
    static const std::array<Real, 11> values = {
        decimal<Real>("0", 0.0),
        decimal<Real>("1.4887433898163121088482600112971998462e-1", 1.48874338981631211e-1),
        decimal<Real>("2.9439286270146019813112660310386556616e-1", 2.94392862701460198e-1),
        decimal<Real>("4.3339539412924719079926594316578416220e-1", 4.33395394129247191e-1),
        decimal<Real>("5.6275713466860468333900009927269414084e-1", 5.62757134668604683e-1),
        decimal<Real>("6.7940956829902440623432736511487357577e-1", 6.79409568299024406e-1),
        decimal<Real>("7.8081772658641689706371757834504237716e-1", 7.80817726586416897e-1),
        decimal<Real>("8.6506336668898451073209668842349304853e-1", 8.65063366688984511e-1),
        decimal<Real>("9.3015749135570822600120718005950834623e-1", 9.30157491355708226e-1),
        decimal<Real>("9.7390652851717172007796401208445205343e-1", 9.73906528517171720e-1),
        decimal<Real>("9.9565716302580808073552728068900284792e-1", 9.95657163025808081e-1)
    };
    return values;
}

template<class Real>
const std::array<Real, 11>& kronrod_weights()
{
    static const std::array<Real, 11> values = {
        decimal<Real>("1.4944555400291690566493646838982120375e-1", 1.49445554002916906e-1),
        decimal<Real>("1.4773910490133849137484151597206804552e-1", 1.47739104901338491e-1),
        decimal<Real>("1.4277593857706008079709427313871706089e-1", 1.42775938577060081e-1),
        decimal<Real>("1.3470921731147332592805400177170683276e-1", 1.34709217311473326e-1),
        decimal<Real>("1.2349197626206585107795810983107415951e-1", 1.23491976262065851e-1),
        decimal<Real>("1.0938715880229764189921059032580496027e-1", 1.09387158802297642e-1),
        decimal<Real>("9.3125454583697605535065465083366344390e-2", 9.31254545836976055e-2),
        decimal<Real>("7.5039674810919952767043140916190009395e-2", 7.50396748109199528e-2),
        decimal<Real>("5.4755896574351996031381300244580176374e-2", 5.47558965743519960e-2),
        decimal<Real>("3.2558162307964727478818972459389760617e-2", 3.25581623079647275e-2),
        decimal<Real>("1.1694638867371874278064396062192048396e-2", 1.16946388673718743e-2)
    };
    return values;
}

template<class Real>
const std::array<Real, 5>& gauss_weights()
{
    static const std::array<Real, 5> values = {
        decimal<Real>("2.9552422471475287017389299465133832942e-1", 2.95524224714752870e-1),
        decimal<Real>("2.6926671930999635509122692156946935286e-1", 2.69266719309996355e-1),
        decimal<Real>("2.1908636251598204399553493422816319246e-1", 2.19086362515982044e-1),
        decimal<Real>("1.4945134915058059314577633965769733240e-1", 1.49451349150580593e-1),
        decimal<Real>("6.6671344308688137593568809893331792858e-2", 6.66713443086881376e-2)
    };
    return values;
}

template<class Real>
struct interval_estimate {
    Real a{};
    Real b{};
    Real value{};
    Real error{};
};

template<class Real, class Function>
interval_estimate<Real> evaluate_interval(Function& function, Real a, Real b)
{
    const auto& xgk = kronrod_abscissas<Real>();
    const auto& wgk = kronrod_weights<Real>();
    const auto& wg = gauss_weights<Real>();

    const Real centre = (a + b) / Real(2);
    const Real half_length = (b - a) / Real(2);
    const Real abs_half_length = abs_value(half_length);

    const Real centre_value = Real(std::invoke(function, centre));
    std::array<Real, 10> lower_values{};
    std::array<Real, 10> upper_values{};

    Real kronrod = wgk[0] * centre_value;
    Real gauss{};
    Real result_abs = wgk[0] * abs_value(centre_value);

    for (std::size_t i = 1; i < xgk.size(); ++i) {
        const Real offset = half_length * xgk[i];
        const Real lower = Real(std::invoke(function, centre - offset));
        const Real upper = Real(std::invoke(function, centre + offset));
        lower_values[i - 1] = lower;
        upper_values[i - 1] = upper;

        const Real pair_sum = lower + upper;
        kronrod += wgk[i] * pair_sum;
        result_abs += wgk[i] * (abs_value(lower) + abs_value(upper));

        if ((i & 1U) != 0U)
            gauss += wg[(i - 1) / 2] * pair_sum;
    }

    // QUADPACK-style resasc scaling makes the embedded-rule error estimate
    // much less optimistic on irregular integrands.
    const Real mean = kronrod / Real(2);
    Real result_asc = wgk[0] * abs_value(centre_value - mean);
    for (std::size_t i = 1; i < xgk.size(); ++i) {
        result_asc += wgk[i] *
            (abs_value(lower_values[i - 1] - mean) +
             abs_value(upper_values[i - 1] - mean));
    }

    result_abs *= abs_half_length;
    result_asc *= abs_half_length;
    Real error = abs_value((kronrod - gauss) * half_length);

    if (result_asc != Real(0) && error != Real(0)) {
        const Real scale = Real(200) * error / result_asc;
        error = scale < Real(1)
              ? result_asc * scale * sqrt_value(scale)
              : result_asc;
    }

    const Real roundoff_floor = Real(50) *
        Real(std::numeric_limits<Real>::epsilon()) * result_abs;
    if (error < roundoff_floor)
        error = roundoff_floor;

    return {a, b, kronrod * half_length, error};
}

template<class Real>
struct interval_error_less {
    bool operator()(const interval_estimate<Real>& lhs,
                    const interval_estimate<Real>& rhs) const
    {
        return lhs.error < rhs.error;
    }
};

} // namespace detail

template<class Real>
Real default_tolerance()
{
    static_assert(detail::is_supported_real_v<Real>,
                  "general_integrator supports double, dd_128, and float128");
    return detail::sqrt_value(Real(std::numeric_limits<Real>::epsilon()));
}

template<class Real>
struct options {
    Real epsabs = default_tolerance<Real>();
    Real epsrel = default_tolerance<Real>();
    std::size_t limit = 50;
    std::vector<Real> points;
};

template<class Value, class Real>
struct result {
    Value value{};
    Real absolute_error{};
    Real real_error{};
    Real imaginary_error{};
    std::size_t evaluations = 0;
    std::size_t subintervals = 0;
    bool converged = false;
};

namespace detail {

template<class Real, class Function>
result<Real, Real> integrate_real(Function& function,
                                  Real a,
                                  Real b,
                                  const options<Real>& settings)
{
    if (settings.limit == 0)
        throw std::invalid_argument("general_integrator: limit must be positive");
    if (settings.epsabs < Real(0) || settings.epsrel < Real(0) ||
        (settings.epsabs == Real(0) && settings.epsrel == Real(0))) {
        throw std::invalid_argument(
            "general_integrator: epsabs and epsrel must be non-negative and not both zero");
    }

    result<Real, Real> output;
    if (a == b) {
        output.converged = true;
        return output;
    }

    const bool reversed = b < a;
    const Real lower_bound = reversed ? b : a;
    const Real upper_bound = reversed ? a : b;

    std::vector<Real> breakpoints;
    breakpoints.reserve(settings.points.size() + 2);
    breakpoints.push_back(lower_bound);
    for (const Real& point : settings.points) {
        if (point < lower_bound || upper_bound < point)
            throw std::invalid_argument(
                "general_integrator: every point must lie in [a,b]");
        if (point != lower_bound && point != upper_bound)
            breakpoints.push_back(point);
    }
    breakpoints.push_back(upper_bound);
    std::sort(breakpoints.begin(), breakpoints.end());
    breakpoints.erase(std::unique(breakpoints.begin(), breakpoints.end()),
                      breakpoints.end());

    const std::size_t initial_intervals = breakpoints.size() - 1;
    if (initial_intervals > settings.limit) {
        throw std::invalid_argument(
            "general_integrator: limit is smaller than the number of intervals created by points");
    }

    using interval = interval_estimate<Real>;
    std::priority_queue<interval,
                        std::vector<interval>,
                        interval_error_less<Real>> queue;

    Real total_value{};
    Real total_error{};
    for (std::size_t i = 0; i < initial_intervals; ++i) {
        interval current = evaluate_interval<Real>(
            function, breakpoints[i], breakpoints[i + 1]);
        total_value += current.value;
        total_error += current.error;
        queue.push(std::move(current));
        output.evaluations += 21;
    }

    auto target_error = [&]() {
        const Real relative = settings.epsrel * abs_value(total_value);
        return settings.epsabs < relative ? relative : settings.epsabs;
    };

    bool stalled = false;
    while (total_error > target_error() && queue.size() < settings.limit) {
        interval worst = queue.top();
        queue.pop();

        const Real midpoint = (worst.a + worst.b) / Real(2);
        if (midpoint == worst.a || midpoint == worst.b) {
            queue.push(std::move(worst));
            stalled = true;
            break;
        }

        interval left = evaluate_interval<Real>(
            function, worst.a, midpoint);
        interval right = evaluate_interval<Real>(
            function, midpoint, worst.b);
        output.evaluations += 42;

        total_value += left.value + right.value - worst.value;
        const Real replacement_error = left.error + right.error;
        if (total_error < worst.error)
            total_error = replacement_error;
        else
            total_error += replacement_error - worst.error;

        queue.push(std::move(left));
        queue.push(std::move(right));
    }

    output.value = reversed ? -total_value : total_value;
    output.absolute_error = total_error;
    output.real_error = total_error;
    output.subintervals = queue.size();
    output.converged = !stalled && total_error <= target_error();
    return output;
}

} // namespace detail

// Integrate f over the finite real interval [a,b].  f may return Real or
// std::complex<Real>.  Complex functions are integrated as independent real
// and imaginary problems, matching scipy.integrate.quad(complex_func=True).
template<class Real, class Function>
auto integrate(Function&& function,
               Real a,
               Real b,
               const options<Real>& settings = {})
{
    static_assert(detail::is_supported_real_v<Real>,
                  "general_integrator supports double, dd_128, and float128");

    using function_type = std::remove_reference_t<Function>;
    using value_type = std::remove_cvref_t<
        std::invoke_result_t<function_type&, Real>>;

    static_assert(std::is_same_v<value_type, Real> ||
                  (detail::is_std_complex_v<value_type> &&
                   std::is_same_v<typename detail::is_std_complex<value_type>::real_type,
                                  Real>),
                  "the integrand must return Real or std::complex<Real>");

    function_type& function_ref = function;
    if constexpr (std::is_same_v<value_type, Real>) {
        return detail::integrate_real<Real>(function_ref, a, b, settings);
    } else {
        auto real_function = [&](Real x) -> Real {
            return Real(std::invoke(function_ref, x).real());
        };
        auto imaginary_function = [&](Real x) -> Real {
            return Real(std::invoke(function_ref, x).imag());
        };

        const auto real_result = detail::integrate_real<Real>(
            real_function, a, b, settings);
        const auto imaginary_result = detail::integrate_real<Real>(
            imaginary_function, a, b, settings);

        result<value_type, Real> output;
        output.value = value_type(real_result.value, imaginary_result.value);
        output.real_error = real_result.absolute_error;
        output.imaginary_error = imaginary_result.absolute_error;
        output.absolute_error = detail::sqrt_value(
            output.real_error * output.real_error +
            output.imaginary_error * output.imaginary_error);
        output.evaluations = real_result.evaluations + imaginary_result.evaluations;
        output.subintervals = real_result.subintervals +
                              imaginary_result.subintervals;
        output.converged = real_result.converged && imaginary_result.converged;
        return output;
    }
}

// Positional convenience overload resembling scipy.integrate.quad.
template<class Real, class Function>
auto quad(Function&& function,
          Real a,
          Real b,
          Real epsabs = default_tolerance<Real>(),
          Real epsrel = default_tolerance<Real>(),
          std::size_t limit = 50,
          std::vector<Real> points = {})
{
    options<Real> settings;
    settings.epsabs = epsabs;
    settings.epsrel = epsrel;
    settings.limit = limit;
    settings.points = std::move(points);
    return integrate<Real>(std::forward<Function>(function), a, b, settings);
}

} // namespace general_integrator
