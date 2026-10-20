#pragma once

#include "mps_base.h"

#include <pi.h>
#include <grid.h>
#include <type_int128.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace analytic_mps_construction {

/// Construct the rank-one basis MPS that is one at the supplied bit string
/// and zero at every other bit string. bits[k] selects the physical index
/// of core k; bits[0] is the least-significant bit in the QTGrid convention.
/// Scalar defaults to double and may also be complex. Returns only the MPS:
/// a bit string does not specify a coordinate interval or grid.
template<typename Scalar = double>
MPS<Scalar> make_basis_mps(const std::vector<int>& bits)
{
    if (bits.empty()) {
        throw std::invalid_argument("make_basis_mps: bits must not be empty");
    }

    std::vector<Tensor3D<Scalar>> cores;
    cores.reserve(bits.size());
    for (int bit : bits) {
        if (bit != 0 && bit != 1) {
            throw std::invalid_argument("make_basis_mps: each bit must be 0 or 1");
        }
        Tensor3D<Scalar> core(1, 2, 1); // zero-initialized
        core(0, bit, 0) = Scalar(1);
        cores.push_back(std::move(core));
    }

    return MPS<Scalar>(std::move(cores), /*max_bond_dim=*/0,
                       /*reltol=*/-1, /*w=*/-1);
}

/// Construct H(x-x0), or H(x0-x) when rev is true, as a rank-at-most-two MPS.
/// Returns {mps, grid} on [a,b), with least-significant-bit-first cores.
/// fixed_to_left=true snaps an off-grid x0 to the neighboring point on its
/// left; false snaps to the neighboring point on its right. The selected
/// discontinuity_mode sets the value at the snapped jump: one, half, or zero.
/// A right snap beyond the last sample lies at b and gives a constant state.
/// Jumps outside [a,b) also give the corresponding constant state.
/// Scalar defaults to double and may be complex; coordinates use its real type.
/// No TCI, truncation, or canonicalization is performed.
template<typename Scalar = double, typename Sint = util::i128>
std::pair<MPS<Scalar>, QTGrid<typename Eigen::NumTraits<Scalar>::Real, Sint>>
make_heaviside_mps(
    typename Eigen::NumTraits<Scalar>::Real a,
    typename Eigen::NumTraits<Scalar>::Real b,
    int nbit,
    typename Eigen::NumTraits<Scalar>::Real x0,
    bool rev = false,
    bool fixed_to_left = false,
    const std::string& discontinuity_mode = "one")
{
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Grid = QTGrid<RealScalar, Sint>;
    using std::isfinite;

    if (nbit <= 0) {
        throw std::invalid_argument("make_heaviside_mps: nbit must be positive");
    }
    if (!isfinite(x0)) {
        throw std::invalid_argument("make_heaviside_mps: x0 must be finite");
    }

    Scalar at_jump;
    if (discontinuity_mode == "one") {
        at_jump = Scalar(1);
    } else if (discontinuity_mode == "half") {
        at_jump = Scalar(1) / Scalar(2);
    } else if (discontinuity_mode == "zero") {
        at_jump = Scalar(0);
    } else {
        throw std::invalid_argument(
            "make_heaviside_mps: discontinuity_mode must be one, half, or zero");
    }

    Grid grid(a, b, nbit);
    const Sint count = grid.get_N();
    Sint jump;
    if (x0 < a) {
        jump = Sint(-1);
    } else if (x0 >= b) {
        jump = count;
    } else {
        // Find the first sample >= x0 using integer-index binary search.
        // This implements ceiling/floor snapping without converting a large
        // floating-point index to Sint (important for dd_128 and 128-bit grids).
        Sint lower = 0;
        Sint upper = count;
        while (lower < upper) {
            const Sint middle = lower + (upper - lower) / Sint(2);
            const RealScalar coordinate = a + RealScalar(middle) * grid.get_dx();
            if (coordinate < x0) {
                lower = middle + Sint(1);
            } else {
                upper = middle;
            }
        }
        jump = lower;
        if (fixed_to_left &&
            (jump == count || a + RealScalar(jump) * grid.get_dx() != x0)) {
            --jump;
        }
    }

    const bool constant = jump < Sint(0) || jump >= count;
    const Scalar constant_value = (jump < Sint(0)) != rev ? Scalar(1) : Scalar(0);
    std::vector<Tensor3D<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(nbit));

    for (int bit = 0; bit < nbit; ++bit) {
        if (constant) {
            Tensor3D<Scalar> core(1, 2, 1);
            core(0, 0, 0) = core(0, 1, 0) = bit == 0 ? constant_value : Scalar(1);
            cores.push_back(std::move(core));
            continue;
        }

        const bool first = bit == 0;
        const bool last = bit == nbit - 1;
        const int jump_bit = static_cast<int>((jump >> bit) & Sint(1));
        Tensor3D<Scalar> core(first ? 1 : 2, 2, last ? 1 : 2);
        for (int physical = 0; physical < 2; ++physical) {
            const bool equal = physical == jump_bit;
            const Scalar decided = (rev ? physical < jump_bit : physical > jump_bit)
                ? Scalar(1) : Scalar(0);
            // Track the row [f,1], initially [H(0),1]. Equal bits preserve f;
            // a differing, more-significant bit overrides it with decided.
            if (first) {
                core(0, physical, 0) = equal ? at_jump : decided;
                if (!last) {
                    core(0, physical, 1) = Scalar(1);
                }
            } else {
                core(0, physical, 0) = equal ? Scalar(1) : Scalar(0);
                core(1, physical, 0) = equal ? Scalar(0) : decided;
                if (!last) {
                    core(1, physical, 1) = Scalar(1);
                }
            }
        }
        cores.push_back(std::move(core));
    }

    MPS<Scalar> mps(std::move(cores), /*max_bond_dim=*/0,
                    /*reltol=*/-1, /*w=*/-1);
    return {std::move(mps), std::move(grid)};
}

/// Construct a window H(x-x0) * H(x1-x), requiring x0 < x1.
/// Returns {mps, grid} on [a,b), LSB-first, with bond rank at most three.
/// Both jumps use make_heaviside_mps's snapping and discontinuity conventions.
/// At distinct snapped endpoints, each endpoint has the selected jump value.
/// If snapping makes the endpoints coincide, their jump values multiply
/// (one -> 1, half -> 1/4, zero -> 0 at that single sample).
/// Directly tracks [H(x-x0), H(x1-x), 1] and selects their sum minus one
/// for distinct endpoints. Constants and coincident jumps use rank one;
/// windows with only one visible endpoint reuse the rank-two Heaviside.
/// No product MPS, TCI, SVD, truncation, or canonicalization is used.
template<typename Scalar = double, typename Sint = util::i128>
std::pair<MPS<Scalar>, QTGrid<typename Eigen::NumTraits<Scalar>::Real, Sint>>
make_step_mps(
    typename Eigen::NumTraits<Scalar>::Real a,
    typename Eigen::NumTraits<Scalar>::Real b,
    int nbit,
    typename Eigen::NumTraits<Scalar>::Real x0,
    typename Eigen::NumTraits<Scalar>::Real x1,
    bool fixed_to_left = false,
    const std::string& discontinuity_mode = "one")
{
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Grid = QTGrid<RealScalar, Sint>;
    using Result = std::pair<MPS<Scalar>, Grid>;
    using std::isfinite;

    if (nbit <= 0) {
        throw std::invalid_argument("make_step_mps: nbit must be positive");
    }
    if (!isfinite(x0) || !isfinite(x1) || !(x0 < x1)) {
        throw std::invalid_argument("make_step_mps: require finite x0 < x1");
    }

    Scalar at_jump;
    if (discontinuity_mode == "one") {
        at_jump = Scalar(1);
    } else if (discontinuity_mode == "half") {
        at_jump = Scalar(1) / Scalar(2);
    } else if (discontinuity_mode == "zero") {
        at_jump = Scalar(0);
    } else {
        throw std::invalid_argument(
            "make_step_mps: discontinuity_mode must be one, half, or zero");
    }

    Grid grid(a, b, nbit);
    const Sint count = grid.get_N();
    const auto snapped_jump = [&](RealScalar x) -> Sint {
        if (x < a) return Sint(-1);
        if (x >= b) return count;
        Sint lower = 0;
        Sint upper = count;
        while (lower < upper) {
            const Sint middle = lower + (upper - lower) / Sint(2);
            const RealScalar coordinate = a + RealScalar(middle) * grid.get_dx();
            if (coordinate < x) lower = middle + Sint(1);
            else upper = middle;
        }
        if (fixed_to_left &&
            (lower == count || a + RealScalar(lower) * grid.get_dx() != x)) {
            --lower;
        }
        return lower;
    };
    const Sint begin = snapped_jump(x0);
    const Sint end = snapped_jump(x1);

    const auto constant_result = [&](Scalar value) -> Result {
        std::vector<Tensor3D<Scalar>> cores;
        cores.reserve(static_cast<std::size_t>(nbit));
        for (int bit = 0; bit < nbit; ++bit) {
            Tensor3D<Scalar> core(1, 2, 1);
            core(0, 0, 0) = core(0, 1, 0) = bit == 0 ? value : Scalar(1);
            cores.push_back(std::move(core));
        }
        return {MPS<Scalar>(std::move(cores), 0, -1, -1), std::move(grid)};
    };
    if (end < Sint(0) || begin >= count) {
        return constant_result(Scalar(0));
    }
    if (begin < Sint(0) && end >= count) {
        return constant_result(Scalar(1));
    }
    if (begin < Sint(0)) {
        return make_heaviside_mps<Scalar, Sint>(
            a, b, nbit, x1, /*rev=*/true, fixed_to_left, discontinuity_mode);
    }
    if (end >= count) {
        return make_heaviside_mps<Scalar, Sint>(
            a, b, nbit, x0, /*rev=*/false, fixed_to_left, discontinuity_mode);
    }
    if (begin == end) {
        std::vector<int> bits(static_cast<std::size_t>(nbit));
        for (int bit = 0; bit < nbit; ++bit) {
            bits[static_cast<std::size_t>(bit)] =
                static_cast<int>((begin >> bit) & Sint(1));
        }
        auto mps = make_basis_mps<Scalar>(bits);
        mps *= at_jump * at_jump;
        return {std::move(mps), std::move(grid)};
    }

    std::vector<Tensor3D<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(nbit));
    for (int bit = 0; bit < nbit; ++bit) {
        const bool first = bit == 0;
        const bool last = bit == nbit - 1;
        const int begin_bit = static_cast<int>((begin >> bit) & Sint(1));
        const int end_bit = static_cast<int>((end >> bit) & Sint(1));
        Tensor3D<Scalar> core(first ? 1 : 3, 2, last ? 1 : 3);
        for (int physical = 0; physical < 2; ++physical) {
            const bool keep_begin = physical == begin_bit;
            const bool keep_end = physical == end_bit;
            const Scalar rising = physical > begin_bit ? Scalar(1) : Scalar(0);
            const Scalar falling = physical < end_bit ? Scalar(1) : Scalar(0);
            if (first) {
                const Scalar first_rising = keep_begin ? at_jump : rising;
                const Scalar first_falling = keep_end ? at_jump : falling;
                if (last) {
                    core(0, physical, 0) = first_rising + first_falling - Scalar(1);
                } else {
                    core(0, physical, 0) = first_rising;
                    core(0, physical, 1) = first_falling;
                    core(0, physical, 2) = Scalar(1);
                }
            } else if (last) {
                // Contract the final state [rising,falling,1] with [1,1,-1].
                core(0, physical, 0) = keep_begin ? Scalar(1) : Scalar(0);
                core(1, physical, 0) = keep_end ? Scalar(1) : Scalar(0);
                core(2, physical, 0) = rising + falling - Scalar(1);
            } else {
                // A differing, more-significant bit overrides its comparison.
                core(0, physical, 0) = keep_begin ? Scalar(1) : Scalar(0);
                core(1, physical, 1) = keep_end ? Scalar(1) : Scalar(0);
                core(2, physical, 0) = rising;
                core(2, physical, 1) = falling;
                core(2, physical, 2) = Scalar(1);
            }
        }
        cores.push_back(std::move(core));
    }

    MPS<Scalar> mps(std::move(cores), /*max_bond_dim=*/0,
                    /*reltol=*/-1, /*w=*/-1);
    return {std::move(mps), std::move(grid)};
}

/// Construct rho * exp(x * phi) directly as a rank-one quantics MPS.
/// Scalar may be real or complex; a and b use its corresponding real type.
/// Returns {mps, grid}, with grid points x_j = a + j*(b-a)/2^nbit on [a,b).
/// Cores are least-significant-bit first, matching QTGrid.
///
/// Since x = a + dx * sum_k bit_k * 2^k, each core has entries
/// [1, exp(phi * dx * 2^k)]. The first core also carries rho * exp(a * phi).
/// No TCI, truncation, or canonicalization is performed.
template<typename Scalar, typename Sint = util::i128>
std::pair<MPS<Scalar>, QTGrid<typename Eigen::NumTraits<Scalar>::Real, Sint>>
make_exp_mps(
    typename Eigen::NumTraits<Scalar>::Real a,
    typename Eigen::NumTraits<Scalar>::Real b,
    int nbit,
    Scalar rho,
    Scalar phi)
{
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Grid = QTGrid<RealScalar, Sint>;
    using std::exp;

    if (nbit <= 0) {
        throw std::invalid_argument("make_exp_mps: nbit must be positive");
    }
    Grid grid(a, b, nbit);

    const Scalar origin_exponent = Scalar(a) * phi;
    const Scalar amplitude = rho * exp(origin_exponent);
    RealScalar step = grid.get_dx();
    std::vector<Tensor3D<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(nbit));

    for (int bit = 0; bit < nbit; ++bit) {
        const Scalar exponent = Scalar(step) * phi;
        const Scalar factor = exp(exponent);
        Tensor3D<Scalar> core(1, 2, 1);
        core(0, 0, 0) = bit == 0 ? amplitude : Scalar(1);
        core(0, 1, 0) = bit == 0 ? amplitude * factor : factor;
        cores.push_back(std::move(core));
        if (bit + 1 < nbit) {
            step *= RealScalar(2);
        }
    }

    MPS<Scalar> mps(std::move(cores), /*max_bond_dim=*/0,
                    /*reltol=*/-1, /*w=*/-1);
    return {std::move(mps), std::move(grid)};
}

/// Construct A * cos(B*x + C) directly as a quantics MPS of bond rank at
/// most two. Scalar may be real or complex; a and b use its real type.
/// Returns {mps, grid} on QTGrid's half-open interval [a,b), LSB-first.
///
/// The first core carries A*[cos(B*a+C), sin(B*a+C)] with the first-bit
/// phase added. Subsequent bits apply the angle-addition rotation
/// [[cos(delta), sin(delta)], [-sin(delta), cos(delta)]], where
/// delta = B*dx*2^k for bit k equal to one, or zero otherwise.
/// The last core selects the cosine component. For nbit == 1 both
/// boundary contractions are contained in the single rank-one core.
/// No TCI, truncation, or canonicalization is performed.
template<typename Scalar, typename Sint = util::i128>
std::pair<MPS<Scalar>, QTGrid<typename Eigen::NumTraits<Scalar>::Real, Sint>>
make_cos_mps(
    typename Eigen::NumTraits<Scalar>::Real a,
    typename Eigen::NumTraits<Scalar>::Real b,
    int nbit,
    Scalar A,
    Scalar B,
    Scalar C)
{
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Grid = QTGrid<RealScalar, Sint>;
    using std::cos;
    using std::sin;

    if (nbit <= 0) {
        throw std::invalid_argument("make_cos_mps: nbit must be positive");
    }
    Grid grid(a, b, nbit);

    const Scalar origin_angle = B * Scalar(a) + C;
    RealScalar step = grid.get_dx();
    std::vector<Tensor3D<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(nbit));

    for (int bit = 0; bit < nbit; ++bit) {
        const int left_dim = bit == 0 ? 1 : 2;
        const int right_dim = bit == nbit - 1 ? 1 : 2;
        const Scalar angle_step = B * Scalar(step);
        Tensor3D<Scalar> core(left_dim, 2, right_dim); // zero-initialized

        if (bit == 0) {
            for (int physical = 0; physical < 2; ++physical) {
                const Scalar angle = origin_angle + Scalar(physical) * angle_step;
                core(0, physical, 0) = A * cos(angle);
                if (right_dim == 2) {
                    core(0, physical, 1) = A * sin(angle);
                }
            }
        } else {
            const Scalar cosine = cos(angle_step);
            const Scalar sine = sin(angle_step);
            // A zero bit leaves the angle unchanged; a one bit rotates it.
            core(0, 0, 0) = Scalar(1);
            core(0, 1, 0) = cosine;
            core(1, 1, 0) = -sine;
            if (right_dim == 2) {
                core(1, 0, 1) = Scalar(1);
                core(0, 1, 1) = sine;
                core(1, 1, 1) = cosine;
            }
        }

        cores.push_back(std::move(core));
        if (bit + 1 < nbit) {
            step *= RealScalar(2);
        }
    }

    MPS<Scalar> mps(std::move(cores), /*max_bond_dim=*/0,
                    /*reltol=*/-1, /*w=*/-1);
    return {std::move(mps), std::move(grid)};
}

/// Construct A * sin(B*x + C) by reusing make_cos_mps with phase C-pi/2.
/// Returns {mps, grid} with the same grid and rank bounds as make_cos_mps.
template<typename Scalar, typename Sint = util::i128>
std::pair<MPS<Scalar>, QTGrid<typename Eigen::NumTraits<Scalar>::Real, Sint>>
make_sin_mps(
    typename Eigen::NumTraits<Scalar>::Real a,
    typename Eigen::NumTraits<Scalar>::Real b,
    int nbit,
    Scalar A,
    Scalar B,
    Scalar C)
{
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    const Scalar phase = C - Scalar(::pi<RealScalar>() / RealScalar(2));
    return make_cos_mps<Scalar, Sint>(a, b, nbit, A, B, phase);
}

/// Construct sum_{j=0}^p coefficients[j] * x^j as a quantics MPS.
/// Coefficients are in ascending order: {A,B,C,D,...} means A+B*x+C*x^2+D*x^3+...
/// Returns {mps, grid} on [a,b), LSB-first, with bond rank at most p+1.
/// Real and complex coefficients are supported. The construction uses
/// (x+delta)^j = sum_{i=0}^j binomial(j,i)*x^i*delta^(j-i).
/// The first core initializes the powers at a plus the first-bit shift;
/// interior cores translate them, and the last contracts with the coefficients.
/// No TCI, truncation, or canonicalization is performed; only floating-point
/// rounding limits the exact polynomial representation.
template<typename Scalar, typename Sint = util::i128>
std::pair<MPS<Scalar>, QTGrid<typename Eigen::NumTraits<Scalar>::Real, Sint>>
make_polynomial_orderp_mps(
    typename Eigen::NumTraits<Scalar>::Real a,
    typename Eigen::NumTraits<Scalar>::Real b,
    int nbit,
    const std::vector<Scalar>& coefficients)
{
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Grid = QTGrid<RealScalar, Sint>;

    if (nbit <= 0) {
        throw std::invalid_argument("make_polynomial_orderp_mps: nbit must be positive");
    }
    if (coefficients.empty()) {
        throw std::invalid_argument("make_polynomial_orderp_mps: coefficients must not be empty");
    }
    Grid grid(a, b, nbit);
    const std::size_t rank = coefficients.size();
    const Eigen::Index bond_dim = static_cast<Eigen::Index>(rank);

    std::vector<std::vector<RealScalar>> binomial(
        rank, std::vector<RealScalar>(rank, RealScalar(0)));
    for (std::size_t j = 0; j < rank; ++j) {
        binomial[j][0] = binomial[j][j] = RealScalar(1);
        for (std::size_t i = 1; i < j; ++i) {
            binomial[j][i] = binomial[j-1][i-1] + binomial[j-1][i];
        }
    }

    RealScalar step = grid.get_dx();
    std::vector<Tensor3D<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(nbit));
    for (int bit = 0; bit < nbit; ++bit) {
        const bool first = bit == 0;
        const bool last = bit == nbit - 1;
        Tensor3D<Scalar> core(first ? 1 : bond_dim, 2, last ? 1 : bond_dim);

        for (int physical = 0; physical < 2; ++physical) {
            const RealScalar delta = physical == 0 ? RealScalar(0) : step;
            if (first) {
                const RealScalar x = a + delta;
                if (last) {
                    // A single-bit grid has both boundary contractions here.
                    Scalar value = coefficients.back();
                    for (std::size_t j = rank - 1; j > 0; --j) {
                        value = value * Scalar(x) + coefficients[j-1];
                    }
                    core(0, physical, 0) = value;
                } else {
                    RealScalar power = RealScalar(1);
                    for (std::size_t j = 0; j < rank; ++j) {
                        core(0, physical, static_cast<Eigen::Index>(j)) = Scalar(power);
                        if (j + 1 < rank) {
                            power *= x;
                        }
                    }
                }
            } else {
                std::vector<RealScalar> powers(rank, RealScalar(1));
                for (std::size_t j = 1; j < rank; ++j) {
                    powers[j] = powers[j-1] * delta;
                }
                for (std::size_t i = 0; i < rank; ++i) {
                    for (std::size_t j = i; j < rank; ++j) {
                        const Scalar translation = Scalar(binomial[j][i] * powers[j-i]);
                        const auto left = static_cast<Eigen::Index>(i);
                        if (last) {
                            core(left, physical, 0) += translation * coefficients[j];
                        } else {
                            core(left, physical, static_cast<Eigen::Index>(j)) = translation;
                        }
                    }
                }
            }
        }
        cores.push_back(std::move(core));
        if (bit + 1 < nbit) {
            step *= RealScalar(2);
        }
    }

    MPS<Scalar> mps(std::move(cores), /*max_bond_dim=*/0,
                    /*reltol=*/-1, /*w=*/-1);
    return {std::move(mps), std::move(grid)};
}

} // namespace analytic_mps_construction

