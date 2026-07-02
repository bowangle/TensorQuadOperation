#pragma once

// ============================================================
// -1. EIGEN INDEX TYPE (THE ACTUAL BUILD FIX)
// ------------------------------------------------------------
// Eigen writes `RealScalar(rows)` internally (e.g. in
// ColPivHouseholderQR.h and BDCSVD.h). With the default index
// type `long`, that becomes `dd_real(long)`, which is ambiguous
// because long->int and long->double are equal-rank conversions
// and dd_real only defines dd_real(int) / dd_real(double).
//
// Forcing Eigen's index type to `int` makes those calls resolve
// to dd_real(int) exactly, fixing every such site at once.
//
// REQUIREMENT: this header MUST be included before any Eigen
// header. A #define after Eigen is already parsed has no effect,
// so we fail loudly below instead of silently letting the
// ambiguity come back.
//
// ODR WARNING: EVERY translation unit that touches Eigen must
// see the same index type. Relying on include order per-file is
// fragile; the robust way to guarantee it project-wide is a
// global build flag (this block then becomes a harmless no-op):
//
//     add_compile_definitions(EIGEN_DEFAULT_DENSE_INDEX_TYPE=int)
//
// Trade-off of int indices: each matrix dimension is capped at
// ~2e9 elements. Fine for dense SVD workloads like this one.
// ============================================================
#if defined(EIGEN_CORE_H) && !defined(EIGEN_DEFAULT_DENSE_INDEX_TYPE)
#  error "type_double_double.h must be included BEFORE any Eigen header, or define EIGEN_DEFAULT_DENSE_INDEX_TYPE=int as a global build flag."
#endif

#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#  define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif

#include <qd/dd_real.h>
#include <complex>
#include <cmath>
#include <Eigen/Core>
#include <type_traits>
#include <cstddef>

using dd_128 = dd_real;
using Cdd_128 = std::complex<dd_128>;

// ============================================================
// 0. SAFE SCALAR CONVERSION HELPERS
// ------------------------------------------------------------
// Kept for use in YOUR code. Note: these cannot fix the calls
// inside Eigen (those are hard-coded RealScalar(rows)); the
// index-type define above is what fixes those.
// ============================================================
inline dd_real dd_from_double(double x) {
    return dd_real(x);
}
inline dd_real dd_from_index(Eigen::Index x) {
    return dd_real(static_cast<double>(x));
}
inline dd_real dd_from_int(int x) {
    return dd_real(static_cast<double>(x));
}
inline dd_real dd_from_size(std::size_t x) {
    return dd_real(static_cast<double>(x));
}
// Generic safe cast
template <typename T>
inline dd_real to_dd(T x) {
    return dd_real(static_cast<double>(x));
}

// ============================================================
// 1. MATH SUPPORT (Eigen ADL)
// ============================================================
inline dd_real hypot(const dd_real& a, const dd_real& b) {
    dd_real ax = abs(a), ay = abs(b);
    if (ax == dd_real(0.0)) return ay;
    if (ay == dd_real(0.0)) return ax;
    dd_real hi = (ax > ay) ? ax : ay;
    dd_real lo = (ax > ay) ? ay : ax;
    dd_real r = lo / hi;
    return hi * sqrt(dd_real(1.0) + r * r);
}

// ============================================================
// 2. Eigen TRAITS
// ============================================================
namespace Eigen {
template<>
struct NumTraits<dd_real> : GenericNumTraits<dd_real> {
    using Real = dd_real;
    using NonInteger = dd_real;
    using Nested = dd_real;
    using Literal = dd_real;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 2,
        AddCost = 16,
        MulCost = 32
    };
    static Real epsilon()         { return dd_real::_eps; }
    static Real dummy_precision() { return dd_real(1e-30); }
    static Real highest()         { return dd_real::_max; }
    static Real lowest()          { return -dd_real::_max; }
    static int digits10()         { return 31; }
    static int max_digits10()     { return 33; }
};
}



// ------------------------------------------------------------
// (Removed) boost::is_floating_point<dd_real> specialization.
//
// It broke the build: this header is included before anything
// pulls in Boost, so the name boost::is_floating_point was not
// yet declared as a template. The specialization implicitly
// declared it as a NON-template, which then collided with the
// real Boost template once <boost/.../is_floating_point.hpp> was
// included (via mat_decomp.h), cascading into is_arithmetic /
// is_float / lexical_cast errors.
//
// It was also undefined behavior (specializing a Boost trait for
// a non-fundamental type) and served no purpose here: dd_real is
// not a Boost.Multiprecision backend, so nothing depends on it.
//
// If you ever truly need it, you must (a) #include
// <boost/type_traits/is_floating_point.hpp> BEFORE specializing,
// and (b) accept the UB. Not recommended.
// ------------------------------------------------------------