#include "type_double_double.h"
#include "type_float128_boost.h"
#include "tensor.h"
#include "tt_base.h"
#include <iostream>
#include <complex>
#include <string>
#include <cassert>
#include <chrono>
#include <Eigen/Dense>

// ---------------------------------------------------------------------------
// Helper: cast size_t index to double first to avoid ambiguous conversions
// for types like dd_real that have dd_real(int) and dd_real(double) but not
// dd_real(size_t).
// ---------------------------------------------------------------------------
template<class Real>
static Real fromIndex(size_t i)
{
    return Real((double)i);
}

// ---------------------------------------------------------------------------
// Deterministic fill: t(l,p,r) = (l+1) + 10*(p+1) + 100*(r+1)
// ---------------------------------------------------------------------------
template<class Scalar>
static void fillDeterministic(Tensor3D<Scalar>& t)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    const size_t L = t.n_left, P = t.n_phys, R = t.n_right;

    for (size_t r = 0; r < R; ++r)
        for (size_t p = 0; p < P; ++p)
            for (size_t l = 0; l < L; ++l)
            {
                Real val = fromIndex<Real>(l + 1)
                         + Real(10.0) * fromIndex<Real>(p + 1)
                         + Real(100.0) * fromIndex<Real>(r + 1);
                if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
                    t(l, p, r) = Scalar(val, Real(0.0));
                else
                    t(l, p, r) = Scalar(val);
            }
}

// ---------------------------------------------------------------------------
// Individual named sub-tests
// ---------------------------------------------------------------------------

template<class Scalar>
static void testAccessCorrectness(Tensor3D<Scalar>& t, const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    const size_t L = t.n_left, P = t.n_phys, R = t.n_right;

    auto t0 = std::chrono::steady_clock::now();
    bool ok = true;

    for (size_t r = 0; r < R && ok; ++r)
        for (size_t p = 0; p < P && ok; ++p)
            for (size_t l = 0; l < L && ok; ++l)
            {
                Real expected = fromIndex<Real>(l + 1)
                              + Real(10.0) * fromIndex<Real>(p + 1)
                              + Real(100.0) * fromIndex<Real>(r + 1);
                Real got;
                if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
                    got = t(l, p, r).real();
                else
                    got = t(l, p, r);
                ok = (got == expected);
            }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][ACCESS     ] ok=" << ok << "  time=" << ms << " ms\n";
    assert(ok);
}

template<class Scalar>
static void testRightView(Tensor3D<Scalar>& t, const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    auto t0 = std::chrono::steady_clock::now();

    Scalar sentinel;
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
        sentinel = Scalar(Real(9999.0), Real(0.0));
    else
        sentinel = Scalar(Real(9999.0));

    auto v = t.right(0);
    v(0, 0) = sentinel;
    bool aliases = (t(0, 0, 0) == sentinel);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][RIGHT_VIEW ] aliases=" << aliases << "  time=" << ms << " ms\n";
    assert(aliases);
}

template<class Scalar>
static void testPhysView(Tensor3D<Scalar>& t, const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    auto t0 = std::chrono::steady_clock::now();

    Scalar sentinel;
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
        sentinel = Scalar(Real(8888.0), Real(0.0));
    else
        sentinel = Scalar(Real(8888.0));

    auto v = t.phys(1);
    v(0, 0) = sentinel;
    bool aliases = (t(0, 1, 0) == sentinel);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][PHYS_VIEW  ] aliases=" << aliases << "  time=" << ms << " ms\n";
    assert(aliases);
}

template<class Scalar>
static void testLeftView(Tensor3D<Scalar>& t, const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    auto t0 = std::chrono::steady_clock::now();

    Scalar sentinel;
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
        sentinel = Scalar(Real(7777.0), Real(0.0));
    else
        sentinel = Scalar(Real(7777.0));

    auto v = t.left(1);
    v(0, 0) = sentinel;
    bool aliases = (t(1, 0, 0) == sentinel);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][LEFT_VIEW  ] aliases=" << aliases << "  time=" << ms << " ms\n";
    assert(aliases);
}

template<class Scalar>
static void testRightCopy(Tensor3D<Scalar>& t, const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    auto t0 = std::chrono::steady_clock::now();

    Scalar sentinel;
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
        sentinel = Scalar(Real(12345.0), Real(0.0));
    else
        sentinel = Scalar(Real(12345.0));

    // snapshot what t(0,0,0) holds before the copy write
    Scalar before = t(0, 0, 0);
    auto c = t.right_copy(0);
    c(0, 0) = sentinel;
    bool independent = (t(0, 0, 0) == before);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][RIGHT_COPY ] independent=" << independent << "  time=" << ms << " ms\n";
    assert(independent);
}

template<class Scalar>
static void testPhysCopy(Tensor3D<Scalar>& t, const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    auto t0 = std::chrono::steady_clock::now();

    Scalar sentinel;
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
        sentinel = Scalar(Real(54321.0), Real(0.0));
    else
        sentinel = Scalar(Real(54321.0));

    Scalar before = t(0, 0, 0);
    auto c = t.phys_copy(0);
    c(0, 0) = sentinel;
    bool independent = (t(0, 0, 0) == before);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][PHYS_COPY  ] independent=" << independent << "  time=" << ms << " ms\n";
    assert(independent);
}

template<class Scalar>
static void testFlattenPhysLeft(Tensor3D<Scalar>& t, const std::string& label)
{
    const size_t L = t.n_left, P = t.n_phys, R = t.n_right;
    auto t0 = std::chrono::steady_clock::now();

    auto f = t.flatten_phys(true);
    bool ok = true;
    for (size_t r = 0; r < R && ok; ++r)
        for (size_t p = 0; p < P && ok; ++p)
            for (size_t l = 0; l < L && ok; ++l)
                ok = (f((Eigen::Index)(l + p * L), (Eigen::Index)r) == t(l, p, r));

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][FLAT_LEFT  ] ok=" << ok << "  time=" << ms << " ms\n";
    assert(ok);
}

template<class Scalar>
static void testFlattenPhysRight(Tensor3D<Scalar>& t, const std::string& label)
{
    const size_t L = t.n_left, P = t.n_phys, R = t.n_right;
    auto t0 = std::chrono::steady_clock::now();

    auto f = t.flatten_phys(false);
    bool ok = true;
    for (size_t r = 0; r < R && ok; ++r)
        for (size_t p = 0; p < P && ok; ++p)
            for (size_t l = 0; l < L && ok; ++l)
                ok = (f((Eigen::Index)l, (Eigen::Index)(r * P + p)) == t(l, p, r));

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][FLAT_RIGHT ] ok=" << ok << "  time=" << ms << " ms\n";
    assert(ok);
}

template<class Scalar>
static void testViewCopyConsistency(Tensor3D<Scalar>& t, const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    auto t0 = std::chrono::steady_clock::now();

    Scalar sv, sc;
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
        sv = Scalar(Real(1111.0), Real(0.0));
        sc = Scalar(Real(2222.0), Real(0.0));
    } else {
        sv = Scalar(Real(1111.0));
        sc = Scalar(Real(2222.0));
    }

    auto v = t.right(0);
    auto c = t.right_copy(0);
    v(0, 1) = sv;
    c(0, 1) = sc;

    bool view_aliases = (t(0, 1, 0) == sv);
    bool copy_indep   = (c(0, 1)    == sc) && (t(0, 1, 0) != sc);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[" << label << "][VIEW_COPY  ] view_aliases=" << view_aliases
              << "  copy_indep=" << copy_indep << "  time=" << ms << " ms\n";
    assert(view_aliases);
    assert(copy_indep);
}

// ---------------------------------------------------------------------------
// Run all sub-tests for one tensor instance (fill first, then test)
// ---------------------------------------------------------------------------
template<class Scalar>
static void runTensorTests(Tensor3D<Scalar>& t, const std::string& label)
{
    fillDeterministic      (t);   // sets known state
    testAccessCorrectness  (t, label);
    // view/copy tests mutate the tensor; refill before each group
    fillDeterministic      (t);
    testRightView          (t, label);
    fillDeterministic      (t);
    testPhysView           (t, label);
    fillDeterministic      (t);
    testLeftView           (t, label);
    fillDeterministic      (t);
    testRightCopy          (t, label);
    fillDeterministic      (t);
    testPhysCopy           (t, label);
    fillDeterministic      (t);
    testFlattenPhysLeft    (t, label);
    fillDeterministic      (t);
    testFlattenPhysRight   (t, label);
    fillDeterministic      (t);
    testViewCopyConsistency(t, label);
}

// ---------------------------------------------------------------------------
// All tests for one scalar type
// ---------------------------------------------------------------------------
template<class Scalar>
static void runAll(const std::string& type, const std::string& tt_file)
{
    std::cout << "\n===== Tensor3D tests for " << type << " =====\n";

    // load_vector_tensor internally does `using RealT = T::value_type` which
    // only compiles for std::complex<X>.  Additionally, Cfloat128 triggers a
    // Boost assign_components ADL ordering bug on some Boost versions.
    // => only load from file for Cdd_128 (the one type we know works).
    if constexpr (std::is_same_v<Scalar, Cdd_128>)
    {
        auto tt = load_vector_tensor<Scalar>(tt_file);
        Tensor3D<Scalar> t = tt[5];
        runTensorTests(t, type + " TT[5]");
    }

    // Synthetic shapes run for every scalar type
    { Tensor3D<Scalar> t(3, 2, 4);  runTensorTests(t, type + " SMALL(3,2,4)"); }
    { Tensor3D<Scalar> t(10, 3, 2); runTensorTests(t, type + " TALL(10,3,2)"); }
    { Tensor3D<Scalar> t(2, 3, 10); runTensorTests(t, type + " WIDE(2,3,10)"); }
}

// ================================= MAIN ====================================
int main()
{
    const std::string tt_file = "gE<_site[[1, 0], [0, 0]].tt";

    runAll<double>            ("double",             tt_file);
    runAll<std::complex<double>>("complex double",   tt_file);
    runAll<dd_128>            ("double double",      tt_file);
    runAll<Cdd_128>           ("complex double double", tt_file);
    runAll<float128>          ("float128",           tt_file);
    runAll<Cfloat128>         ("Cfloat128",          tt_file);

    std::cout << "\nTensor3D TEST PASSED\n";
    return 0;
}