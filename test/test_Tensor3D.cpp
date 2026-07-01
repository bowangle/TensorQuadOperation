#include "tensor.h"
#include "tt_base.h"
#include "type_float128_boost.h"
#include <iostream>
#include <complex>

static void test_tensor3d(Tensor3D<Cfloat128>& t)
{
    using T = Cfloat128;

    const size_t L = t.n_left;
    const size_t P = t.n_phys;
    const size_t R = t.n_right;

    // ============================================================
    // 1. Fill tensor with deterministic pattern
    // ============================================================
    for (size_t r = 0; r < R; ++r)
        for (size_t p = 0; p < P; ++p)
            for (size_t l = 0; l < L; ++l)
                t(l,p,r) = T((l + 1) + 10*(p + 1) + 100*(r + 1), 0);

    // ============================================================
    // 2. operator() correctness
    // ============================================================
    for (size_t r = 0; r < R; ++r)
        for (size_t p = 0; p < P; ++p)
            for (size_t l = 0; l < L; ++l)
                assert(t(l,p,r).real() ==
                       (l + 1) + 10*(p + 1) + 100*(r + 1));

    // ============================================================
    // 3. RIGHT VIEW (mutability + aliasing)
    // ============================================================
    {
        auto v = t.right(0);
        v(0,0) = T(9999,0);
        assert(t(0,0,0).real() == 9999);
    }

    // ============================================================
    // 4. PHYS VIEW
    // ============================================================
    {
        auto v = t.phys(1);
        v(0,0) = T(8888,0);
        assert(t(0,1,0).real() == 8888);
    }

    // ============================================================
    // 5. LEFT VIEW
    // ============================================================
    {
        auto v = t.left(1);
        v(0,0) = T(7777,0);
        assert(t(1,0,0).real() == 7777);
    }

    // ============================================================
    // 6. COPY INDEPENDENCE (right_copy)
    // ============================================================
    {
        auto c = t.right_copy(0);
        c(0,0) = T(12345,0);
        assert(t(0,0,0).real() != 12345);
    }

    // ============================================================
    // 7. COPY INDEPENDENCE (phys_copy)
    // ============================================================
    {
        auto c = t.phys_copy(0);
        c(0,0) = T(54321,0);
        assert(t(0,0,0).real() != 54321);
    }

    // ============================================================
    // 8. FLATTEN (phys_on_left = true)
    // ============================================================
    {
        auto f = t.flatten_phys(true);

        for (size_t r = 0; r < R; ++r)
            for (size_t p = 0; p < P; ++p)
                for (size_t l = 0; l < L; ++l)
                {
                    size_t row = l + p * L;
                    assert(f(row, r) == t(l,p,r));
                }
    }

    // ============================================================
    // 9. FLATTEN (phys_on_left = false)
    // ============================================================
    {
        auto f = t.flatten_phys(false);

        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index p = 0; p < P; ++p)
                for (Eigen::Index l = 0; l < L; ++l) {
                    Eigen::Index col = r * P + p;
                    assert(f(l, col) == t(l, p, r));
                }
    }

    // ============================================================
    // 10. VIEW vs COPY consistency
    // ============================================================
    {
        auto v = t.right(0);
        auto c = t.right_copy(0);

        v(0,1) = T(1111,0);
        c(0,1) = T(2222,0);

        assert(t(0,1,0) == T(1111,0));
        assert(c(0,1) == T(2222,0));
    }
}

int main()
{
    auto tt = load_vector_tensor<Cfloat128>("gE<_site[[1, 0], [0, 0]].tt");
    int nBit = tt.size();
    int i = 5;

    // i want to test

    Tensor3D tensor = tt[i];

    test_tensor3d(tensor);

    Eigen::Index a = 5;

    std::cout << "All Tensor3D tests passed!\n";
}