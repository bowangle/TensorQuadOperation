# QuanticsIntegration

`QuanticsIntegration` fits functions on binary quantics grids and integrates their MPS or TTN representations. It uses the tensor operations in [TensorQuadOperation](../TensorQuadOperation/README.md). Public integrators are templated on the scalar and grid-index types; the default index type is `util::i128`.

| Public header | What's available |
|---|---|
| [`quantics_integrator.hpp`](quantics_integrator.hpp) | `QuanticsIntegrator1D`: construct from an MPS and `Grid`, reload saved files, or `fit()` with 1D tensor cross interpolation (TCI); full-grid, interval, and batched interval integration; prefix primitives and causal convolution responses. |
| [`quantics_integrator_kD.hpp`](quantics_integrator_kD.hpp) | `QuanticsIntegrator_kD`: construct from a TTN and `mGrid`, reload saved files, or `fit()` with tree TCI; full-grid and rectangular integration, separable-kernel convolution, and ordered-simplex integration. |
| [`indefinite_integral.hpp`](indefinite_integral.hpp) | Heaviside MPO construction and grid-aware helpers for 1D primitives and convolution. |
| [`simplex_helper.hpp`](simplex_helper.hpp) | Lower-level routines for the simplex subtractor, kernel weighting, and branch elimination. Start with `make_integration_simplex()` for normal use. |

## Definite integration and rectangular primitives

- `integrate()` sums over the stored grid and multiplies by the cell volume. `integrate(IndexRange)` accepts half-open **index** bounds `[begin, end)`, not physical coordinates. In kD, `begin` and `end` each contain one index per axis for **one box**. Direct index ranges can use `end = N` (or `N[i]`) to include the last stored sample. In 1D, `integrate_lrange()` accepts a batch of intervals, and `sum()` gives unscaled sums.
- `make_indefinite_integrator_from_conv()` constructs an exclusive prefix sum with a Heaviside kernel that vanishes at lag zero. Use `integrate_indefinite()` in 1D or `eval_rect_indefinite_integrator()` in kD to query physical bounds. In kD, `eval_cube_indefinite_integrator()` broadcasts the same lower and upper bounds to each axis. The rectangle evaluator uses inclusion–exclusion over the stored prefix values; it does not accept the upper domain endpoint, which is not stored.
- `make_indefinite_integrator_conv()` takes a supplied lag kernel (one per axis for kD) and returns a **quadrature-scaled causal response**. It is not generally a primitive whose differences give arbitrary weighted subinterval integrals. The convolution-based integrator builders require a nonempty save prefix and write `<prefix>_conv_indefinite_int.tt`. Do not scale these results again.
- `fit()` uses `xfac_quad` runners and saves the approximation and grid under the requested nonempty prefix. Use the full-precision grid JSON (`_grid_E.json` in 1D, `_multgrid_E.json` in kD) when loading fitted data.

## Ordered-simplex integration

Provide a Tucker TTN for the **gap-coordinate** function $f(v_1,\ldots,v_d)$, with one ordered branch per gap and the same zero-based grid `[0,b)` on every axis. With $u_0=t$ and $v_k=u_{k-1}-u_k$, the result is a single MPS on the observation time $t$. For $t=nh$, it represents the inclusive sum

$$
Q^{(h)}_{d,\lambda}(nh)=h^d
\sum_{0\le m_d\le\cdots\le m_1\le n}
 f\bigl((n-m_1)h,(m_1-m_2)h,\ldots,(m_{d-1}-m_d)h\bigr)
 \prod_{k=1}^d\lambda(m_kh).
$$

Pass **one optional kernel MPS** on a matching **1D `Grid`**. The same samples of $\lambda$ weight every integrated time, not the gaps or the observation time $t$. Pass `nullptr` for the unweighted sum; `kernel_grid` is then ignored. Zero gaps and equal times are included, so the discrete value at $t=0$ need not vanish. The upper endpoint `b` is excluded.

Example using complex-double tensors, a 50-bit grid, and $\lambda(u)=1+u$:

```cpp
#include <complex>
#include "QuanticsIntegration/quantics_integrator_kD.hpp"
#include "TensorQuadOperation/generic_construction.hpp"

using Scalar = std::complex<double>;
using Integrator = QuanticsIntegration::QuanticsIntegrator_kD<Scalar>;

Scalar weighted_at_half(const Integrator::mState& values) {
    // values is a 3D Tucker TTN for f(v1, v2, v3) on this grid.
    const Integrator::mGrid grid({0, 0, 0}, {1, 1, 1}, {50, 50, 50});
    const auto [kernel, kernel_grid] =
        TensorQuadOperation::analytic_mps_construction::make_polynomial_orderp_mps<Scalar>(
            0, 1, 50, {Scalar(1), Scalar(1)}, /*ordering=*/2);
    const auto [primitive, time_grid] = Integrator::make_integration_simplex(
        values, grid, &kernel, kernel_grid, /*save_prefix=*/"");
    return Integrator::integrate_simplex(primitive, time_grid, 0.5);
}
```

`integrate_simplex()` also accepts a vector of times and returns values in the same order. The MPS already includes all quadrature factors. An optional save prefix writes the MPS and time grid. An optional `path_logger` prefix writes timestamped step ranks to `<path_logger>logger_integration_simplex.log`.

The default backend is `"zip-up"`; `"qrsvd"` and `"optimize"` are also available. By default, `merge_trucation=true` allows the backend to apply the configured tolerance and bond cap before absorbing the junction. Set it to `false` to defer **merge** truncation until after junction absorption; this does not disable truncation elsewhere. Early truncation can improve speed and reduce intermediate ranks, but should be checked against a reference for new inputs.

See the [simplex contraction derivation](../../latex_note/simplex_contraction_scheme/simplex_contraction.tex), [convolution note](../../latex_note/conv/conv.tex), and [simplex test/plot instructions](../../test/simplex_plot/README.md).
