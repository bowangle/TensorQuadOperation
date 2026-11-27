# TensorQuadOperation

The `TensorQuadOperation` namespace provides tensor representations and operations without requiring an integration grid. For fitting and quadrature, see [QuanticsIntegration](../QuanticsIntegration/README.md).

## What's available

| Public header | Types and operations |
|---|---|
| [`tt_base.hpp`](tt_base.hpp) | `TT<T>`: tensor-train construction, evaluation, batch evaluation, arithmetic, conjugation, QR canonicalization, SVD compression, and save/load. |
| [`mps_base.hpp`](mps_base.hpp) | `MPS<T>`: matrix product states, overlaps (`dot`), squared norms (`norm2`), open-boundary contractions (`contract_env`), translations, and causal convolution. |
| [`mpo_base.hpp`](mpo_base.hpp) | `MPO<T>`: matrix product operators with separate input/output dimensions, diagonal construction (`from_mps`), diagonal extraction (`diag`), transpose, and application to an MPS (`_mul`). |
| [`ttn_base.hpp`](ttn_base.hpp) | `TTN<T>`: tree tensor networks backed by `xfac_quad::TensorTree<T>`, with evaluation, save/load, rank and storage inspection, scalar scaling, canonical-center moves, tree-wide SVD compression, branch operations, and convolution with one 1D kernel per branch. |
| [`utc.hpp`](utc.hpp) | Lower-level contraction routines in `utc::zip_up`, `utc::qrsvd`, and `utc::optimized`. |
| [`generic_construction.hpp`](generic_construction.hpp) | Exact core constructions in `analytic_mps_construction`, including ones, basis states, point defects, steps, exponentials, sine/cosine, and polynomials. Some factories also return the associated 1D grid. |
| [`magic_tensor.hpp`](magic_tensor.hpp) | Binary carry tensors used for addition, subtraction constraints, and convolution. |

For example, apply a diagonal operator to a binary MPS:

```cpp
#include <complex>
#include "TensorQuadOperation/generic_construction.hpp"
#include "TensorQuadOperation/mpo_base.hpp"

using Scalar = std::complex<double>;
using TensorQuadOperation::MPO;

const auto state = TensorQuadOperation::analytic_mps_construction::make_one_mps<Scalar>(8);
auto op = MPO<Scalar>::from_mps(state);
const auto result = op._mul(state, "zip-up");
```

## Layout, gauges, and contraction methods

- Train storage orderings `0`, `1`, and `2` are `(physical, left, right)`, `(left, physical, right)`, and `(left, right, physical)`. `TT<T>::change_ordering()` changes storage layout, not site order. The binary convolution and analytic quantics constructions use LSB-first sites.
- An MPO stores a fused physical index `output * input_dim + input`. MPO–MPS application contracts the input leg and preserves the output leg. Use matching storage orderings on the operands.
- `w = -1` means no canonical center is assigned. For trains, `shift_w()` moves the center; for trees, use `canonify_to_center()` or `move_canonical_center()`. Tree bond axes follow the topology's neighbor order. `TTN::mul_branch()` applies an MPO to a branch and reinserts the result.
- `"zip-up"` contracts and can truncate as it proceeds. `"qrsvd"` forms a QR contraction and then compresses in a separate SVD sweep. `"optimize"` uses variational sweeps. A rank cap limits retained ranks where compression is applied, not necessarily temporary dimensions. A local relative tolerance is not a global error guarantee.
- `MPS::conv()` and `TTN::conv()` compute discrete **causal, nonperiodic** convolution. They do not conjugate or apply grid spacing. In the tree case, the kernels are separable, but the input TTN need not be.

Core storage and decomposition come from `xfac_quad`. Use `xfac_quad::Tensor3D<T>` for cores and `xfac_quad::MatQR`, `xfac_quad::SVDDecomp`, or `xfac_quad::MatSVDFixedTol` from `xfac_quad/matrix/mat_decomp.h` for standalone decompositions.

Train persistence uses the `.tt` format with cores saved in ordering 1; tree persistence stores both topology and cores. See the [project README](../../README.md) for build instructions and [tensor-operation tests](../../test/TensorQuadOperation/) for more examples.
