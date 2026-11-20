# TensorQuadOperation

A high-performance C++20 library for **Matrix Product States (MPS)** and **Matrix Product Operators (MPO)** — also known as Tensor Trains (TT) — with support for **quadruple‑precision arithmetic** (float128, double‑double) and generic scalar types via Eigen.

The library provides construction, canonicalisation, compression, arithmetic, and contraction of MPS/MPO tensors, as well as TCI (Tensor Cross Interpolation) for constructing tensor trains from black‑box functions.

---

## Installation

### Prerequisites

- **C++20** compiler (GCC ≥ 12 or Clang ≥ 16)
- **CMake** ≥ 3.21
- **Eigen** (fetched automatically as part of the extern chain)
- **Boost** (for `boost::multiprecision::float128`)
- **OpenMP**
- **spdlog**, **QD** (double‑double) — all handled by the external submodules

### Step 1 – Clone

```bash
git clone https://github.com/bowangle/TensorQuadOperation.git
cd TensorQuadOperation
```

### Step 2 – Install externals

```bash
bash install_extern.sh
```

This initialises all submodules and sets up the merged `xfac_quad` dependency (QTGrid, numerical core, and runners), including Eigen, QD, spdlog, JSON, and numeric‑type‑quad.

### Step 3 – Compile

```bash
bash compile.sh
```

Runs CMake (Release, `-march=native`) and builds every test target into `build/`.

### Step 4 – Run the tests (optional)

```bash
bash run_test.sh
```

The test executables run in parallel; their logs are saved beside the grouped sources in `test/TensorQuadOperation/` and `test/QuanticsIntegration/`. The script returns a failure status if any test fails.

---

## Library overview

Public headers are split into two folders with matching C++ namespaces:

- `TensorQuadOperation`: tensor types, arithmetic, construction, and contraction helpers. The nested helper namespaces are `TensorQuadOperation::utc` and `TensorQuadOperation::analytic_mps_construction`.
- `QuanticsIntegration`: grid-aware fitting and integration algorithms, depending on `TensorQuadOperation`, not the reverse.

The CMake target `tensorquad_deps` exposes the `include/` root and dependencies for both layers. Use the prefixed include paths and qualified names; the old flat headers and global project names are no longer provided. Serialized tensor and grid formats are unchanged.

```cpp
#include "TensorQuadOperation/mps_base.h"
#include "TensorQuadOperation/ttn_base.hpp"
#include "QuanticsIntegration/quantics_integrator.hpp"
#include "QuanticsIntegration/quantics_integrator_kD.hpp"

using State = TensorQuadOperation::MPS<double>;
using TreeState = TensorQuadOperation::TTN<double>;
using Integrator1D = QuanticsIntegration::QuanticsIntegrator1D<double>;
using IntegratorKD = QuanticsIntegration::QuanticsIntegrator_kD<double>;
```

| Header | Purpose |
|---|---|
| `xfac_quad/cubemat_helper.h` (dependency) | Shared `xfac_quad::Tensor3D<T>` — column-major 3-index tensors with views, copies, and serialization |
| `TensorQuadOperation/tt_base.h` | `TT<T>` — generic Tensor Train: canonicalisation, addition, scalar ops, conjugation, evaluation, save/load |
| `TensorQuadOperation/mps_base.h` | `MPS<T>` — Matrix Product State (phys = 2): `dot`, `norm2`, convolution, translation, promotion from `TT` |
| `TensorQuadOperation/mpo_base.h` | `MPO<T>` — Matrix Product Operator (phys = 4): `from_mps`, `diag`, `transpose`, `_mul` (apply to MPS) |
| `TensorQuadOperation/ttn_base.hpp` | `TTN<T>` — tree tensor networks, canonicalisation, compression, branch operations, and separable-kernel convolution |
| `TensorQuadOperation/utc.h` | `utc::zip_up_mpo_mps`, `utc::qrsvd_contract_mpo_mps`, `utc::optimize_dm_generic` — contraction kernels |
| `TensorQuadOperation/mat_decomp.h` | QR and SVD decompositions with truncation (reltol + max bond dim), dispatched to LAPACK BDCSVD for standard types and JacobiSVD for extended‑precision |
| `TensorQuadOperation/matrix.h` | Thin wrapper pulling Eigen into the project |
| `TensorQuadOperation/magic_tensor.hpp` | Binary addition tensors and convolution operators |
| `TensorQuadOperation/generic_construction.hpp` | Analytic MPS construction in `analytic_mps_construction` |
| `QuanticsIntegration/quantics_integrator.hpp` | `QuanticsIntegrator1D<Scalar, Sint>` and its `QUANTICS_INT_1D` alias — 1D fitting and integration |
| `QuanticsIntegration/quantics_integrator_kD.hpp` | `QuanticsIntegrator_kD<Scalar, Sint>` — multidimensional tree fitting and integration |
| `QuanticsIntegration/indefinite_integral.hpp` | Heaviside MPOs and grid-aware indefinite-integration helpers |

---

## Using MPS

### Construction

```cpp
#include "TensorQuadOperation/mps_base.h"

using TensorQuadOperation::MPS;

// Build from a vector of shared Tensor3D cores
std::vector<xfac_quad::Tensor3D<std::complex<double>>> cores = { /* ... */ };
MPS<std::complex<double>> psi(cores);

// Load from a TCI‑generated file (Armadillo cube text format)
MPS<std::complex<double>> psi("path/to/file.tt");

// Load with truncation parameters
MPS<std::complex<float128>> psi("file.tt",
    /*max_bond_dim=*/32,
    /*reltol=*/1e-12,
    /*w=*/0);          // canonical centre at site 0
```

### Core operations

```cpp
// --- Evaluation ---
std::vector<int> idx = {0, 1, 0, 1, ...};
auto value = psi.eval(idx);                    // single point
auto pts   = psi.generate_points(1000);        // random points
auto vals  = psi.eval_list(pts);               // batch evaluation

// --- Arithmetic ---
auto sum  = psi1 + psi2;                       // TT addition (block‑diagonal direct sum)
auto diff = psi1 - psi2;                       // subtraction
auto neg  = -psi;                              // unary minus
auto scaled = psi * std::complex<double>(3.0); // scalar multiplication
psi += psi2;                                   // in‑place addition
psi *= 2.0;                                    // in‑place scalar multiply

// --- Conjugation ---
auto psi_conj = psi.conj();                    // element‑wise complex conjugate

// --- Canonicalisation ---
psi._initialize_w(center);                     // bring to canonical form with centre at `center`
psi.shift_w(5);                                // move canonical centre to site 5
psi.check_canonical();                         // verify orthogonality constraints

// --- Compression ---
psi.compress_svd(1e-12, 32);                   // SVD truncation: reltol + max bond dim

// --- Dot product & norm ---
auto overlap  = psi1.dot(psi2);                // <psi1 | psi2>
auto norm_sq  = psi1.norm2();                  // <psi | psi> (real, ≥ 0)
```

### Save / Load

```cpp
psi.save("my_state");     // writes my_state.tt (Armadillo cube text format)
MPS<T> reloaded("my_state.tt");
```

---

## Using MPO

### Construction from an MPS (diagonal operator)

```cpp
#include "TensorQuadOperation/mpo_base.h"

using TensorQuadOperation::MPS;
using TensorQuadOperation::MPO;

MPS<std::complex<double>> psi("state.tt");
MPO<std::complex<double>> rho = MPO<std::complex<double>>::from_mps(psi);
// rho represents diag(psi) — the diagonal density‑matrix operator
```

### MPO‑specific operations

```cpp
// --- Transpose ---
MPO<std::complex<double>> O_T = O.transpose();  // swap row/col on each physical leg

// --- Extract diagonal (MPO → MPS) ---
MPS<std::complex<double>> diag_mps = O.diag();  // keep only (0,0) and (1,1) entries

// --- Apply MPO to MPS (three contraction methods) ---
MPS<std::complex<double>> result;

// 1. zip‑up: efficient direct contraction with SVD truncation
result = O._mul(psi, "zip-up");

// 2. qrsvd: QR + SVD contraction
result = O._mul(psi, "qrsvd");

// 3. optimize: variational optimisation with 1‑site or 2‑site sweeps
result = O._mul(psi, "optimize", /*order=*/2, /*n_sweeps=*/2,
                /*previous=*/nullptr);
```

### Arithmetic (inherited from `TT`)

All `TT` arithmetic operators work on `MPO` as well: `+`, `-`, `*scalar`, `/scalar`, `conj()`, `compress_svd(...)`, `save(...)`, etc.

---

## Supported scalar types

The library is fully templated on the scalar type `T`. The following are tested:

| Scalar type | Description |
|---|---|
| `std::complex<double>` | Standard double precision |
| `std::complex<float128>` | Boost 128‑bit float (quad precision) |
| `std::complex<dd_128>` | Double‑double (QD library) |

The `TensorQuadOperation/mat_decomp.h` SVD dispatches to **LAPACK BDCSVD** for `float`/`double`/`complex<float>`/`complex<double>` (3–10× faster) and falls back to **Eigen JacobiSVD** for extended‑precision types.

---

## Test suite

Test sources mirror the two public-header folders. Paths below are relative to `test/`; existing executable and CTest names are unchanged. Both CTest and `run_test.sh` use the project root as the working directory, so shared data stays under `test/output/`.

| Test source | What it covers |
|---|---|
| `TensorQuadOperation/test.cpp` | Standalone tensor-file I/O diagnostic (built separately from the script's test suite) |
| `TensorQuadOperation/test_Tensor3D.cpp` | `Tensor3D` indexing, slice views, flattening, Armadillo I/O |
| `TensorQuadOperation/test_Mat_QR.cpp` | QR decomposition correctness |
| `TensorQuadOperation/test_Mat_SVDDecomp.cpp` | SVD decomposition, reltol truncation, max bond dim |
| `TensorQuadOperation/test_tt_base.cpp` | `TT` construction, canonicalisation sweeps, arithmetic, conjugation, save/reload |
| `TensorQuadOperation/test_mps.cpp` | MPS operations, arithmetic, and copy independence |
| `TensorQuadOperation/test_compress_svd.cpp` | Compression quality across bond‑dimension sweeps |
| `TensorQuadOperation/test_mpo.cpp` | MPO construction, multiplication, diagonal extraction, transpose, and save/reload |
| `TensorQuadOperation/test_ttn.cpp` | Tree canonicalisation, compression, scaling, and branch operations |
| `TensorQuadOperation/test_analytics_construction.cpp` | Analytic MPS construction |
| `TensorQuadOperation/test_convolution.cpp` | MPS convolution and translation |
| `QuanticsIntegration/test_indefinite_integral.cpp` | Heaviside MPO for exclusive-prefix integration |
| `QuanticsIntegration/test_quantics_integrator.cpp` | One-dimensional fitting, range sums, and indefinite integration |
| `QuanticsIntegration/test_quantics_integrator_kD.cpp` | Tree fitting, rectangular integration, and convolution-based integration comparisons |

Run all tests at once:

```bash
bash run_test.sh
```

---

## Project structure

```
TensorQuadOperation/
├── include/                         # Public include root
│   ├── TensorQuadOperation/         # TensorQuadOperation namespace
│   │   ├── matrix.h
│   │   ├── mat_decomp.h
│   │   ├── tt_base.h
│   │   ├── mps_base.h
│   │   ├── mpo_base.h
│   │   ├── ttn_base.hpp
│   │   ├── utc.h
│   │   ├── magic_tensor.hpp
│   │   └── generic_construction.hpp
│   └── QuanticsIntegration/         # QuanticsIntegration namespace
│       ├── quantics_integrator.hpp
│       ├── quantics_integrator_kD.hpp
│       └── indefinite_integral.hpp
├── test/
│   ├── TensorQuadOperation/         # Tensor-operation test sources and logs
│   │   ├── test.cpp
│   │   ├── test_Tensor3D.cpp
│   │   ├── test_Mat_QR.cpp
│   │   ├── test_Mat_SVDDecomp.cpp
│   │   ├── test_tt_base.cpp
│   │   ├── test_mps.cpp
│   │   ├── test_mpo.cpp
│   │   ├── test_ttn.cpp
│   │   ├── test_compress_svd.cpp
│   │   ├── test_analytics_construction.cpp
│   │   └── test_convolution.cpp
│   ├── QuanticsIntegration/         # Integration test sources and logs
│   │   ├── test_indefinite_integral.cpp
│   │   ├── test_quantics_integrator.cpp
│   │   └── test_quantics_integrator_kD.cpp
│   └── output/                     # Shared generated fixtures (ignored)
├── extern/               # External dependencies (xfac_quad and Catch2)
├── build/                # CMake build directory
├── CMakeLists.txt
├── compile.sh            # CMake configure + build
├── install_extern.sh     # Init submodules + build extern chain
└── run_test.sh           # Run all tests in parallel
```

## License

[Specify your license here]
