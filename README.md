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

This initialises all submodules and builds the `xfac_quad_runner` dependency chain (Eigen, QD, spdlog, QTgrid‑quad, numeric‑type‑quad).

### Step 3 – Compile

```bash
bash compile.sh
```

Runs CMake (Release, `-march=native`) and builds every test target into `build/`.

### Step 4 – Run the tests (optional)

```bash
bash run_test.sh
```

All 7 test executables run in parallel; their output is saved to `test/test_*.txt`.

---

## Library overview

| Header | Purpose |
|---|---|
| `tensor.h` | `Tensor3D<T>` — the 3‑index building block (left × phys × right) with stride‑aware slice views |
| `tt_base.h` | `TT<T>` — generic Tensor Train: canonicalisation, addition, scalar ops, conjugation, evaluation, save/load |
| `mps_base.h` | `MPS<T>` — Matrix Product State (phys = 2): `dot`, `norm2`, promotion from `TT` |
| `mpo_base.h` | `MPO<T>` — Matrix Product Operator (phys = 4): `from_mps`, `diag`, `transpose`, `_mul` (apply to MPS/MPO) |
| `utc.h` | `utc::zip_up_mpo_mps`, `utc::qrsvd_contract_mpo_mps`, `utc::optimize_dm_generic` — contraction kernels |
| `mat_decomp.h` | QR and SVD decompositions with truncation (reltol + max bond dim), dispatched to LAPACK BDCSVD for standard types and JacobiSVD for extended‑precision |
| `matrix.h` | Thin wrapper pulling Eigen into the project |

---

## Using MPS

### Construction

```cpp
#include "mps_base.h"

// Build from a vector of Tensor3D<T> cores
std::vector<Tensor3D<std::complex<double>>> cores = { /* ... */ };
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
#include "mpo_base.h"

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

The `mat_decomp.h` SVD dispatches to **LAPACK BDCSVD** for `float`/`double`/`complex<float>`/`complex<double>` (3–10× faster) and falls back to **Eigen JacobiSVD** for extended‑precision types.

---

## Test suite

| Test file | What it covers |
|---|---|
| `test_Tensor3D` | `Tensor3D` indexing, slice views, flattening, Armadillo I/O |
| `test_Mat_QR` | QR decomposition correctness |
| `test_Mat_SVDDecomp` | SVD decomposition, reltol truncation, max bond dim |
| `test_tt_base` | `TT` construction, canonicalisation sweeps, arithmetic, conjugation, save/reload |
| `test_mps` | `MPS` dot product, norm, `_initialize_w`, `shift_w`, `compress_svd`, arithmetic, copy independence |
| `test_compress_svd` | Compression quality across bond‑dimension sweeps |
| `test_mpo` | `MPO::from_mps`, `_mul` (zip‑up / qrsvd / optimize), `diag()`, `transpose()`, save/reload |

Run all tests at once:

```bash
bash run_test.sh
```

---

## Project structure

```
TensorQuadOperation/
├── include/              # Public headers
│   ├── tensor.h          # Tensor3D<T>
│   ├── matrix.h          # Eigen include wrapper
│   ├── tt_base.h         # TT<T> base class
│   ├── mps_base.h        # MPS<T>
│   ├── mpo_base.h        # MPO<T>
│   ├── utc.h             # Contraction kernels
│   └── mat_decomp.h      # QR + SVD with truncation
├── test/                 # Test sources + output
│   ├── test.cpp
│   ├── test_Tensor3D.cpp
│   ├── test_Mat_QR.cpp
│   ├── test_Mat_SVDDecomp.cpp
│   ├── test_tt_base.cpp
│   ├── test_mps.cpp
│   ├── test_compress_svd.cpp
│   └── test_mpo.cpp
├── extern/               # External dependencies (xfac_quad_runner & friends)
├── build/                # CMake build directory
├── CMakeLists.txt
├── compile.sh            # CMake configure + build
├── install_extern.sh     # Init submodules + build extern chain
└── run_test.sh           # Run all tests in parallel
```

## License

[Specify your license here]
