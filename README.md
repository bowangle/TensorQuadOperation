# TensorQuadOperation

A C++20 library for tensor trains, tree tensor networks, quantics integration, and tensor-network Fourier transforms. It supports standard and extended precision through Eigen and `xfac_quad`, including `double`, Boost `float128`, and QD double-double (`dd_128`), with real and complex scalars where supported by the operation.

## Documentation

| Namespace | What it provides |
|---|---|
| [**TensorQuadOperation**](include/TensorQuadOperation/README.md) | `TT`, `MPS`, `MPO`, and `TTN`; tensor construction, arithmetic, canonicalization, compression, contraction, and discrete convolution. |
| [**QuanticsIntegration**](include/QuanticsIntegration/README.md) | Grid-aware TCI fitting, definite and indefinite integration, rectangular queries, and weighted or unweighted simplex integration. |
| [`fft_tn_quadp`](src/fft-tn-quadp/README.md) | Tensor-network FFT using QFT MPOs, quadrature and discontinuity corrections, and QFT precomputation. See also the [FFT derivation](latex_note/fft_notes_v2/fft_note_v2.tex). |
| [`quad_integration`](include/quad_integration/integration_method.hpp) | A 1D interface for adaptive quadrature, quantics fitting and integration, and saving or loading integration parameters. |

`QuanticsIntegration` and `fft_tn_quadp` use the core tensor operations. The `quad_integration` interface brings together adaptive quadrature from `fft_tn_quadp` and the 1D quantics integrator.

## Installation

### Prerequisites

- A C++20 compiler (GCC 12+ or Clang 16+) with OpenMP support
- CMake 3.21+, Git, Bash, and Boost headers
- `gnuplot` for automatic plots when running the full simplex test suite

The external submodules provide `xfac_quad`, Eigen, QD, spdlog, JSON, and Catch2.

### Clone, install dependencies, and build

```sh
git clone https://github.com/bowangle/TensorQuadOperation.git
cd TensorQuadOperation
bash install_extern.sh
bash compile.sh
```

`install_extern.sh` initializes the submodules and sets up the external dependencies. `compile.sh` configures a Release build with `-march=native` and builds the tools and test executables in `build/`. To build without this project's test targets, configure CMake with `-DTQ_BUILD_TESTS=OFF`.

## How to use it

Add the repository as a CMake subdirectory and link your executable against the interface target. It exposes the public headers under `include/` and their dependencies.

```cmake
add_subdirectory(TensorQuadOperation)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE tensorquad_deps)
```

Use the namespace-prefixed headers. This example constructs a constant MPS, evaluates a sample, and integrates it on a 1D grid:

```cpp
#include <complex>
#include <iostream>
#include "TensorQuadOperation/generic_construction.hpp"
#include "QuanticsIntegration/quantics_integrator.hpp"

int main() {
    using Scalar = std::complex<double>;
    using Integrator = QuanticsIntegration::QuanticsIntegrator1D<Scalar>;
    constexpr int bits = 8;

    const Integrator::Grid grid(0.0, 1.0, bits);
    const auto values =
        TensorQuadOperation::analytic_mps_construction::make_one_mps<Scalar>(bits);
    std::cout << values.eval(grid.coord_to_id(0.5)) << '\n';

    const Integrator integrator(grid, values);
    std::cout << integrator.integrate() << '\n';  // Integral of 1 on [0, 1).
}
```

See the [tensor operations guide](include/TensorQuadOperation/README.md) for MPO application and contraction methods, and the [integration guide](include/QuanticsIntegration/README.md) for TCI fitting, rectangular queries, and weighted simplex integration.

### Executables

- `build/precompute_qft` builds and saves QFT MPOs in `mpo_data/data/`. Run `./build/precompute_qft --help` from the repository root for options. FFT tests can also generate missing cached MPOs on demand.
- `build/integration_method` is a smoke executable for the `quad_integration` interface. Use `#include "quad_integration/integration_method.hpp"` from C++ to call the integration routines.

## Tests

Run from the repository root:

```sh
bash run_test.sh                     # All three project test groups
bash run_test_TensorQuadOperation.sh
bash run_test_QuanticsIntegration.sh  # Includes simplex integration
bash run_test_fft-tn-quadp.sh
# Or use CTest:
ctest --test-dir build --output-on-failure
# Run the simplex suite on its own:
./build/tq_test_test_simplex_integration
```

The runners keep console logs beside the sources in `test/TensorQuadOperation/`, `test/QuanticsIntegration/`, and `test/fft-tn-quadp/`. FFT tests run serially in the shell runner because they may populate a shared QFT cache; the other groups run their tests in parallel. Generated tensors, grids, and plots go under the ignored `test/output/` directory. See the [simplex plotting README](test/simplex_plot/README.md) for output locations and how to replot CSVs.

## Project structure

```text
TensorQuadOperation/
├── include/
│   ├── TensorQuadOperation/    # TT, MPS, MPO, TTN, and contraction helpers
│   ├── QuanticsIntegration/   # Grid-aware fitting and integration
│   ├── fft-tn-quadp/         # FFT, QFT MPOs, and adaptive quadrature
│   └── quad_integration/     # High-level 1D integration interface
├── src/
│   ├── fft-tn-quadp/         # QFT precompute executable and plotting helper
│   └── quad_integration/     # Integration interface smoke executable
├── test/
│   ├── TensorQuadOperation/    # Tensor-operation tests and runner logs
│   ├── QuanticsIntegration/   # Fitting and integration tests and runner logs
│   ├── fft-tn-quadp/          # FFT/QFT tests and runner logs
│   ├── simplex_plot/          # Plotting script and instructions
│   └── output/               # Generated fixtures, diagnostic logs, and plots (ignored)
├── latex_note/                # Convolution, simplex, and FFT derivations
├── extern/                    # xfac_quad and Catch2 submodules
├── build/                     # Generated CMake build directory
├── CMakeLists.txt
├── install_extern.sh          # Initialize and set up dependencies
├── compile.sh                 # Configure and build
└── run_test.sh                # Run the test executables
```

## License

[Specify your license here]
