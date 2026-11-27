# TensorQuadOperation

A C++20 library for tensor trains, tree tensor networks, and quantics integration. It supports standard and extended precision through Eigen and `xfac_quad`, including `double`, Boost `float128`, and QD double-double (`dd_128`), with real and complex scalars where supported by the operation.

## Documentation

| Namespace | What it provides |
|---|---|
| [**TensorQuadOperation**](include/TensorQuadOperation/README.md) | `TT`, `MPS`, `MPO`, and `TTN`; tensor construction, arithmetic, canonicalization, compression, contraction, and discrete convolution. |
| [**QuanticsIntegration**](include/QuanticsIntegration/README.md) | Grid-aware TCI fitting, definite and indefinite integration, rectangular queries, and weighted or unweighted simplex integration. |

The integration layer uses the tensor layer, not the other way around.

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

`install_extern.sh` initializes the submodules and sets up the external dependencies. `compile.sh` configures a Release build with `-march=native` and builds the test executables in `build/`. To build without this project's test targets, configure CMake with `-DTQ_BUILD_TESTS=OFF`.

## How to use it

Add the repository as a CMake subdirectory and link your executable against the interface target. It exposes both public header trees and their dependencies.

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

## Tests

Run from the repository root:

```sh
bash run_test.sh
# Or use CTest:
ctest --test-dir build --output-on-failure
# Run the simplex suite on its own:
./build/tq_test_test_simplex_integration
```

The runner executes tests in parallel and keeps their console logs beside the sources in `test/TensorQuadOperation/` and `test/QuanticsIntegration/`. Generated tensors, grids, and plots go under the ignored `test/output/` directory. See the [simplex plotting README](test/simplex_plot/README.md) for output locations and how to replot CSVs.

## Project structure

```text
TensorQuadOperation/
├── include/
│   ├── TensorQuadOperation/    # TT, MPS, MPO, TTN, and contraction helpers
│   └── QuanticsIntegration/   # Grid-aware fitting and integration
├── test/
│   ├── TensorQuadOperation/    # Tensor-operation tests and runner logs
│   ├── QuanticsIntegration/   # Fitting and integration tests and runner logs
│   ├── simplex_plot/          # Plotting script and instructions
│   └── output/               # Generated fixtures, diagnostic logs, and plots (ignored)
├── latex_note/                # Convolution and simplex derivations
├── extern/                    # xfac_quad and Catch2 submodules
├── build/                     # Generated CMake build directory
├── CMakeLists.txt
├── install_extern.sh          # Initialize and set up dependencies
├── compile.sh                 # Configure and build
└── run_test.sh                # Run the test executables
```

## License

[Specify your license here]
