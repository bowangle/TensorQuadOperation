# fft-tn-quadp

Quad-precision QFT MPO precomputation using tensor networks.

## Quick Start

```bash
# 1. Install external dependencies
./install_extern.sh

# 2. Compile (tests + executable)
./compile.sh

# 3. Precompute QFT MPOs
./build/precompute_qft +double +dd128 +float128 +C max_a=127
python3 src/plot_precompute_qft.py

# 4. Run tests
./run_test.sh
python3 test/plot_errors_qft_mpo.py

# Figures are saved after each script:
#   src/plot_precompute_qft.py   → mpo_data/figure/:
#     - precompute_qft_chi_err.*  : chi & relative error vs nBit
#     - precompute_qft_timing.*   : t_construction & t_all vs nBit
#   test/plot_errors_qft_mpo.py  → test/figure/:
#     - errors_chi_max{...}.*     : rms & relative error vs chi
#     - size_chi_max{...}.*       : tot_nb_value & relative error vs max bond dim
```

## Dependencies

- CMake ≥ 3.21
- C++20 compiler (GCC or Clang)
- Python 3 with `matplotlib` and `numpy` (for plotting, steps 3 & 4)
- Boost (system-installed, found via `find_package`)

`install_extern.sh` fetches and builds all external dependencies into `extern/`.

```
fft-tn-quadp
└── TensorQuadOperation     MPS/MPO contractions & compression
    + OpenMP
    └── xfac_quad_runner    tensor-cross interpolation runner
        + OpenMP
        + spdlog (header-only)
        └── xfac_quad       tensor-cross interpolation
            + OpenMP
            + Boost (headers)
            └── QTgrid-quad  grid discretization
                + nlohmann/json (header-only)
                └── numeric-type-quad  extended-precision types
                    + Eigen (header-only)
                    + QD        double-double & quad-double
                    + quadmath  (GCC only)
```

Each level only knows its immediate child via `add_subdirectory()`.

## Build Targets

| Target            | Description                          |
|-------------------|--------------------------------------|
| `Test_qft_mpo`    | Unit tests for the QFT MPO builder   |
| `calc_MPO_QFT`    | Precompute MPOs and save to disk     |

## CLI Usage — `calc_MPO_QFT`

```bash
Usage: calc_MPO_QFT [+double] [+float128] [+dd128] [+all]
                     [+compress|+C] [min_a=N] [max_a=N] [chi=N] [eps_factor|ef=N]
```

### Type toggles (all off by default)

| Flag         | Description            |
|--------------|------------------------|
| `+double`    | `std::complex<double>` |
| `+float128`  | `boost::float128`      |
| `+dd128`     | double-double (qd)     |
| `+all`       | enable all three       |

### Other flags

| Flag              | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `+compress`, `+C` | off     | enable SVD compression                   |
| `min_a=N`         | 10      | minimum number of bits                   |
| `max_a=N`         | 100     | maximum number of bits                   |
| `chi=N`           | 35      | bond dimension for MPO construction      |
| `eps_factor=N`, `ef=N` | 2 | tolerance = 10^N × ε (for compression)   |

### Examples

```bash
# Precompute with all types, compression, up to 60 bits
./build/calc_MPO_QFT +double +dd128 +float128 +C max_a=60

# Quick run with just double, no compression, 10-30 bits
./build/calc_MPO_QFT +double min_a=10 max_a=30 chi=20
```

## Output

Precomputed MPOs and info files are saved to `mpo_data/data/`. Figures are saved to `mpo_data/figure/`.
