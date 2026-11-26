# Simplex integration plots

From the `TensorQuadOperation/` root, run the simplex test suite with `gnuplot` on your `PATH`:

```sh
OMP_NUM_THREADS=2 ./build/tq_test_test_simplex_integration
```

The full run tests `zip-up` and `qrsvd` with merge truncation on and off, and `optimize` with merge truncation on only. It generates 57 high-resolution CSVs and SVGs, including two `dd_128` degree-29 datasets (hand-built and TCI-fitted) with truncated `zip-up` only. Each dataset stem has one of five suffixes: no suffix (untruncated zip-up), `_trunc` (truncated zip-up), `_qrsvd`, `_qrsvd_trunc`, or `_optimize_trunc`. To replot the CSVs, run `gnuplot test/simplex_plot/plot.gp`.

The CSVs, SVGs, and integration logs go into five sibling folders under `test/output/simplex_plot/`: `1D/`, `2D/`, `3D/`, `3Drank30/`, and `3Drank30_tci/`. For example, `3D/double_3D.svg` shows the low-degree polynomial. `3Drank30/double_3D_rank30.svg` uses the explicitly constructed degree-29 TTN; `3Drank30_tci/double_3D_rank30_tci.svg` uses the same polynomial fitted by `QuanticsIntegrator_kD::fit`. The TCI fit artifacts also go in `3Drank30_tci/`. Both degree-29 cases use `(1 + c*(v1 + v2 + v3))^29` with `c = (10^(4/29) - 1)/3`, so the integrand ranges from 1 to just under `1e4` on the fitting cube. They use 50 bits in double and 100 bits in `dd_128`, with 1,000 observation times in both precisions. The degree-29 fits and integrations use a bond cap of 100 and `reltol = 1e4 * epsilon` in the tested precision. The fitted tree and TCI diagnostics use the `double_3D_rank30_tci_fit` prefix. Sweep progress and pivot errors are flushed to `double_3D_rank30_tci_fit.log` as the fit runs.

Each figure shows the MPS and both reference curves, absolute error on a log-y scale, and relative error on log-log axes. The CSV stores error magnitudes computed in the tested precision before plotting. The absolute-error panel includes the discrete-to-continuous quadrature difference. The relative-error panel omits `t=0`, and exact zero errors are not drawn on log axes.
