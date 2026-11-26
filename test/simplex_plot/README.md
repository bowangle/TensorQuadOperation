# Simplex integration plots

From the `TensorQuadOperation/` root, run the simplex test suite with `gnuplot` on your `PATH`:

```sh
OMP_NUM_THREADS=2 ./build/tq_test_test_simplex_integration
```

The full run writes high-resolution CSVs and generates the SVGs automatically. To replot existing CSVs, run `gnuplot test/simplex_plot/plot.gp`.

The CSVs, SVGs, and per-dataset integration logs go to `test/output/simplex_plot/`. For example, `double_3D.svg` shows the low-degree polynomial. `double_3D_rank30.svg` uses the explicitly constructed degree-29 TTN; `double_3D_rank30_tci.svg` uses the same polynomial fitted by `QuanticsIntegrator_kD::fit`. Both degree-29 cases use `(1 + c*(v1 + v2 + v3))^29` with `c = (10^(4/29) - 1)/3`, so the integrand ranges from 1 to just under `1e4` on the fitting cube. They use 50 bits and 1,000 observation times. The TCI fit and both degree-29 integrations use a bond cap of 100 and `reltol = 1e4 * epsilon(double)`. The fitted tree and TCI diagnostics use the `double_3D_rank30_tci_fit` prefix. Sweep progress and pivot errors are flushed to `double_3D_rank30_tci_fit.log` as the fit runs.

Each figure shows the MPS and both reference curves, absolute error on a log-y scale, and relative error on log-log axes. The CSV stores error magnitudes computed in the tested precision before plotting. The absolute-error panel includes the discrete-to-continuous quadrature difference. The relative-error panel omits `t=0`, and exact zero errors are not drawn on log axes.
