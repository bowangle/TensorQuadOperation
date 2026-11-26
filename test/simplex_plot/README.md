# Simplex integration plots

From the `TensorQuadOperation/` root, run the simplex test suite. It runs all resolutions and writes the high-resolution CSVs without any filter. Then plot them with gnuplot:

```sh
OMP_NUM_THREADS=2 ./build/tq_test_test_simplex_integration
gnuplot test/simplex_plot/plot.gp
```

The CSVs and SVGs go to `test/output/simplex_plot/`, one of each per dimension and precision. For example, open `test/output/simplex_plot/double_3D.svg` for the 50-bit 3D double-precision case.

Each figure shows the MPS and both reference curves, absolute error on a log-y scale, and relative error on log-log axes. The CSV stores error magnitudes computed in the tested precision before plotting. The absolute-error panel includes the discrete-to-continuous quadrature difference. The relative-error panel omits `t=0`, and exact zero errors are not drawn on log axes.
