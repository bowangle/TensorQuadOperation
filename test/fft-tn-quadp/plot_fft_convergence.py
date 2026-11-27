#!/usr/bin/env python3
"""Plot double and dd128 FFT convergence results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCALAR_TYPES = ("double", "dd128")


def plot_scalar_type(scalar_type: str, csv_file: Path, show: bool):
    with csv_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["function"], row["method"], int(row["padding_bits"]))].append(
            (int(row["n_bit"]), float(row["max_error"]))
        )

    output_dir = csv_file.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    functions = sorted({key[0] for key in grouped})
    for function in functions:
        fig, axis = plt.subplots(figsize=(8, 5))
        function_values = []
        for (name, method, padding), values in sorted(grouped.items()):
            if name != function:
                continue
            values.sort()
            function_values.extend(values)
            n_bits = [value[0] for value in values]
            errors = [value[1] for value in values]
            data_line, = axis.plot(
                n_bits,
                errors,
                marker="o",
                linestyle="-" if padding == 0 else "--",
                label=f"{method}, padding={padding}",
            )

            # Anchor both convergence guides to this curve's first point.
            if values and errors[0] > 0:
                anchor_n = n_bits[0]
                anchor_error = errors[0]
                guide_options = {
                    "color": data_line.get_color(),
                    "linewidth": 1.0,
                    "alpha": 0.22,
                    "label": "_nolegend_",
                }
                axis.plot(
                    n_bits,
                    [anchor_error * 2.0 ** (-(n_bit - anchor_n))
                     for n_bit in n_bits],
                    linestyle=":",
                    **guide_options,
                )
                axis.plot(
                    n_bits,
                    [anchor_error * 2.0 ** (-2 * (n_bit - anchor_n))
                     for n_bit in n_bits],
                    linestyle="-.",
                    **guide_options,
                )

        positive_errors = [error for _, error in function_values if error > 0]
        if positive_errors:
            # Generic legend entries for the faint per-curve guides.
            axis.plot([], [], color="gray", linestyle=":", alpha=0.45,
                      linewidth=1.0, label=r"$1/N$ guide")
            axis.plot([], [], color="gray", linestyle="-.", alpha=0.45,
                      linewidth=1.0, label=r"$1/N^2$ guide")
            # Keep reference guides from expanding the plot beyond the data.
            axis.set_ylim(min(positive_errors) / 3, max(positive_errors) * 3)

        axis.set(
            title=f"FFT convergence ({scalar_type}): {function}",
            xlabel="nBit",
            ylabel="maximum absolute error",
            yscale="log",
        )
        axis.grid(True, which="both", linestyle=":")
        axis.legend(fontsize="small")
        fig.tight_layout()
        filename = output_dir / f"fft_convergence_{function}.png"
        fig.savefig(filename, dpi=200)
        print(f"Saved {filename}")
        if show:
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_root",
        nargs="?",
        type=Path,
        default=Path("test/output_fft_conv"),
        help="directory containing the double and dd128 result folders",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    found = False
    for scalar_type in SCALAR_TYPES:
        csv_file = args.output_root / scalar_type / "fft_convergence.csv"
        if not csv_file.exists():
            print(f"Skipping missing {csv_file}")
            continue
        found = True
        plot_scalar_type(scalar_type, csv_file, args.show)

    if not found:
        raise FileNotFoundError(
            f"no convergence CSV found below {args.output_root}"
        )


if __name__ == "__main__":
    main()
