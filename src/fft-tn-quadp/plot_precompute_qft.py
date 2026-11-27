#!/usr/bin/env python3
"""
Load precomputed QFT MPO info files from mpo_data/ and plot:

  Single figure with four subplots:
    1. chi (before & after compression) vs nBit
    2. relative rms error (before & after compression) vs nBit
    3. t_construction vs nBit
    4. t_all vs nBit

All data types are overlaid on the same figure.

Info file format (one data line after a #-comment header):
    (real,0)  chi_before  (real,0)  chi_after  t_construction  t_all

Usage:
    python plot_precompute_qft.py [--dir mpo_data] [--save plot.png]
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(fname):
    """
    Parse a filename like:
        QFT_nBit=30_Cdouble_info.txt
        compress_QFT_nBit=30_Cdouble_info.txt
    Returns dict with keys: nBit, type_name, compressed (bool)
    """
    base = os.path.basename(fname).replace("_info.txt", "")
    m = re.match(r"compress_QFT_nBit=(\d+)_(.+)", base)
    if m:
        return {
            "nBit": int(m.group(1)),
            "type_name": m.group(2),
            "compressed": True,
        }
    m = re.match(r"QFT_nBit=(\d+)_(.+)", base)
    if m:
        return {
            "nBit": int(m.group(1)),
            "type_name": m.group(2),
            "compressed": False,
        }
    return None


def load_info_file(fname):
    """
    Load an info file.
    Returns (error_before, chi_before, error_after, chi_after, t_construction, t_all).
    The file has one #-comment line then one data line.
    """
    with open(fname, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise ValueError(f"Empty data in {fname}")
    parts = lines[0].split()
    # parts: [(real,0), chi_before, (real,0), chi_after, t_construction, t_all]
    if len(parts) < 4:
        raise ValueError(f"Unexpected format in {fname}: {lines[0]}")

    def parse_complex(s):
        s = s.strip("()")
        vals = s.split(",")
        return float(vals[0])

    error_before = parse_complex(parts[0])
    chi_before    = int(parts[1])
    error_after  = parse_complex(parts[2])
    chi_after     = int(parts[3])
    t_construction = float(parts[4]) if len(parts) >= 5 else float('nan')
    t_all          = float(parts[5]) if len(parts) >= 6 else float('nan')
    return error_before, chi_before, error_after, chi_after, t_construction, t_all


def plot_all(data_dir):
    """Find all info files and plot them."""
    figure_dir = "mpo_data/figure"
    os.makedirs(figure_dir, exist_ok=True)

    pattern = os.path.join(data_dir, "*_info.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    # Collect entries
    entries = []
    for f in files:
        meta = parse_filename(f)
        if meta is None:
            print(f"WARNING: Could not parse filename: {f}")
            continue
        try:
            err_b, chi_b, err_a, chi_a, t_constr, t_all = load_info_file(f)
        except Exception as e:
            print(f"WARNING: {e}")
            continue
        entries.append({
            "nBit": meta["nBit"],
            "type_name": meta["type_name"],
            "compressed": meta["compressed"],
            "error_before": err_b,
            "chi_before": chi_b,
            "error_after": err_a,
            "chi_after": chi_a,
            "t_construction": t_constr,
            "t_all": t_all,
        })

    if not entries:
        print("No valid entries found.")
        sys.exit(1)

    # ---- Assign colors by type_name, markers/linestyles by compress status ----
    type_colors = {}
    color_pool = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def get_color(type_name):
        if type_name not in type_colors:
            type_colors[type_name] = color_pool[len(type_colors) % len(color_pool)]
        return type_colors[type_name]

    # marker + linestyle by (compressed, before/after)
    # uncompressed: circle + solid
    # compressed before: square + dotted
    # compressed after:  square + dashed
    style_map = {
        (True,  "before"):  dict(marker='o', ls=':',  label='compr before'),
        (True,  "after"):   dict(marker='s', ls='--', label='compr after'),
    }

    # Separate chi and error for compressed/uncompressed per type
    # For uncompressed files: error_before == error_after, chi_before == chi_after
    # For compressed files:   error_before/after differ, chi_before/after differ

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle("Precomputed QFT MPO — chi & error vs nBit", fontsize=14)

    # Collect types present
    types_seen = sorted(set(e["type_name"] for e in entries))

    for t in types_seen:
        c = get_color(t)

        # Only compressed data
        comp = [e for e in entries if e["type_name"] == t and e["compressed"]]

        if comp:
            comp.sort(key=lambda e: e["nBit"])
            nbits = [e["nBit"] for e in comp]

            chi_b = [e["chi_before"] for e in comp]
            chi_a = [e["chi_after"]  for e in comp]
            err_b = [e["error_before"] for e in comp]
            err_a = [e["error_after"]  for e in comp]

            sb = style_map[(True, "before")]
            sa = style_map[(True, "after")]

            label_b = f"{t} ({sb['label']})"
            label_a = f"{t} ({sa['label']})"

            ax1.plot(nbits, chi_b, marker=sb['marker'], color=c, linestyle=sb['ls'],
                     markersize=5, label=label_b)
            ax1.plot(nbits, chi_a, marker=sa['marker'], color=c, linestyle=sa['ls'],
                     markersize=5, label=label_a)

            ax2.plot(nbits, err_b, marker=sb['marker'], color=c, linestyle=sb['ls'],
                     markersize=5, label=label_b)
            ax2.plot(nbits, err_a, marker=sa['marker'], color=c, linestyle=sa['ls'],
                     markersize=5, label=label_a)

    # Add faint horizontal line at last chi_after for each type
    for t in types_seen:
        comp = [e for e in entries if e["type_name"] == t and e["compressed"]]
        if comp:
            last = max(comp, key=lambda e: e["nBit"])
            ax1.axhline(y=last["chi_after"], color=get_color(t), linestyle='--', alpha=0.3, linewidth=1)
            ax1.text(ax1.get_xlim()[1], last["chi_after"], f" {last['chi_after']}",
                     color=get_color(t), fontsize=7, va='center', ha='left')

    ax1.set_xlabel("nBit")
    ax1.set_ylabel("chi")
    ax1.set_title("chi vs nBit")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("nBit")
    ax2.set_ylabel("relative rms |MPO - F|")
    ax2.set_title("relative rms error vs nBit")
    ax2.set_yscale("log")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Add faint horizontal lines at last error_before & error_after for each type
    for t in types_seen:
        comp = [e for e in entries if e["type_name"] == t and e["compressed"]]
        if comp:
            last = max(comp, key=lambda e: e["nBit"])
            for val, lbl in [(last["error_before"], "before"), (last["error_after"], "after")]:
                ax2.axhline(y=val, color=get_color(t), linestyle='--', alpha=0.3, linewidth=1)
                ax2.text(ax2.get_xlim()[1], val, f" {t} {lbl}={val:.2e}",
                         color=get_color(t), fontsize=7, va='center', ha='left')

    # ---- Figure 2: timing ----
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Precomputed QFT MPO — timing vs nBit", fontsize=14)

    for t in types_seen:
        c = get_color(t)
        comp = [e for e in entries if e["type_name"] == t and e["compressed"]]

        if comp:
            comp.sort(key=lambda e: e["nBit"])
            nbits = [e["nBit"] for e in comp]

            t_constr = [e["t_construction"] for e in comp]
            t_all_v  = [e["t_all"] for e in comp]

            sb = style_map[(True, "before")]
            sa = style_map[(True, "after")]

            label_b = f"{t} ({sb['label']})"
            label_a = f"{t} ({sa['label']})"

            ax3.plot(nbits, t_constr, marker=sb['marker'], color=c, linestyle=sb['ls'],
                     markersize=5, label=label_b)
            ax4.plot(nbits, t_all_v, marker=sa['marker'], color=c, linestyle=sa['ls'],
                     markersize=5, label=label_a)

    ax3.set_xlabel("nBit")
    ax3.set_ylabel("t_construction (s)")
    ax3.set_title("t_construction vs nBit")
    ax3.set_yscale("log")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel("nBit")
    ax4.set_ylabel("t_all (s)")
    ax4.set_title("t_all vs nBit")
    ax4.set_yscale("log")
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig1_save = os.path.join(figure_dir, f"precompute_qft_chi_err.{fmt}")
        fig1.savefig(fig1_save, dpi=150)
        print(f"Saved: {fig1_save}")
        fig2_save = os.path.join(figure_dir, f"precompute_qft_timing.{fmt}")
        fig2.savefig(fig2_save, dpi=150)
        print(f"Saved: {fig2_save}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot precomputed QFT MPO info data")
    parser.add_argument("--dir", default="mpo_data/data",
                        help="Directory containing *_info.txt files")
    args = parser.parse_args()

    plot_all(args.dir)
