#!/usr/bin/env python3
"""
Load error data files from test/output/ and test/output_compress/ and plot:

  Figure 1 (error vs chi):
    1. rms |MPO - F|
    2. relative rms |MPO - F|

  Figure 2 (size / compression):
    1. total number of tensor values (tot_nb_value) vs chi
    2. rms |MPO - F| vs max_bond_dim
    3. relative rms |MPO - F| vs max_bond_dim

All r values and all types are overlaid on the same figures.

Usage:
    python plot_errors_qft_mpo.py [--dir test/output] [--save plot.png]
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
        error_chi_max40_r30_complex_double.txt
        error_compress_chi_max40_r30_complex_double.txt
    Returns dict with keys: chi_max, r, type_name, compressed (bool)
    """
    base = os.path.basename(fname).replace(".txt", "")
    m = re.match(r"error_compress_chi_max(\d+)_r(\d+)_(.+)", base)
    if m:
        return {
            "chi_max": int(m.group(1)),
            "r": int(m.group(2)),
            "type_name": m.group(3).rstrip('_'),
            "compressed": True,
        }
    m = re.match(r"error_chi_max(\d+)_r(\d+)_(.+)", base)
    if m:
        return {
            "chi_max": int(m.group(1)),
            "r": int(m.group(2)),
            "type_name": m.group(3).rstrip('_'),
            "compressed": False,
        }
    return None


def load_file(fname):
    """Load a data file, return (chi, errors, tot_nb, max_bd)."""
    data = np.loadtxt(fname, comments="#")
    chi = data[:, 0].astype(int)
    # columns: chi, max|MPO-F|, rms|MPO-F|, rms|F|, relative_rms, tot_nb_value, max_bond_dim, [list_chi...], [nb_value_core...]
    errors = data[:, 1:5]       # max, rms_err, rms_F, rel_rms
    tot_nb = data[:, 5]          # tot_nb_value
    max_bd = data[:, 6]          # max_bond_dim
    return chi, errors, tot_nb, max_bd


def plot_all(data_dir, compress_dir):
    """Find all error files and plot them."""
    figure_dir = "test/figure"
    os.makedirs(figure_dir, exist_ok=True)

    # Original files
    pattern = os.path.join(data_dir, "error_chi_max*_r*_*.txt")
    files = sorted(glob.glob(pattern))

    # Compressed files
    pattern_comp = os.path.join(compress_dir, "error_compress_chi_max*_r*_*.txt")
    files_comp = sorted(glob.glob(pattern_comp))

    all_files = files + files_comp

    if not all_files:
        print(f"No files found matching patterns: {pattern} or {pattern_comp}")
        sys.exit(1)

    print(f"Found {len(files)} original + {len(files_comp)} compressed = {len(all_files)} file(s):")
    for f in all_files:
        tag = "[compress]" if "compress" in f else "[orig]   "
        print(f"  {tag} {f}")

    # Collect all entries: list of dicts
    entries = []
    chi_max_set = set()
    for f in all_files:
        meta = parse_filename(f)
        if meta is None:
            print(f"WARNING: Could not parse filename: {f}")
            continue
        chi, errors, tot_nb, max_bd = load_file(f)
        chi_max_set.add(meta["chi_max"])
        entries.append({
            "chi_max": meta["chi_max"],
            "type_name": meta["type_name"],
            "r": meta["r"],
            "compressed": meta["compressed"],
            "chi": chi,
            "rms_err": errors[:, 1],
            "rel_rms": errors[:, 3],
            "tot_nb": tot_nb,
            "max_bd": max_bd,
        })

    # marker by type_name, color by r, linestyle by compressed
    type_markers = {}  # filled on first encounter
    marker_pool = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    r_colors = {}  # filled on first encounter
    color_pool = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def get_marker(type_name):
        if type_name not in type_markers:
            type_markers[type_name] = marker_pool[len(type_markers) % len(marker_pool)]
        return type_markers[type_name]

    def get_color(r):
        if r not in r_colors:
            r_colors[r] = color_pool[len(r_colors) % len(color_pool)]
        return r_colors[r]

    def get_ls(compressed):
        return ':' if compressed else '-'

    for chi_max in sorted(chi_max_set):
        # Filter entries for this chi_max and sort for consistent legend
        cur = [e for e in entries if e["chi_max"] == chi_max]
        cur.sort(key=lambda e: (e["r"], e["compressed"], e["type_name"]))

        # ---- Figure 1: error vs chi (original only) ----
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle(f"chi_max = {chi_max}  —  error vs chi", fontsize=14)

        for e in cur:
            if e["compressed"]:
                continue
            label = f"{e['type_name']}, r={e['r']}"
            m = get_marker(e["type_name"])
            c = get_color(e["r"])
            ls = get_ls(e["compressed"])
            ax1.plot(e["chi"], e["rms_err"], marker=m, color=c, linestyle=ls, markersize=4, label=label)
            ax2.plot(e["chi"], e["rel_rms"], marker=m, color=c, linestyle=ls, markersize=4, label=label)

        ax1.set_xlabel("chi")
        ax1.set_ylabel("rms |MPO - F|")
        ax1.set_title("rms |MPO - F|")
        ax1.set_yscale("log")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("chi")
        ax2.set_ylabel("relative rms |MPO - F|")
        ax2.set_title("relative rms |MPO - F|")
        ax2.set_yscale("log")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        for fmt in ['png', 'pdf']:
            fig1_save = os.path.join(figure_dir, f"errors_chi_max{chi_max}.{fmt}")
            fig1.savefig(fig1_save, dpi=150)
            print(f"Saved: {fig1_save}")

        # ---- Figure 2: size / compression ----
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))


        # Shared color/marker/linestyle for both plots (r=30 only, color by type, marker+ls by compressed)
        type_colors = {}
        color_pool2 = plt.rcParams['axes.prop_cycle'].by_key()['color']
        def get_type_color(type_name):
            if type_name not in type_colors:
                type_colors[type_name] = color_pool2[len(type_colors) % len(color_pool2)]
            return type_colors[type_name]
        comp_marker = {False: 'o', True: 's'}
        comp_ls = {False: '-', True: ':'}

        for e in cur:
            if e["r"] != 30:
                continue
            tag = "compress " if e["compressed"] else ""
            label = f"{tag}{e['type_name']}"
            m = comp_marker[e["compressed"]]
            ls = comp_ls[e["compressed"]]
            c = get_type_color(e["type_name"])
            ax3.plot(e["chi"], e["tot_nb"], marker=m, color=c, linestyle=ls, markersize=5, label=label)
            ax4.plot(e["max_bd"], e["rel_rms"], marker=m, color=c, linestyle=ls, markersize=5, label=label)

        ax3.set_xlabel("chi_construction")
        ax3.set_ylabel("total number of values")
        ax3.set_title("tot_nb_value vs chi_construction")
        ax3.set_yscale("log")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        ax4.set_xlabel("max bond dim")
        ax4.set_ylabel("relative rms |MPO - F|")
        ax4.set_title("relative rms |MPO - F| vs max_bond_dim")
        ax4.set_yscale("log")
        for y, lbl in [(1e-14, "1e-14"), (1e-29, "1e-29"), (1e-32, "1e-32")]:
            ax4.axhline(y=y, linestyle="--", color="gray", alpha=0.5)
            ax4.text(ax4.get_xlim()[1] * 0.98, y, lbl, ha="right", va="bottom", fontsize=6, color="gray")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        for fmt in ['png', 'pdf']:
            fig2_save = os.path.join(figure_dir, f"size_chi_max{chi_max}.{fmt}")
            fig2.savefig(fig2_save, dpi=150)
            print(f"Saved: {fig2_save}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QFT MPO error data")
    parser.add_argument("--dir", default="test/output",
                        help="Directory containing error data files")
    parser.add_argument("--compress-dir", default="test/output_compress",
                        help="Directory containing compressed error data files")
    args = parser.parse_args()

    plot_all(args.dir, args.compress_dir)
