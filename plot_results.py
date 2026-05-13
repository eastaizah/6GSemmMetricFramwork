#!/usr/bin/env python3
"""
Plot generation script for:

    "A Multi-Dimensional Semantic Metric Standardization Framework
     for Evaluating AI-Native Systems in 6G Networks"

Generates the two most representative figures as high-quality PNG files:

  Figure 1 — Semantic Fidelity vs. Bottleneck Dimension k
             (RSE and S³I at SNR = 10 dB, AWGN channel)
             Source: simulation_results/results_combined.npz

  Figure 2 — TSR vs. SNR: Graceful Degradation (k = 32, all channels)
             (Proposed semantic system vs. classical JPEG2000+LDPC baseline)
             Source: simulation_results/results_k32.npz

Usage
-----
    python plot_results.py [--output-dir <dir>]

Output files (in --output-dir, default: simulation_results/):
    fig_semantic_fidelity_vs_k.png
    fig_tsr_vs_snr.png

Requirements: numpy, matplotlib
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_HERE, "simulation_results")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load(fname):
    path = os.path.join(_SIM_DIR, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Simulation result file not found: {path}\n"
            "Run simulate_semantic_metrics.py first."
        )
    return np.load(path, allow_pickle=True)


def _classical_tsr(snr_db, cliff=6.0, width=1.5, mx=0.95):
    """Logistic cliff-effect model for JPEG2000+LDPC baseline."""
    return mx / (1.0 + np.exp(-(np.asarray(snr_db) - cliff) / width))


# ─────────────────────────────────────────────────────────────
# Figure 1 — Semantic Fidelity vs. k
# ─────────────────────────────────────────────────────────────

def plot_fidelity_vs_k(output_dir: str):
    """Fig. 4 / Fig. 7 in the article.

    RSE (right axis, 0–0.03) and S³I (left axis, 0–1) vs. k ∈ {8,16,32,64,128}
    at SNR = 10 dB, AWGN channel.  Source: results_combined.npz.
    """
    rc = _load("results_combined.npz")

    k_values = [8, 16, 32, 64, 128]
    rho_pct  = [k / 512 * 100 for k in k_values]

    rse_vals  = [float(rc[f"k{k}_AWGN_snr10.0__RSE"])  for k in k_values]
    s3i_vals  = [float(rc[f"k{k}_AWGN_snr10.0__S3I"])  for k in k_values]
    nsmi_vals = [float(rc[f"k{k}_AWGN_snr10.0__NSMI"]) for k in k_values]
    swd_vals  = [float(rc[f"k{k}_AWGN_snr10.0__SWD"])  for k in k_values]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    # S³I on left axis
    l1, = ax1.plot(k_values, s3i_vals, color="#d62728", linestyle="--",
                   marker="s", markersize=8, linewidth=2.0,
                   label=r"$\mathrm{S^3I}$ (left axis)")
    # RSE on right axis
    l2, = ax2.plot(k_values, rse_vals, color="#1f77b4", linestyle="-",
                   marker="o", markersize=8, linewidth=2.0,
                   label="RSE (right axis)")
    # NSMI on right axis (dashed, same colour family as RSE)
    l3, = ax2.plot(k_values, nsmi_vals, color="#aec7e8", linestyle="-.",
                   marker="^", markersize=7, linewidth=1.5,
                   label="NSMI (right axis)")

    # Annotations for key operating point k=32
    ax1.annotate(
        rf"$k=32$: $\mathrm{{S^3I}}={s3i_vals[2]:.3f}$",
        xy=(32, s3i_vals[2]), xytext=(45, s3i_vals[2] - 0.04),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        fontsize=9, color="#d62728",
    )
    ax2.annotate(
        f"$k=32$: RSE$={rse_vals[2]:.3f}$",
        xy=(32, rse_vals[2]), xytext=(45, rse_vals[2] + 0.002),
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
        fontsize=9, color="#1f77b4",
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(k_values)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlabel(r"Bottleneck Dimension $k$", fontsize=12)
    ax1.set_ylabel(r"$\mathrm{S^3I}$", fontsize=12, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0.30, 0.60)
    ax1.set_xlim(6, 160)

    ax2.set_ylabel("RSE / NSMI", fontsize=12, color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0.00, 0.030)

    # Secondary top X-axis: compression ratio ρ
    ax3 = ax1.twiny()
    ax3.set_xscale("log", base=2)
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(k_values)
    ax3.set_xticklabels([f"{r:.2f}%" for r in rho_pct], fontsize=8)
    ax3.set_xlabel(r"Compression Ratio $\rho = k/512$", fontsize=10)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_title(
        "Semantic Fidelity vs. Bottleneck Dimension $k$\n"
        "(SNR = 10 dB, AWGN, $N=1{,}000$, seed = 42)",
        fontsize=11,
    )
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    out_path = os.path.join(output_dir, "fig_semantic_fidelity_vs_k.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# Figure 2 — TSR vs. SNR (Graceful Degradation)
# ─────────────────────────────────────────────────────────────

def plot_tsr_vs_snr(output_dir: str):
    """Fig. 5 / Fig. 8 in the article.

    TSR vs. SNR for k = 32 across all five channel models, plus the
    classical JPEG2000+LDPC cliff-effect baseline.
    Source: simulation_results/results_k32.npz.
    """
    r32 = _load("results_k32.npz")

    snr_vals = np.arange(-5.0, 27.5, 2.5)

    channels = {
        "AWGN":       {"color": "#1f77b4", "ls": "-",   "marker": "o", "label": "Proposed ($k=32$, AWGN)"},
        "Rayleigh":   {"color": "#d62728", "ls": "-",   "marker": "s", "label": "Proposed ($k=32$, Rayleigh)"},
        "Rician_K5":  {"color": "#2ca02c", "ls": "--",  "marker": "^", "label": r"Proposed ($k=32$, Rician $K$=5 dB)"},
        "Rician_K10": {"color": "#9467bd", "ls": "--",  "marker": "D", "label": r"Proposed ($k=32$, Rician $K$=10 dB)"},
        "TDL-A":      {"color": "#ff7f0e", "ls": "-.",  "marker": "x", "label": "Proposed ($k=32$, TDL-A 3GPP)"},
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plot semantic channels
    for ch_name, style in channels.items():
        tsr_list  = []
        ci_lo_list = []
        ci_hi_list = []
        for snr in snr_vals:
            key = f"{ch_name}__snr{snr:.1f}__TSR"
            if key in r32:
                t   = float(r32[key])
                lo  = float(r32[f"{ch_name}__snr{snr:.1f}__TSR_CI_lo"])
                hi  = float(r32[f"{ch_name}__snr{snr:.1f}__TSR_CI_hi"])
            else:
                t, lo, hi = np.nan, np.nan, np.nan
            tsr_list.append(t)
            ci_lo_list.append(lo)
            ci_hi_list.append(hi)

        tsr_arr = np.array(tsr_list)
        lo_arr  = np.array(ci_lo_list)
        hi_arr  = np.array(ci_hi_list)
        valid   = ~np.isnan(tsr_arr)

        ax.plot(snr_vals[valid], tsr_arr[valid],
                color=style["color"], linestyle=style["ls"],
                marker=style["marker"], markersize=5,
                linewidth=2.0 if ch_name == "AWGN" else 1.5,
                label=style["label"])

        # Confidence-interval shading for AWGN only (cleaner plot)
        if ch_name == "AWGN":
            ax.fill_between(snr_vals[valid], lo_arr[valid], hi_arr[valid],
                            color=style["color"], alpha=0.15,
                            label="AWGN 95% Wilson CI")

    # Classical baseline
    snr_dense = np.linspace(-5, 25, 200)
    ax.plot(snr_dense, _classical_tsr(snr_dense),
            color="gray", linestyle=":", linewidth=2.0,
            label="Classical (JPEG2000+LDPC)")

    # Annotate graceful-degradation zone
    ax.axvspan(-5, 5, alpha=0.06, color="blue")
    ax.text(0.0, 0.05, "Graceful\nDegradation\nZone",
            ha="center", va="bottom", fontsize=8,
            color="#1f77b4", style="italic")

    # Annotate 0 dB advantage
    awgn_0 = float(r32.get("AWGN__snr0.0__TSR", np.nan))
    cl_0   = _classical_tsr(0.0)
    if not np.isnan(awgn_0):
        if cl_0 > 0:
            ratio_str = f"≈ {awgn_0 / cl_0:.0f}×"
        else:
            ratio_str = "»"
        ax.annotate(
            f"{ratio_str} at 0 dB\n({awgn_0:.3f} vs {cl_0:.3f})",
            xy=(0, awgn_0), xytext=(3, awgn_0 - 0.12),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=8,
        )

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Task Success Rate (TSR)", fontsize=12)
    ax.set_xlim(-5, 25)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(-5, 26, 5))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, ncol=1)
    ax.set_title(
        "TSR vs. SNR — Graceful Degradation ($k=32$)\n"
        "($N=1{,}000$, seed = 42)",
        fontsize=11,
    )

    out_path = os.path.join(output_dir, "fig_tsr_vs_snr.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate PNG figures from semantic-metrics simulation results."
    )
    parser.add_argument(
        "--output-dir", default=_SIM_DIR,
        help=f"Directory to write PNG files (default: {_SIM_DIR})"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating figures …")
    p1 = plot_fidelity_vs_k(args.output_dir)
    p2 = plot_tsr_vs_snr(args.output_dir)
    print(f"\nDone.  Two PNG files written:")
    print(f"  {p1}")
    print(f"  {p2}")


if __name__ == "__main__":
    main()
