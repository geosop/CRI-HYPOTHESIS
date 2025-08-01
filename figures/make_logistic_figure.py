# -*- coding: utf-8 -*-
"""
figures/make_logistic_figure.py  •  CRI v0.1‑SIM
-----------------------------------------------------------------
Generate the logistic “tipping‑point” figure (gate probability
vs. cue probability) for the Perspective / SI.

Outputs a vector PDF (88 mm wide, 6:4 aspect).  All data are
synthetic — see logistic_gate/ for CSV + YAML inputs.

Usage
-----
    python make_logistic_figure.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Matplotlib style (journal‑ready) ─────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "Arial",
    "font.size":        8,
    "axes.linewidth":   0.5,
    "lines.linewidth":  0.75,
    "legend.fontsize":  6,
    "xtick.labelsize":  6,
    "ytick.labelsize":  6,
})
# ─────────────────────────────────────────────────────────────────────────────

def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["logistic"]

def logistic(p, p0, alpha):
    """Gate G(p) = 1 / (1 + exp(-(p - p0)/alpha))."""
    return 1.0 / (1.0 + np.exp(-(p - p0) / alpha))

def derivative(p, p0, alpha):
    """Analytic derivative dG/dp."""
    k = 1.0 / alpha
    exp_t = np.exp(-k * (p - p0))
    return k * exp_t / (1.0 + exp_t) ** 2

def main():
    here     = os.path.dirname(__file__)
    repo     = os.path.abspath(os.path.join(here, os.pardir))
    gate_dir = os.path.join(repo, "logistic_gate")
    out_dir  = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # ── load synthetic data ─────────────────────────────────────────────────
    params  = load_params(os.path.join(gate_dir, "default_params.yml"))
    df_obs  = pd.read_csv(os.path.join(gate_dir, "output", "logistic_data.csv"))
    df_fit  = pd.read_csv(os.path.join(gate_dir, "output", "fit_logistic_results.csv"))

    p0_hat    = df_fit["p0_hat"].iloc[0]
    alpha_hat = df_fit["alpha_hat"].iloc[0]
    noise     = params["noise_std"]

    # ── continuous curve for plotting ───────────────────────────────────────
    p_cont = np.linspace(params["p_min"], params["p_max"], params["n_points"])
    G_fit  = logistic(p_cont, p0_hat, alpha_hat)
    upper  = np.clip(G_fit + noise, 0.0, 1.0)
    lower  = np.clip(G_fit - noise, 0.0, 1.0)

    # ── figure (88 mm single‑column) ────────────────────────────────────────
    width_mm, height_mm = 88, 88 / 1.5
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4),
                            constrained_layout=True)

    # main panel
    ax.plot(p_cont, G_fit, color="C0", label="Fitted logistic")
    ax.fill_between(p_cont, lower, upper, color="C0", alpha=0.30,
                    label=rf"±{noise:.2f} CI")
    ax.scatter(df_obs["p"], df_obs["G_obs"], s=25, edgecolors="k",
               facecolors="C1", label="Observed")
    ax.axvline(p0_hat, color="grey", linestyle="--",
               label=rf"$\hat{{p}}_0={p0_hat:.2f}$")

    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$G(p \mid a)$")
    ax.legend(loc="upper left")

    # inset: derivative
    ax_ins = inset_axes(ax, width="75%", height="75%",
                        loc="lower left",
                        bbox_to_anchor=(0.65, 0.30, 0.4, 0.4),
                        bbox_transform=ax.transAxes)
    ax_ins.plot(p_cont, derivative(p_cont, p0_hat, alpha_hat),
                linewidth=0.75, color="C0")
    ax_ins.set_title(r"$\mathrm{d}G/\mathrm{d}p$", fontsize=8)
    ax_ins.set_xlabel(r"$p$", fontsize=7)
    ax_ins.set_ylabel("Rate", fontsize=7)
    ax_ins.tick_params(labelsize=6)

    # ── export ──────────────────────────────────────────────────────────────
    pdf_path = os.path.join(out_dir, "CRI_logistic_refined_fig.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved logistic figure ➜ {pdf_path}")

if __name__ == "__main__":
    main()

