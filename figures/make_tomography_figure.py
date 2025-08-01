# -*- coding: utf-8 -*-
"""
figures/make_tomography_figure.py  •  CRI v0.1‑SIM
-----------------------------------------------------------------
Generate the synthetic rate‑estimation figure:

  • Left: excited‑state population decay P(t) for two γ_b values
  • Right: inferred jump‑rate ratio R vs. environmental coupling λ_env,
           plus a linear regression fit  (simulation only)

Outputs a vector PDF (180 mm wide).  All inputs come from
qpt/output/*.  They are purely synthetic; no experimental
tomography data are included in v0.1‑SIM.

Usage
-----
    python make_tomography_figure.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Matplotlib style (journal quality) ───────────────────────────────────────
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
    """Load regression‑simulation parameters."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["qpt"]

def main():
    here     = os.path.dirname(__file__)
    repo     = os.path.abspath(os.path.join(here, os.pardir))
    qpt_dir  = os.path.join(repo, "qpt")
    out_dir  = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # ── load synthetic arrays ───────────────────────────────────────────────
    params   = load_params(os.path.join(qpt_dir, "default_params.yml"))
    sim      = np.load(os.path.join(qpt_dir, "output", "qpt_sim_data.npz"))
    required = {"t", "gamma_b_vals", "pops", "lambda_env", "R_obs"}
    if not required.issubset(sim.files):
        raise KeyError(f"Simulation NPZ must contain keys {required}")

    t            = sim["t"]
    gamma_b_vals = sim["gamma_b_vals"]
    pops_list    = sim["pops"]
    lambda_env   = sim["lambda_env"]
    R_obs        = sim["R_obs"]

    # linear‑fit coefficients (slope, intercept) from CSV
    df_fit   = pd.read_csv(os.path.join(qpt_dir, "output", "qpt_R_fit.csv"))
    slope    = df_fit["slope"].iloc[0]
    intercept= df_fit["intercept"].iloc[0]

    # ── figure canvas (180 mm × 60 mm) ──────────────────────────────────────
    width_mm, height_mm = 180, 180 / 3
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # ── left panel : P(t) decay ─────────────────────────────────────────────
    for gb, pop in zip(gamma_b_vals, pops_list):
        ax1.plot(t, pop, label=rf"$\gamma_b={gb:.1f}$")
    ax1.set_xlabel(r"Time $t$ (s)")
    ax1.set_ylabel(r"Population $P(t)$")
    ax1.set_title("Simulated Excited State Populations")
    ax1.legend(title="Backward rates", loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── right panel : R vs λ_env ────────────────────────────────────────────
    ax2.scatter(lambda_env, R_obs, s=25, label="Observed $R$")
    lam_cont = np.linspace(params["lambda_env_min"],
                           params["lambda_env_max"], 200)
    ax2.plot(lam_cont, slope * lam_cont + intercept,
             linestyle="--",
             label=rf"Fit: $R = {slope:.2f}\,\lambda_{{env}} + {intercept:.2f}$")
    ax2.set_xlabel(r"Environmental coupling $\lambda_{env}$")
    ax2.set_ylabel(r"Jump rate ratio $R$")
    ax2.set_title("Inferred $R$ vs. $\lambda_{env}$")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ── export ──────────────────────────────────────────────────────────────
    pdf_path = os.path.join(out_dir, "CRI_rate_estimation_fig.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rate‑estimation figure ➜ {pdf_path}")

if __name__ == "__main__":
    main()
