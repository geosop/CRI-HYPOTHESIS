# -*- coding: utf-8 -*-
"""
figures/make_tomography_figure.py  •  CRI v0.1-SIM

Box-2(c):
  Left: P(t) for two γ_b; each decays as exp[-(γ_f+γ_b) t / 2]
  Right: dark-teal dots = bootstrapped R(λ_env) estimates; dashed theoretical
         line R(λ_env)=λ_env; shaded band = fixed ±0.10 (95% CI surrogate).

Outputs: PDF (vector, 180 mm wide) + 1200 dpi PNG.
"""
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Portable, CI-friendly font setup
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,  # embed TrueType
    "ps.fonttype": 42,
})

def load_params(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        cfg = yaml.safe_load(f)
    return cfg["qpt"]

def main():
    here     = os.path.dirname(__file__)
    repo     = os.path.abspath(os.path.join(here, os.pardir))
    qpt_dir  = os.path.join(repo, 'qpt')
    out_fig  = os.path.join(here, 'output')
    os.makedirs(out_fig, exist_ok=True)

    p   = load_params(os.path.join(qpt_dir, 'default_params.yml'))
    sim = np.load(os.path.join(qpt_dir, 'output', 'qpt_sim_data.npz'), allow_pickle=True)

    t            = sim['t']                 # seconds
    gamma_b_vals = sim['gamma_b_vals']      # e.g., [0.2, 0.8]
    pops         = sim['pops']              # populations P(t) per γ_b
    lambda_env   = sim['lambda_env']        # s^{-1}
    R_mean       = sim['R_mean']
    R_low        = sim['R_ci_low']
    R_high       = sim['R_ci_high']

    # Fitted slope/intercept for R vs λ_env (if available)
    slope, intercept = np.nan, np.nan
    fit_csv = os.path.join(qpt_dir, 'output', 'qpt_R_fit.csv')
    if os.path.exists(fit_csv):
        try:
            import pandas as pd
            df_fit = pd.read_csv(fit_csv)
            slope, intercept = float(df_fit['slope'][0]), float(df_fit['intercept'][0])
        except Exception:
            pass

    # Figure canvas (two panels)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(180/25.4, 60/25.4), constrained_layout=True
    )

    # ---------------- Left panel: populations ----------------
    for gb, pop in zip(gamma_b_vals, pops):
        ax1.plot(t, pop, label=rf"$\gamma_b={float(gb):.1f}$")
    ax1.set(
        xlabel=r"Time $t$ (s)",
        ylabel=r"Population $P(t)$",
        title=r"$P(t)=\exp[-(\gamma_f+\gamma_b)t/2]$"
    )
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(alpha=0.25, linestyle="--")

    # ---------------- Right panel: R vs λ_env ----------------
    col_teal = "#136F63"  # dark teal dots
    col_band = "#B9E3D6"  # light teal band

    # Theoretical dashed line: R(λ_env) = λ_env
    lam_min = float(p['lambda_env_min'])
    lam_max = float(p['lambda_env_max'])
    lam_dense = np.linspace(lam_min, lam_max, 400)
    ax2.plot(
        lam_dense, lam_dense, linestyle="--", color="grey",
        label=r"Theory $R(\lambda_{\mathrm{env}})=\lambda_{\mathrm{env}}$"
    )

    # Fixed 95% CI band ±0.10 around theoretical line (clipped to ≥0)
    halfw = float(p.get('fixed_ci_halfwidth', 0.10))  # default ±0.10 if missing
    band_low  = np.clip(lam_dense - halfw, 0, None)
    band_high = lam_dense + halfw
    ax2.fill_between(
        lam_dense, band_low, band_high,
        color=col_band, alpha=0.8, edgecolor="none",
        label=r"95% CI ($\pm0.10$)"
    )

    # Dark-teal dots: bootstrapped R estimates (means per λ_env)
    ax2.scatter(
        lambda_env, R_mean, s=28, color=col_teal,
        edgecolors="black", linewidths=0.4, label=r"Bootstrapped $R$"
    )
    # Light vertical error bars from per-λ bootstrap
    ax2.vlines(lambda_env, R_low, R_high, colors=col_teal, alpha=0.35, linewidth=0.8)

    # Axis labels/titles (use "Jump-weight ratio" per notation)
    if np.isfinite(slope) and np.isfinite(intercept):
        title_str = rf"$R$ vs. $\lambda_{{\mathrm{{env}}}}$  (fit: $R\!=\!{slope:.2f}\lambda_{{\mathrm{{env}}}}{intercept:+.2f}$)"
    else:
        title_str = r"$R$ vs. $\lambda_{\mathrm{env}}$"

    ax2.set(
        xlabel=r"$\lambda_{\mathrm{env}}$ (s$^{-1}$)",
        ylabel=r"Jump-weight ratio $R$",
        title=title_str
    )
    ax2.set_xlim(lam_min - 0.02, lam_max + 0.02)
    ax2.set_ylim(0, max(1.05, (np.nanmax(R_high) + 0.05)))
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(alpha=0.25, linestyle="--")

    # Save (PDF + PNG)
    pdf_path = os.path.join(out_fig, 'Box2c_rate_refined.pdf')
    png_path = os.path.join(out_fig, 'Box2c_rate_refined.png')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=int(p.get('figure_dpi', 1200)), bbox_inches='tight')
    plt.close(fig)
    print("Saved Box-2(c) figure →", pdf_path, "and", png_path)

if __name__=='__main__':
    main()
