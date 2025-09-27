# -*- coding: utf-8 -*-
"""
figures/make_tomography_figure.py  •  CRI v0.1-SIM

Box-2(c) — two panels:
  Left (ms axis): P(t) for two γ_b; each decays as
      P(t) = exp[-(κ0/2) (γ_fwd + γ_b) t]        (t in seconds; axis shown in ms)
  Right (dimensionless normalization): dark-teal dots = bootstrapped R(λ_env);
      dashed theory R = λ_env / κ0 ; shaded band = fixed ±0.10 (95% CI surrogate).

Outputs: PDF (vector, 180 mm wide) + 1200 dpi PNG.
"""
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- Matplotlib (portable, CI-friendly) -------------------------
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

LIGHT_GRAY = "#f2f2f2"

# ---------------- Helpers -----------------------------------------------------
def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def load_qpt_params(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("qpt", cfg)  # tolerate flat YAML

def derive_kappa0(p):
    """
    Returns κ0 [s^-1] using the following priority:
      1) p['kappa0'] or p['kappa_0']
      2) 1 / p['tau_mem_s']           (if available)
      3) 1000 / p['tau_mem_ms']
      4) fallback 8.0  [s^-1]  (≈125 ms)
    """
    for key in ("kappa0", "kappa_0"):
        if key in p:
            return _safe_float(p[key], 8.0)
    if "tau_mem_s" in p:
        val = _safe_float(p["tau_mem_s"], 0.125)
        return 1.0 / val if val > 0 else 8.0
    if "tau_mem_ms" in p:
        val_ms = _safe_float(p["tau_mem_ms"], 125.0)
        return 1000.0 / val_ms if val_ms > 0 else 8.0
    return 8.0

def main():
    here     = os.path.dirname(__file__)
    repo     = os.path.abspath(os.path.join(here, os.pardir))
    qpt_dir  = os.path.join(repo, "qpt")
    out_dir  = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Load params & sim --------------------------------------
    p = load_qpt_params(os.path.join(qpt_dir, "default_params.yml"))
    kappa0 = derive_kappa0(p)  # [s^-1]
    gamma_fwd = _safe_float(p.get("gamma_fwd", 1.0), 1.0)

    sim_path = os.path.join(qpt_dir, "output", "qpt_sim_data.npz")
    sim = np.load(sim_path, allow_pickle=True)

    # Expect t (seconds), gamma_b_vals, pops (same length as gamma_b_vals)
    t            = sim["t"]                 # seconds
    gamma_b_vals = sim["gamma_b_vals"]      # e.g., [0.2, 0.8]
    pops         = sim["pops"]              # shape: (len(gamma_b_vals), len(t))

    # For the right panel
    lambda_env = sim["lambda_env"]          # s^{-1}, array-like
    R_mean     = sim["R_mean"]              # per-λ mean
    # Optional CIs; if absent, create narrow CI to avoid errors
    R_low  = sim["R_ci_low"]  if "R_ci_low"  in sim.files else (R_mean - 0.05)
    R_high = sim["R_ci_high"] if "R_ci_high" in sim.files else (R_mean + 0.05)

    lam_min = _safe_float(p.get("lambda_env_min", np.nanmin(lambda_env)), np.nanmin(lambda_env))
    lam_max = _safe_float(p.get("lambda_env_max", np.nanmax(lambda_env)), np.nanmax(lambda_env))
    halfw   = _safe_float(p.get("fixed_ci_halfwidth", 0.10), 0.10)  # ±0.10 band

    # ---------------- Figure canvas -----------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180/25.4, 60/25.4), constrained_layout=True)

    # ---------------- Left panel: P(t) in ms --------------------------------
    t_ms = 1000.0 * np.asarray(t)  # axis in milliseconds
    for gb, pop in zip(gamma_b_vals, pops):
        ax1.plot(t_ms, pop, label=rf"$\gamma_b={float(gb):.1f}$")
    ax1.set(
        xlabel=r"Time $t$ (ms)",
        ylabel=r"Population $P(t)$",
        title=rf"$P(t)=\exp[-(\kappa_0/2)\,(\gamma_{{\mathrm{{fwd}}}}+\gamma_b)\,t]$"
    )
    # Set a CRI-consistent window (default 0–200 ms)
    ax1.set_xlim(0.0, _safe_float(p.get("left_panel_tmax_ms", 200.0), 200.0))
    # Annotate κ0 and γ_fwd for clarity
    ax1.text(
        0.02, 0.95,
        rf"$\kappa_0={kappa0:.2f}\,\mathrm{{s}}^{{-1}},\ \gamma_{{\mathrm{{fwd}}}}={gamma_fwd:.1f}$",
        transform=ax1.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="0.8", linewidth=0.6),
        fontsize=6.5
    )
    # Legend with light gray background
    leg1 = ax1.legend(loc="upper right", frameon=True, fancybox=True)
    leg1.get_frame().set_facecolor(LIGHT_GRAY)
    leg1.get_frame().set_edgecolor("0.80")
    leg1.get_frame().set_linewidth(0.6)
    ax1.grid(alpha=0.25, linestyle="--")

    # ---------------- Right panel: R vs λ_env (normalized) -------------------
    col_teal = "#136F63"  # dark teal dots
    col_band = "#B9E3D6"  # light teal band

    lam_dense = np.linspace(lam_min, lam_max, 400)
    theory = lam_dense / kappa0  # R = λ_env / κ0 (dimensionless)

    # CI band around theory (clip to ≥0)
    band_low  = np.clip(theory - halfw, 0, None)
    band_high = theory + halfw
    ax2.fill_between(lam_dense, band_low, band_high, color=col_band, alpha=0.9, edgecolor="none",
                     label=r"95\% CI ($\pm 0.10$)")

    # Theory dashed line
    ax2.plot(lam_dense, theory, linestyle="--", color="grey",
             label=rf"Theory $R=\lambda_{{\mathrm{{env}}}}/\kappa_0$")

    # Bootstrapped estimates with vertical CIs
    ax2.scatter(lambda_env, R_mean, s=28, color=col_teal, edgecolors="black", linewidths=0.4,
                label=r"Bootstrapped $R$")
    ax2.vlines(lambda_env, R_low, R_high, colors=col_teal, alpha=0.35, linewidth=0.8)

    # Labels/limits
    ax2.set(
        xlabel=r"$\lambda_{\mathrm{env}}$ (s$^{-1}$)",
        ylabel=r"Jump-weight ratio $R$",
        title=rf"$R$ vs. $\lambda_{{\mathrm{{env}}}}$   ($\kappa_0={kappa0:.2f}\,\mathrm{{s}}^{{-1}}$)"
    )
    ax2.set_xlim(lam_min - 0.02*(lam_max-lam_min), lam_max + 0.02*(lam_max-lam_min))
    y_top = np.nanmax([np.nanmax(R_high), np.nanmax(theory)+halfw, 1.05])
    ax2.set_ylim(0, y_top)
    # Legend with light gray background
    leg2 = ax2.legend(loc="upper left", frameon=True, fancybox=True)
    leg2.get_frame().set_facecolor(LIGHT_GRAY)
    leg2.get_frame().set_edgecolor("0.80")
    leg2.get_frame().set_linewidth(0.6)
    ax2.grid(alpha=0.25, linestyle="--")

    # ---------------- Save ---------------------------------------------------
    pdf_path = os.path.join(out_dir, "Box2c_rate_refined.pdf")
    png_path = os.path.join(out_dir, "Box2c_rate_refined.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=int(_safe_float(p.get("figure_dpi", 1200), 1200)), bbox_inches="tight")
    plt.close(fig)
    print("Saved Box-2(c) figure →", pdf_path, "and", png_path)

if __name__ == "__main__":
    main()
