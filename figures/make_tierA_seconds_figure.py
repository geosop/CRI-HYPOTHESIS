# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:41:48 2025

@author: ADMIN

Tier-A (seconds-scale) synthetic simulation for CRI — slope panel + gate-saturation check.

Outputs:
  figures/output/TierA_decay_loglinear.pdf/.png
  figures/output/TierA_gate_saturation.pdf/.png

CRI alignment:
- Gate-on selection for slope fit uses a tight threshold (default 0.98) and
  requires enough gate-on samples at each τ_f before using that point.
- Gate-saturation panel normalizes per arousal quantile:
    G_norm = (G - p0_q) / (pinf_q - p0_q), clipped to [0, 1],
  where p0_q and pinf_q are robust medians from early/late τ_f windows.
- τ95 (earliest τ_f with G_norm ≥ 0.95) is marked for each quantile.

You can override defaults with a TierA section in default_params.yml.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# -----------------------------
# Defaults (can be overridden)
# -----------------------------
DEFAULTS = dict(
    T0=1.0,                    # seconds-scale horizon
    N_TAU=20,                  # number of tau_f grid points in [0, T0]
    tau_fut=0.10,              # s, the “future” time constant
    p_base=0.20,               # baseline in P-space
    B_a=0.50,                  # arousal bump amplitude
    mu_a=0.0,                  # arousal index center
    sigma_a=1.0,               # arousal width
    alpha=0.05,                # logistic scale
    gate_thresh=0.98,          # *** tighter selection for slope fit ***
    min_gate_frac=0.25,        # require ≥ this fraction of trials gate-on at a τ_f
    min_gate_n=50,             # ... and at least this many trials gate-on
    N_TRIALS=5000,             # number of trials
    a_state_mean=0.0,          # mean arousal (align to mu_a)
    a_state_sd=0.75,           # arousal spread
    A0=1.0,                    # signal scale
    C_true=0.05,               # additive baseline to remove before log
    x_noise=0.90,              # noise suppression index (Tier-A high)
    sigma0=0.30,               # baseline noise scale
    seed=42,                   # RNG seed
    arousal_quantiles=(0.2, 0.5, 0.8),  # for gate-saturation grouping
    # normalization windows for panel (b)
    baseline_max_s=0.05,       # “early” window upper bound (s) for p0
    plateau_frac=0.90,         # use last 10% of τ_f range for pinf
    sat_level=0.95             # dashed line + τ95 definition
)

def load_yaml_overrides(path="default_params.yml"):
    if not HAVE_YAML or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
        return y.get("TierA", {}) or {}
    except Exception:
        return {}

# -----------------------------
# Model pieces
# -----------------------------
def p0_of_a(a, p_base, B_a, mu_a, sigma_a):
    return p_base + B_a * np.exp(- (a - mu_a)**2 / (2.0 * sigma_a**2))

def G_of_x(x, alpha):
    return 1.0 / (1.0 + np.exp(-x / alpha))

def P_of_tau(tau, T0):
    return tau / T0

def simulate_trial_amplitude(a, tau_vals, params, rng):
    p0a = p0_of_a(a, params["p_base"], params["B_a"], params["mu_a"], params["sigma_a"])
    P_vals = P_of_tau(tau_vals, params["T0"])
    G_vals = G_of_x(P_vals - p0a, params["alpha"])
    clean = params["A0"] * G_vals * np.exp(-tau_vals / params["tau_fut"])
    sigma_noise = params["sigma0"] * (1.0 - params["x_noise"])**2
    noise = rng.normal(loc=0.0, scale=sigma_noise, size=clean.shape)
    return clean + params["C_true"] + noise, G_vals

# -----------------------------
# Robust gate normalization (panel b)
# -----------------------------
def robust_p0_pinf(y, t, baseline_max_s=0.05, plateau_frac=0.90):
    """Median p0 from early τ_f, median pinf from last (1-plateau_frac) fraction."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    # early window
    m0 = t <= baseline_max_s
    p0 = np.nanmedian(y[m0]) if np.any(m0) else np.nanmedian(y[:max(1, int(0.05*len(y)))])
    # plateau window
    m1 = t >= plateau_frac * t.max()
    pinf = np.nanmedian(y[m1]) if np.any(m1) else np.nanmedian(y[-max(1, int(0.1*len(y))):])
    if not np.isfinite(pinf) or pinf <= p0:
        pinf = p0 + max(1e-6, float(np.nanmax(y) - p0))
    return p0, pinf

def normalize_gate(y, t, baseline_max_s, plateau_frac):
    p0, pinf = robust_p0_pinf(y, t, baseline_max_s, plateau_frac)
    G = (np.asarray(y, float) - p0) / (pinf - p0 + 1e-12)
    return np.clip(G, 0.0, 1.0), p0, pinf

def tau_at_level(t, y, thr):
    i = np.where(y >= thr)[0]
    return float(t[i[0]]) if i.size else np.nan

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures/output", help="output directory for figures")
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed")
    args = ap.parse_args()

    # Params
    params = DEFAULTS.copy()
    params.update(load_yaml_overrides("default_params.yml"))
    if args.seed is not None:
        params["seed"] = args.seed

    rng = np.random.default_rng(params["seed"])
    os.makedirs(args.outdir, exist_ok=True)

    # τ_f grid
    tau_grid = np.linspace(0.0, params["T0"], int(params["N_TAU"]))
    N_TAU = tau_grid.size

    # Simulate trials
    a_samples = rng.normal(loc=params["a_state_mean"], scale=params["a_state_sd"], size=params["N_TRIALS"])
    A_trials = np.zeros((params["N_TRIALS"], N_TAU))
    G_trials = np.zeros((params["N_TRIALS"], N_TAU))
    for i, a in enumerate(a_samples):
        A_i, G_i = simulate_trial_amplitude(a, tau_grid, params, rng)
        A_trials[i] = A_i
        G_trials[i] = G_i

    # ============================================================
    # Panel (a): log-linear fit on gate-on trials (robust selection)
    # ============================================================
    A_bc = A_trials - params["C_true"]  # remove additive baseline before log
    A_med = np.full(N_TAU, np.nan)
    used_mask = np.zeros(N_TAU, dtype=bool)

    for j in range(N_TAU):
        gate_on = G_trials[:, j] >= params["gate_thresh"]
        if gate_on.sum() >= max(params["min_gate_n"], int(params["min_gate_frac"] * params["N_TRIALS"])):
            vals = A_bc[gate_on, j]
            vals = vals[vals > 0]  # positivity for log
            if vals.size:
                A_med[j] = np.median(vals)
                used_mask[j] = True

    tau_fit = tau_grid[used_mask & np.isfinite(A_med) & (A_med > 0)]
    y_fit = np.log(A_med[used_mask & np.isfinite(A_med) & (A_med > 0)])

    slope = np.nan
    slope_se = np.nan
    intercept = np.nan
    if tau_fit.size >= 2:
        coeffs, cov = np.polyfit(tau_fit, y_fit, deg=1, cov=True)
        slope, intercept = coeffs[0], coeffs[1]
        slope_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = 1.96
    slope_ci = (slope - z * slope_se, slope + z * slope_se) if np.isfinite(slope) else (np.nan, np.nan)
    tau_fut_hat = -1.0 / slope if (np.isfinite(slope) and slope < 0) else np.nan

    # Plot panel (a)
    fig_a, ax_a = plt.subplots(figsize=(5.0, 4.0))
    ax_a.plot(tau_fit, y_fit, marker='o', linestyle='none',
              label='Median ln $A_{\\mathrm{pre}}$ (gate-on)')
    if np.isfinite(slope):
        y_line = intercept + slope * tau_fit
        ax_a.plot(tau_fit, y_line, linestyle='-', label='Linear fit')
        title_a = (f"log-linear fit: slope={slope:.3f} "
                   f"(95% CI {slope_ci[0]:.3f},{slope_ci[1]:.3f}); "
                   f"$\\widehat{{\\tau}}_{{\\mathrm{{fut}}}}$={tau_fut_hat:.3f}s")
    else:
        title_a = "log-linear fit (insufficient gate-on points)"
    ax_a.set_title(title_a)
    ax_a.set_xlabel(r"$\tau_f$ (s)")
    ax_a.set_ylabel(r"$\ln A_{\mathrm{pre}}(\tau_f)$")
    ax_a.legend(frameon=False)
    ax_a.grid(True, alpha=0.3)
    fig_a.tight_layout()
    for ext in ("pdf", "png"):
        fig_a.savefig(os.path.join(args.outdir, f"TierA_decay_loglinear.{ext}"),
                      bbox_inches="tight")

    # ============================================================
    # Panel (b): gate saturation across arousal quantiles (normalized)
    # ============================================================
    q = params["arousal_quantiles"]
    # 3 bins: low (≤q20), mid (q20–q80), high (>q80)
    q_edges = np.quantile(a_samples, q)
    bins = (-np.inf, q_edges[0], q_edges[2], np.inf)
    labels = ["low a", "mid a", "high a"]

    # median gate per τ_f within each arousal bin
    G_medians = []
    for b in range(3):
        in_bin = (a_samples > bins[b]) & (a_samples <= bins[b+1])
        G_medians.append(np.median(G_trials[in_bin, :], axis=0))

    # normalize per quantile (robust)
    G_norms, taus_95 = [], []
    for med in G_medians:
        Gn, p0, pinf = normalize_gate(
            med, tau_grid,
            baseline_max_s=params["baseline_max_s"],
            plateau_frac=params["plateau_frac"]
        )
        G_norms.append(Gn)
        taus_95.append(tau_at_level(tau_grid, Gn, params["sat_level"]))

    # plot
    fig_b, ax_b = plt.subplots(figsize=(5.0, 4.0))
    for Gn, lab in zip(G_norms, labels):
        ax_b.plot(tau_grid, Gn, label=lab)
    ax_b.axhline(params["sat_level"], linestyle='--', linewidth=1.0)

    # τ95 markers (reviewer-friendly)
    for τ, lab in zip(taus_95, labels):
        if np.isfinite(τ):
            ax_b.axvline(τ, ls=":", lw=0.8, alpha=0.6)
            ax_b.text(τ, params["sat_level"] + 0.02,
                      f"τ₉₅({lab})={τ*1e3:.0f} ms",
                      rotation=90, va="bottom", ha="center", fontsize=7)

    ax_b.set_title(r"Gate saturation: normalized $G$ across arousal quantiles")
    ax_b.set_xlabel(r"$\tau_f$ (s)")
    ax_b.set_ylabel(r"Gate $G$ (normalized)")
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.legend(frameon=False)
    ax_b.grid(True, alpha=0.3)
    fig_b.tight_layout()
    for ext in ("pdf", "png"):
        fig_b.savefig(os.path.join(args.outdir, f"TierA_gate_saturation.{ext}"),
                      bbox_inches="tight")

    print("Wrote:",
          os.path.join(args.outdir, "TierA_decay_loglinear.pdf"),
          os.path.join(args.outdir, "TierA_gate_saturation.pdf"),
          file=sys.stderr)

if __name__ == "__main__":
    main()
