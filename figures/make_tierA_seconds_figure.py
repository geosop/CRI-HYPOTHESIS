# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:41:48 2025

@author: ADMIN

Tier-A (seconds-scale) synthetic simulation for CRI — slope panel + gate-saturation check.

Outputs (both formats):
  figures/output/TierA_decay_loglinear.pdf
  figures/output/TierA_decay_loglinear.png
  figures/output/TierA_gate_saturation.pdf
  figures/output/TierA_gate_saturation.png

What it does (aligned with SI S6 Tier-A):
- Uniform scheduler: P(tau_f) = tau_f / T0 on [0, T0] (seconds-scale).
- Logistic gate G(x) with input x = P(tau_f) - p0(a).
- Threshold: p0(a) = p_base + B_a * exp(-(a - mu_a)^2/(2*sigma_a^2)).
- Synthetic amplitude: A_pre = A0 * G * exp(-tau_f / tau_fut) + C_true + Gaussian noise.
- Baseline removal (subtract C_true) before log-linear fit.
- Gate-on selection for slope fit: G >= gate_thresh.
- Saves both PDF and PNG into figures/output/.

"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml  # optional
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# -----------------------------
# Defaults (can be overridden by YAML or CLI)
# -----------------------------
DEFAULTS = dict(
    T0=1.0,                   # seconds-scale horizon
    N_TAU=20,                 # number of tau_f grid points in [0, T0]
    tau_fut=0.10,             # s, e.g., 100 ms
    p_base=0.20,              # dimensionless
    B_a=0.50,                 # dimensionless
    mu_a=0.0,                 # arousal index center
    sigma_a=1.0,              # arousal width
    alpha=0.05,               # logistic scale
    gate_thresh=0.95,         # selection threshold for slope fit
    N_TRIALS=5000,            # number of trials
    a_state_mean=0.0,         # mean arousal (align to mu_a)
    a_state_sd=0.75,          # arousal spread
    A0=1.0,                   # signal scale
    C_true=0.05,              # additive baseline to remove before log
    x_noise=0.90,             # noise suppression index (Tier-A high)
    sigma0=0.30,              # baseline noise scale
    seed=42,                  # RNG seed
    arousal_quantiles=(0.2, 0.5, 0.8)  # for gate-saturation panel
)

def load_yaml_overrides(path="default_params.yml"):
    if not HAVE_YAML or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            y = yaml.safe_load(f)
        return (y or {}).get("TierA", {})
    except Exception:
        return {}

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

    # Prepare
    rng = np.random.default_rng(params["seed"])
    os.makedirs(args.outdir, exist_ok=True)

    # Grid
    tau_grid = np.linspace(0.0, params["T0"], int(params["N_TAU"]))

    # Simulate trials
    a_samples = rng.normal(loc=params["a_state_mean"], scale=params["a_state_sd"], size=params["N_TRIALS"])
    N_TAU = tau_grid.size
    A_trials = np.zeros((params["N_TRIALS"], N_TAU))
    G_trials = np.zeros((params["N_TRIALS"], N_TAU))
    for i, a in enumerate(a_samples):
        A_i, G_i = simulate_trial_amplitude(a, tau_grid, params, rng)
        A_trials[i] = A_i
        G_trials[i] = G_i

    # ---------- Panel (a): log-linear fit on gate-on trials ----------
    A_bc = A_trials - params["C_true"]  # remove baseline before log
    A_med = np.zeros(N_TAU)
    for j in range(N_TAU):
        gate_on = G_trials[:, j] >= params["gate_thresh"]
        vals = A_bc[gate_on, j]
        vals = vals[vals > 0]  # log domain positivity
        A_med[j] = np.median(vals) if vals.size > 0 else np.nan

    valid_idx = np.isfinite(A_med) & (A_med > 0)
    tau_fit = tau_grid[valid_idx]
    y_fit = np.log(A_med[valid_idx])

    slope = np.nan
    slope_se = np.nan
    intercept = np.nan
    if tau_fit.size >= 2:
        coeffs, cov = np.polyfit(tau_fit, y_fit, deg=1, cov=True)
        slope, intercept = coeffs[0], coeffs[1]
        slope_se = float(np.sqrt(cov[0, 0]))
    z = 1.96
    slope_ci = (slope - z * slope_se, slope + z * slope_se) if np.isfinite(slope) else (np.nan, np.nan)
    tau_fut_hat = -1.0 / slope if (np.isfinite(slope) and slope < 0) else np.nan

    # Plot panel (a)
    fig_a, ax_a = plt.subplots(figsize=(5.0, 4.0))
    ax_a.plot(tau_fit, y_fit, marker='o', linestyle='none', label='Median ln $A_{\\mathrm{pre}}$ (gate-on)')
    if np.isfinite(slope):
        y_line = intercept + slope * tau_fit
        ax_a.plot(tau_fit, y_line, linestyle='-', label='Linear fit')
        title_a = (f"log-linear fit: slope={slope:.3f} (95% CI {slope_ci[0]:.3f},{slope_ci[1]:.3f}); "
                   f"$\\widehat{{\\tau}}_{{\\mathrm{{fut}}}}$={-1/slope:.3f}s")
    else:
        title_a = "log-linear fit"
    ax_a.set_title(title_a)
    ax_a.set_xlabel(r"$\tau_f$ (s)")
    ax_a.set_ylabel(r"$\ln A_{\mathrm{pre}}(\tau_f)$")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    fig_a.tight_layout()
    for ext in ("pdf", "png"):
        fig_a.savefig(os.path.join(args.outdir, f"TierA_decay_loglinear.{ext}"), bbox_inches="tight")

    # ---------- Panel (b): gate saturation across arousal quantiles ----------
    q = params["arousal_quantiles"]
    q_edges = np.quantile(a_samples, q)
    bins = (-np.inf, q_edges[0], q_edges[2], np.inf)
    labels = ["low a", "mid a", "high a"]

    G_medians = []
    for b in range(3):
        in_bin = (a_samples > bins[b]) & (a_samples <= bins[b+1])
        G_medians.append(np.median(G_trials[in_bin, :], axis=0))

    fig_b, ax_b = plt.subplots(figsize=(5.0, 4.0))
    for med, lab in zip(G_medians, labels):
        ax_b.plot(tau_grid, med, label=lab)
    ax_b.axhline(params["gate_thresh"], linestyle='--')
    ax_b.set_title(r"Gate saturation: $G(P(\tau_f)-p_0(a))$ across arousal quantiles")
    ax_b.set_xlabel(r"$\tau_f$ (s)")
    ax_b.set_ylabel(r"Gate $G$")
    ax_b.set_ylim(0.0, 1.05)
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)
    fig_b.tight_layout()
    for ext in ("pdf", "png"):
        fig_b.savefig(os.path.join(args.outdir, f"TierA_gate_saturation.{ext}"), bbox_inches="tight")

    print("Wrote:",
          os.path.join(args.outdir, "TierA_decay_loglinear.pdf"),
          os.path.join(args.outdir, "TierA_gate_saturation.pdf"),
          file=sys.stderr)

if __name__ == "__main__":
    main()

