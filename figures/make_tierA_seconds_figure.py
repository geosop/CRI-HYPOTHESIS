# -*- coding: utf-8 -*-
"""
figures/make_tierA_seconds_figure.py

Tier-A (seconds-scale) synthetic simulation for CRI — slope panel + gate-saturation check.

Outputs:
  figures/output/TierA_decay_loglinear.pdf/.png
  figures/output/TierA_gate_saturation.pdf/.png

CRI alignment highlights:
- STRICT per-τ gate-on rule for selection into the slope fit:
    median raw G(τ) ≥ gate_thresh  AND  at least min_gate_n & min_gate_frac trials gate-on.
- Log–linear slope fit is **ordinary least squares (OLS)** with **heteroskedasticity-robust HC3 SEs**.
- Optional early-window restriction after gate turn-on to avoid late noise-floor points.
- Gate-saturation panel normalizes medians by arousal quantile and marks τ95 (s).

Author: ADMIN
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Silence Arial not found spam; DejaVu is available on CI runners
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "dejavusans"

# Optional YAML overrides (figures/default_params.yml with a "TierA" block)
try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False


# -----------------------------
# Defaults (can be overridden)
# -----------------------------
DEFAULTS = dict(
    # Experiment geometry (Tier-A seconds regime)
    T0=1.0,                   # horizon in seconds
    N_TAU=30,                 # number of τ_f grid points in [0, T0]

    # Mechanistic "truth"
    tau_fut=0.28,             # seconds; CRI-consistent seconds-scale constant

    # Gate parameters G(P(τ_f) - p0(a))
    p_base=0.15,
    B_a=0.40,
    mu_a=0.0,
    sigma_a=1.0,
    alpha=0.08,               # softened logistic → broader gate-on window

    # Gate-on logic for the slope fit (STRICT)
    gate_thresh=0.98,         # threshold for "gate-on"
    min_gate_n=800,           # require at least this many gate-on trials at a τ
    min_gate_frac=0.15,       # and at least this fraction of all trials

    # (Optional) restrict fit to an early window after turn-on (avoid ultra-late noise floor)
    use_fit_window=True,
    fit_window_s=0.40,        # fit over [τ_on, τ_on + fit_window_s]

    # Trials & noise (high SNR in gate-on band)
    N_TRIALS=20000,
    a_state_mean=0.0,
    a_state_sd=0.75,
    A0=1.0,
    C_true=0.03,              # additive baseline; removed before log
    x_noise=0.995,            # strong suppression (Tier-A high-precision)
    sigma0=0.05,              # baseline noise scale
    seed=42,

    # Gate-saturation (panel b)
    arousal_quantiles=(0.2, 0.5, 0.8),
    baseline_max_s=0.05,      # early window for p0 (seconds)
    plateau_frac=0.90,        # late window start fraction for p_inf
    sat_level=0.95,           # dashed level and τ95 definition

    # Plotting niceties
    show_hc3_band=True        # draw a 95% HC3 confidence band around the OLS line
)


def load_yaml_overrides(path: str) -> dict:
    if not HAVE_YAML or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
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
    """Return (A_pre, G_raw) arrays for one trial across τ_f grid (Tier-A gate: G(P(τ_f)-p0(a)))."""
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
    """Median p0 from early τ_f, median p_inf from the last (1 - plateau_frac) tail."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    # early window
    m0 = t <= baseline_max_s
    p0 = np.nanmedian(y[m0]) if np.any(m0) else np.nanmedian(y[:max(1, int(0.05 * len(y)))] )
    # late window
    m1 = t >= plateau_frac * np.nanmax(t)
    p_inf = np.nanmedian(y[m1]) if np.any(m1) else np.nanmedian(y[-max(1, int(0.1 * len(y))):])
    if not np.isfinite(p_inf) or p_inf <= p0:
        p_inf = p0 + max(1e-6, float(np.nanmax(y) - p0))
    return p0, p_inf

def normalize_gate(y, t, baseline_max_s, plateau_frac):
    p0, p_inf = robust_p0_pinf(y, t, baseline_max_s, plateau_frac)
    G = (np.asarray(y, float) - p0) / (p_inf - p0 + 1e-12)
    return np.clip(G, 0.0, 1.0), p0, p_inf

def tau_at_level(t, y, thr):
    """Earliest t where y >= thr; NaN if none."""
    idx = np.where(y >= thr)[0]
    return float(t[idx[0]]) if idx.size else np.nan


# -----------------------------
# OLS + HC3 utilities
# -----------------------------
def ols_hc3(X, y):
    """
    OLS coefficients and HC3 (heteroskedasticity-robust) covariance.
    Returns: beta, cov_HC3
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)           # (p x 1)
    resid = (y - X @ beta).ravel()       # (n,)

    # leverage h_i = x_i^T (X^T X)^{-1} x_i
    H_diag = np.sum((X @ XtX_inv) * X, axis=1)  # (n,)

    # HC3 meat: X^T diag(e_i^2 / (1 - h_i)^2) X
    scale = (resid**2) / np.maximum(1.0 - H_diag, 1e-12)**2
    DX = X * scale[:, None]
    meat = X.T @ DX
    cov_hc3 = XtX_inv @ meat @ XtX_inv
    return beta.ravel(), cov_hc3


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures/output", help="output directory for figures")
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed")
    args = ap.parse_args()

    # Resolve paths
    here = os.path.dirname(__file__)
    yaml_path = os.path.join(here, "default_params.yml")

    # Params
    params = DEFAULTS.copy()
    params.update(load_yaml_overrides(yaml_path))
    if args.seed is not None:
        params["seed"] = args.seed

    rng = np.random.default_rng(params["seed"])
    os.makedirs(args.outdir, exist_ok=True)

    # τ_f grid (seconds)
    tau_grid = np.linspace(0.0, params["T0"], int(params["N_TAU"]))
    N_TAU = tau_grid.size

    # Simulate trials
    a_samples = rng.normal(loc=params["a_state_mean"], scale=params["a_state_sd"],
                           size=params["N_TRIALS"])
    A_trials = np.zeros((params["N_TRIALS"], N_TAU))
    G_trials = np.zeros((params["N_TRIALS"], N_TAU))
    for i, a in enumerate(a_samples):
        A_i, G_i = simulate_trial_amplitude(a, tau_grid, params, rng)
        A_trials[i] = A_i
        G_trials[i] = G_i

    # ============================================================
    # Panel (a): CRI-grade slope fit with strict gate-on selection
    # ============================================================
    # Per-τ diagnostics
    G_med = np.median(G_trials, axis=0)                              # median raw G at each τ
    n_on  = (G_trials >= params["gate_thresh"]).sum(axis=0)          # # gate-on trials at τ
    need_n = max(params["min_gate_n"], int(params["min_gate_frac"] * params["N_TRIALS"]))
    gate_mask = (G_med >= params["gate_thresh"]) & (n_on >= need_n)  # τ usable for fit

    # Baseline-corrected amplitude medians (over gate-on trials only)
    A_bc = A_trials - params["C_true"]
    A_med_gate = np.full(N_TAU, np.nan)
    for j in range(N_TAU):
        if gate_mask[j]:
            sel = (G_trials[:, j] >= params["gate_thresh"])
            vals = A_bc[sel, j]
            vals = vals[vals > 0]  # positivity for log
            if vals.size:
                A_med_gate[j] = np.median(vals)

    use = gate_mask & np.isfinite(A_med_gate) & (A_med_gate > 0)

    # (Optional) restrict to first window after turn-on for stability
    if params.get("use_fit_window", True) and np.any(use):
        hit = np.where(G_med >= params["gate_thresh"])[0]
        if hit.size:
            tau_on = tau_grid[hit[0]]
            mwin = (tau_grid >= tau_on) & (tau_grid <= min(params["T0"], tau_on + params.get("fit_window_s", 0.4)))
            use = use & mwin

    tau_fit = tau_grid[use]
    y_fit   = np.log(A_med_gate[use])

    # OLS + HC3 robust SEs (no weights)
    slope = intercept = np.nan
    slope_se = intercept_se = np.nan
    ci_low = ci_high = np.nan
    z = 1.96  # 95% normal approx

    if tau_fit.size >= 2:
        X = np.column_stack([np.ones_like(tau_fit), tau_fit])   # [1, τ]
        beta, cov_hc3 = ols_hc3(X, y_fit)
        intercept, slope = beta[0], beta[1]
        intercept_se = float(np.sqrt(max(cov_hc3[0, 0], 0.0)))
        slope_se     = float(np.sqrt(max(cov_hc3[1, 1], 0.0)))
        ci_low, ci_high = slope - z * slope_se, slope + z * slope_se
        tau_fut_hat = -1.0 / slope if (np.isfinite(slope) and slope < 0) else np.nan
    else:
        tau_fut_hat = np.nan

    # Fitted line and optional 95% HC3 band
    y_line = intercept + slope * tau_fit if np.isfinite(slope) else None
    y_lo = y_hi = None
    if params.get("show_hc3_band", True) and np.isfinite(slope):
        # For each x0 = [1, τ], Var(yhat) = x0^T Cov(beta) x0
        X0 = np.column_stack([np.ones_like(tau_fit), tau_fit])
        var_yhat = np.sum(X0 @ cov_hc3 * X0, axis=1)
        se_yhat = np.sqrt(np.maximum(var_yhat, 0.0))
        y_lo = y_line - z * se_yhat
        y_hi = y_line + z * se_yhat

    # Plot panel (a)
    fig_a, ax_a = plt.subplots(figsize=(5.2, 4.0))
    ax_a.plot(tau_fit, y_fit, marker='o', linestyle='none',
              label='Median ln $A_{\\mathrm{pre}}$ (gate-on)')

    if y_line is not None:
        ax_a.plot(tau_fit, y_line, linestyle='-', label='OLS fit')
        if y_lo is not None:
            # Requested: make the 95% HC3 band green and clearly visible
            ax_a.fill_between(
                tau_fit, y_lo, y_hi,
                color='green', alpha=0.50, linewidth=0, zorder=1.3,
                label='95% HC3 band'
            )

        title_a = (f"log-linear fit (OLS, 95% HC3 CI): "
                   f"slope={slope:.3f} (CI {ci_low:.3f},{ci_high:.3f}); "
                   f"$\\widehat{{\\tau}}_{{\\mathrm{{fut}}}}$={tau_fut_hat:.3f}s")
    else:
        title_a = "log-linear fit (insufficient gate-on τ)"

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

    # Requested: "low a" should be 'navy'; keep others as they are
    colors = {"low a": "navy", "mid a": "tab:orange", "high a": "tab:green"}

    # Median raw gate per τ_f within each arousal bin
    G_medians = []
    for b in range(3):
        in_bin = (a_samples > bins[b]) & (a_samples <= bins[b + 1])
        G_medians.append(np.median(G_trials[in_bin, :], axis=0))

    # Normalize per quantile (robust) & find τ95 in SECONDS
    G_norms, taus_95 = [], []
    for med in G_medians:
        Gn, p0, p_inf = normalize_gate(
            med, tau_grid,
            baseline_max_s=params["baseline_max_s"],
            plateau_frac=params["plateau_frac"]
        )
        G_norms.append(Gn)
        taus_95.append(tau_at_level(tau_grid, Gn, params["sat_level"]))

    # Plot gate saturation (draw green last so it isn't covered)
    fig_b, ax_b = plt.subplots(figsize=(5.2, 4.0))
    lw_low = 1.6
    lw_map = {"low a": lw_low, "mid a": 1.2 * lw_low, "high a": 2.0 * lw_low}
    plot_order = ["mid a", "high a", "low a"]  # ensures green on top
    for lab in plot_order:
        Gn = G_norms[labels.index(lab)]
        ax_b.plot(tau_grid, Gn, label=lab, color=colors[lab], lw=lw_map[lab], zorder=2)

    # Horizontal saturation line
    ax_b.axhline(params["sat_level"], color="0.3", ls="--", lw=1.0, zorder=0)

    ax_b.set_title(r"Gate saturation: normalized $G$ across arousal quantiles")
    ax_b.set_xlabel(r"$\tau_f$ (s)")
    ax_b.set_ylabel(r"Gate $G$ (normalized)")
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.margins(y=0.02)
    ax_b.grid(True, alpha=0.3)

    # Keep legend order as low, mid, high
    handles, leg_labels = ax_b.get_legend_handles_labels()
    order = [leg_labels.index("low a"), leg_labels.index("mid a"), leg_labels.index("high a")]
    ax_b.legend([handles[i] for i in order], [leg_labels[i] for i in order], frameon=False)

    # --- τ95 markers: vertical blue guides + bottom-anchored labels with auto-offset
    vline_kw = dict(color="tab:blue", ls=":", lw=1.15, zorder=1)
    base_y   = ax_b.get_ylim()[0]
    offset_pts = 2  # lift labels slightly above x-axis
    eps = 0.01      # seconds → treat τ95 as coincident if |Δ| < 10 ms
    t95_map = dict(zip(labels, taus_95))

    def _offset_for(label):
        # If low/high coincide, push low left and high right; others centered
        if label == "low a":
            return -8, "right"
        if label == "high a":
            return +8, "left"
        return 0, "center"

    for lab in labels:
        t95 = t95_map.get(lab, np.nan)
        if not np.isfinite(t95):
            continue
        ax_b.axvline(t95, **vline_kw)
        dx_pt, ha = 0, "center"
        for prev in labels:
            if prev == lab:
                break
            t_prev = t95_map.get(prev, np.nan)
            if np.isfinite(t_prev) and abs(t95 - t_prev) < eps:
                dx_pt, ha = _offset_for(lab)
                break

        ax_b.annotate(
            rf"$\tau_{{95}}$({lab}) = {t95:.2f} s",
            xy=(t95, base_y), xycoords="data",
            xytext=(dx_pt, offset_pts), textcoords="offset points",
            rotation=90, va="bottom", ha=ha, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            clip_on=True,
        )

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
