# -*- coding: utf-8 -*- 
"""
@author: ADMIN
figures/make_tierB_tempered_figure.py
(A) AIC/LRT; (B) early-time log-survival curvature (ms); (C) 95% CIs
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # ok to keep even if unused

# CI-safe font (prevents Arial warnings on GitHub runners)
mpl.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parents[1]
TIERB = ROOT / "tierB_tempered"
OUT_TIERB = TIERB / "output"
OUT_FIG = Path(__file__).resolve().parent / "output"
OUT_FIG.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- helpers ----------
def km_log_survival(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return x, np.full_like(x, np.nan, dtype=float)
    xs, counts = np.unique(np.sort(x), return_counts=True)
    n = x.size
    S, tvals, logs = 1.0, [], []
    for xi, di in zip(xs, counts):
        if n <= 0:
            break
        S *= (1.0 - di / n)
        n -= di
        if S <= 0:
            break
        tvals.append(xi)
        logs.append(np.log(S))
    return np.asarray(tvals), np.asarray(logs)

def logS_2exp(t, eta, tf, ts):
    return np.log((1.0 - eta) * np.exp(-t / tf) + eta * np.exp(-t / ts))

def ensure_data():
    need = [
        OUT_TIERB / "aic_lrt.csv",
        OUT_TIERB / "curvature_demo.csv",
        OUT_TIERB / "ci_2exp.csv",
        OUT_TIERB / "fit_1exp.csv",
        OUT_TIERB / "fit_2exp.csv",
    ]
    if all(p.exists() for p in need):
        return
    os.environ.setdefault("TIERB_BOOT", "60")
    from tierB_tempered.simulate_and_fit import main as build
    print("Tier-B data missing → running simulate_and_fit.py (quick bootstrap=60)")
    build()

def draw_two_exp(n, eta, tf, ts, seed=20250701):
    rng = np.random.default_rng(seed)
    z = rng.random(n)
    t = np.empty(n, float)
    slow = (z < eta)
    t[slow] = rng.exponential(ts, slow.sum())
    t[~slow] = rng.exponential(tf, (~slow).sum())
    return t

# ---------- main ----------
def main():
    ensure_data()
    aic = pd.read_csv(OUT_TIERB / "aic_lrt.csv")
    curv = pd.read_csv(OUT_TIERB / "curvature_demo.csv")
    ci = pd.read_csv(OUT_TIERB / "ci_2exp.csv", index_col=0)
    f1 = pd.read_csv(OUT_TIERB / "fit_1exp.csv")
    f2 = pd.read_csv(OUT_TIERB / "fit_2exp.csv")

    # Pull scalars safely (avoids FutureWarning)
    aic1 = float(aic["AIC_1exp"].iloc[0])
    aic2 = float(aic["AIC_2exp"].iloc[0])
    dAIC = float(aic["Delta_AIC"].iloc[0]) if "Delta_AIC" in aic.columns else (aic1 - aic2)
    lrt  = float(aic["LRT"].iloc[0])
    p    = float(aic["p_value"].iloc[0])

    eta = float(f2["eta"].iloc[0])
    tf  = float(f2["tau_fast"].iloc[0])
    ts  = float(f2["tau_slow"].iloc[0])
    tau1 = float(f1["value"].iloc[0])

    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.28)

    # (A) AIC / LRT
    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("Model comparison (Tier-B)", fontsize=11)
    bars = axA.bar([0, 1], [aic1, aic2], tick_label=["1-exp", "2-exp"])
    for b in bars:
        axA.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)
    # AIC/LRT/p summary at top-left, smaller font
    axA.text(
        0.02, 0.98,
        f"ΔAIC={dAIC:.1f}   LRT={lrt:.2f}   p={p:.3g}",
        transform=axA.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.2"),
    )
    axA.set_ylabel("AIC")

    # (B) Curvature on log scale (early, ms)
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("Log-survival curvature (early, ms)", fontsize=11)

    # KM/ECDF with Greenwood band (uncensored)
    def km_with_var(x: np.ndarray):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        xs, counts = np.unique(np.sort(x), return_counts=True)
        n = x.size
        S = 1.0
        S_list, t_list, varS_list = [], [], []
        cum = 0.0
        for xi, di in zip(xs, counts):
            if n <= 0:
                break
            S *= (1.0 - di / n)
            if n - di > 0:
                cum += di / (n * (n - di))  # Greenwood increment
            varS = (S ** 2) * cum
            n -= di
            if S <= 0:
                break
            S_list.append(S); t_list.append(xi); varS_list.append(varS)
        S = np.asarray(S_list); t = np.asarray(t_list); varS = np.asarray(varS_list)
        logS = np.log(S)
        se_logS = np.sqrt(varS) / S  # delta method
        return t, logS, se_logS

    # Prefer raw synthetic sample → KM; else use precomputed columns
    sample_path = OUT_TIERB / "sample_B.csv"
    if sample_path.exists():
        df_s = pd.read_csv(sample_path)
        s = (df_s["t"] if "t" in df_s.columns else df_s.iloc[:, 0]).to_numpy()
        t_emp_s, log_emp, se_log_emp = km_with_var(s)
    else:
        t_emp_s = curv["t"].to_numpy()
        log_emp = curv["log_surv_emp"].to_numpy()
        se_log_emp = None  # no band available

    # Convert to ms and keep an early window that shows curvature clearly
    t_emp_ms = 1e3 * t_emp_s
    t_mod_ms = 1e3 * curv["t"].to_numpy()
    tau_slow_ms = max(60.0, min(200.0, ts * 1e3 * 5.0))
    m_emp = t_emp_ms <= tau_slow_ms
    m_mod = t_mod_ms <= tau_slow_ms

    # Plot KM curve and 95% CI band
    if se_log_emp is not None:
        lo = log_emp - 1.96 * se_log_emp
        hi = log_emp + 1.96 * se_log_emp
        axB.fill_between(t_emp_ms[m_emp], lo[m_emp], hi[m_emp], alpha=0.15, label="KM 95% CI")
    axB.plot(t_emp_ms[m_emp], log_emp[m_emp], marker="o", lw=0, ms=2.5, label="KM/ECDF")

    # Overlay model lines
    axB.plot(t_mod_ms[m_mod], curv["log_surv_1exp"].to_numpy()[m_mod], lw=2, label="1-exp fit")
    axB.plot(t_mod_ms[m_mod], curv["log_surv_2exp"].to_numpy()[m_mod], lw=2, label="2-exp fit")

    axB.set_xlabel("time (ms)")
    axB.set_ylabel("log survival")
    axB.set_xlim(0, tau_slow_ms)
    axB.legend(fontsize=9, frameon=False, loc="lower left")

    # Curvature index κ with a light bootstrap
    def _slope(x, y):
        if x.size < 2:
            return np.nan
        return np.polyfit(x, y, 1)[0]

    W1 = (t_emp_ms >= 10) & (t_emp_ms <= 50)
    W2 = (t_emp_ms >= 120) & (t_emp_ms <= 200)
    k_hat = _slope(t_emp_ms[W1], log_emp[W1]) - _slope(t_emp_ms[W2], log_emp[W2])

    ci_available = False
    if sample_path.exists():
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(300):  # light bootstrap for CI
            xb = rng.choice(s, size=s.size, replace=True)
            tb, lb, _seb = km_with_var(xb)
            tb_ms = 1e3 * tb
            W1b = (tb_ms >= 10) & (tb_ms <= 50)
            W2b = (tb_ms >= 120) & (tb_ms <= 200)
            kb = _slope(tb_ms[W1b], lb[W1b]) - _slope(tb_ms[W2b], lb[W2b])
            if np.isfinite(kb):
                boots.append(kb)
        if boots:
            lo_k, hi_k = np.percentile(boots, [2.5, 97.5])
            ci_available = True

    # κ inset (multiline only if CI available)
    k_lines = [r"$\kappa$ = {:.3f}".format(k_hat)]
    if ci_available:
        k_lines.append("95% CI [{:.3f}, {:.3f}]".format(lo_k, hi_k))
    axB.text(
        0.02, 0.98, "\n".join(k_lines),
        transform=axB.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.2"),
    )

    # (C) CIs
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("95% CIs for (η, τ_fast, τ_slow)", fontsize=11)
    labels = [r"η", r"τ$_{\rm fast}$ (ms)", r"τ$_{\rm slow}$ (ms)"]
    centers = [eta, tf*1e3, ts*1e3]
    lo = [ci.loc["lo", "eta"], ci.loc["lo", "tau_fast"]*1e3, ci.loc["lo", "tau_slow"]*1e3]
    hi = [ci.loc["hi", "eta"], ci.loc["hi", "tau_fast"]*1e3, ci.loc["hi", "tau_slow"]*1e3]
    x = np.arange(len(labels))
    axC.errorbar(
        x, centers,
        yerr=[np.array(centers)-np.array(lo), np.array(hi)-np.array(centers)],
        fmt="o", capsize=4
    )
    axC.set_xticks(x, labels)
    axC.set_xlim(-0.5, len(labels)-0.5)
    axC.grid(axis="y", alpha=0.3)

    # Summary box at top-left
    axC.text(
        0.02, 0.98,
        (f"1-exp $\\hat{{\\tau}}$ = {tau1*1e3:.1f} ms\n"
         f"2-exp $\\hat{{\\eta}}$ = {eta:.2f}\n"
         f"$\\hat{{\\tau}}_\\mathrm{{fast}}$ = {tf*1e3:.1f} ms\n"
         f"$\\hat{{\\tau}}_\\mathrm{{slow}}$ = {ts*1e3:.1f} ms"),
        transform=axC.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.2"),
    )

    fig.suptitle("Tier-B tempered mixtures: 1-exp vs 2-exp", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"tierB_tempered_modelcomp.{ext}", bbox_inches="tight", dpi=300)
    print(f"Saved figures → {OUT_FIG}/tierB_tempered_modelcomp.[png|pdf]")

if __name__ == "__main__":
    main()
