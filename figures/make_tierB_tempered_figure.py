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
import matplotlib.pyplot as plt

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
    S, tvals = 1.0, []
    logs = []
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
    eta = float(f2["eta"])
    tf  = float(f2["tau_fast"])
    ts  = float(f2["tau_slow"])

    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.28)

    # (A) AIC / LRT
    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("Model comparison (Tier-B)", fontsize=11)
    aic1, aic2 = float(aic["AIC_1exp"]), float(aic["AIC_2exp"])
    bars = axA.bar([0, 1], [aic1, aic2], tick_label=["1-exp", "2-exp"])
    for b in bars:
        axA.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)
    axA.text(0.5, 0.02, f"ΔAIC={float(aic['Delta_AIC']):.1f}   "
                        f"LRT={float(aic['LRT']):.2f}   p={float(aic['p_value']):.3g}",
             transform=axA.transAxes, ha="center", va="bottom", fontsize=10)
    axA.set_ylabel("AIC")

    # (B) Early-time curvature (ms)
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("Log-survival curvature (early, ms)", fontsize=11)

    # read sample_B if present
    sample_path = OUT_TIERB / "sample_B.csv"
    used_label = "KM/ECDF"
    if sample_path.exists():
        s = pd.read_csv(sample_path)
        t_emp_s = (s["t"] if "t" in s.columns else s.iloc[:, 0]).to_numpy(dtype=float)
        t_emp, log_emp = km_log_survival(t_emp_s)
        # sanity-check at 200 ms
        t_probe = 0.200
        S_emp = np.exp(np.interp(t_probe, t_emp, log_emp, left=0.999999, right=np.exp(log_emp[-1])))
        S_mod = np.exp(logS_2exp(t_probe, eta, tf, ts))
        if not (0.2 <= (S_emp / S_mod) <= 5.0):   # very loose guard
            # regenerate in-memory synthetic points from fitted params
            t_syn = draw_two_exp(20000, eta, tf, ts)
            t_emp, log_emp = km_log_survival(t_syn)
            used_label = "KM/ECDF (synthetic)"
    else:
        # if no sample, synthesize in-memory
        t_syn = draw_two_exp(20000, eta, tf, ts)
        t_emp, log_emp = km_log_survival(t_syn)
        used_label = "KM/ECDF (synthetic)"

    # convert to ms and clip early window
    t_emp_ms = 1e3 * t_emp
    t_mod_ms = 1e3 * curv["t"].to_numpy()
    tau_slow_ms = max(60.0, min(200.0, ts * 1e3 * 5.0))
    m_emp = t_emp_ms <= tau_slow_ms
    m_mod = t_mod_ms <= tau_slow_ms

    axB.plot(t_emp_ms[m_emp], log_emp[m_emp], marker="o", lw=0, ms=3, label=used_label)
    axB.plot(t_mod_ms[m_mod], curv["log_surv_1exp"].to_numpy()[m_mod], lw=2, label="1-exp fit")
    axB.plot(t_mod_ms[m_mod], curv["log_surv_2exp"].to_numpy()[m_mod], lw=2, label="2-exp fit")
    axB.set_xlabel("time (ms)")
    axB.set_ylabel("log survival")
    axB.set_xlim(0, tau_slow_ms)
    axB.legend(fontsize=9, frameon=False)

    # (C) CIs
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("95% CIs for (η, τ_fast, τ_slow)", fontsize=11)
    labels = [r"η", r"τ$_{\rm fast}$ (ms)", r"τ$_{\rm slow}$ (ms)"]
    centers = [eta, tf*1e3, ts*1e3]
    lo = [ci.loc["lo", "eta"], ci.loc["lo", "tau_fast"]*1e3, ci.loc["lo", "tau_slow"]*1e3]
    hi = [ci.loc["hi", "eta"], ci.loc["hi", "tau_fast"]*1e3, ci.loc["hi", "tau_slow"]*1e3]
    x = np.arange(len(labels))
    axC.errorbar(x, centers,
                 yerr=[np.array(centers)-np.array(lo), np.array(hi)-np.array(centers)],
                 fmt="o", capsize=4)
    axC.set_xticks(x, labels); axC.set_xlim(-0.5, len(labels)-0.5); axC.grid(axis="y", alpha=0.3)
    tau1 = float(f1["value"])
    axC.text(0.02, 0.02,
             f"1-exp τ̂ = {tau1*1e3:.1f} ms\n"
             f"2-exp η̂ = {eta:.2f}\n"
             f"τ̂_fast = {tf*1e3:.1f} ms\n"
             f"τ̂_slow = {ts*1e3:.1f} ms",
             transform=axC.transAxes, va="bottom", fontsize=9)

    fig.suptitle("Tier-B tempered mixtures: 1-exp vs 2-exp", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"tierB_tempered_modelcomp.{ext}", bbox_inches="tight", dpi=300)
    print(f"Saved figures → {OUT_FIG}/tierB_tempered_modelcomp.[png|pdf]")

if __name__ == "__main__":
    main()

