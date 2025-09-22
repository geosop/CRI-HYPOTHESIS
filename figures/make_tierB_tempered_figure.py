# -*- coding: utf-8 -*- 
"""
@author: ADMIN

Figures for Tier-B tempered mixture:
  (A) AIC / LRT comparison (1-exp vs 2-exp)
  (B) Log-survival curvature demo
  (C) 95% CIs for (eta, tau_fast, tau_slow)

Saves PDF/PNG in figures/output/.
If tierB_tempered outputs are missing, we generate them on-the-fly with small boot.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TIERB = ROOT / "tierB_tempered"
OUT_TIERB = TIERB / "output"
OUT_FIG = Path(__file__).resolve().parent / "output"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ensure module on path for fallback generation
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- ECDF / Kaplan–Meier survival (no censoring) ------------------------------
def km_log_survival(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return {'t', 'log_surv_emp'} for an ECDF/Kaplan–Meier survival built
    directly from the sample x. We drop the final step where S would hit 0
    to avoid log(0). Works with purely uncensored samples.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"t": x, "log_surv_emp": np.full_like(x, np.nan, dtype=float)}

    xs, counts = np.unique(np.sort(x), return_counts=True)
    n_risk = x.size
    S_vals = []
    T_vals = []
    S = 1.0

    for xi, di in zip(xs, counts):
        if n_risk <= 0:
            break
        # KM step at distinct time xi with d_i events
        S *= (1.0 - di / n_risk)
        n_risk -= di
        if S <= 0:
            # would be log(0); stop before it
            break
        S_vals.append(S)
        T_vals.append(xi)

    T_vals = np.asarray(T_vals, float)
    S_vals = np.asarray(S_vals, float)
    logS = np.log(S_vals)
    return {"t": T_vals, "log_surv_emp": logS}


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
    # Fallback: generate quickly with fewer boot reps to keep CI fast
    os.environ.setdefault("TIERB_BOOT", "60")
    from tierB_tempered.simulate_and_fit import main as build
    print("Tier-B data missing → running simulate_and_fit.py (quick bootstrap=60)")
    build()


def main():
    ensure_data()

    aic = pd.read_csv(OUT_TIERB / "aic_lrt.csv")
    curv = pd.read_csv(OUT_TIERB / "curvature_demo.csv")
    ci = pd.read_csv(OUT_TIERB / "ci_2exp.csv", index_col=0)
    f1 = pd.read_csv(OUT_TIERB / "fit_1exp.csv")
    f2 = pd.read_csv(OUT_TIERB / "fit_2exp.csv")

    # ----- Figure layout -----
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
    dAIC = float(aic["Delta_AIC"])
    lrt, p = float(aic["LRT"]), float(aic["p_value"])
    axA.text(0.5, 0.02, f"ΔAIC={dAIC:.1f}   LRT={lrt:.2f}   p={p:.3g}",
             transform=axA.transAxes, ha="center", va="bottom", fontsize=10)
    axA.set_ylabel("AIC")

    # (B) Curvature on log scale
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("Log-survival curvature", fontsize=11)

    # Prefer KM/ECDF from a raw synthetic sample if available
    # Expecting a simple one-column CSV of event times in seconds
    sample_path = OUT_TIERB / "sample_B.csv"
    if sample_path.exists():
        s = pd.read_csv(sample_path).to_numpy().ravel()
        km = km_log_survival(s)
        t_emp, log_emp = km["t"], km["log_surv_emp"]
        emp_label = "empirical (KM/ECDF)"
    else:
        # Fallback to precomputed column (keeps backward compatibility)
        t_emp = curv["t"].to_numpy()
        log_emp = curv["log_surv_emp"].to_numpy()
        emp_label = "empirical"

    axB.plot(t_emp, log_emp, marker="o", lw=0, ms=3, label=emp_label)
    axB.plot(curv["t"], curv["log_surv_1exp"], lw=2, label="1-exp fit")
    axB.plot(curv["t"], curv["log_surv_2exp"], lw=2, label="2-exp fit")
    axB.set_xlabel("time (s)")
    axB.set_ylabel("log survival")
    axB.legend(fontsize=9, frameon=False)

    # (C) CIs for (eta, tau_fast, tau_slow)
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("95% CIs for (η, τ_fast, τ_slow)", fontsize=11)

    # point estimates:
    eta_hat = float(f2["eta"])
    tf_hat = float(f2["tau_fast"])
    ts_hat = float(f2["tau_slow"])

    labels = [r"η", r"τ$_{\rm fast}$ (ms)", r"τ$_{\rm slow}$ (ms)"]
    centers = [eta_hat, tf_hat*1e3, ts_hat*1e3]
    lo = [ci.loc["lo", "eta"], ci.loc["lo", "tau_fast"]*1e3, ci.loc["lo", "tau_slow"]*1e3]
    hi = [ci.loc["hi", "eta"], ci.loc["hi", "tau_fast"]*1e3, ci.loc["hi", "tau_slow"]*1e3]

    x = np.arange(len(labels))
    axC.errorbar(x, centers, yerr=[np.array(centers)-np.array(lo), np.array(hi)-np.array(centers)],
                 fmt="o", capsize=4)
    axC.set_xticks(x, labels)
    axC.set_xlim(-0.5, len(labels)-0.5)
    axC.grid(axis="y", alpha=0.3)

    # neat summary box
    tau1 = float(f1["value"])
    axC.text(0.02, 0.02,
             f"1-exp τ̂ = {tau1*1e3:.1f} ms\n"
             f"2-exp η̂ = {eta_hat:.2f}\n"
             f"τ̂_fast = {tf_hat*1e3:.1f} ms\n"
             f"τ̂_slow = {ts_hat*1e3:.1f} ms",
             transform=axC.transAxes, va="bottom", fontsize=9)

    fig.suptitle("Tier-B tempered mixtures: 1-exp vs 2-exp", fontsize=12, y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"tierB_tempered_modelcomp.{ext}", bbox_inches="tight", dpi=300)
    print(f"Saved figures → {OUT_FIG}/tierB_tempered_modelcomp.[png|pdf]")


if __name__ == "__main__":
    main()

