# -*- coding: utf-8 -*- 
"""
@author: ADMIN

Figures for Tier-B tempered mixture:
  (A) AIC / LRT comparison (1-exp vs 2-exp)
  (B) Log-survival curvature (early, ms) — KM/ECDF from sample_B.csv,
      analytic 1-exp & 2-exp curves on the same grid
  (C) 95% CIs for (eta, tau_fast, tau_slow)

Saves PDF/PNG in figures/output/.
If tierB_tempered outputs are missing, we generate them on the fly with small boot.
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
    Return {'t', 'log_surv_emp'} for a Kaplan–Meier survival built directly
    from sample x (in seconds). We drop the final step where S→0 to avoid log(0).
    Works with uncensored samples.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"t": x, "log_surv_emp": np.full_like(x, np.nan, dtype=float)}

    xs, counts = np.unique(np.sort(x), return_counts=True)
    n_risk = x.size
    S_vals, T_vals = [], []
    S = 1.0
    for xi, di in zip(xs, counts):
        if n_risk <= 0:
            break
        S *= (1.0 - di / n_risk)
        n_risk -= di
        if S <= 0:
            break  # avoid log(0)
        S_vals.append(S)
        T_vals.append(xi)

    T_vals = np.asarray(T_vals, float)
    S_vals = np.asarray(S_vals, float)
    return {"t": T_vals, "log_surv_emp": np.log(S_vals)}


def ensure_data():
    # We want these to exist; if not, run the quick generator.
    need = [
        OUT_TIERB / "aic_lrt.csv",
        OUT_TIERB / "ci_2exp.csv",
        OUT_TIERB / "fit_1exp.csv",
        OUT_TIERB / "fit_2exp.csv",
        OUT_TIERB / "sample_B.csv",     # <- require the synthetic sample for KM dots
    ]
    if all(p.exists() for p in need):
        return
    os.environ.setdefault("TIERB_BOOT", "60")
    from tierB_tempered.simulate_and_fit import main as build
    print("Tier-B data missing → running simulate_and_fit.py (quick bootstrap=60)")
    build()


def main():
    ensure_data()

    aic = pd.read_csv(OUT_TIERB / "aic_lrt.csv")
    ci  = pd.read_csv(OUT_TIERB / "ci_2exp.csv", index_col=0)
    f1  = pd.read_csv(OUT_TIERB / "fit_1exp.csv")
    f2  = pd.read_csv(OUT_TIERB / "fit_2exp.csv")

    # ----- Figure layout -----
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.28)

    # (A) AIC / LRT
    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("Model comparison (Tier-B)", fontsize=11)
    aic1, aic2 = float(aic["AIC_1exp"]), float(aic["AIC_2exp"])
    bars = axA.bar([0, 1], [aic1, aic2], tick_label=["1-exp", "2-exp"])
    for b in bars:
        axA.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    dAIC = float(aic["Delta_AIC"])
    lrt, p = float(aic["LRT"]), float(aic["p_value"])
    axA.text(0.5, 0.02, f"ΔAIC={dAIC:.1f}   LRT={lrt:.2f}   p={p:.3g}",
             transform=axA.transAxes, ha="center", va="bottom", fontsize=10)
    axA.set_ylabel("AIC")

    # (B) Log-survival curvature (early, ms) — KM from sample_B, model curves analytic
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("Log-survival curvature (early, ms)", fontsize=11)

    # --- KM dots from the synthetic sample (seconds → ms)
    df_s = pd.read_csv(OUT_TIERB / "sample_B.csv")
    s = (df_s["t"] if "t" in df_s.columns else df_s.iloc[:, 0]).to_numpy()  # seconds
    km = km_log_survival(s)
    t_emp_ms = km["t"] * 1e3
    log_emp  = km["log_surv_emp"]

    # --- analytic curves on the same early-time grid
    tau1 = float(f1["value"])               # seconds
    eta  = float(f2["eta"])
    tf   = float(f2["tau_fast"])            # seconds
    ts   = float(f2["tau_slow"])            # seconds

    # early window: up to 200 ms or 5×tau_slow (whichever smaller; min cap 60 ms)
    tmax_ms = min(200.0, 5.0 * ts * 1e3)
    tmax_ms = max(60.0, tmax_ms)
    tgrid_ms = np.linspace(0.0, tmax_ms, 200)
    tgrid_s  = tgrid_ms / 1e3

    log1 = -tgrid_s / tau1
    log2 = np.log(eta * np.exp(-tgrid_s / tf) + (1.0 - eta) * np.exp(-tgrid_s / ts))

    # restrict dots to the same window
    m_emp = t_emp_ms <= tmax_ms

    axB.plot(t_emp_ms[m_emp], log_emp[m_emp], 'o', ms=3, label="KM/ECDF (synthetic)")
    axB.plot(tgrid_ms, log1, lw=2, label="1-exp fit")
    axB.plot(tgrid_ms, log2, lw=2, label="2-exp fit")
    axB.set_xlabel("time (ms)")
    axB.set_ylabel("log survival")
    axB.set_xlim(0, tmax_ms)
    axB.legend(fontsize=9, frameon=False)

    # (C) CIs for (eta, tau_fast, tau_slow)
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("95% CIs for (η, τ_fast, τ_slow)", fontsize=11)

    eta_hat = float(f2["eta"])
    tf_hat  = float(f2["tau_fast"])
    ts_hat  = float(f2["tau_slow"])

    labels  = [r"η", r"τ$_{\rm fast}$ (ms)", r"τ$_{\rm slow}$ (ms)"]
    centers = [eta_hat, tf_hat * 1e3, ts_hat * 1e3]
    lo = [ci.loc["lo", "eta"],      ci.loc["lo", "tau_fast"] * 1e3, ci.loc["lo", "tau_slow"] * 1e3]
    hi = [ci.loc["hi", "eta"],      ci.loc["hi", "tau_fast"] * 1e3, ci.loc["hi", "tau_slow"] * 1e3]

    x = np.arange(len(labels))
    axC.errorbar(
        x, centers,
        yerr=[np.array(centers) - np.array(lo), np.array(hi) - np.array(centers)],
        fmt="o", capsize=4
    )
    axC.set_xticks(x, labels)
    axC.set_xlim(-0.5, len(labels) - 0.5)
    axC.grid(axis="y", alpha=0.3)

    # neat summary box
    axC.text(
        0.02, 0.02,
        (f"1-exp τ̂ = {tau1*1e3:.1f} ms\n"
         f"2-exp η̂ = {eta_hat:.2f}\n"
         f"τ̂_fast = {tf_hat*1e3:.1f} ms\n"
         f"τ̂_slow = {ts_hat*1e3:.1f} ms"),
        transform=axC.transAxes, va="bottom", fontsize=9
    )

    fig.suptitle("Tier-B tempered mixtures: 1-exp vs 2-exp", fontsize=12, y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"tierB_tempered_modelcomp.{ext}",
                    bbox_inches="tight", dpi=300)
    print(f"Saved figures → {OUT_FIG}/tierB_tempered_modelcomp.[png|pdf]")


if __name__ == "__main__":
    main()
