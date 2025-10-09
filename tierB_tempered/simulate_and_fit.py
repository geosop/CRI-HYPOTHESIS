# -*- coding: utf-8 -*-
"""
Admin

Tier-B tempered mixture (simulation + fitting)

- Simulate mixture-of-exponentials lifetimes (Tier-B)
- MLE fit: 1-exp and 2-exp (η = FAST-component weight)
- Export AIC & AICc for each model
- Likelihood-ratio test (Wilks χ² with df=2) + parametric-bootstrap p-value under 1-exp null
- Bootstrap CIs for (eta, tau_fast, tau_slow)
- Curvature demo on log-survival (diagnostic only)

Outputs → tierB_tempered/output/
  - tierB_times.csv
  - fit_1exp.csv (tau, loglik, aic, aicc)
  - fit_2exp.csv (eta, tau_fast, tau_slow, loglik, aic, aicc)
  - aic_lrt.csv   (AIC/AICc for both, Δ’s, log-liks, LRT, df, p_wilks, p_boot, metric)
  - curvature_demo.csv (empirical + model log-survival)
  - bootstrap_2exp.csv, ci_2exp.csv
  - manifest.json

Environment overrides:
  TIERB_BOOT      → bootstrap draws for CI of 2-exp params   (default from YAML)
  TIERB_LRT_BOOT  → parametric-bootstrap LRT replicates       (default 0 → skip)
"""

from __future__ import annotations

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from math import log
from dataclasses import dataclass
from pathlib import Path
from scipy import optimize, stats, special

# allow utilities import (seed_manager)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utilities.seed_manager import load_state, save_state  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    with open(HERE / "default_params.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["tierB_tempered"]


# ---------- helpers ----------
def aicc(aic: float, n: int, k: int) -> float:
    """
    AICc = AIC + 2k(k+1)/(n-k-1), when n > k + 1; else return NaN.
    """
    denom = (n - k - 1)
    if denom <= 0:
        return float("nan")
    return aic + (2.0 * k * (k + 1)) / denom


# ---------- Simulation ----------
def sample_mixture_exp(n: int, eta_fast: float, tau_fast: float, tau_slow: float, rng) -> np.ndarray:
    """
    Draw lifetimes from a FAST/SLOW two-exponential mixture.
    η is the FAST weight (slow weight = 1 - η).
    """
    z = rng.random(n) < eta_fast
    n_f = int(z.sum())
    n_s = n - n_f
    t_fast = rng.exponential(scale=tau_fast, size=n_f)
    t_slow = rng.exponential(scale=tau_slow, size=n_s)
    return np.concatenate([t_fast, t_slow])


# ---------- Likelihoods (log-domain stable) ----------
def _ll_1exp(times: np.ndarray, tau: float) -> float:
    # closed-form MLE is tau = mean(times), but we keep generic LL for symmetry
    n = times.size
    if tau <= 0:
        return -np.inf
    lam = 1.0 / tau
    return n * log(lam) - lam * float(times.sum())


def _ll_2exp(times: np.ndarray, eta_fast: float, tau_fast: float, tau_slow: float) -> float:
    """
    Mixture log-likelihood (η = fast weight).
    Enforce tau_fast <= tau_slow to avoid label switching.
    """
    if not (0.0 < eta_fast < 1.0) or tau_fast <= 0 or tau_slow <= 0:
        return -np.inf

    # Relabel if needed
    if tau_fast > tau_slow:
        tau_fast, tau_slow = tau_slow, tau_fast
        eta_fast = 1.0 - eta_fast

    lam_f, lam_s = 1.0 / tau_fast, 1.0 / tau_slow
    a = np.log(eta_fast) + np.log(lam_f) - times * lam_f
    b = np.log(1.0 - eta_fast) + np.log(lam_s) - times * lam_s
    ll = special.logsumexp(np.vstack([a, b]), axis=0).sum()
    return float(ll)


# ---------- Parameterizations for unconstrained optimization ----------
def _unpack_theta_1exp(theta: np.ndarray) -> float:
    # theta[0] = log(tau)
    return float(np.exp(theta[0]))


def _unpack_theta_2exp(theta: np.ndarray) -> tuple[float, float, float]:
    # theta[0] = logit(eta_fast)          → eta_fast in (0,1)
    # theta[1] = log(tau_fast)
    # theta[2] = log( tau_slow - tau_fast ), so tau_slow = tau_fast + exp(theta[2])
    eta_fast = 1.0 / (1.0 + np.exp(-theta[0]))
    tau_fast = float(np.exp(theta[1]))
    tau_slow = tau_fast + float(np.exp(theta[2]))
    return eta_fast, tau_fast, tau_slow


def nll_1exp(theta: np.ndarray, times: np.ndarray) -> float:
    tau = _unpack_theta_1exp(theta)
    return -_ll_1exp(times, tau)


def nll_2exp(theta: np.ndarray, times: np.ndarray) -> float:
    eta_fast, tau_fast, tau_slow = _unpack_theta_2exp(theta)
    return -_ll_2exp(times, eta_fast, tau_fast, tau_slow)


@dataclass
class FitResult:
    params: dict
    loglik: float
    aic: float
    aicc: float
    success: bool
    nit: int


def fit_1exp(times: np.ndarray, max_iter: int = 500) -> FitResult:
    n = times.size
    k = 1
    # sensible start at log(mean)
    tau0 = float(times.mean())
    res = optimize.minimize(
        fun=nll_1exp,
        x0=np.array([np.log(tau0 + 1e-12)], dtype=float),
        args=(times,),
        method="L-BFGS-B",
        options=dict(maxiter=max_iter),
    )
    tau = _unpack_theta_1exp(res.x)
    ll = -float(res.fun)
    aic = 2 * k - 2 * ll
    return FitResult(
        params=dict(tau=tau),
        loglik=ll,
        aic=aic,
        aicc=aicc(aic, n=n, k=k),
        success=bool(res.success),
        nit=int(res.nit),
    )


def fit_2exp(times: np.ndarray, max_iter: int = 500) -> FitResult:
    n = times.size
    k = 3
    # heuristic start from 1-exp fit split into slower+faster
    tau_hat = float(times.mean())
    theta0 = np.array([0.0, np.log(tau_hat * 0.5), np.log(tau_hat * 0.5)], dtype=float)
    res = optimize.minimize(
        fun=nll_2exp,
        x0=theta0,
        args=(times,),
        method="L-BFGS-B",
        options=dict(maxiter=max_iter),
    )
    eta_fast, tau_fast, tau_slow = _unpack_theta_2exp(res.x)
    ll = -float(res.fun)
    aic = 2 * k - 2 * ll
    return FitResult(
        params=dict(eta=eta_fast, tau_fast=tau_fast, tau_slow=tau_slow),
        loglik=ll,
        aic=aic,
        aicc=aicc(aic, n=n, k=k),
        success=bool(res.success),
        nit=int(res.nit),
    )


# ---------- Curvature demo ----------
def curvature_demo(times: np.ndarray, bins: int, t_max: float) -> pd.DataFrame:
    """
    Empirical survival via histogram/CDF (diagnostic). Values are NOT used for AIC/LRT.
    """
    times = times[(times >= 0) & (times <= t_max)]
    hist, edges = np.histogram(times, bins=bins, range=(0.0, t_max))
    cdf = np.cumsum(hist) / max(1, hist.sum())
    surv = 1.0 - cdf
    t_mid = 0.5 * (edges[1:] + edges[:-1])
    surv = np.clip(surv, 1e-12, None)  # avoid log(0)
    return pd.DataFrame({"t": t_mid, "log_surv_emp": np.log(surv)})


# ---------- Bootstrap CIs for 2-exp ----------
def bootstrap_2exp(times: np.ndarray, n_boot: int, max_iter: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parametric bootstrap centered at the fitted 2-exp parameters.
    """
    results = []
    f2 = fit_2exp(times, max_iter=max_iter)
    eta, tau_f, tau_s = f2.params["eta"], f2.params["tau_fast"], f2.params["tau_slow"]
    n = times.size
    for _ in range(int(n_boot)):
        tb = sample_mixture_exp(n, eta, tau_f, tau_s, rng)
        fb = fit_2exp(tb, max_iter=max_iter)
        results.append(fb.params)
    df = pd.DataFrame(results)
    ci = df.quantile([0.025, 0.5, 0.975]).rename(index={0.025: "lo", 0.5: "med", 0.975: "hi"})
    return df, ci


# ---------- Parametric-bootstrap LRT under 1-exp null ----------
def lrt_parametric_boot(times: np.ndarray, f1: FitResult, f2: FitResult, B: int, max_iter: int, rng) -> float:
    """
    Bootstrap p-value for LRT under H0: 1-exp.
    Simulate from Exp(tau_hat), refit both models, compute LRT_b = 2*(LL2-LL1).
    Return p_boot = mean(LRT_b >= LRT_obs) (with +1/(B+1) continuity correction).
    """
    if B <= 0:
        return float("nan")

    n = times.size
    tau_hat = float(f1.params["tau"])
    lrt_obs = 2.0 * (f2.loglik - f1.loglik)
    count = 0
    valid = 0
    for _ in range(int(B)):
        tb = np.random.default_rng(rng.integers(0, 2**31 - 1)).exponential(scale=tau_hat, size=n)
        # Fit both models to bootstrap sample
        f1b = fit_1exp(tb, max_iter=max_iter)
        f2b = fit_2exp(tb, max_iter=max_iter)
        if np.isfinite(f1b.loglik) and np.isfinite(f2b.loglik):
            lrt_b = 2.0 * (f2b.loglik - f1b.loglik)
            valid += 1
            if lrt_b >= lrt_obs:
                count += 1
    if valid == 0:
        return float("nan")
    # add +1 / (valid+1) to avoid zero p-values in small samples
    return (count + 1.0) / (valid + 1.0)


def main():
    # RNG bookkeeping like the rest of your repo
    load_state()
    P = load_params()
    rng = np.random.default_rng(P.get("seed", 7))
    n_boot = int(os.environ.get("TIERB_BOOT", P.get("n_boot", 120)))
    max_iter = int(P.get("max_iter", 500))
    lrt_boot = int(os.environ.get("TIERB_LRT_BOOT", "0"))  # default: skip bootstrap LRT

    # simulate Tier-B lifetimes
    times = sample_mixture_exp(
        n=P["n_samples"],
        eta_fast=P["eta_true"],
        tau_fast=P["tau_fast_true_s"],
        tau_slow=P["tau_slow_true_s"],
        rng=rng,
    )
    save_state()

    n = times.size

    # save raw times (used by the figure for KM/ECDF)
    pd.DataFrame({"t": times}).to_csv(OUT / "tierB_times.csv", index=False)

    # fits
    f1 = fit_1exp(times, max_iter=max_iter)
    f2 = fit_2exp(times, max_iter=max_iter)

    # export fits (include AICc)
    pd.DataFrame([{
        "param": "tau",
        "value": f1.params["tau"],
        "loglik": f1.loglik,
        "aic": f1.aic,
        "aicc": f1.aicc,
    }]).to_csv(OUT / "fit_1exp.csv", index=False)

    pd.DataFrame([{
        "eta": f2.params["eta"],            # FAST weight
        "tau_fast": f2.params["tau_fast"],  # seconds
        "tau_slow": f2.params["tau_slow"],  # seconds
        "loglik": f2.loglik,
        "aic": f2.aic,
        "aicc": f2.aicc,
    }]).to_csv(OUT / "fit_2exp.csv", index=False)

    # AIC / AICc deltas
    dAIC = f1.aic - f2.aic
    dAICc = (f1.aicc - f2.aicc) if (np.isfinite(f1.aicc) and np.isfinite(f2.aicc)) else float("nan")

    # LRT (Wilks; note mixture non-regular → approximate)
    lrt = 2.0 * (f2.loglik - f1.loglik)
    p_wilks = 1.0 - stats.chi2.cdf(lrt, df=2)

    # Parametric bootstrap under 1-exp null (optional, costly)
    p_boot = lrt_parametric_boot(times, f1, f2, B=lrt_boot, max_iter=max_iter, rng=rng)

    # Decide which metric label to recommend for plotting
    # (common rule-of-thumb: use AICc if n/k < 40 for any model; here check the larger k=3)
    metric = "AICc" if (n / 3.0) < 40.0 and np.isfinite(f1.aicc) and np.isfinite(f2.aicc) else "AIC"

    pd.DataFrame([{
        "metric": metric,
        "AIC_1exp": f1.aic,
        "AIC_2exp": f2.aic,
        "AICc_1exp": f1.aicc,
        "AICc_2exp": f2.aicc,
        "Delta_AIC": dAIC,        # positive favors 2-exp
        "Delta_AICc": dAICc,      # positive favors 2-exp (if defined)
        "logLik_1exp": f1.loglik,
        "logLik_2exp": f2.loglik,
        "LRT": lrt,
        "df": 2,
        "p_wilks": p_wilks,
        "p_boot": p_boot
    }]).to_csv(OUT / "aic_lrt.csv", index=False)

    # curvature demo on log-survival + model overlays (diagnostic)
    curv = curvature_demo(times, bins=int(P["curvature_bins"]), t_max=float(P["t_max_s"]))
    t = curv["t"].values
    tau1 = f1.params["tau"]
    eta, tf, ts = f2.params["eta"], f2.params["tau_fast"], f2.params["tau_slow"]
    S1 = np.exp(-t / tau1)
    S2 = eta * np.exp(-t / tf) + (1.0 - eta) * np.exp(-t / ts)
    curv["log_surv_1exp"] = np.log(np.clip(S1, 1e-15, None))
    curv["log_surv_2exp"] = np.log(np.clip(S2, 1e-15, None))
    curv.to_csv(OUT / "curvature_demo.csv", index=False)

    # bootstrap CIs for (eta, tau_fast, tau_slow)
    boot_df, ci = bootstrap_2exp(times, n_boot=n_boot, max_iter=max_iter, rng=rng)
    boot_df.to_csv(OUT / "bootstrap_2exp.csv", index=False)
    ci.to_csv(OUT / "ci_2exp.csv")

    # manifest
    manifest = {
        "files": [
            "tierB_times.csv", "fit_1exp.csv", "fit_2exp.csv",
            "aic_lrt.csv", "curvature_demo.csv", "bootstrap_2exp.csv", "ci_2exp.csv"
        ]
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Tier-B tempered mixture outputs → {OUT}")

    # --- also generate a matching raw sample for KM/ECDF plotting (optional) ---
    try:
        from tierB_tempered.make_sample_B import main as _make_sample_B
        _make_sample_B()
    except Exception as e:
        print(f"[WARN] make_sample_B skipped: {e}")


if __name__ == "__main__":
    main()

