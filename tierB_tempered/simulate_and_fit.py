# -*- coding: utf-8 -*-
"""
Admin

Tier-B tempered mixture:
 - simulate mixture-of-exponentials data
 - MLE-fit 1-exp and 2-exp
 - export AIC + LRT
 - bootstrap CIs for (eta, tau_fast, tau_slow)
 - curvature demo on log-survival

Outputs -> tierB_tempered/output/
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from math import log, exp
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


# ---------- Simulation ----------
def sample_mixture_exp(n, eta, tau_f, tau_s, rng) -> np.ndarray:
    z = rng.random(n) < eta
    n_f = int(z.sum())
    n_s = n - n_f
    t_fast = rng.exponential(scale=tau_f, size=n_f)
    t_slow = rng.exponential(scale=tau_s, size=n_s)
    return np.concatenate([t_fast, t_slow])


# ---------- Likelihoods (log-domain stable) ----------
def _ll_1exp(times, tau):
    # closed-form MLE is tau = mean(times), but we still implement generic LL
    n = times.size
    if tau <= 0:
        return -np.inf
    lam = 1.0 / tau
    return n * log(lam) - lam * times.sum()


def _ll_2exp(times, eta, tau_f, tau_s):
    if not (0 < eta < 1) or tau_f <= 0 or tau_s <= 0:
        return -np.inf
    # ensure fast < slow to avoid label switching; swap if needed
    if tau_f > tau_s:
        tau_f, tau_s = tau_s, tau_f
        eta = 1.0 - eta
    lam_f, lam_s = 1.0 / tau_f, 1.0 / tau_s
    a = np.log(eta) + np.log(lam_f) - times * lam_f
    b = np.log(1 - eta) + np.log(lam_s) - times * lam_s
    # log-sum-exp across the two components
    ll = special.logsumexp(np.vstack([a, b]), axis=0).sum()
    return float(ll)


# ---------- Parameterizations for unconstrained optimization ----------
def _unpack_theta_1exp(theta):
    # theta[0] = log(tau)
    return np.exp(theta[0])


def _unpack_theta_2exp(theta):
    # theta[0] = logit(eta), eta in (0,1)
    # theta[1] = log(tau_fast)
    # theta[2] = log( tau_slow - tau_fast ), so tau_slow = tau_fast + exp(theta[2])
    eta = 1 / (1 + np.exp(-theta[0]))
    tau_f = np.exp(theta[1])
    tau_s = tau_f + np.exp(theta[2])
    return eta, tau_f, tau_s


def nll_1exp(theta, times):
    tau = _unpack_theta_1exp(theta)
    return -_ll_1exp(times, tau)


def nll_2exp(theta, times):
    eta, tau_f, tau_s = _unpack_theta_2exp(theta)
    return -_ll_2exp(times, eta, tau_f, tau_s)


@dataclass
class FitResult:
    params: dict
    loglik: float
    aic: float
    success: bool
    nit: int


def fit_1exp(times, max_iter=500) -> FitResult:
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
    ll = -res.fun
    k = 1
    return FitResult(
        params=dict(tau=tau),
        loglik=ll,
        aic=2 * k - 2 * ll,
        success=bool(res.success),
        nit=int(res.nit),
    )


def fit_2exp(times, max_iter=500) -> FitResult:
    # heuristics: start from 1-exp fit split into slower+faster
    tau_hat = float(times.mean())
    theta0 = np.array([0.0, np.log(tau_hat * 0.5), np.log(tau_hat * 0.5)], dtype=float)
    res = optimize.minimize(
        fun=nll_2exp,
        x0=theta0,
        args=(times,),
        method="L-BFGS-B",
        options=dict(maxiter=max_iter),
    )
    eta, tau_f, tau_s = _unpack_theta_2exp(res.x)
    ll = -res.fun
    k = 3
    return FitResult(
        params=dict(eta=eta, tau_fast=tau_f, tau_slow=tau_s),
        loglik=ll,
        aic=2 * k - 2 * ll,
        success=bool(res.success),
        nit=int(res.nit),
    )


def curvature_demo(times, bins, t_max):
    # empirical survival via histogram/CDF
    times = times[(times >= 0) & (times <= t_max)]
    hist, edges = np.histogram(times, bins=bins, range=(0, t_max))
    cdf = np.cumsum(hist) / hist.sum()
    surv = 1.0 - cdf
    t_mid = 0.5 * (edges[1:] + edges[:-1])
    # avoid log(0)
    surv = np.clip(surv, 1e-12, None)
    return pd.DataFrame({"t": t_mid, "log_surv_emp": np.log(surv)})


def bootstrap_2exp(times, n_boot, max_iter, rng):
    results = []
    # fit once to center bootstrap
    f2 = fit_2exp(times, max_iter=max_iter)
    eta, tau_f, tau_s = f2.params["eta"], f2.params["tau_fast"], f2.params["tau_slow"]
    n = times.size
    for _ in range(n_boot):
        tb = sample_mixture_exp(n, eta, tau_f, tau_s, rng)
        fb = fit_2exp(tb, max_iter=max_iter)
        results.append(fb.params)
    df = pd.DataFrame(results)
    ci = df.quantile([0.025, 0.5, 0.975]).rename(index={0.025: "lo", 0.5: "med", 0.975: "hi"})
    return df, ci


def main():
    # RNG bookkeeping like the rest of your repo
    load_state()
    P = load_params()
    rng = np.random.default_rng(P.get("seed", 7))
    n_boot = int(os.environ.get("TIERB_BOOT", P.get("n_boot", 120)))
    max_iter = int(P.get("max_iter", 500))

    # simulate Tier-B
    times = sample_mixture_exp(
        n=P["n_samples"],
        eta=P["eta_true"],
        tau_f=P["tau_fast_true_s"],
        tau_s=P["tau_slow_true_s"],
        rng=rng,
    )
    save_state()

    # save raw times
    pd.DataFrame({"t": times}).to_csv(OUT / "tierB_times.csv", index=False)

    # fits
    f1 = fit_1exp(times, max_iter=max_iter)
    f2 = fit_2exp(times, max_iter=max_iter)

    # export fits
    pd.DataFrame([{"param": "tau", "value": f1.params["tau"], "loglik": f1.loglik, "aic": f1.aic}]) \
      .to_csv(OUT / "fit_1exp.csv", index=False)

    pd.DataFrame([{
        "eta": f2.params["eta"],
        "tau_fast": f2.params["tau_fast"],
        "tau_slow": f2.params["tau_slow"],
        "loglik": f2.loglik,
        "aic": f2.aic
    }]).to_csv(OUT / "fit_2exp.csv", index=False)

    # AIC + LRT (χ² with df=2; caveat: mixture on boundary not exact; OK for SI)
    dAIC = f1.aic - f2.aic
    lrt = 2.0 * (f2.loglik - f1.loglik)
    pval = 1.0 - stats.chi2.cdf(lrt, df=2)
    pd.DataFrame([{
        "AIC_1exp": f1.aic,
        "AIC_2exp": f2.aic,
        "Delta_AIC": dAIC,     # positive favors 2-exp
        "logLik_1exp": f1.loglik,
        "logLik_2exp": f2.loglik,
        "LRT": lrt,
        "df": 2,
        "p_value": pval
    }]).to_csv(OUT / "aic_lrt.csv", index=False)

    # curvature demo on log-survival + model overlays
    curv = curvature_demo(times, bins=int(P["curvature_bins"]), t_max=float(P["t_max_s"]))
    # overlay 1-exp and 2-exp theoretical log-survival
    t = curv["t"].values
    tau1 = f1.params["tau"]
    eta, tf, ts = f2.params["eta"], f2.params["tau_fast"], f2.params["tau_slow"]
    S1 = np.exp(-t / tau1)
    S2 = eta * np.exp(-t / tf) + (1 - eta) * np.exp(-t / ts)
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


if __name__ == "__main__":
    main()
