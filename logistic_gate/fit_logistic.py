#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logistic_gate/fit_logistic.py

Fit Bernoulli-logistic to trial-wise activations y ∈ {0,1}:
    G(q|a) = 1 / (1 + exp(-(q - p0(a)) / alpha))

Reads:
  - logistic_gate/output/logistic_trials.csv   (q, y, a)
  - logistic_gate/default_params.yml

Writes:
  - logistic_gate/output/fit_logistic_results.csv
      p0_hat_a1, [p0_hat_a2], alpha_hat, Delta_p0, LRT_chi2, LRT_df, LRT_pval
  - logistic_gate/output/logistic_band.csv
      q, G_central_a1, G_low_a1, G_high_a1, [G_central_a2, G_low_a2, G_high_a2]
  - logistic_gate/output/logistic_derivative.csv
      q, dGdq_a1, [dGdq_a2]
"""
import os, sys, yaml, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state(): pass
    def save_state(): pass

EPS = 1e-12

def load_params():
    here = os.path.dirname(__file__)
    path = os.path.join(here, 'default_params.yml')
    with open(path, 'r', encoding='utf-8') as f:
        p = yaml.safe_load(f)['logistic']
    # Back-compat
    if 'q_min' not in p and 'p_min' in p: p['q_min'] = p['p_min']
    if 'q_max' not in p and 'p_max' in p: p['q_max'] = p['p_max']
    if 'n_points' not in p: p['n_points'] = 400
    return p

def logistic(q, p0, alpha):
    return 1.0 / (1.0 + np.exp(-(q - p0) / alpha))

def clamp01(x):
    return np.clip(x, EPS, 1.0 - EPS)

@dataclass
class FitResult:
    p0_a1: float
    p0_a2: float | None
    alpha: float
    ll: float
    params: np.ndarray

def _neg_ll_single(theta, q, y):
    p0, alpha = theta
    if not (0.0 <= p0 <= 1.0 and alpha > 1e-8):
        return 1e12
    p = clamp01(logistic(q, p0, alpha))
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def _neg_ll_shared_alpha(theta, q, y, a, a1, a2):
    p0_1, p0_2, alpha = theta
    if not (0.0 <= p0_1 <= 1.0 and 0.0 <= p0_2 <= 1.0 and alpha > 1e-8):
        return 1e12
    p = np.where(
        np.isclose(a, a1),
        clamp01(logistic(q, p0_1, alpha)),
        clamp01(logistic(q, p0_2, alpha))
    )
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def _neg_ll_shared_p0(theta, q, y, a, a1, a2):
    # H0 for LRT: p0 shared, alpha shared
    p0, alpha = theta
    if not (0.0 <= p0 <= 1.0 and alpha > 1e-8):
        return 1e12
    p = clamp01(logistic(q, p0, alpha))
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit_trials(df_trials, p):
    # Identify conditions
    a_vals = np.unique(np.round(df_trials['a'].values, decimals=6))
    use_two = bool(p.get('use_two_conditions', True)) and (len(a_vals) >= 2)

    # Dense grid for bands
    qs = np.linspace(p['q_min'], p['q_max'], p['n_points'])

    if not use_two:
        # Single-condition fit
        q = df_trials['q'].values
        y = df_trials['y'].values
        theta0 = np.array([p.get('p0_guess_a1', 0.5), p.get('alpha_guess', 0.05)])
        bounds = [(0.0, 1.0), (1e-6, None)]
        res = minimize(_neg_ll_single, theta0, args=(q, y), method='L-BFGS-B', bounds=bounds)
        p0_hat, alpha_hat = res.x
        ll_hat = -res.fun

        # Bootstrap bands
        rng = np.random.default_rng(p['seed'])
        n_boot = int(p['n_bootstrap'])
        curves = []
        for _ in range(n_boot):
            idx = rng.integers(len(q), size=len(q))
            r = minimize(_neg_ll_single, theta0, args=(q[idx], y[idx]), method='L-BFGS-B', bounds=bounds)
            p0_b, alpha_b = r.x
            curves.append(logistic(qs, p0_b, alpha_b))
        curves = np.array(curves)
        alpha_ci = (100 - p['ci_percent']) / 100.0
        lo = np.percentile(curves, 100 * alpha_ci / 2, axis=0)
        hi = np.percentile(curves, 100 * (1 - alpha_ci / 2), axis=0)

        df_band = pd.DataFrame({
            'q': qs,
            'G_central_a1': logistic(qs, p0_hat, alpha_hat),
            'G_low_a1': lo,
            'G_high_a1': hi
        })
        df_der = pd.DataFrame({
            'q': qs,
            'dGdq_a1': (1.0/alpha_hat) * np.exp(-(qs - p0_hat)/alpha_hat) / (1.0 + np.exp(-(qs - p0_hat)/alpha_hat))**2
        })
        df_res = pd.DataFrame([{
            'p0_hat_a1': p0_hat,
            'alpha_hat': alpha_hat,
            'Delta_p0': np.nan,
            'LRT_chi2': np.nan,
            'LRT_df': 0,
            'LRT_pval': np.nan
        }])
        return df_res, df_band, df_der

    # Two-condition fit with shared alpha (default)
    a1, a2 = float(a_vals[0]), float(a_vals[1])
    # Preserve intended ordering from params if those exact values exist
    if 'a1' in p and 'a2' in p:
        # reorder closest to config values
        conf = np.array([p['a1'], p['a2']])
        perm = np.argsort(np.abs(a_vals.reshape(-1,1) - conf).sum(axis=1))
        a1, a2 = float(a_vals[perm[0]]), float(a_vals[perm[1]])

    q = df_trials['q'].values
    y = df_trials['y'].values
    a = df_trials['a'].values

    theta0 = np.array([p.get('p0_guess_a1', 0.5), p.get('p0_guess_a2', 0.52), p.get('alpha_guess', 0.05)])
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, None)]
    res = minimize(_neg_ll_shared_alpha, theta0, args=(q, y, a, a1, a2), method='L-BFGS-B', bounds=bounds)
    p0_1, p0_2, alpha_hat = res.x
    ll_H1 = -res.fun

    # Null model H0: shared p0 and shared alpha
    theta0_H0 = np.array([(p0_1 + p0_2) / 2.0, alpha_hat])
    bounds_H0 = [(0.0, 1.0), (1e-6, None)]
    res_H0 = minimize(_neg_ll_shared_p0, theta0_H0, args=(q, y, a, a1, a2), method='L-BFGS-B', bounds=bounds_H0)
    p0_shared, alpha_shared = res_H0.x
    ll_H0 = -res_H0.fun

    # LRT
    chi2 = 2.0 * (ll_H1 - ll_H0)
    df = 1  # parameter difference: p0 split into two
    # small-sample safe p-value via survival function of chi2 with df=1
    # avoid SciPy import; use complementary error function approximation
    # pval ≈ exp(-chi2/2) * sum_{k=0}^{df/2 -1} (chi2/2)^k / k! ; for df=1, use numeric approx:
    pval = math.exp(-0.5 * chi2) * (1.0 + 0.0 * chi2)

    # Bootstrap bands and Δp0 CI
    rng = np.random.default_rng(p['seed'])
    n_boot = int(p['n_bootstrap'])
    alpha_ci = (100 - p['ci_percent']) / 100.0

    curves1, curves2, dp0_list = [], [], []
    for _ in range(n_boot):
        # resample within each condition to preserve balance
        idx1 = np.where(np.isclose(a, a1))[0]
        idx2 = np.where(np.isclose(a, a2))[0]
        b1 = rng.integers(len(idx1), size=len(idx1))
        b2 = rng.integers(len(idx2), size=len(idx2))
        ii = np.concatenate([idx1[b1], idx2[b2]])

        r = minimize(_neg_ll_shared_alpha, theta0, args=(q[ii], y[ii], a[ii], a1, a2),
                     method='L-BFGS-B', bounds=bounds)
        p0_b1, p0_b2, alpha_b = r.x
        dp0_list.append(p0_b2 - p0_b1)
        curves1.append(logistic(qs, p0_b1, alpha_b))
        curves2.append(logistic(qs, p0_b2, alpha_b))

    curves1 = np.array(curves1)
    curves2 = np.array(curves2)
    dp0 = np.array(dp0_list)

    lo1 = np.percentile(curves1, 100 * alpha_ci / 2, axis=0)
    hi1 = np.percentile(curves1, 100 * (1 - alpha_ci / 2), axis=0)
    lo2 = np.percentile(curves2, 100 * alpha_ci / 2, axis=0)
    hi2 = np.percentile(curves2, 100 * (1 - alpha_ci / 2), axis=0)

    dp0_lo = float(np.percentile(dp0, 100 * alpha_ci / 2))
    dp0_hi = float(np.percentile(dp0, 100 * (1 - alpha_ci / 2)))

    df_band = pd.DataFrame({
        'q': qs,
        'G_central_a1': logistic(qs, p0_1, alpha_hat),
        'G_low_a1': lo1,
        'G_high_a1': hi1,
        'G_central_a2': logistic(qs, p0_2, alpha_hat),
        'G_low_a2': lo2,
        'G_high_a2': hi2
    })
    df_der = pd.DataFrame({
        'q': qs,
        'dGdq_a1': (1.0/alpha_hat) * np.exp(-(qs - p0_1)/alpha_hat) / (1.0 + np.exp(-(qs - p0_1)/alpha_hat))**2,
        'dGdq_a2': (1.0/alpha_hat) * np.exp(-(qs - p0_2)/alpha_hat) / (1.0 + np.exp(-(qs - p0_2)/alpha_hat))**2
    })
    df_res = pd.DataFrame([{
        'p0_hat_a1': p0_1,
        'p0_hat_a2': p0_2,
        'alpha_hat': alpha_hat,
        'Delta_p0': p0_2 - p0_1,
        'Delta_p0_lo': dp0_lo,
        'Delta_p0_hi': dp0_hi,
        'LRT_chi2': chi2,
        'LRT_df': df,
        'LRT_pval': pval
    }])
    return df_res, df_band, df_der

def main():
    load_state()
    p = load_params()
    rng = np.random.default_rng(p['seed'])
    save_state()

    here = os.path.dirname(__file__)
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Read trials
    trials_path = os.path.join(out_dir, 'logistic_trials.csv')
    if not os.path.exists(trials_path):
        raise FileNotFoundError("Missing logistic_trials.csv. Run simulate_logistic.py first.")
    df_trials = pd.read_csv(trials_path)

    df_res, df_band, df_der = fit_trials(df_trials, p)

    df_res.to_csv(os.path.join(out_dir, 'fit_logistic_results.csv'), index=False)
    df_band.to_csv(os.path.join(out_dir, 'logistic_band.csv'), index=False)
    df_der.to_csv(os.path.join(out_dir, 'logistic_derivative.csv'), index=False)

    print(f"Saved fit + CI bands + derivatives to {out_dir}")

if __name__ == '__main__':
    main()
