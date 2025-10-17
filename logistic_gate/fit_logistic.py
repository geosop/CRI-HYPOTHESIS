#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logistic_gate/fit_logistic.py

Backbone: trial-wise Bernoulli logistic fits for two arousal levels with shared alpha.
Diagnostics (SI): kernel-smoothed curves + calibration curve and metrics.

Reads:
  - logistic_gate/output/logistic_trials.csv   (q, y, a)
  - logistic_gate/default_params.yml

Writes:
  - logistic_gate/output/fit_logistic_results.csv
      p0_hat_a1, [p0_hat_a2], alpha_hat, Delta_p0, Delta_p0_lo, Delta_p0_hi,
      LRT_chi2, LRT_df, LRT_pval,
      Brier_a1, Brier_a2, Brier_all, ECE_a1, ECE_a2, ECE_all
  - logistic_gate/output/logistic_band.csv
      q, G_central_a1, G_low_a1, G_high_a1, [G_central_a2, G_low_a2, G_high_a2]
  - logistic_gate/output/logistic_derivative.csv
      q, dGdq_a1, [dGdq_a2]
  - logistic_gate/output/logistic_kernel.csv            (diagnostic)
      q, Gk_central_a1, Gk_low_a1, Gk_high_a1, [Gk_central_a2, Gk_low_a2, Gk_high_a2]
  - logistic_gate/output/logistic_calibration.csv       (diagnostic)
      bin_left, bin_right, bin_center, n_bin, pred_mean, obs_rate, lo, hi, a
  - logistic_gate/output/logistic_calibration_metrics.csv (diagnostic)
      metric, a, value
"""
import os, sys, yaml, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from math import erf, sqrt

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
    if 'q_min' not in p and 'p_min' in p: p['q_min'] = p['p_min']
    if 'q_max' not in p and 'p_max' in p: p['q_max'] = p['p_max']
    if 'n_points' not in p: p['n_points'] = 400
    return p

def logistic(q, p0, alpha):
    # numerically safe logistic
    z = -(q - p0) / max(alpha, 1e-12)
    out = np.empty_like(z, dtype=float)
    # stable computation depending on sign
    pos = z >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def clamp01(x):
    return np.clip(x, EPS, 1.0 - EPS)

# ----------- Negative log-likelihoods -----------
def _nll_single(theta, q, y):
    p0, alpha = theta
    if not (0.0 <= p0 <= 1.0 and alpha > 1e-8): return 1e12
    p = clamp01(logistic(q, p0, alpha))
    return -np.sum(y*np.log(p) + (1-y)*np.log(1-p))

def _nll_shared_alpha(theta, q, y, a, a1, a2):
    p0_1, p0_2, alpha = theta
    if not (0.0 <= p0_1 <= 1.0 and 0.0 <= p0_2 <= 1.0 and alpha > 1e-8): return 1e12
    p = np.where(
        np.isclose(a, a1),
        clamp01(logistic(q, p0_1, alpha)),
        clamp01(logistic(q, p0_2, alpha))
    )
    return -np.sum(y*np.log(p) + (1-y)*np.log(1-p))

def _nll_shared_p0(theta, q, y):
    # H0: shared p0 and alpha
    p0, alpha = theta
    if not (0.0 <= p0 <= 1.0 and alpha > 1e-8): return 1e12
    p = clamp01(logistic(q, p0, alpha))
    return -np.sum(y*np.log(p) + (1-y)*np.log(1-p))

# ----------- Kernel smoother (Gaussian) -----------
def gaussian_kernel(u):
    return np.exp(-0.5 * u*u)

def kernel_smoother(q_grid, q_obs, y_obs, h):
    """Nadaraya–Watson estimator with Gaussian kernel."""
    preds = np.empty_like(q_grid, dtype=float)
    for i, q0 in enumerate(q_grid):
        w = gaussian_kernel((q0 - q_obs) / h)
        s = np.sum(w)
        preds[i] = np.sum(w * y_obs) / s if s > 0 else np.nan
    return preds

# ----------- Wilson interval for a binomial proportion -----------
# Brown, Cai, DasGupta (2001) Statistical Science 16(2)
def wilson_interval(s, n, z=1.959963984540054):  # ~ N(0,1), 95%
    if n == 0:
        return (np.nan, np.nan)
    phat = s / n
    denom = 1.0 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * sqrt((phat*(1-phat)/n) + (z*z/(4*n*n))) / denom
    return (center - half, center + half)

# ----------- Calibration metrics -----------
def brier_score(y, p):
    return float(np.mean((y - p)**2))

def ece_score(y, p, m=10):
    edges = np.linspace(0, 1, m+1)
    total = len(y)
    ece = 0.0
    for i in range(m):
        idx = np.where((p >= edges[i]) & (p < edges[i+1]))[0]
        if i == m-1:  # include right edge
            idx = np.where((p >= edges[i]) & (p <= edges[i+1]))[0]
        if len(idx) == 0: 
            continue
        obs = np.mean(y[idx])
        pred = np.mean(p[idx])
        ece += (len(idx)/total) * abs(obs - pred)
    return float(ece)

# ----------- Fitting pipeline -----------
def fit_two_condition(df_trials, p):
    # Identify a1, a2
    a_vals = np.unique(np.round(df_trials['a'].values, 6))
    if len(a_vals) < 2:
        raise RuntimeError("Two-condition fit requested but only one arousal level present.")
    a1, a2 = float(a_vals[0]), float(a_vals[1])
    if 'a1' in p and 'a2' in p:
        # assign closest to configured values
        conf = np.array([p['a1'], p['a2']])
        perm = np.argsort(np.abs(a_vals.reshape(-1,1) - conf).sum(axis=1))
        a1, a2 = float(a_vals[perm[0]]), float(a_vals[perm[1]])

    q = df_trials['q'].values
    y = df_trials['y'].values
    a = df_trials['a'].values

    theta0 = np.array([p.get('p0_guess_a1', 0.5), p.get('p0_guess_a2', 0.52), p.get('alpha_guess', 0.05)])
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, None)]
    res = minimize(_nll_shared_alpha, theta0, args=(q, y, a, a1, a2), method='L-BFGS-B', bounds=bounds)
    p0_1, p0_2, alpha_hat = res.x
    ll_H1 = -res.fun

    # Null: shared p0, shared alpha
    theta0_H0 = np.array([(p0_1 + p0_2) / 2.0, alpha_hat])
    bounds_H0 = [(0.0, 1.0), (1e-6, None)]
    res_H0 = minimize(_nll_shared_p0, theta0_H0, args=(q, y), method='L-BFGS-B', bounds=bounds_H0)
    p0_shared, alpha_shared = res_H0.x
    ll_H0 = -res_H0.fun

    chi2 = 2.0 * (ll_H1 - ll_H0)
    df = 1
    # chi2_1 sf approx (no SciPy.stats): sf(x)=erfc(sqrt(x/2))/2
    pval = 0.5 * (1.0 - erf(sqrt(max(chi2, 0.0)/2.0)))

    # Bands via bootstrap (resample within each condition)
    rng = np.random.default_rng(p['seed'])
    n_boot = int(p['n_bootstrap'])
    qs = np.linspace(p['q_min'], p['q_max'], p['n_points'])

    idx1 = np.where(np.isclose(a, a1))[0]
    idx2 = np.where(np.isclose(a, a2))[0]
    q1, y1 = q[idx1], y[idx1]
    q2, y2 = q[idx2], y[idx2]

    curves1, curves2, dp0 = [], [], []
    for _ in range(n_boot):
        b1 = rng.integers(len(idx1), size=len(idx1))
        b2 = rng.integers(len(idx2), size=len(idx2))
        q_b = np.concatenate([q1[b1], q2[b2]])
        y_b = np.concatenate([y1[b1], y2[b2]])
        a_b = np.concatenate([np.full_like(b1, a1, dtype=float), np.full_like(b2, a2, dtype=float)])
        r = minimize(_nll_shared_alpha, theta0, args=(q_b, y_b, a_b, a1, a2), method='L-BFGS-B', bounds=bounds)
        p0_b1, p0_b2, alpha_b = r.x
        dp0.append(p0_b2 - p0_b1)
        curves1.append(logistic(qs, p0_b1, alpha_b))
        curves2.append(logistic(qs, p0_b2, alpha_b))

    alpha_ci = (100 - p['ci_percent']) / 100.0
    curves1 = np.array(curves1)
    curves2 = np.array(curves2)
    lo1 = np.percentile(curves1, 100*alpha_ci/2, axis=0)
    hi1 = np.percentile(curves1, 100*(1 - alpha_ci/2), axis=0)
    lo2 = np.percentile(curves2, 100*alpha_ci/2, axis=0)
    hi2 = np.percentile(curves2, 100*(1 - alpha_ci/2), axis=0)
    dp0 = np.array(dp0)
    dp0_lo = float(np.percentile(dp0, 100*alpha_ci/2))
    dp0_hi = float(np.percentile(dp0, 100*(1 - alpha_ci/2)))

    # Diagnostics: kernel smoother (per condition)
    kernel_csv = None
    if bool(p.get('kernel_enabled', True)):
        h = float(p.get('kernel_bandwidth', 0.08))
        n_b = int(p.get('kernel_bootstrap', p['n_bootstrap']))
        grid = qs
        Gk1 = kernel_smoother(grid, q1, y1, h)
        Gk2 = kernel_smoother(grid, q2, y2, h)
        # bootstrap CIs
        rng = np.random.default_rng(p['seed'])
        c1, c2 = [], []
        for _ in range(n_b):
            b1 = rng.integers(len(q1), size=len(q1))
            b2 = rng.integers(len(q2), size=len(q2))
            c1.append(kernel_smoother(grid, q1[b1], y1[b1], h))
            c2.append(kernel_smoother(grid, q2[b2], y2[b2], h))
        c1 = np.array(c1); c2 = np.array(c2)
        lo1_k = np.percentile(c1, 100*alpha_ci/2, axis=0)
        hi1_k = np.percentile(c1, 100*(1 - alpha_ci/2), axis=0)
        lo2_k = np.percentile(c2, 100*alpha_ci/2, axis=0)
        hi2_k = np.percentile(c2, 100*(1 - alpha_ci/2), axis=0)
        kernel_csv = pd.DataFrame({
            'q': grid,
            'Gk_central_a1': Gk1, 'Gk_low_a1': lo1_k, 'Gk_high_a1': hi1_k,
            'Gk_central_a2': Gk2, 'Gk_low_a2': lo2_k, 'Gk_high_a2': hi2_k
        })

    # Calibration diagnostics
    calib_bins = max(2, int(p.get('calib_bins', 10)))
    # logistic predictions (fitted H1)
    p_pred = np.where(np.isclose(a, a1),
                      clamp01(logistic(q, p0_1, alpha_hat)),
                      clamp01(logistic(q, p0_2, alpha_hat)))
    edges = np.linspace(0, 1, calib_bins+1)
    rows = []
    for val in [a1, a2]:
        sel = np.isclose(a, val)
        yv = y[sel]; pv = p_pred[sel]
        for i in range(calib_bins):
            left, right = edges[i], edges[i+1]
            idx = (pv >= left) & (pv < right) if i < calib_bins-1 else (pv >= left) & (pv <= right)
            n = int(np.sum(idx))
            if n == 0:
                continue
            s = int(np.sum(yv[idx]))
            obs = s / n
            lo, hi = wilson_interval(s, n)
            rows.append({
                'bin_left': left, 'bin_right': right, 'bin_center': 0.5*(left+right),
                'n_bin': n, 'pred_mean': float(np.mean(pv[idx])), 'obs_rate': obs, 'lo': lo, 'hi': hi,
                'a': float(val)
            })
    calib_csv = pd.DataFrame(rows)

    # Metrics: Brier & ECE
    metrics = []
    for label, mask in [('a1', np.isclose(a, a1)), ('a2', np.isclose(a, a2)), ('all', np.full_like(a, True, dtype=bool))]:
        yv = y[mask]; pv = p_pred[mask]
        metrics.append({'metric': 'Brier', 'a': label, 'value': brier_score(yv, pv)})
        metrics.append({'metric': 'ECE',   'a': label, 'value': ece_score(yv, pv, m=calib_bins)})
    metrics_csv = pd.DataFrame(metrics)

    # Assemble outputs
    qs = np.linspace(p['q_min'], p['q_max'], p['n_points'])
    band_csv = pd.DataFrame({
        'q': qs,
        'G_central_a1': logistic(qs, p0_1, alpha_hat),
        'G_low_a1': lo1, 'G_high_a1': hi1,
        'G_central_a2': logistic(qs, p0_2, alpha_hat),
        'G_low_a2': lo2, 'G_high_a2': hi2
    })
    der_csv = pd.DataFrame({
        'q': qs,
        'dGdq_a1': (1.0/alpha_hat) * np.exp(-(qs - p0_1)/alpha_hat) / (1.0 + np.exp(-(qs - p0_1)/alpha_hat))**2,
        'dGdq_a2': (1.0/alpha_hat) * np.exp(-(qs - p0_2)/alpha_hat) / (1.0 + np.exp(-(qs - p0_2)/alpha_hat))**2
    })
    res_csv = pd.DataFrame([{
        'p0_hat_a1': p0_1, 'p0_hat_a2': p0_2, 'alpha_hat': alpha_hat,
        'Delta_p0': p0_2 - p0_1, 'Delta_p0_lo': dp0_lo, 'Delta_p0_hi': dp0_hi,
        'LRT_chi2': chi2, 'LRT_df': df, 'LRT_pval': pval,
        'Brier_a1': float(metrics_csv.query("metric=='Brier' and a=='a1'")['value'].iloc[0]),
        'Brier_a2': float(metrics_csv.query("metric=='Brier' and a=='a2'")['value'].iloc[0]),
        'Brier_all': float(metrics_csv.query("metric=='Brier' and a=='all'")['value'].iloc[0]),
        'ECE_a1': float(metrics_csv.query("metric=='ECE' and a=='a1'")['value'].iloc[0]),
        'ECE_a2': float(metrics_csv.query("metric=='ECE' and a=='a2'")['value'].iloc[0]),
        'ECE_all': float(metrics_csv.query("metric=='ECE' and a=='all'")['value'].iloc[0])
    }])

    return res_csv, band_csv, der_csv, kernel_csv, calib_csv, metrics_csv

def fit_single_condition(df_trials, p):
    q = df_trials['q'].values
    y = df_trials['y'].values
    theta0 = np.array([p.get('p0_guess_a1', 0.5), p.get('alpha_guess', 0.05)])
    bounds = [(0.0, 1.0), (1e-6, None)]
    res = minimize(_nll_single, theta0, args=(q, y), method='L-BFGS-B', bounds=bounds)
    p0_hat, alpha_hat = res.x

    # Bootstrap band
    rng = np.random.default_rng(p['seed'])
    n_boot = int(p['n_bootstrap'])
    qs = np.linspace(p['q_min'], p['q_max'], p['n_points'])
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(len(q), size=len(q))
        r = minimize(_nll_single, theta0, args=(q[idx], y[idx]), method='L-BFGS-B', bounds=bounds)
        p0_b, alpha_b = r.x
        curves.append(logistic(qs, p0_b, alpha_b))
    curves = np.array(curves)
    alpha_ci = (100 - p['ci_percent']) / 100.0
    lo = np.percentile(curves, 100*alpha_ci/2, axis=0)
    hi = np.percentile(curves, 100*(1 - alpha_ci/2), axis=0)

    band_csv = pd.DataFrame({
        'q': qs,
        'G_central_a1': logistic(qs, p0_hat, alpha_hat),
        'G_low_a1': lo, 'G_high_a1': hi
    })
    der_csv = pd.DataFrame({
        'q': qs,
        'dGdq_a1': (1.0/alpha_hat) * np.exp(-(qs - p0_hat)/alpha_hat) / (1.0 + np.exp(-(qs - p0_hat)/alpha_hat))**2
    })

    # Diagnostics (kernel + calibration) still computed for a1
    kernel_csv = None
    if bool(p.get('kernel_enabled', True)):
        h = float(p.get('kernel_bandwidth', 0.08))
        n_b = int(p.get('kernel_bootstrap', p['n_bootstrap']))
        Gk = kernel_smoother(qs, q, y, h)
        rng = np.random.default_rng(p['seed'])
        B = []
        for _ in range(n_b):
            idx = rng.integers(len(q), size=len(q))
            B.append(kernel_smoother(qs, q[idx], y[idx], h))
        B = np.array(B)
        lo_k = np.percentile(B, 100*alpha_ci/2, axis=0)
        hi_k = np.percentile(B, 100*(1 - alpha_ci/2), axis=0)
        kernel_csv = pd.DataFrame({'q': qs, 'Gk_central_a1': Gk, 'Gk_low_a1': lo_k, 'Gk_high_a1': hi_k})

    p_pred = clamp01(logistic(q, p0_hat, alpha_hat))
    calib_bins = max(2, int(p.get('calib_bins', 10)))
    edges = np.linspace(0, 1, calib_bins+1)
    rows = []
    for i in range(calib_bins):
        left, right = edges[i], edges[i+1]
        idx = (p_pred >= left) & (p_pred < right) if i < calib_bins-1 else (p_pred >= left) & (p_pred <= right)
        n = int(np.sum(idx))
        if n == 0:
            continue
        s = int(np.sum(y[idx]))
        obs = s / n
        lo, hi = wilson_interval(s, n)
        rows.append({'bin_left': left, 'bin_right': right, 'bin_center': 0.5*(left+right),
                    'n_bin': n, 'pred_mean': float(np.mean(p_pred[idx])),
                    'obs_rate': obs, 'lo': lo, 'hi': hi, 'a': 'a1'})
    calib_csv = pd.DataFrame(rows)
    metrics_csv = pd.DataFrame([
        {'metric': 'Brier', 'a': 'a1', 'value': brier_score(y, p_pred)},
        {'metric': 'ECE',   'a': 'a1', 'value': ece_score(y, p_pred, m=calib_bins)}
    ])
    res_csv = pd.DataFrame([{
        'p0_hat_a1': p0_hat, 'alpha_hat': alpha_hat,
        'Delta_p0': np.nan, 'Delta_p0_lo': np.nan, 'Delta_p0_hi': np.nan,
        'LRT_chi2': np.nan, 'LRT_df': 0, 'LRT_pval': np.nan,
        'Brier_a1': float(metrics_csv.query("metric=='Brier' and a=='a1'")['value'].iloc[0]),
        'Brier_a2': np.nan, 'Brier_all': float(metrics_csv.query("metric=='Brier' and a=='a1'")['value'].iloc[0]),
        'ECE_a1': float(metrics_csv.query("metric=='ECE' and a=='a1'")['value'].iloc[0]),
        'ECE_a2': np.nan, 'ECE_all': float(metrics_csv.query("metric=='ECE' and a=='a1'")['value'].iloc[0])
    }])
    return res_csv, band_csv, der_csv, kernel_csv, calib_csv, metrics_csv

def main():
    load_state()
    p = load_params()
    save_state()

    here = os.path.dirname(__file__)
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    trials_path = os.path.join(out_dir, 'logistic_trials.csv')
    if not os.path.exists(trials_path):
        raise FileNotFoundError("Missing logistic_trials.csv. Run simulate_logistic.py first.")
    df_trials = pd.read_csv(trials_path)

    a_levels = np.unique(np.round(df_trials['a'].values, 6))
    two = bool(p.get('use_two_conditions', True)) and (len(a_levels) >= 2)

    if two:
        df_res, df_band, df_der, df_kernel, df_calib, df_metrics = fit_two_condition(df_trials, p)
    else:
        df_res, df_band, df_der, df_kernel, df_calib, df_metrics = fit_single_condition(df_trials, p)

    df_res.to_csv(os.path.join(out_dir, 'fit_logistic_results.csv'), index=False)
    df_band.to_csv(os.path.join(out_dir, 'logistic_band.csv'), index=False)
    df_der.to_csv(os.path.join(out_dir, 'logistic_derivative.csv'), index=False)
    if df_kernel is not None:
        df_kernel.to_csv(os.path.join(out_dir, 'logistic_kernel.csv'), index=False)
    df_calib.to_csv(os.path.join(out_dir, 'logistic_calibration.csv'), index=False)
    df_metrics.to_csv(os.path.join(out_dir, 'logistic_calibration_metrics.csv'), index=False)

    print(f"Saved logistic fit, bands, derivatives, and diagnostics to {out_dir}")

if __name__ == '__main__':
    main()

