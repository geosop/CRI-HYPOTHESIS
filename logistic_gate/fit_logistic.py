#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logistic_gate/fit_logistic.py

Fit logistic to (p, G_obs) and build a 95% bootstrap band for G(p|a).

Reads:
  - logistic_gate/output/logistic_data.csv
  - logistic_gate/default_params.yml

Writes:
  - logistic_gate/output/fit_logistic_results.csv
  - logistic_gate/output/logistic_band.csv  (p, G_central, G_low, G_high)
"""
import os, sys, yaml
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state(): pass
    def save_state(): pass

def load_params():
    here = os.path.dirname(__file__)
    path = os.path.join(here, 'default_params.yml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['logistic']

def logistic_fn(p, p0, alpha):
    return 1.0 / (1.0 + np.exp(-(p - p0) / alpha))

def fit_logistic(p, G, p0_guess, alpha_guess):
    popt, pcov = curve_fit(
        logistic_fn, p, G,
        p0=[p0_guess, alpha_guess],
        bounds=([0.0, 1e-6], [1.0, np.inf]),
        maxfev=10000
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def bootstrap_band(p_dense, p_obs, G_obs, p0_g, alpha_g, n_boot, ci, seed):
    rng = np.random.default_rng(seed)
    n = len(p_obs)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(n, size=n)  # bootstrap resample indices
        (p0_b, a_b), _ = fit_logistic(p_obs[idx], G_obs[idx], p0_g, alpha_g)
        curves.append(logistic_fn(p_dense, p0_b, a_b))
    curves = np.array(curves)
    alpha = (100 - ci) / 100.0
    low  = np.percentile(curves, 100*alpha/2, axis=0)
    high = np.percentile(curves, 100*(1 - alpha/2), axis=0)
    return low, high

def main():
    load_state()
    p = load_params()
    rng = np.random.default_rng(p['seed'])
    save_state()

    here = os.path.dirname(__file__)
    df = pd.read_csv(f'{here}/output/logistic_data.csv')
    p_obs, G_obs = df['p'].values, df['G_obs'].values

    # Central fit
    (p0_hat, alpha_hat), (p0_se, alpha_se) = fit_logistic(
        p_obs, G_obs, p['p0_guess'], p['alpha_guess']
    )

    # Dense grid for band
    ps = np.linspace(p['p_min'], p['p_max'], p['n_points'])
    G_central = logistic_fn(ps, p0_hat, alpha_hat)

    # Bootstrap CI band over the curve
    G_low, G_high = bootstrap_band(
        ps, p_obs, G_obs, p['p0_guess'], p['alpha_guess'],
        p['n_bootstrap'], p['ci_percent'], p['seed']
    )

    out_dir = f'{here}/output'
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{
        'p0_hat': p0_hat, 'alpha_hat': alpha_hat,
        'p0_se': p0_se, 'alpha_se': alpha_se
    }]).to_csv(f'{out_dir}/fit_logistic_results.csv', index=False)

    pd.DataFrame({
        'p': ps,
        'G_central': G_central,
        'G_low': G_low,
        'G_high': G_high
    }).to_csv(f'{out_dir}/logistic_band.csv', index=False)

    print(f"Saved fit + CI band to {out_dir}")

if __name__ == '__main__':
    main()
