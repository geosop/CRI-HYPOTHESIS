# -*- coding: utf-8 -*-
"""
logistic_gate/simulate_logistic.py

Simulate noisy observations of the logistic “tipping-point” curve:
    G(p|a) = 1 / (1 + exp(-(p - p0) / alpha))

Writes:
  - logistic_gate/output/logistic_curve.csv    (dense noiseless curve)
  - logistic_gate/output/logistic_data.csv     (noisy observed points)

Usage:
    python logistic_gate/simulate_logistic.py
"""
import os, sys, yaml
import numpy as np
import pandas as pd

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

def main():
    load_state()
    p = load_params()
    rng = np.random.default_rng(p['seed'])
    save_state()

    # Dense grid and noiseless curve
    ps = np.linspace(p['p_min'], p['p_max'], p['n_points'])
    G_true = 1.0 / (1.0 + np.exp(-(ps - p['p0']) / p['alpha']))

    # Random subset for “observed” points with additive noise
    n_obs = int(p['n_obs'])
    if n_obs < 20:
        print(f"[warn] n_obs={n_obs} < 20 — manuscript suggests ≥20 for CI reliability.")
    idx = rng.choice(len(ps), size=n_obs, replace=False)
    G_obs = np.clip(G_true[idx] + rng.normal(0, p['noise_std'], size=n_obs), 0, 1)

    out = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({'p': ps, 'G_true': G_true}).to_csv(f'{out}/logistic_curve.csv', index=False)
    pd.DataFrame({'p': ps[idx], 'G_obs': G_obs}).to_csv(f'{out}/logistic_data.csv', index=False)
    print(f"Saved logistic data → {out}")

if __name__ == '__main__':
    main()
