# -*- coding: utf-8 -*-
"""
logistic_gate/simulate_logistic.py

Simulate trial-wise Bernoulli activations for the logistic “tipping point”:
    G(q|a) = 1 / (1 + exp(-(q - p0(a)) / alpha))

Writes:
  - logistic_gate/output/logistic_curve.csv
      q, G_a1, [G_a2]
  - logistic_gate/output/logistic_trials.csv
      q, y, a   (trial-wise Bernoulli outcomes)
  - logistic_gate/output/logistic_bins.csv
      q_bin_center, rate_mean, n_bin, a  (bin means for visualization ONLY)

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
        p = yaml.safe_load(f)['logistic']
    # Back-compat if older keys exist
    if 'q_min' not in p and 'p_min' in p: p['q_min'] = p['p_min']
    if 'q_max' not in p and 'p_max' in p: p['q_max'] = p['p_max']
    if 'n_points' not in p: p['n_points'] = 400
    if 'n_bins' not in p: p['n_bins'] = 15
    return p

def logistic(q, p0, alpha):
    return 1.0 / (1.0 + np.exp(-(q - p0) / alpha))

def _sample_q(rng, n, q_min, q_max, mode):
    if str(mode).lower() == "uniform":
        return np.linspace(q_min, q_max, n)
    # random iid in [q_min, q_max]
    return rng.uniform(q_min, q_max, size=n)

def main():
    load_state()
    p = load_params()
    rng = np.random.default_rng(p['seed'])
    save_state()

    out = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out, exist_ok=True)

    q_grid = np.linspace(p['q_min'], p['q_max'], p['n_points'])

    # Ground-truth curves
    G_a1 = logistic(q_grid, p['p0_a1'], p['alpha'])
    data = {'q': q_grid, 'G_a1': G_a1}

    trials = []

    # Condition a1
    q_a1 = _sample_q(rng, int(p['n_trials_a1']), p['q_min'], p['q_max'], p['q_sampling'])
    prob_a1 = logistic(q_a1, p['p0_a1'], p['alpha'])
    y_a1 = rng.binomial(1, prob_a1)
    trials.append(pd.DataFrame({'q': q_a1, 'y': y_a1, 'a': np.full_like(q_a1, fill_value=p['a1'], dtype=float)}))

    # Optional condition a2
    use_two = bool(p.get('use_two_conditions', True))
    if use_two:
        G_a2 = logistic(q_grid, p['p0_a2'], p['alpha'])
        data['G_a2'] = G_a2

        q_a2 = _sample_q(rng, int(p['n_trials_a2']), p['q_min'], p['q_max'], p['q_sampling'])
        prob_a2 = logistic(q_a2, p['p0_a2'], p['alpha'])
        y_a2 = rng.binomial(1, prob_a2)
        trials.append(pd.DataFrame({'q': q_a2, 'y': y_a2, 'a': np.full_like(q_a2, fill_value=p['a2'], dtype=float)}))

    # Write dense noiseless curves
    pd.DataFrame(data).to_csv(os.path.join(out, 'logistic_curve.csv'), index=False)

    # Write trials
    df_trials = pd.concat(trials, ignore_index=True)
    df_trials = df_trials.sort_values('q').reset_index(drop=True)
    pd.DataFrame(df_trials).to_csv(os.path.join(out, 'logistic_trials.csv'), index=False)

    # Bin means for visualization only
    bins = int(p['n_bins'])
    edges = np.linspace(p['q_min'], p['q_max'], bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for a_val, df_a in df_trials.groupby('a'):
        counts, _ = np.histogram(df_a['q'].values, bins=edges)
        sums, _   = np.histogram(df_a['q'].values, bins=edges, weights=df_a['y'].values.astype(float))
        with np.errstate(invalid='ignore', divide='ignore'):
            means = np.where(counts > 0, sums / counts, np.nan)
        for c, m, n in zip(bin_centers, means, counts):
            if n > 0:
                rows.append({'q_bin_center': c, 'rate_mean': m, 'n_bin': int(n), 'a': float(a_val)})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'logistic_bins.csv'), index=False)

    print(f"Saved logistic_curve.csv, logistic_trials.csv, logistic_bins.csv → {out}")

if __name__ == '__main__':
    main()
