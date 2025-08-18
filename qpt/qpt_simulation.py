# -*- coding: utf-8 -*-
"""
qpt/qpt_simulation.py

Generates synthetic data for Box-2(c):
  - Left: P(t) = exp(-(γ_f + γ_b) t / 2) for γ_b in {0.2, 0.8}, γ_f = 1.0
  - Right: For each λ_env in a grid, simulate R_samples ≈ λ_env + noise
           (n_trials_per_lambda per λ) → mean & 95% CI per λ

Writes qpt/output/qpt_sim_data.npz with:
  t, gamma_b_vals, pops(2D float array), lambda_env, R_mean, R_ci_low, R_ci_high
"""
import os, sys, yaml
import numpy as np

# ensure utilities on path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state(): pass
    def save_state(): pass

DEFAULTS = {
    'gamma_f': 1.0,
    'gamma_b_vals': [0.2, 0.8],
    't_max': 10.0,
    'n_t': 200,
    'lambda_env_min': 0.0,
    'lambda_env_max': 1.0,
    'n_lambda_env': 11,
    'noise_R': 0.05,
    'n_trials_per_lambda': 200,
    'ci_percent': 95,
    'seed': 52,
}

def load_params():
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'default_params.yml')
    with open(cfg_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        cfg = yaml.safe_load(f) or {}
    user = (cfg.get('qpt') or {})
    return {**DEFAULTS, **user}

def simulate_populations(t, gamma_f, gamma_b_vals):
    # return a 2D float array: rows correspond to γ_b
    curves = [np.exp(-(gamma_f + gb) * t / 2.0) for gb in gamma_b_vals]
    return np.vstack(curves).astype(float)

def main():
    load_state()
    p = load_params()
    rng = np.random.default_rng(p['seed'])
    save_state()

    # Left panel data
    t = np.linspace(0.0, float(p['t_max']), int(p['n_t']))
    pops = simulate_populations(t, float(p['gamma_f']), [float(g) for g in p['gamma_b_vals']])

    # Right panel data
    lambda_env = np.linspace(float(p['lambda_env_min']),
                             float(p['lambda_env_max']),
                             int(p['n_lambda_env']))
    n_trials = int(p['n_trials_per_lambda'])
    noise = float(p['noise_R'])
    alpha = (100 - float(p['ci_percent'])) / 100.0

    R_mean, R_low, R_high = [], [], []
    for lam in lambda_env:
        samples = lam + rng.normal(0.0, noise, size=n_trials)
        samples = np.clip(samples, 0, None)
        R_mean.append(np.mean(samples))
        R_low.append(np.percentile(samples, 100*alpha/2))
        R_high.append(np.percentile(samples, 100*(1 - alpha/2)))

    out = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'qpt_sim_data.npz'),
             t=t.astype(float),
             gamma_b_vals=np.array(p['gamma_b_vals'], dtype=float),
             pops=pops,  # 2D float array (n_gb × n_t)
             lambda_env=lambda_env.astype(float),
             R_mean=np.array(R_mean, dtype=float),
             R_ci_low=np.array(R_low, dtype=float),
             R_ci_high=np.array(R_high, dtype=float))
    print("Saved qpt_sim_data.npz  (pops shape:", pops.shape, ")")

if __name__ == '__main__':
    main()
