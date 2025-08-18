# -*- coding: utf-8 -*-
"""
decay/simulate_decay.py

Generate synthetic exponential-decay data (Box-2(a)):
    ln A_pre = ln(A0) − Δ / τ_fut

Writes:
  - decay/output/decay_data.csv    (discrete noisy points)
  - decay/output/decay_curve.csv   (noiseless continuous curve)

Usage:
    python decay/simulate_decay.py
"""
import os, sys, yaml
import numpy as np
import pandas as pd

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
try:
    from utilities.seed_manager import load_state, save_state
except Exception:
    def load_state(): pass
    def save_state(): pass

def load_params():
    cfg_path = os.path.join(os.path.dirname(__file__), "default_params.yml")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['decay']

def main():
    load_state()
    p = load_params()
    rng = np.random.default_rng(p.get("seed", 0))
    save_state()

    # {0,5,10,15,20} ms in seconds
    deltas = np.arange(p['delta_start'],
                       p['delta_end'] + p['delta_step']*0.5,
                       p['delta_step'])

    # Noiseless model
    A_pre_true = p['A0'] * np.exp(-deltas / p['tau_f'])
    lnA_true   = np.log(A_pre_true)

    # Add log-domain noise
    sigma_log = float(p['noise_log'])
    lnA_noisy = lnA_true + rng.normal(0.0, sigma_log, size=len(deltas))

    # Continuous curve (noiseless)
    deltas_cont  = np.linspace(p['delta_start'], p['delta_end'], p['n_cont'])
    lnA_cont     = np.log(p['A0']) - deltas_cont/p['tau_f']

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({
        'delta':    deltas,
        'lnA_pre':  lnA_noisy,
        'lnA_true': lnA_true,
        'se_lnA':   sigma_log
    }).to_csv(os.path.join(out_dir, 'decay_data.csv'), index=False)

    pd.DataFrame({
        'delta_cont':   deltas_cont,
        'lnA_pre_cont': lnA_cont
    }).to_csv(os.path.join(out_dir, 'decay_curve.csv'), index=False)

    print(f"Saved decay data to {out_dir}")

if __name__ == '__main__':
    main()
