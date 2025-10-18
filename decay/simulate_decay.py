# -*- coding: utf-8 -*-
"""
decay/simulate_decay.py  •  CRI v0.3-SIM

Generates:
  - decay/output/decay_data.csv        (delta, lnA_pre, se_lnA)  <-- used for WLS/OLS/Tobit
  - decay/output/decay_curve.csv       (delta_cont, lnA_pre_cont) <-- used to draw the line
  - decay/output/decay_data_raw.csv    (delta, lnA_pre_raw)      <-- replicates (SI)
"""
import os, yaml
import numpy as np
import pandas as pd

def _load_params():
    here = os.path.dirname(__file__)
    path = os.path.join(here, 'default_params.yml')
    with open(path, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f)
    p = y['decay'] if isinstance(y, dict) and 'decay' in y else y
    return {
        'seed':        int(p.get('seed', 52)),
        'A0':          float(p.get('A0', 1.0)),
        'tau_f':       float(p.get('tau_f', 0.02)),
        'noise_log':   float(p.get('noise_log', 0.10)),
        'delta_start': float(p.get('delta_start', 0.0)),
        'delta_end':   float(p.get('delta_end', 0.02)),
        'delta_step':  float(p.get('delta_step', 0.005)),
        'n_cont':      int(p.get('n_cont', 300)),
        # NEW: number of replicate trials per discrete delay (for SEs)
        'n_rep':       int(p.get('n_rep', 40)),
    }

def main():
    p = _load_params()
    rng = np.random.default_rng(p['seed'])

    here = os.path.dirname(__file__)
    outd = os.path.join(here, 'output')
    os.makedirs(outd, exist_ok=True)

    # Discrete delays for orange points (e.g., 0, 5, 10, 15, 20 ms)
    deltas = np.arange(p['delta_start'], p['delta_end'] + 1e-12, p['delta_step'])
    # True mean in log domain
    lnA_true = np.log(p['A0']) - deltas / p['tau_f']

    # Simulate replicates in log-domain (homoscedastic log-noise)
    raw_rows = []
    for d, mu in zip(deltas, lnA_true):
        y = mu + rng.normal(0.0, p['noise_log'], size=p['n_rep'])
        for v in y:
            raw_rows.append({'delta': float(d), 'lnA_pre_raw': float(v)})
    pd.DataFrame(raw_rows).to_csv(os.path.join(outd, 'decay_data_raw.csv'), index=False)

    # Aggregate to mean ± SE per delay (this is what WLS will use)
    df_raw = pd.DataFrame(raw_rows)
    agg = df_raw.groupby('delta', as_index=False).agg(
        lnA_pre=('lnA_pre_raw', 'mean'),
        sd=('lnA_pre_raw', 'std'),
        n=('lnA_pre_raw', 'size')
    )
    # Standard error of the mean in log-domain
    agg['se_lnA'] = agg['sd'] / np.sqrt(agg['n'])
    agg[['delta', 'lnA_pre', 'se_lnA']].to_csv(os.path.join(outd, 'decay_data.csv'), index=False)

    # Dense noiseless curve for drawing the central line/band background
    x_cont = np.linspace(p['delta_start'], p['delta_end'], p['n_cont'])
    lnA_cont = np.log(p['A0']) - x_cont / p['tau_f']
    pd.DataFrame({'delta_cont': x_cont, 'lnA_pre_cont': lnA_cont}).to_csv(
        os.path.join(outd, 'decay_curve.csv'), index=False
    )

    print(f"Saved discrete data with SEs → {os.path.join(outd, 'decay_data.csv')}")
    print(f"Saved dense curve → {os.path.join(outd, 'decay_curve.csv')}")

if __name__ == '__main__':
    main()
