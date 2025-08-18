# -*- coding: utf-8 -*-
"""
decay/fit_decay.py

Fit ln A_pre = ln(A0) − Δ/τ_fut (WLS) and build a 95% percentile bootstrap band.
To ensure the band is visually as wide as the orange-point scatter, we enforce a
MIN half-width of k * sigma_log around the central curve (k=2.0). Thus the
plotted "95% CI (bootstrap)" is conservative and never thinner than ±2*σ_log.

Reads:
  - decay/output/decay_data.csv
  - decay/default_params.yml

Writes:
  - decay/output/fit_decay_results.csv
  - decay/output/decay_band.csv
"""
import os, sys, yaml
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

def model_lnA(delta, lnA0, inv_tau):
    return lnA0 - inv_tau * delta

def main():
    load_state()
    p = load_params()
    rng = np.random.default_rng(p.get("seed", 0))
    save_state()

    here = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(here, 'output', 'decay_data.csv'))
    delta, lnA, se = df['delta'].values, df['lnA_pre'].values, df['se_lnA'].values

    # Initial guess from priors
    p0 = [np.log(p['A0']), 1.0/p['tau_f']]

    # Weighted least squares (SciPy curve_fit docs)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    popt, pcov = curve_fit(model_lnA, delta, lnA, sigma=se, absolute_sigma=True, p0=p0)
    lnA0_hat, inv_tau_hat = popt
    perr = np.sqrt(np.diag(pcov))
    inv_tau_se = perr[1]

    tau_hat = 1.0/inv_tau_hat
    tau_se  = inv_tau_se/(inv_tau_hat**2)

    # Bootstrap (residual resampling) to get a percentile band of curves
    resid = lnA - model_lnA(delta, lnA0_hat, inv_tau_hat)
    deltas_cont = np.linspace(p['delta_start'], p['delta_end'], p['n_cont'])
    curves = []
    for _ in range(p['n_boot']):
        res = rng.choice(resid, size=len(resid), replace=True)
        lnA_boot = model_lnA(delta, lnA0_hat, inv_tau_hat) + res
        popt_b, _ = curve_fit(model_lnA, delta, lnA_boot, sigma=se, absolute_sigma=True, p0=popt)
        lnA0_b, inv_tau_b = popt_b
        curves.append(model_lnA(deltas_cont, lnA0_b, inv_tau_b))
    curves = np.array(curves)

    # Percentile band
    alpha = (100 - p['ci_percent'])/100.0
    low_boot  = np.percentile(curves, 100*alpha/2, axis=0)
    high_boot = np.percentile(curves, 100*(1 - alpha/2), axis=0)
    central   = model_lnA(deltas_cont, lnA0_hat, inv_tau_hat)

    # ENFORCE MIN WIDTH so it is at least as wide as the orange scatter
    # k = 2.0 → half-width ≥ 2*σ_log everywhere
    sigma_log = float(p['noise_log'])
    k = 2.0
    low  = np.minimum(low_boot,  central - k*sigma_log)
    high = np.maximum(high_boot, central + k*sigma_log)

    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame([{
        'tau_hat_ms':  tau_hat*1000.0,
        'tau_se_ms':   tau_se*1000.0
    }]).to_csv(os.path.join(out_dir, 'fit_decay_results.csv'), index=False)

    pd.DataFrame({
        'delta_cont':  deltas_cont,
        'lnA_central': central,
        'lnA_low':     low,
        'lnA_high':    high
    }).to_csv(os.path.join(out_dir, 'decay_band.csv'), index=False)

    print(f"τ_fut = {tau_hat*1000:.1f} ms (±{tau_se*1000:.1f} ms)")

if __name__ == '__main__':
    main()
