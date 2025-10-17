# -*- coding: utf-8 -*-
"""
figures/make_logistic_diagnostics.py  •  CRI v0.2-SIM (SI diagnostics)

Panel (top): model-free kernel smoother (faint) overlaid on fitted logistic curves.
Panel (bottom): calibration curves (observed vs predicted) with Wilson intervals,
plus Brier and ECE annotations per condition.
"""
import os, yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def load_gate_params(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['logistic']

def main():
    here   = os.path.dirname(__file__)
    repo   = os.path.abspath(os.path.join(here, os.pardir))
    gate   = os.path.join(repo, 'logistic_gate')
    params = load_gate_params(os.path.join(gate, 'default_params.yml'))

    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    band   = pd.read_csv(os.path.join(gate, 'output', 'logistic_band.csv'))
    kern   = pd.read_csv(os.path.join(gate, 'output', 'logistic_kernel.csv'))
    calib  = pd.read_csv(os.path.join(gate, 'output', 'logistic_calibration.csv'))
    metr   = pd.read_csv(os.path.join(gate, 'output', 'logistic_calibration_metrics.csv'))

    two = 'G_central_a2' in band.columns

    # Colors
    col_teal1 = "#6EC5B8"; col_blue1 = "#1f77b4"
    col_teal2 = "#9ADBD2"; col_purp2 = "#7f3c8d"
    col_kern  = "#888888"  # faint gray for kernel

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(88/25.4, 110/25.4), gridspec_kw={'hspace': 0.35})

    # --- Top: kernel smoother vs logistic fits ---
    ax1.fill_between(band['q'], band['G_low_a1'], band['G_high_a1'], facecolor=col_teal1, alpha=0.25, edgecolor='none')
    ax1.plot(band['q'], band['G_central_a1'], color=col_blue1, lw=1.2, label=r"Fitted $G(q\mid a_1)$")
    ax1.plot(kern['q'], kern['Gk_central_a1'], color=col_kern, lw=1.0, alpha=0.9, label="Kernel smoother (a1)")
    ax1.fill_between(kern['q'], kern['Gk_low_a1'], kern['Gk_high_a1'], color=col_kern, alpha=0.12, edgecolor='none')

    if two:
        ax1.fill_between(band['q'], band['G_low_a2'], band['G_high_a2'], facecolor=col_teal2, alpha=0.25, edgecolor='none')
        ax1.plot(band['q'], band['G_central_a2'], color=col_purp2, lw=1.2, label=r"Fitted $G(q\mid a_2)$")
        ax1.plot(kern['q'], kern['Gk_central_a2'], color=col_kern, lw=1.0, alpha=0.9, linestyle='--', label="Kernel smoother (a2)")
        ax1.fill_between(kern['q'], kern['Gk_low_a2'], kern['Gk_high_a2'], color=col_kern, alpha=0.12, edgecolor='none')

    ax1.set_xlim(0,1); ax1.set_ylim(0,1)
    ax1.set_xlabel(r"$q$"); ax1.set_ylabel(r"$G(q)$")
    ax1.legend(loc='lower right', frameon=True)

    # --- Bottom: calibration curve with Wilson CIs ---
    for a_val, df_a in calib.groupby('a'):
        ax2.errorbar(df_a['pred_mean'], df_a['obs_rate'],
                     yerr=[df_a['obs_rate']-df_a['lo'], df_a['hi']-df_a['obs_rate']],
                     fmt='o', ms=3.5, capsize=1.5, label=f"Calibration (a={a_val})")
    ax2.plot([0,1],[0,1], ls='--', color='0.5', lw=1.0, label="Perfect calibration")

    # Annotate Brier/ECE
    def metric_val(name, a):
        row = metr[(metr['metric']==name) & (metr['a']==a)]
        return float(row['value'].iloc[0]) if not row.empty else np.nan
    if two:
        text = (f"a1: Brier={metric_val('Brier','a1'):.3f}, ECE={metric_val('ECE','a1'):.3f}    "
                f"a2: Brier={metric_val('Brier','a2'):.3f}, ECE={metric_val('ECE','a2'):.3f}")
    else:
        text = f"a1: Brier={metric_val('Brier','a1'):.3f}, ECE={metric_val('ECE','a1'):.3f}"
    ax2.text(0.02, 0.95, text, transform=ax2.transAxes, va='top', ha='left', fontsize=7)

    ax2.set_xlim(0,1); ax2.set_ylim(0,1)
    ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Observed rate")
    ax2.legend(loc='upper left', frameon=True)

    pdf = os.path.join(out_dir, 'Box2b_logistic_diagnostics.pdf')
    png = os.path.join(out_dir, 'Box2b_logistic_diagnostics.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=int(params.get('figure_dpi', 1200)), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved SI diagnostics figure → {pdf} and {png}")

if __name__ == '__main__':
    main()
