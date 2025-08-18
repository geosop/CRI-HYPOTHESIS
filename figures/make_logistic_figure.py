# -*- coding: utf-8 -*-
"""
figures/make_logistic_figure.py  •  CRI v0.1-SIM

Render Box-2(b): logistic “tipping point” with a light-teal 95% bootstrap CI,
solid blue fitted sigmoid, orange observed points, dashed vertical line at p0̂,
and an inset showing dG/dp peaking at p0̂.

Usage:
    python figures/make_logistic_figure.py
"""
import os, yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams.update({
    "font.family":"Arial","font.size":8,
    "axes.linewidth":0.6,"lines.linewidth":1.0,
    "legend.fontsize":6,"xtick.labelsize":7,"ytick.labelsize":7,
    "pdf.fonttype":42,"ps.fonttype":42,  # embed TrueType
})

def load_params(path):
    with open(path,'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['logistic']

def logistic(p, p0, alpha):
    return 1.0/(1.0 + np.exp(-(p - p0)/alpha))

def dlogistic_dp(p, p0, alpha):
    k = 1.0/alpha
    e = np.exp(-k*(p - p0))
    return k*e/(1.0 + e)**2

def main():
    here   = os.path.dirname(__file__)
    repo   = os.path.abspath(os.path.join(here, os.pardir))
    gate   = os.path.join(repo, 'logistic_gate')
    params = load_params(os.path.join(gate, 'default_params.yml'))

    # Inputs
    df_curve = pd.read_csv(f'{gate}/output/logistic_curve.csv')    # p, G_true (not strictly needed)
    df_band  = pd.read_csv(f'{gate}/output/logistic_band.csv')     # p, G_central, G_low, G_high
    df_fit   = pd.read_csv(f'{gate}/output/fit_logistic_results.csv')
    df_obs   = pd.read_csv(f'{gate}/output/logistic_data.csv')

    p_cont  = df_band['p'].values
    G_ctr   = df_band['G_central'].values
    G_low   = df_band['G_low'].values
    G_high  = df_band['G_high'].values
    p_obs   = df_obs['p'].values
    G_obs   = df_obs['G_obs'].values

    p0_hat  = df_fit['p0_hat'].iloc[0]
    alpha_hat = df_fit['alpha_hat'].iloc[0]

    # Colors (solid blue line, light-teal CI, orange circles)
    col_blue = "#1f77b4"
    col_teal = "#6EC5B8"
    col_orng = "#FF8C1A"

    # Figure
    fig, ax = plt.subplots(figsize=(88/25.4, (88/1.5)/25.4))

    # CI band (light-teal, visible)
    ax.fill_between(p_cont, G_low, G_high,
                    facecolor=col_teal, alpha=0.45, edgecolor=col_teal, linewidth=0.6,
                    label=f"{params['ci_percent']}% CI (bootstrap)")

    # Fitted logistic (solid blue line)
    ax.plot(p_cont, G_ctr, color=col_blue, linewidth=1.2, label=r"Fitted logistic $G(p\mid a)$")

    # Observed orange circles with thin black edge
    ax.scatter(p_obs, G_obs, s=18, facecolors=col_orng, edgecolors='black', linewidths=0.4,
               label="Observed")

    # Vertical dashed at p0̂
    ax.axvline(p0_hat, color='grey', linestyle='--', linewidth=0.8, label=rf"$p_0={p0_hat:.2f}$")

    # Axes & legend
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$G(p\mid a)$")
    ax.legend(loc='upper left', frameon=True)

    # Inset: derivative dG/dp; dashed line at p0̂
    ax_ins = inset_axes(ax, width='70%', height='70%',
                        loc='lower left', bbox_to_anchor=(0.62, 0.27, 0.38, 0.38),
                        bbox_transform=ax.transAxes)
    ax_ins.plot(p_cont, dlogistic_dp(p_cont, p0_hat, alpha_hat), color=col_blue, linewidth=0.9)
    ax_ins.axvline(p0_hat, color='grey', linestyle='--', linewidth=0.7)
    ax_ins.set_title(r'$\mathrm{d}G/\mathrm{d}p$', fontsize=8)
    ax_ins.set_xlabel(r"$p$", fontsize=7)
    ax_ins.set_ylabel("Rate", fontsize=7)
    ax_ins.set_xlim(0, 1)
    ax_ins.tick_params(labelsize=6)

    # Save
    out = os.path.join(here, 'output')
    os.makedirs(out, exist_ok=True)
    pdf = os.path.join(out, 'Box2b_logistic_refined.pdf')
    png = os.path.join(out, 'Box2b_logistic_refined.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=params.get('figure_dpi', 1200), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved logistic figure → {pdf} and {png}")

if __name__=='__main__':
    main()
