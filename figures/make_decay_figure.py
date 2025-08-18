# -*- coding: utf-8 -*-
"""
figures/make_decay_figure.py  •  CRI v0.1-SIM

Edits per request:
- Legend kept off the detection bound (moved up, same size)
- Inset moved vertically higher
- No orange connecting line in inset (black line + orange dots)
- Wide, clearly visible 95% CI (matches orange scatter)
- x-axis fixed to 0–20 ms

Matplotlib references:
- Legends: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
- Inset axes: https://matplotlib.org/stable/api/toolkits/axes_grid1_api.html#insetlocator
"""
import os, yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms as mtransforms

mpl.rcParams.update({
    "font.family":      "Arial",
    "font.size":        8,
    "axes.linewidth":   0.6,
    "lines.linewidth":  0.9,
    "legend.fontsize":  5.5,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

def load_params(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['decay']

def main():
    here         = os.path.dirname(__file__)
    repo_root    = os.path.abspath(os.path.join(here, os.pardir))
    decay_folder = os.path.join(repo_root, 'decay')
    out_folder   = os.path.join(here, 'output')

    p = load_params(os.path.join(decay_folder, 'default_params.yml'))

    pts  = pd.read_csv(os.path.join(decay_folder, 'output', 'decay_data.csv'))
    band = pd.read_csv(os.path.join(decay_folder, 'output', 'decay_band.csv'))

    # to ms
    pts['delta_ms']  = pts['delta'] * 1000.0
    band['delta_ms'] = band['delta_cont'] * 1000.0

    A_pts     = np.exp(pts['lnA_pre'].values)
    A_central = np.exp(band['lnA_central'].values)

    fig, ax = plt.subplots(figsize=(88/25.4, 58/25.4))

    # Wide & visible 95% CI
    ax.fill_between(band['delta_ms'], band['lnA_low'], band['lnA_high'],
                    facecolor='#5B8FD9', alpha=0.58, zorder=1,
                    edgecolor='#3E6FB8', linewidth=0.9,
                    label=f"{p['ci_percent']}% CI (bootstrap)")

    # Central fitted curve (black)
    ax.plot(band['delta_ms'], band['lnA_central'],
            color='black', linewidth=1.3, zorder=2, label=r"$\ln A_{\mathrm{pre}}(\tau_f)$")

    # Orange dots with error bars (NO connecting orange line in main)
    ax.errorbar(pts['delta_ms'], pts['lnA_pre'], yerr=pts['se_lnA'],
                fmt='o', color='#FF8C1A', markersize=3.8, elinewidth=0.75,
                capsize=1.8, zorder=3, label="Sampled delays")

    # Detection bound
    eps = float(p.get('epsilon_detection', 0.01))
    ax.axhline(np.log(eps), linestyle='--', color='#D62728', linewidth=1.0,
               label=r"Detection bound: $\ln \epsilon$")

    # Axes & limits
    ax.set_xlabel(r"$\tau_f$ (ms)")
    ax.set_ylabel(r"$\ln A_{\mathrm{pre}}(\tau_f)$")
    ax.set_xlim(-0.5, 20.5)

    # Legend: move upward so it never covers the red dashed bound
    # (anchored ~30% up from bottom-left corner)
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.30), frameon=True)

   
    # Slope/τ annotation: move VERTICALLY ONLY (upwards) and make smaller.
    # Place it just BELOW the blue 95% CI band, at the band’s lowest edge minus a small margin.
    try:
       fit = pd.read_csv(os.path.join(decay_folder, 'output', 'fit_decay_results.csv')).iloc[0]
       tau_ms = fit['tau_hat_ms']
       text = (r"$\mathrm{slope}\approx -1/\tau_{\mathrm{fut}}$"
              + "\n" + rf"$\tau_{{\mathrm{{fut}}}}={tau_ms:.1f}\,\mathrm{{ms}}$")
    except Exception:
       text = r"$\mathrm{slope}\approx -1/\tau_{\mathrm{fut}}$"

    # Compute y just below the band (in data coords); keep x the same (axes fraction).
    y_ann = float(band['lnA_low'].min()) - 0.10  # 0.10 log-units margin
    x_ann_axes = 0.58                             # unchanged x position

    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(x_ann_axes, y_ann, text, transform=trans, fontsize=6.0, va="top",
          bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, edgecolor="none"))

    


    # Inset moved higher; leave only black connecting line + orange dots
    ax_ins = inset_axes(ax, width='58%', height='52%',
                        loc='lower right',
                        bbox_to_anchor=(0.40, 0.18, 0.58, 0.52),  # y shifted from 0.06 → 0.18
                        bbox_transform=ax.transAxes)
    ax_ins.plot(band['delta_ms'], A_central, color='black', linewidth=0.9, zorder=2)  # black line
    # no orange connecting line here by request
    ax_ins.scatter(pts['delta_ms'], A_pts, s=12, color='#FF8C1A', zorder=3)           # orange dots
    ax_ins.axhline(eps, linestyle='--', color='#D62728', linewidth=0.7)

    ax_ins.set_title(r"Raw $A_{\mathrm{pre}}(\tau_f)$", fontsize=6.5, pad=1.5)
    ax_ins.set_xlabel(r"$\tau_f$ (ms)", fontsize=6.5)
    ax_ins.set_ylabel(r"$A_{\mathrm{pre}}$", fontsize=6.5)
    ax_ins.tick_params(labelsize=6)
    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_ins.set_xlim(-0.5, 20.5)

    fig.tight_layout()

    os.makedirs(out_folder, exist_ok=True)
    out_pdf = os.path.join(out_folder, 'Box2a_decay_refined.pdf')
    out_png = os.path.join(out_folder, 'Box2a_decay_refined.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=p.get('figure_dpi', 1200), bbox_inches='tight')
    plt.close(fig)
    print("Saved Box-2(a) to", out_pdf, "and", out_png)

if __name__ == "__main__":
    main()
