#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# One-click driver for the Goldilocks_CRI pipeline
#
# Prerequisite: your Conda env 'seismic_mon' is up-to-date and installed.
# Usage:
#   chmod +x run_all.sh
#   conda activate seismic_mon
#   ./run_all.sh
# -----------------------------------------------------------------------------

echo
echo "⏱ 1) Schedule Psychopy cues"
python stimulus_presentation/psychopy_cue_scheduler.py

echo
echo "🔬 2) Decay simulation & fitting"
python decay/simulate_decay.py
python decay/fit_decay.py

echo
echo "🔬 3) Logistic‐gating simulation & fitting"
python logistic_gate/simulate_logistic.py
python logistic_gate/fit_logistic.py

echo
echo "🔬 4) Quantum Process Tomography simulation & fitting"
python qpt/qpt_simulation.py
python qpt/qpt_fit.py

echo
echo "🎛 5) Synthetic EEG generation"
python synthetic_EEG/make_synthetic_eeg.py

echo
echo "🧹 6) EEG preprocessing & artifact removal"
python preprocessing/artifact_pipeline.py

echo
echo "📦 7) Epoch extraction & feature computation"
python epochs_features/extract_epochs.py
python epochs_features/compute_x_t.py

echo
echo "📊 8) Statistical tests & power analysis"
python statistics/permutation_test.py
python statistics/power_analysis.py

echo
echo "🖼 9) Figure generation"
python figures/make_decay_figure.py
python figures/make_logistic_figure.py
python figures/make_tomography_figure.py
python figures/EEG_flowchart_SIfigure1.py
python figures/make_tierA_seconds_figure.py

# --- CRI Main Figure 1 (TikZ → PDF & PNG) ---
# 1) Compile TikZ to PDF (cropped), writing into figures/output/
latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error \
  -output-directory=figures/output \
  figures/CRI-manuscript_figure_1.tex

# 2) Convert the PDF to a high-res PNG (prefer pdftocairo; fallback to ImageMagick)
if command -v pdftocairo >/dev/null 2>&1; then
  pdftocairo -png -singlefile -r 600 \
    figures/output/CRI-manuscript_figure_1.pdf \
    figures/output/CRI-manuscript_figure_1
elif command -v magick >/dev/null 2>&1; then
  magick -density 600 figures/output/CRI-manuscript_figure_1.pdf \
         -quality 100 figures/output/CRI-manuscript_figure_1.png
else
  echo "WARNING: Neither 'pdftocairo' nor 'magick' found; PNG not generated." 1>&2
fi

echo
echo "✅ All pipelines complete!"

