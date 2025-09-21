#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# One-click driver for the Goldilocks_CRI pipeline
#
# Local full pipeline:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# CI / figures-only:
#   CRI_FIGS_ONLY=1 ./run_all.sh
#
# Optional flags:
#   CRI_SKIP_TEX=1      # skip LaTeX→PDF/PNG even in figures mode
#   CRI_AUTO_INSTALL=1  # pip install -r requirements.txt before running
# -----------------------------------------------------------------------------

cd "$(dirname "$0")"

# Helpers
section () { echo -e "\n$1"; }

# Read flags
FIGS_ONLY="${CRI_FIGS_ONLY:-0}"
SKIP_TEX="${CRI_SKIP_TEX:-0}"

# Ensure output dir exists
mkdir -p figures/output

# Optional: lightweight pip install for CI (avoids conda entirely)
if [[ "${CRI_AUTO_INSTALL:-0}" == "1" ]]; then
  echo "📦 Installing Python deps (pip)…"
  python -m pip install --upgrade pip
  if [[ -f requirements.txt ]]; then
    python -m pip install -r requirements.txt
  else
    python -m pip install numpy matplotlib pyyaml
  fi
fi

if [[ "$FIGS_ONLY" != "1" ]]; then
  # ---------------- Full local pipeline ----------------
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
fi

# ---------------- Figures (always run) ----------------
echo
echo "🖼 9) Figure generation (auto-discovery via generate_figures.py)"
python generate_figures.py

# --- CRI Main Figure 1 (TikZ → PDF & PNG) ---
if [[ "${SKIP_TEX}" != "1" ]]; then
  if command -v latexmk >/dev/null 2>&1; then
    echo
    echo "🧩 Compile TikZ (latexmk)"
    latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error \
      -output-directory=figures/output \
      figures/CRI-manuscript_figure_1.tex

    # Convert PDF → PNG at high resolution (pdftocairo preferred)
    if command -v pdftocairo >/dev/null 2>&1; then
      echo
      echo "🖼 PDF → PNG (pdftocairo)"
      pdftocairo -png -singlefile -r 600 \
        figures/output/CRI-manuscript_figure_1.pdf \
        figures/output/CRI-manuscript_figure_1
    elif command -v magick >/dev/null 2>&1; then
      echo
      echo "🖼 PDF → PNG (ImageMagick)"
      magick -density 600 figures/output/CRI-manuscript_figure_1.pdf \
             -quality 100 figures/output/CRI-manuscript_figure_1.png
    else
      echo "⚠️  Neither 'pdftocairo' nor 'magick' found; PNG not generated." >&2
    fi
  else
    echo "ℹ️  'latexmk' not found; skipping TikZ compile. Set CRI_SKIP_TEX=1 to silence." >&2
  fi
else
  echo "ℹ️  CRI_SKIP_TEX=1 — skipping TikZ compile."
fi

echo
echo "✅ Pipeline complete!"


