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
#
# Reproducibility envs (with safe defaults for local runs):
#   CRI_SEED           - global seed for all scripts (default 52)
#   PYTHONHASHSEED     - python hashing determinism (default 0)
#   *BLAS OMP threads  - set to 1 for deterministic numerics
# -----------------------------------------------------------------------------

cd "$(dirname "$0")"

# --------- Reproducibility defaults (overridden by env if already set) ---------
: "${CRI_SEED:=52}";            export CRI_SEED
: "${PYTHONHASHSEED:=0}";       export PYTHONHASHSEED
: "${OMP_NUM_THREADS:=1}";      export OMP_NUM_THREADS
: "${OPENBLAS_NUM_THREADS:=1}"; export OPENBLAS_NUM_THREADS
: "${MKL_NUM_THREADS:=1}";      export MKL_NUM_THREADS
: "${NUMEXPR_NUM_THREADS:=1}";  export NUMEXPR_NUM_THREADS
: "${MPLBACKEND:=Agg}";         export MPLBACKEND  # headless matplotlib

echo "=== CRI_Goldilocks pipeline ==="
echo "CRI_SEED=${CRI_SEED} | PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "Threads: OMP=${OMP_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, NUMEXPR=${NUMEXPR_NUM_THREADS}"
echo "MPLBACKEND=${MPLBACKEND}"
echo

# Helpers
section () { echo -e "\n$1"; }
run_py () {
  local script="$1"
  echo "▶ python ${script}"
  python "${script}"
}

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
    # minimal local fallback
    python -m pip install numpy pandas scipy matplotlib PyYAML
  fi
fi

if [[ "$FIGS_ONLY" != "1" ]]; then
  # ---------------- Full local pipeline ----------------
  echo
  echo "⏱ 1) Schedule Psychopy cues"
  run_py stimulus_presentation/psychopy_cue_scheduler.py

  echo
  echo "🔬 2) Decay simulation & fitting"
  run_py decay/simulate_decay.py
  run_py decay/fit_decay.py

  echo
  echo "🔬 2b) Tier-B tempered mixtures"
  run_py tierB_tempered/simulate_and_fit.py

  echo
  echo "🔬 3) Logistic‐gating simulation & fitting"
  run_py logistic_gate/simulate_logistic.py
  run_py logistic_gate/fit_logistic.py

  echo
  echo "🔬 4) Quantum Process Tomography simulation & fitting"
  run_py qpt/qpt_simulation.py
  run_py qpt/qpt_fit.py

  echo
  echo "🎛 5) Synthetic EEG generation"
  run_py synthetic_EEG/make_synthetic_eeg.py

  echo
  echo "🧹 6) EEG preprocessing & artifact removal"
  run_py preprocessing/artifact_pipeline.py

  echo
  echo "📦 7) Epoch extraction & feature computation"
  run_py epochs_features/extract_epochs.py
  run_py epochs_features/compute_x_t.py

  echo
  echo "📊 8) Statistical tests & power analysis"
  run_py statistics/permutation_test.py
  run_py statistics/power_analysis.py
fi

# ---------------- Figures (always run) ----------------
echo
echo "🖼 9) Figure generation (auto-discovery via generate_figures.py)"
run_py generate_figures.py

# --- SI EEG Flowchart (Python) ---
if [[ -f figures/EEG_flowchart_SIfigure1.py ]]; then
  echo
  echo "🧠 9b) SI Figure 1 — EEG flowchart (Python)"
  python figures/EEG_flowchart_SIfigure1.py || { echo "❌ EEG flowchart script failed." >&2; exit 1; }
else
  echo "⚠️  figures/EEG_flowchart_SIfigure1.py not found; skipping." >&2
fi

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
