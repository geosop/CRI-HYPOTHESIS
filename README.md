# CRI_Goldilocks

**Reproducibility Pipeline for the "Conscious Retroactive Intervention (CRI)" PERSPECTIVE Manuscript in preparation**

---

## Overview

This repository supports the Perspective manuscript submission:
> **"Conscious Retroactive Intervention: A Time-Reversed Quantum Framework for Predictive Cognition"**

It provides all code, synthetic data, simulations, figures, and numerical pipelines to transparently reproduce and audit the CRI framework and its supplementary analyses.
CRI (Goldilocks-CRI) proposes that neural systems can—under sharply tuned, “just-right” physiological conditions—be influenced by probabilistic information about future outcomes. This model combines predictive coding, quantum process tomography, and novel “Goldilocks” retrocausal gating, producing specific, testable signatures in behavior, EEG, and simulated neural dynamics.

>⚠️ Note: This repository contains code for computational modeling and simulation only. It does not provide pipelines or scripts for empirical EEG or behavioral data analysis.

---

## Repository Structure

```text
Goldilocks_CRI/
├── synthetic_EEG/ # Synthetic EEG generation and outputs
├── preprocessing/ # EEG artifact detection and cleaning
├── decay/ # Retrocausal decay simulations & fits
├── logistic_gate/ # Logistic gating simulations & fits
├── qpt/ # Quantum process tomography analyses
├── simulation_core/ # Core simulation scripts
├── epochs_features/ # Epoch extraction and feature computation
├── statistics/ # Permutation tests and power analyses
├── figures/ # Figure scripts and output (PDF)
├── stimulus_presentation/ # Psychopy cue schedule generator
├── utilities/ # Shared utilities, including environment files
├── run_all.sh # Master pipeline script
├── README.md
└── default_params.yml # Master parameter file

```

---

## Key Features

- **Reproducible Simulations:**  
  All simulations (decay, logistic gating, quantum tomography) and analysis scripts are provided, with outputs matching the manuscript and Supplementary Information.
- **Synthetic EEG Generation:**  
  Realistic EEG with injected artifacts and spindles for robust pipeline testing.
- **Automated Artifact Removal:**  
  Identifies flat, noisy, or artifact-laden channels using peak-to-peak and variance metrics.
- **Transparent SI Reproduction:**  
  All figures and tables in the manuscript can be reproduced from the command line. Output is written to standardized folders.
- **Continuous Integration:**  
  CI is configured to verify reproducibility on every commit using a minimal Conda environment.

---

## Getting Started

### Requirements

- **Python** 3.9+ (Conda recommended)
- All dependencies are specified in `utilities/env-ci.yml`.

### Quick Environment Setup

```text
# 1. Create the Conda environment
conda env create -n goldilocks_cri -f utilities/env-ci.yml

# 2. Activate the environment
conda activate goldilocks_cri
```

## Full Pipeline (One Command)
To reproduce all analyses and figures (using only synthetic data):
```text
chmod +x run_all.sh
bash run_all.sh
```
All results are saved in their respective `output/` subfolders. Figures (PDF) are in `figures/output/`.

## Manual Execution Steps
For detailed control, individual pipeline components can be run as follows:
```text
# 1. Synthetic EEG Generation
python synthetic_EEG/make_synthetic_eeg.py

# 2. EEG Preprocessing & Artifact Removal
python preprocessing/artifact_pipeline.py

# 3. Core Simulations
python simulation_core/toy_model_master_eq.py
python simulation_core/retro_kernel_weight.py

# 4. Decay, Logistic, and QPT Simulations
python decay/fit_decay.py
python logistic_gate/fit_logistic.py
python qpt/qpt_fit.py

# 5. Epoch Extraction & Feature Computation
python epochs_features/extract_epochs.py
python epochs_features/compute_x_t.py

# 6. Statistics & Figures
python statistics/permutation_tests.py
python statistics/power_analysis.py
python figures/make_decay_figure.py
python figures/make_logistic_figure.py
python figures/make_tomography_figure.py
python figures/EEG_flowchart_SIfigure1.py
python figures/make_tierA_seconds_figure.py

```

## Downloading the generated figures (from GitHub Actions)

Every push or pull request to `main` triggers the CI pipeline (`.github/workflows/ci.yml`). The pipeline runs `run_all.sh`, which executes the figure scripts:

- `figures/make_decay_figure.py`
- `figures/make_logistic_figure.py`
- `figures/make_tomography_figure.py`
- `figures/make_tierA_seconds_figure.py`

The resulting figure files (PDF/PNG) are written to `figures/output/` and uploaded as a single workflow artifact named:
```text
pipeline-outputs-YYYY-MM-DD_HH-MM-SS
```

### Download via GitHub web UI
1. Open the repository’s **Actions** tab.
2. Click the most recent **CI** run on the `main` branch.
3. In the run summary, scroll to **Artifacts** and download the ZIP named `pipeline-outputs-YYYY-MM-DD_HH-MM-SS`.
4. Unzip locally and navigate to:

```text
figures/output/
```
The three generated figures (PDF/PNG) will be inside.

### Download via GitHub CLI (optional)
If you use the GitHub CLI (`gh`):

```bash
# Download the most recent run’s artifacts interactively
gh run download

# Or fetch a specific artifact by name into ./artifacts
gh run download -n "pipeline-outputs-YYYY-MM-DD_HH-MM-SS" -D ./artifacts
```
**Supplemental Information Figure 1** (EEG preprocessing flowchart) is generated by:
```text
python figures/EEG_flowchart_SIfigure1.py
```
and saved to:
```text
figures/output/SI_Fig1_EEG_flowchart.png
```
## Main manuscript Figure 1 (TikZ → PDF & PNG)

A **TikZ program** has been added at `figures/CRI-manuscript_figure_1.tex` that compiles to a **vector PDF** and a **high-resolution PNG** into `figures/output/`.

### 1) Create `figures/CRI-manuscript_figure_1.tex`

> Standalone TikZ document (no `sidewaysfigure`/`caption`). Compile once; the output is tightly cropped to the graphic.

## Data and Code Availability
- All simulation code and synthetic data are included.
- All figures in the manuscript are reproducible from the generated outputs.
- Statistical analyses (permutation tests, power) are performed with published parameters.
- No human or empirical data are used in this project.

## EEG Artifact Handling Details
- Robust bad-channel detection for synthetic datasets.
- Peak-to-peak, flatness, and variance thresholds configurable in `default_params.yml`.
- In all simulation outputs, interpolation for bad channels is handled by assigning `NaN` values.

## Known Warnings & Expected Behavior
- `"Interpolation failed: No digitization points found..."` — Expected for synthetic datasets.
- Filename warnings from MNE — Outputs can be renamed for BIDS/MNE compatibility if required.
- `tight_layout` UserWarning — Figure rendering only; no effect on scientific results.

---

## Contact

**Maintained by:** George Sopasakis
Conscious Retroactive Intervention Project, 2025
- For questions regarding possible collaborations, technical or reproducibility issues, please use the repository’s GitHub Issues page.


## Citation

If you use this code or data, please cite:
> Sopasakis, G. (2025). Conscious Retroactive Intervention: A Reversed-Time Framework for Predictive Cognition. Manuscript in preparation.


## License

MIT License. See `LICENSE` for details.


## Disclaimer

This repository contains research code for a Perspective manuscript under review. All data are synthetic or simulated; results are for demonstration and reproducibility purposes only. For clinical or commercial use, independent validation is required.

---
**Last updated:** 2025-07-24


