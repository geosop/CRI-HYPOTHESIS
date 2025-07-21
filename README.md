# Goldilocks_CRI

**Reproducibility Pipeline for the "Conscious Retroactive Intervention (CRI)" Perspective Manuscript**

---

## Overview

This repository provides all code, scripts, and environment specifications required to fully reproduce the simulations and analyses for the manuscript submitted for double-blind review.

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
  All figures and tables in the SI can be reproduced from the command line. Output is written to standardized folders.
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

### Code Ocean Capsule
This repository is directly compatible with Code Ocean capsules.
Reviewers: To reproduce all results, simply launch the capsule and run the provided pipeline script.
- [Code Ocean reproducibility guidelines](https://support.codeocean.com/hc/en-us/articles/360010599893-Use-an-environment-YAML-file-to-specify-your-software-environment)

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

```
## Data and Code Availability
- All simulation code and synthetic data are included.
- No human or empirical data are used in this project.

## EEG Artifact Handling Details
- Robust bad-channel detection for both synthetic and real datasets.
- Peak-to-peak, flatness, and variance thresholds configurable in `default_params.yml`.
- For synthetic data, interpolation sets bad channels to `NaN`.
- For real EEG, ensure digitization/headshape points are present for full spatial interpolation.

## Known Warnings & Expected Behavior
- `"Interpolation failed: No digitization points found..."` — Expected for synthetic datasets.
- Filename warnings from MNE — Outputs can be renamed for BIDS/MNE compatibility if required.
- `tight_layout` UserWarning — Figure rendering only; no effect on scientific results.

## Using with Real EEG Data
To use this pipeline with real EEG data:
- Place `.fif` files in the designated input directory.
- Update `default_params.yml` for channel naming and thresholds.
- Ensure headshape/digitization data are present for full spatial interpolation.
- See `preprocessing/artifact_pipeline.py` for further configuration.

## Contact
- For technical questions or reproducibility issues, please use the repository’s GitHub Issues page.
- Direct personal or institutional contact is intentionally omitted to **preserve author anonymity**.

## Citation
If you use this code or data, please cite:
> [Reference to the associated manuscript. Details omitted for anonymous review. Manuscript in preparation.]



## License

MIT License. See `LICENSE` for details.



## Disclaimer

This repository contains research code for a Perspective manuscript under review. All data are synthetic or simulated; results are for demonstration and reproducibility purposes only. For clinical or commercial use, independent validation is required.

---
**Last updated:** 2025-07-21


