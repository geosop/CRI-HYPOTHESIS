# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:10:18 2025

@author: ADMIN

EEG preprocessing: filtering, bad-channel detection, interpolation, ICA
(Picard if available, else FastICA).

preprocessing/artifact_pipeline.py
"""
from __future__ import annotations

import os
import sys
import yaml
import numpy as np
import mne

# ─── Optional Picard → fallback to FastICA ─────────────────────────────────────
try:
    import picard  # provides the 'picard' ICA method
    _ICA_METHOD = "picard"
    # Some builds may lack __version__; set a benign value so version checks pass
    if not hasattr(picard, "__version__"):
        picard.__version__ = "0.0"
except Exception:
    _ICA_METHOD = "fastica"  # scikit-learn fallback

# ─── Quiet MNE “stupid warnings” about file naming (as in your version) ───────
mne.utils.set_config('MNE_IGNORE_STUPID_WARNINGS', 'true')

# ─── Import utilities from project root ────────────────────────────────────────
here = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(here, '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from utilities.seed_manager import load_state, save_state  # noqa: E402


# ───────────────────────────────────────────────────────────────────────────────
# Config / params
# ───────────────────────────────────────────────────────────────────────────────
def load_params() -> dict:
    with open(os.path.join(here, 'default_params.yml'), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['preprocessing']


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def detect_bad_channels(raw: mne.io.BaseRaw, params: dict) -> list[str]:
    """Detect bad EEG channels using PTP (NumPy ≥2: np.ptp), flatness, and variance."""
    eeg_ch_names = [ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == 'eeg']
    data = raw.get_data(picks='eeg')

    # NumPy ≥2.0: use np.ptp instead of ndarray.ptp
    ptp_vals = np.ptp(data, axis=1)

    ptp_thresh = float(params.get('ptp_threshold', 150e-6))   # volts
    flat_thresh = float(params.get('flat_threshold', 1e-6))   # volts

    bad_ptp = [ch for ch, val in zip(eeg_ch_names, ptp_vals) if val > ptp_thresh]
    bad_flat = [ch for ch, val in zip(eeg_ch_names, ptp_vals) if val < flat_thresh]

    # Variance-based
    vars_ = np.var(data, axis=1)
    thr = vars_.mean() + float(params.get('bads_threshold', 5.0)) * vars_.std()
    bad_var = [ch for ch, val in zip(eeg_ch_names, vars_) if val > thr]

    print(f"EEG PTP (min, max): {ptp_vals.min()*1e6:.2f}µV, {ptp_vals.max()*1e6:.2f}µV")
    print(f"Bad PTP: {len(bad_ptp)}, Bad flat: {len(bad_flat)}, Bad var: {len(bad_var)}")

    bads = sorted(set(bad_ptp + bad_flat + bad_var))
    if bads:
        print(f"Marking bad channels (PTP>{ptp_thresh*1e6:.1f}µV, "
              f"<{flat_thresh*1e6:.1f}µV, or high var): {bads}")
    else:
        print("No bad channels detected.")
    return bads


# ───────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ───────────────────────────────────────────────────────────────────────────────
def artifact_pipeline(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    # 1) Band-pass
    raw.filter(params['l_freq'], params['h_freq'], fir_design='firwin', verbose=False)

    # 2) Notch (single value or list both OK)
    raw.notch_filter(params['notch_freq'], fir_design='firwin', verbose=False)

    # 3) Bad channels
    raw.info['bads'] = detect_bad_channels(raw, params)

    # 4) Interpolate if possible (skip gracefully if montage/dig is absent)
    try:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    except Exception as e:
        print(f"Interpolation skipped ({e}). Proceeding without interpolation.")

    # 5) ICA picks: valid EEG only
    picks_eeg = mne.pick_types(raw.info, eeg=True, exclude='bads')
    valid_eeg = [idx for idx in picks_eeg if not np.all(np.isnan(raw.get_data(picks=[idx])))]
    if not valid_eeg:
        print("All EEG channels invalid after interpolation. Skipping ICA.")
        return raw

    # Robust EOG handling
    eog_ch = params.get('eog_ch', 'EOG 061')
    picks_eog = []
    if eog_ch in raw.ch_names:
        eog_idx = raw.ch_names.index(eog_ch)
        if not np.all(np.isnan(raw.get_data(picks=[eog_idx]))):
            picks_eog = [eog_idx]

    # 6) ICA (Picard if available else FastICA)
    from mne.preprocessing import ICA
    n_components = min(int(params.get('ica_n_components', 20)), len(valid_eeg))
    fit_params = {'ortho': False} if _ICA_METHOD == 'picard' else None

    ica = ICA(
        n_components=n_components,
        method=_ICA_METHOD,
        fit_params=fit_params,
        random_state=97,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw, picks=valid_eeg, verbose=False)

    # 7) Remove EOG components if we have an EOG channel
    if picks_eog:
        try:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch, verbose=False)
            ica.exclude = eog_inds
        except Exception as e:
            print(f"Warning: EOG artifact detection failed ({e}); skipping EOG exclusion.")

    return ica.apply(raw.copy(), verbose=False)


# ───────────────────────────────────────────────────────────────────────────────
# CLI entry
# ───────────────────────────────────────────────────────────────────────────────
def main():
    params = load_params()
    raw_dir = os.path.normpath(os.path.join(here, '..', 'synthetic_EEG', 'output'))
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(raw_dir):
        if not fname.endswith('.fif'):
            continue
        fpath = os.path.join(raw_dir, fname)
        raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
        clean = artifact_pipeline(raw, params)

        # Save a copy that downstream currently expects…
        out_clean = fname.replace('.fif', '_clean.fif')
        clean.save(os.path.join(out_dir, out_clean), overwrite=True, verbose=False)

        # …and also a standards-compliant EEG filename to silence MNE warnings.
        out_clean_eeg = fname.replace('.fif', '_clean_eeg.fif')
        clean.save(os.path.join(out_dir, out_clean_eeg), overwrite=True, verbose=False)

        print(f"Saved cleaned data to {os.path.join(out_dir, out_clean)} "
              f"and {os.path.join(out_dir, out_clean_eeg)}")


if __name__ == '__main__':
    main()

