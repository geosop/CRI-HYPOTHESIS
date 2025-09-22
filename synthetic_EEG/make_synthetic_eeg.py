# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:30:00 2025

@author: ADMIN

synthetic_EEG/make_synthetic_eeg.py

Generate minimal synthetic EEG datasets (.fif) for pipeline testing.
Outputs are saved to synthetic_EEG/output/.
"""

import os
import sys

# ─── allow imports from project root ──────────────────────────────────────────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

from utilities.seed_manager import load_state, save_state
import yaml
import numpy as np
import mne
import pandas as pd


def load_params():
    """Read YAML parameters from default_params.yml in the same folder."""
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'default_params.yml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['synthetic_EEG']


# -------------------------------------------------------------------------
# Noise helpers
# -------------------------------------------------------------------------

def pink_noise(n, alpha=1.0, rng=None):
    """Return 1/f^alpha noise of length n without division-by-zero warnings."""
    rng = np.random.default_rng() if rng is None else rng
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, d=1)
    denom = freqs ** (alpha / 2)
    denom[0] = np.inf
    colored = np.fft.irfft(np.fft.rfft(white) / denom)
    return colored / np.std(colored)


def inject_spindle(sig, fs, onset_s, dur_s, freq=14.0, amp=2.0):
    """Inject a Hann-windowed sine burst into *sig* in place."""
    t = np.arange(0, dur_s, 1 / fs)
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(t)) / len(t)))
    burst = amp * np.sin(2 * np.pi * freq * t) * window
    i0 = int(onset_s * fs)
    sig[i0:i0 + len(burst)] += burst


# -------------------------------------------------------------------------
# Main synthetic generator
# -------------------------------------------------------------------------

def create_raw(params, tier_seed, inject=False):
    """Create an MNE RawArray with synthetic EEG + optional spindles."""
    fs     = params['sfreq']
    n_ch   = params['n_channels']
    n_samp = int(params['record_length_s'] * fs)
    rng    = np.random.default_rng(tier_seed)

    data = []
    for _ in range(n_ch):
        sig = (0.3 * pink_noise(n_samp, alpha=params['pink_alpha'], rng=rng)
               + 0.7 * rng.standard_normal(n_samp))
        sig = sig / np.std(sig)
        sig = sig * 50e-6  # Scale std to 50 μV (typical EEG)
        data.append(sig)

    # Inject artifact in channel 4 (index 3)
    data[3][1000:2000] += 500e-6  # Artifact of 500 μV
    # Flat channel in channel 7 (index 6)
    data[6][:] = 0.0

    # Add EOG channel
    data.append(rng.standard_normal(n_samp) * 50e-6)

    all_data = np.vstack(data)
    print(f"DEBUG: EEG amplitude min={np.min(all_data)*1e6:.1f}µV, max={np.max(all_data)*1e6:.1f}µV")

    if inject:
        for onset in params['spindle_onsets_s']:
            inject_spindle(
                all_data[0], fs, onset,
                params['spindle_duration_s'],
                freq=params['spindle_freq_hz'],
                amp=params['spindle_amp']
            )

    # Channel names/types
    ch_names = [f'EEG {i:03d}' for i in range(n_ch)] + ['EOG 061']
    ch_types = ['eeg'] * n_ch + ['eog']
    info     = mne.create_info(ch_names, sfreq=fs, ch_types=ch_types)
    raw      = mne.io.RawArray(all_data, info, verbose=False)

    if params.get('apply_montage', False):
        raw.set_montage('standard_1020', on_missing='ignore', verbose=False)

    return raw


def save_events(params, out_dir):
    """Save spindle onsets to events.tsv."""
    df = pd.DataFrame({
        'onset_s':    params['spindle_onsets_s'],
        'duration_s': params['spindle_duration_s'],
        'trial_type': 'spindle'
    })
    df.to_csv(os.path.join(out_dir, 'events.tsv'),
              sep='\t', index=False)


def main():
    # ─── reproducibility ────────────────────────────────────────────────────
    load_state()                                # no-op on first run
    params = load_params()
    np.random.seed(params.get('seed', 0))
    save_state()                                # freeze RNG for next scripts
    # ────────────────────────────────────────────────────────────────────────

    here    = os.path.dirname(__file__)
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Tier A with spindles
    raw_A = create_raw(params, tier_seed=1, inject=True)
    raw_A.save(os.path.join(out_dir, 'TierA_sleep_raw.fif'),
               overwrite=True, verbose=False)

    # Tier B baseline
    raw_B = create_raw(params, tier_seed=2, inject=False)
    raw_B.save(os.path.join(out_dir, 'TierB_wake_raw.fif'),
               overwrite=True, verbose=False)

    # Save spindle events
    save_events(params, out_dir)

    # NumPy 2.x compatible peak-to-peak
    data_A = raw_A.get_data(picks='eeg')
    ptp = np.ptp(data_A, axis=1)
    print(f"TierA EEG ptp per channel: min={np.min(ptp)*1e6:.1f}µV, max={np.max(ptp)*1e6:.1f}µV")

    print(f"Synthetic EEG saved to {out_dir}")


if __name__ == '__main__':
    main()


