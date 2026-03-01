"""
features.py - HRV feature calculation
Time-domain, Frequency-domain, Nonlinear (Poincaré, Sample Entropy)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy as scipy_entropy


# ─────────────────────────────────────────────
# Time-Domain Features
# ─────────────────────────────────────────────

def compute_time_domain(rr_ms: np.ndarray) -> dict:
    """
    Compute standard time-domain HRV features.
    Input: RR intervals in milliseconds
    """
    if len(rr_ms) < 2:
        return {k: np.nan for k in ['mean_rr', 'sdnn', 'rmssd', 'pnn50',
                                     'mean_hr', 'min_hr', 'max_hr', 'nn50']}

    diff_rr = np.diff(rr_ms)

    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100

    # Instantaneous HR
    hr = 60000.0 / rr_ms
    mean_hr = np.mean(hr)
    min_hr = np.min(hr)
    max_hr = np.max(hr)

    return {
        'mean_rr': round(mean_rr, 3),
        'sdnn': round(sdnn, 3),
        'rmssd': round(rmssd, 3),
        'nn50': int(nn50),
        'pnn50': round(pnn50, 3),
        'mean_hr': round(mean_hr, 2),
        'min_hr': round(min_hr, 2),
        'max_hr': round(max_hr, 2),
    }


# ─────────────────────────────────────────────
# Frequency-Domain Features
# ─────────────────────────────────────────────

def compute_frequency_domain(rr_interp: np.ndarray, fs: float = 4.0) -> dict:
    """
    Compute frequency-domain HRV features using Welch's method.
    Input: uniformly resampled RR series (4 Hz), sampling rate fs.
    """
    if len(rr_interp) < 32:
        return {k: np.nan for k in ['vlf_power', 'lf_power', 'hf_power',
                                     'lf_hf_ratio', 'lf_nu', 'hf_nu', 'total_power']}

    # Welch PSD
    nperseg = min(256, len(rr_interp) // 4)
    freqs, psd = signal.welch(rr_interp, fs=fs, nperseg=nperseg, scaling='density')

    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.trapezoid(psd[mask], freqs[mask]) if hasattr(np, 'trapezoid') else np.trapz(psd[mask], freqs[mask])

    vlf = band_power(0.003, 0.04)
    lf = band_power(0.04, 0.15)
    hf = band_power(0.15, 0.40)
    total = band_power(0.003, 0.40)

    lf_nu = (lf / (lf + hf)) * 100 if (lf + hf) > 0 else np.nan
    hf_nu = (hf / (lf + hf)) * 100 if (lf + hf) > 0 else np.nan
    lf_hf = lf / hf if hf > 0 else np.nan

    return {
        'vlf_power': round(vlf, 6),
        'lf_power': round(lf, 6),
        'hf_power': round(hf, 6),
        'total_power': round(total, 6),
        'lf_nu': round(lf_nu, 3),
        'hf_nu': round(hf_nu, 3),
        'lf_hf_ratio': round(lf_hf, 4),
        'freqs': freqs,
        'psd': psd,
    }


# ─────────────────────────────────────────────
# Nonlinear Features
# ─────────────────────────────────────────────

def compute_poincare(rr_ms: np.ndarray) -> dict:
    """
    Poincaré plot analysis.
    SD1: short-term HRV (parasympathetic)
    SD2: long-term HRV (sympathetic + parasympathetic)
    """
    if len(rr_ms) < 3:
        return {'sd1': np.nan, 'sd2': np.nan, 'sd1_sd2_ratio': np.nan, 'csi': np.nan, 'cvi': np.nan}

    rr1 = rr_ms[:-1]
    rr2 = rr_ms[1:]

    sd1 = np.sqrt(0.5 * np.std(rr2 - rr1, ddof=1) ** 2)
    sd2 = np.sqrt(2 * np.std(rr_ms, ddof=1) ** 2 - 0.5 * np.std(rr2 - rr1, ddof=1) ** 2)

    ratio = sd1 / sd2 if sd2 > 0 else np.nan
    csi = sd2 / sd1 if sd1 > 0 else np.nan
    cvi = np.log10(sd1 * sd2) if (sd1 > 0 and sd2 > 0) else np.nan

    return {
        'sd1': round(sd1, 3),
        'sd2': round(sd2, 3),
        'sd1_sd2_ratio': round(ratio, 4),
        'csi': round(csi, 4),
        'cvi': round(cvi, 4),
        'poincare_x': rr1,
        'poincare_y': rr2,
    }


def sample_entropy(rr_ms: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample Entropy (SampEn) for nonlinear complexity.
    m: embedding dimension (2)
    r_factor: tolerance as fraction of std
    """
    if len(rr_ms) < 10:
        return np.nan

    N = len(rr_ms)
    r = r_factor * np.std(rr_ms)

    def count_templates(m):
        count = 0
        for i in range(N - m):
            for j in range(i + 1, N - m):
                if np.max(np.abs(rr_ms[i:i + m] - rr_ms[j:j + m])) <= r:
                    count += 1
        return count

    # For efficiency, limit to 500 samples
    rr_short = rr_ms[:500] if len(rr_ms) > 500 else rr_ms
    N = len(rr_short)

    B = 0
    A = 0
    for i in range(N - m - 1):
        for j in range(i + 1, N - m):
            if np.max(np.abs(rr_short[i:i + m] - rr_short[j:j + m])) <= r:
                B += 1
                if np.abs(rr_short[i + m] - rr_short[j + m]) <= r:
                    A += 1

    if B == 0:
        return np.nan
    return -np.log(A / B) if A > 0 else np.inf


def compute_dfa(rr_ms: np.ndarray) -> dict:
    """
    Detrended Fluctuation Analysis - alpha1 (short-term) and alpha2 (long-term).
    """
    if len(rr_ms) < 64:
        return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}

    N = len(rr_ms)
    y = np.cumsum(rr_ms - np.mean(rr_ms))

    def fluctuation(n):
        segments = N // n
        if segments < 1:
            return np.nan
        F_sq = []
        for k in range(segments):
            seg = y[k * n:(k + 1) * n]
            x = np.arange(len(seg))
            coef = np.polyfit(x, seg, 1)
            trend = np.polyval(coef, x)
            F_sq.append(np.mean((seg - trend) ** 2))
        return np.sqrt(np.mean(F_sq))

    scales = np.unique(np.logspace(np.log10(4), np.log10(N // 4), 20).astype(int))
    scales = scales[scales >= 4]

    flucts = np.array([fluctuation(n) for n in scales])
    valid = ~np.isnan(flucts) & (flucts > 0)

    alpha1 = alpha2 = np.nan
    if np.sum(valid) >= 4:
        log_s = np.log10(scales[valid].astype(float))
        log_f = np.log10(flucts[valid])

        # Short-term: scales 4-16
        s1 = scales[valid] <= 16
        if np.sum(s1) >= 2:
            alpha1 = np.polyfit(log_s[s1], log_f[s1], 1)[0]
        # Long-term: scales 16+
        s2 = scales[valid] > 16
        if np.sum(s2) >= 2:
            alpha2 = np.polyfit(log_s[s2], log_f[s2], 1)[0]

    return {'dfa_alpha1': round(alpha1, 4) if not np.isnan(alpha1) else np.nan,
            'dfa_alpha2': round(alpha2, 4) if not np.isnan(alpha2) else np.nan}


# ─────────────────────────────────────────────
# Windowed Feature Extraction
# ─────────────────────────────────────────────

def extract_windowed_features(rr_ms: np.ndarray, rr_times: np.ndarray,
                               rr_resampled: np.ndarray, t_resampled: np.ndarray,
                               window_s: float = 300.0, step_s: float = 60.0,
                               fs_interp: float = 4.0) -> pd.DataFrame:
    """
    Extract HRV features in sliding windows.
    Returns DataFrame with one row per window.
    """
    records = []

    if len(rr_times) < 2:
        return pd.DataFrame()

    t_start = rr_times[0]
    t_end = rr_times[-1]
    window_starts = np.arange(t_start, t_end - window_s, step_s)

    for ws in window_starts:
        we = ws + window_s

        # Time-domain: use original RR
        mask_td = (rr_times >= ws) & (rr_times < we)
        rr_win = rr_ms[mask_td]

        if len(rr_win) < 20:
            continue

        td = compute_time_domain(rr_win)

        # Frequency-domain: use resampled
        mask_fd = (t_resampled >= ws) & (t_resampled < we)
        rr_win_interp = rr_resampled[mask_fd]
        fd = compute_frequency_domain(rr_win_interp, fs=fs_interp)

        # Nonlinear
        nl = compute_poincare(rr_win)
        sampen = sample_entropy(rr_win)

        row = {
            'window_start_s': ws,
            'window_end_s': we,
            'n_beats': len(rr_win),
        }
        # Add all numeric features
        for d in [td, fd, nl]:
            for k, v in d.items():
                if not isinstance(v, np.ndarray):
                    row[k] = v
        row['sample_entropy'] = sampen

        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Full Feature Pipeline
# ─────────────────────────────────────────────

def compute_all_features(record: dict, verbose: bool = True) -> dict:
    """Compute all HRV features for a record."""
    rr_clean = record.get('rr_clean', np.array([]))
    times_clean = record.get('times_clean', np.array([]))
    rr_resampled = record.get('rr_resampled', np.array([]))
    t_resampled = record.get('t_resampled', np.array([]))

    if len(rr_clean) < 10:
        print(f"  Record {record['record_id']}: insufficient RR data")
        return record

    # Global features
    td = compute_time_domain(rr_clean)
    fd = compute_frequency_domain(rr_resampled)
    nl = compute_poincare(rr_clean)
    dfa = compute_dfa(rr_clean)
    sampen = sample_entropy(rr_clean)

    features = {**td, **nl, **dfa, 'sample_entropy': sampen}
    for k, v in fd.items():
        if not isinstance(v, np.ndarray):
            features[k] = v

    record['features'] = features
    record['psd_freqs'] = fd.get('freqs')
    record['psd_values'] = fd.get('psd')

    # Windowed features (5-min windows, 1-min step)
    win_df = extract_windowed_features(
        rr_clean, times_clean, rr_resampled, t_resampled,
        window_s=300.0, step_s=60.0
    )
    record['windowed_features'] = win_df

    if verbose:
        print(f"  Record {record['record_id']}: "
              f"RMSSD={td['rmssd']:.1f}ms, SDNN={td['sdnn']:.1f}ms, "
              f"LF/HF={features.get('lf_hf_ratio', 'N/A')}")

    return record


if __name__ == "__main__":
    from load_data import generate_synthetic_record
    from preprocess import preprocess_record

    rec = generate_synthetic_record(100, duration_s=300, mean_hr=70, hrv_std=35)
    rec = preprocess_record(rec, verbose=True)
    rec = compute_all_features(rec, verbose=True)

    print("\n=== Global HRV Features ===")
    for k, v in rec['features'].items():
        if not isinstance(v, np.ndarray):
            print(f"  {k:20s}: {v}")

    print(f"\nWindowed features shape: {rec['windowed_features'].shape}")
