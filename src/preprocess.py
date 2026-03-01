"""
preprocess.py - R-peak detection, RR interval extraction, outlier removal
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks


SAMPLING_RATE = 360  # Hz
NORMAL_BEAT_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}  # Normal-type beats


# ─────────────────────────────────────────────
# Signal Filtering
# ─────────────────────────────────────────────

def bandpass_filter(ecg: np.ndarray, fs: float = 360.0,
                    lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter to ECG."""
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, ecg)


def derivative_filter(ecg: np.ndarray) -> np.ndarray:
    """5-point derivative filter for Pan-Tompkins."""
    h = np.array([-1, -2, 0, 2, 1]) * (1 / 8)
    return np.convolve(ecg, h, mode='same')


def moving_window_integration(ecg: np.ndarray, window_size: int = 30) -> np.ndarray:
    """Moving window integration (squaring + integration)."""
    squared = ecg ** 2
    kernel = np.ones(window_size) / window_size
    return np.convolve(squared, kernel, mode='same')


# ─────────────────────────────────────────────
# Pan-Tompkins R-peak Detection
# ─────────────────────────────────────────────

def pan_tompkins_detector(ecg: np.ndarray, fs: float = 360.0) -> np.ndarray:
    """
    Pan-Tompkins QRS detection algorithm.
    Returns array of R-peak sample indices.
    """
    # 1. Bandpass filter
    ecg_filt = bandpass_filter(ecg, fs=fs, lowcut=5.0, highcut=15.0)

    # 2. Derivative
    ecg_deriv = derivative_filter(ecg_filt)

    # 3. Squaring
    ecg_sq = ecg_deriv ** 2

    # 4. Moving window integration
    win_size = int(0.15 * fs)
    ecg_int = moving_window_integration(ecg_sq, window_size=win_size)

    # 5. Adaptive thresholding & peak detection
    min_distance = int(0.2 * fs)  # 200ms refractory period
    threshold = 0.5 * np.mean(ecg_int)
    peaks, props = find_peaks(ecg_int, height=threshold, distance=min_distance)

    # 6. Refine: find true R-peak in original signal near each detected peak
    search_window = int(0.05 * fs)
    refined_peaks = []
    for p in peaks:
        start = max(0, p - search_window)
        end = min(len(ecg), p + search_window)
        local_max = start + np.argmax(np.abs(ecg[start:end]))
        refined_peaks.append(local_max)

    return np.array(sorted(set(refined_peaks)))


def annotation_based_rpeaks(annotations: pd.DataFrame,
                             normal_only: bool = True) -> np.ndarray:
    """
    Extract R-peak locations from annotation file.
    Optionally filter to normal beats only.
    """
    if annotations.empty:
        return np.array([])

    if normal_only:
        mask = annotations['symbol'].isin(NORMAL_BEAT_SYMBOLS)
        ann = annotations[mask]
    else:
        ann = annotations

    return ann['sample'].values.astype(int)


# ─────────────────────────────────────────────
# RR Interval Extraction
# ─────────────────────────────────────────────

def compute_rr_intervals(r_peaks: np.ndarray, fs: float = 360.0) -> tuple:
    """
    Compute RR intervals from R-peak positions.
    Returns:
        rr_ms    : RR intervals in milliseconds
        rr_times : time of each interval midpoint (seconds)
    """
    if len(r_peaks) < 2:
        return np.array([]), np.array([])

    rr_samples = np.diff(r_peaks)
    rr_ms = rr_samples * 1000.0 / fs

    # Time = midpoint between consecutive peaks
    rr_times = (r_peaks[:-1] + r_peaks[1:]) / 2.0 / fs

    return rr_ms, rr_times


# ─────────────────────────────────────────────
# Outlier / Ectopic Beat Removal
# ─────────────────────────────────────────────

def remove_ectopic_beats(rr_ms: np.ndarray, rr_times: np.ndarray,
                          method: str = 'percent',
                          low_rr: float = 300.0, high_rr: float = 2000.0,
                          percent_threshold: float = 20.0) -> tuple:
    """
    Remove ectopic beats and outlier RR intervals.

    Methods:
        'absolute'  : remove RR outside [low_rr, high_rr] ms
        'percent'   : remove RR deviating >percent_threshold% from local median
        'combined'  : both methods
    """
    mask = np.ones(len(rr_ms), dtype=bool)

    # Absolute physiological limits
    phys_mask = (rr_ms >= low_rr) & (rr_ms <= high_rr)
    mask &= phys_mask

    if method in ('percent', 'combined'):
        # Local median filter (window of 9)
        window = 9
        local_median = np.array([
            np.median(rr_ms[max(0, i - window // 2): min(len(rr_ms), i + window // 2 + 1)])
            for i in range(len(rr_ms))
        ])
        pct_deviation = np.abs(rr_ms - local_median) / local_median * 100
        mask &= pct_deviation <= percent_threshold

    rr_clean = rr_ms[mask]
    times_clean = rr_times[mask]
    n_removed = np.sum(~mask)

    return rr_clean, times_clean, n_removed


def interpolate_rr(rr_ms: np.ndarray, rr_times: np.ndarray,
                   fs_resample: float = 4.0) -> tuple:
    """
    Resample RR series to uniform time grid using cubic interpolation.
    Required for frequency-domain analysis.
    """
    from scipy.interpolate import interp1d

    if len(rr_ms) < 4:
        return rr_ms, rr_times

    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / fs_resample)
    interp_func = interp1d(rr_times, rr_ms, kind='cubic', bounds_error=False,
                           fill_value=(rr_ms[0], rr_ms[-1]))
    rr_interp = interp_func(t_uniform)

    return rr_interp, t_uniform


# ─────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────

def preprocess_record(record: dict, use_annotations: bool = True,
                       verbose: bool = True) -> dict:
    """
    Full preprocessing pipeline for one record.
    Returns enriched record dict with RR intervals added.
    """
    fs = record['fs']
    ecg_df = record['ecg']
    annotations = record['annotations']
    rid = record['record_id']

    ecg_signal = ecg_df['signal'].values

    # ── R-peak Detection ──
    if use_annotations and not annotations.empty:
        r_peaks = annotation_based_rpeaks(annotations, normal_only=True)
        method_used = 'annotation'
    else:
        r_peaks = pan_tompkins_detector(ecg_signal, fs=fs)
        method_used = 'pan_tompkins'

    if verbose:
        print(f"  Record {rid}: {len(r_peaks)} R-peaks detected via {method_used}")

    # ── RR Intervals ──
    rr_ms, rr_times = compute_rr_intervals(r_peaks, fs=fs)

    # ── Outlier Removal ──
    rr_clean, times_clean, n_removed = remove_ectopic_beats(rr_ms, rr_times)
    if verbose:
        print(f"  Record {rid}: {n_removed} ectopic beats removed, {len(rr_clean)} intervals retained")

    # ── Resampled RR (for freq analysis) ──
    rr_resampled, t_resampled = interpolate_rr(rr_clean, times_clean)

    # Enrich record
    record['r_peaks'] = r_peaks
    record['rr_ms'] = rr_ms
    record['rr_times'] = rr_times
    record['rr_clean'] = rr_clean
    record['times_clean'] = times_clean
    record['rr_resampled'] = rr_resampled
    record['t_resampled'] = t_resampled
    record['n_ectopic_removed'] = n_removed
    record['detection_method'] = method_used

    return record


if __name__ == "__main__":
    from load_data import generate_synthetic_record

    rec = generate_synthetic_record(100, duration_s=60)
    rec = preprocess_record(rec, use_annotations=True)
    print(f"\nRR intervals (first 10): {rec['rr_clean'][:10].round(1)} ms")
    print(f"Mean RR: {rec['rr_clean'].mean():.1f} ms  ({60000/rec['rr_clean'].mean():.0f} bpm)")
