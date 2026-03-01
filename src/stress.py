"""
stress.py - Stress detection, recovery analysis, autonomic balance assessment
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class StressEvent:
    start_s: float
    end_s: float
    peak_hr: float
    baseline_hr: float
    stress_score: float
    lf_hf_ratio: Optional[float] = None
    duration_s: float = 0.0

    def __post_init__(self):
        self.duration_s = self.end_s - self.start_s


@dataclass
class RecoveryPeriod:
    event: StressEvent
    recovery_start_s: float
    recovery_end_s: float
    recovery_time_s: float
    recovery_quality: str  # 'Fast', 'Normal', 'Slow'
    recovery_score: float  # 0-100


# ─────────────────────────────────────────────
# Baseline Estimation
# ─────────────────────────────────────────────

def estimate_baseline_hr(rr_ms: np.ndarray, rr_times: np.ndarray,
                          method: str = 'percentile',
                          window_s: float = 300.0) -> dict:
    """
    Estimate baseline (resting) HR.
    Methods:
        'percentile'  : 10th percentile HR (resting state)
        'min_window'  : lowest HR in any window_s-long window
        'first_window': HR during first window_s seconds
    """
    hr = 60000.0 / rr_ms

    if method == 'percentile':
        baseline = np.percentile(hr, 10)
        baseline_rr = 60000.0 / baseline

    elif method == 'min_window':
        # Sliding window mean HR, take minimum
        window_beats = max(10, int(window_s / np.mean(np.diff(rr_times))))
        min_hr = np.inf
        for i in range(0, len(hr) - window_beats, window_beats // 4):
            win_mean = np.mean(hr[i:i + window_beats])
            if win_mean < min_hr:
                min_hr = win_mean
        baseline = min_hr
        baseline_rr = 60000.0 / baseline

    elif method == 'first_window':
        mask = rr_times < (rr_times[0] + window_s)
        baseline = np.mean(hr[mask]) if np.any(mask) else np.mean(hr)
        baseline_rr = 60000.0 / baseline

    else:
        baseline = np.mean(hr)
        baseline_rr = np.mean(rr_ms)

    baseline_sdnn = np.std(rr_ms, ddof=1)
    baseline_rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))

    return {
        'baseline_hr': round(baseline, 2),
        'baseline_rr_ms': round(baseline_rr, 2),
        'baseline_sdnn': round(baseline_sdnn, 3),
        'baseline_rmssd': round(baseline_rmssd, 3),
        'method': method,
    }


# ─────────────────────────────────────────────
# Stress Detection
# ─────────────────────────────────────────────

def compute_stress_score(hr: float, baseline_hr: float,
                          sdnn: float, baseline_sdnn: float,
                          lf_hf: Optional[float] = None) -> float:
    """
    Compute stress score (0-100).
    High stress = elevated HR + reduced HRV + high LF/HF.
    """
    # HR elevation component (0-40)
    hr_delta = hr - baseline_hr
    hr_score = np.clip(hr_delta / baseline_hr * 100 * 2, 0, 40)

    # HRV reduction component (0-40)
    hrv_ratio = sdnn / (baseline_sdnn + 1e-6)
    hrv_score = np.clip((1 - hrv_ratio) * 60, 0, 40)

    # LF/HF ratio component (0-20) - sympathetic dominance
    lf_hf_score = 0
    if lf_hf is not None:
        lf_hf_score = np.clip((lf_hf - 1.5) / 3.0 * 20, 0, 20)

    return round(hr_score + hrv_score + lf_hf_score, 2)


def detect_stress_events(windowed_features: pd.DataFrame,
                          baseline: dict,
                          stress_hr_threshold: float = 1.15,
                          stress_hrv_threshold: float = 0.7,
                          min_duration_windows: int = 2) -> List[StressEvent]:
    """
    Detect stress events from windowed HRV features.

    Criteria:
        - HR ≥ baseline_hr × stress_hr_threshold
        OR
        - SDNN ≤ baseline_sdnn × stress_hrv_threshold
    """
    if windowed_features.empty:
        return []

    df = windowed_features.copy()
    baseline_hr = baseline['baseline_hr']
    baseline_sdnn = baseline['baseline_sdnn']

    # Stress flags per window
    hr_stress = df['mean_hr'] >= baseline_hr * stress_hr_threshold
    hrv_stress = df['sdnn'] <= baseline_sdnn * stress_hrv_threshold

    df['is_stress'] = (hr_stress | hrv_stress).astype(int)

    events = []
    in_event = False
    event_start = None
    event_windows = []

    for idx, row in df.iterrows():
        if row['is_stress'] and not in_event:
            in_event = True
            event_start = idx
            event_windows = [row]
        elif row['is_stress'] and in_event:
            event_windows.append(row)
        elif not row['is_stress'] and in_event:
            if len(event_windows) >= min_duration_windows:
                win_df = pd.DataFrame(event_windows)
                stress_score = compute_stress_score(
                    hr=win_df['mean_hr'].mean(),
                    baseline_hr=baseline_hr,
                    sdnn=win_df['sdnn'].mean(),
                    baseline_sdnn=baseline_sdnn,
                    lf_hf=win_df['lf_hf_ratio'].mean() if 'lf_hf_ratio' in win_df else None
                )
                ev = StressEvent(
                    start_s=win_df['window_start_s'].iloc[0],
                    end_s=win_df['window_end_s'].iloc[-1],
                    peak_hr=win_df['mean_hr'].max(),
                    baseline_hr=baseline_hr,
                    stress_score=stress_score,
                    lf_hf_ratio=win_df['lf_hf_ratio'].mean() if 'lf_hf_ratio' in win_df else None,
                )
                events.append(ev)
            in_event = False
            event_windows = []

    return events


def analyze_recovery(rr_ms: np.ndarray, rr_times: np.ndarray,
                      stress_events: List[StressEvent],
                      baseline_hr: float,
                      recovery_threshold_bpm: float = 5.0) -> List[RecoveryPeriod]:
    """
    Analyze HR recovery after each stress event.
    Measures time to return within recovery_threshold_bpm of baseline.
    """
    hr = 60000.0 / rr_ms
    recovery_periods = []

    for ev in stress_events:
        # Find HR values after stress event end
        mask_after = rr_times > ev.end_s
        if not np.any(mask_after):
            continue

        hr_after = hr[mask_after]
        times_after = rr_times[mask_after]

        # Find first time HR drops to within threshold of baseline
        recovered = np.abs(hr_after - baseline_hr) <= recovery_threshold_bpm

        if np.any(recovered):
            first_recovery_idx = np.argmax(recovered)
            recovery_time = times_after[first_recovery_idx] - ev.end_s
            recovery_end = times_after[first_recovery_idx]

            # Classify recovery quality
            if recovery_time < 60:
                quality = 'Fast'
                score = 85 + min(15, (60 - recovery_time) / 60 * 15)
            elif recovery_time < 180:
                quality = 'Normal'
                score = 60 + (180 - recovery_time) / 120 * 25
            else:
                quality = 'Slow'
                score = max(10, 60 - (recovery_time - 180) / 300 * 50)

            recovery_periods.append(RecoveryPeriod(
                event=ev,
                recovery_start_s=ev.end_s,
                recovery_end_s=recovery_end,
                recovery_time_s=recovery_time,
                recovery_quality=quality,
                recovery_score=round(score, 1)
            ))
        else:
            # No recovery within observed window
            recovery_periods.append(RecoveryPeriod(
                event=ev,
                recovery_start_s=ev.end_s,
                recovery_end_s=rr_times[-1],
                recovery_time_s=np.inf,
                recovery_quality='Incomplete',
                recovery_score=10.0
            ))

    return recovery_periods


def compute_recovery_capacity(recovery_periods: List[RecoveryPeriod]) -> dict:
    """
    Summarize recovery capacity across all stress events.
    """
    if not recovery_periods:
        return {'recovery_capacity': 'Unknown', 'avg_recovery_time_s': np.nan,
                'recovery_score': 50.0, 'n_events': 0}

    scores = [r.recovery_score for r in recovery_periods]
    times = [r.recovery_time_s for r in recovery_periods if r.recovery_time_s != np.inf]
    qualities = [r.recovery_quality for r in recovery_periods]

    avg_score = np.mean(scores)
    avg_time = np.mean(times) if times else np.inf

    if avg_score >= 75:
        capacity = 'Excellent'
    elif avg_score >= 55:
        capacity = 'Good'
    elif avg_score >= 35:
        capacity = 'Fair'
    else:
        capacity = 'Poor'

    quality_counts = {q: qualities.count(q) for q in set(qualities)}

    return {
        'recovery_capacity': capacity,
        'avg_recovery_time_s': round(avg_time, 1) if avg_time != np.inf else None,
        'avg_recovery_score': round(avg_score, 1),
        'n_stress_events': len(recovery_periods),
        'quality_distribution': quality_counts,
    }


# ─────────────────────────────────────────────
# Full Stress Analysis Pipeline
# ─────────────────────────────────────────────

def analyze_stress(record: dict, verbose: bool = True) -> dict:
    """Run full stress analysis pipeline on a preprocessed record."""
    rr_clean = record.get('rr_clean', np.array([]))
    times_clean = record.get('times_clean', np.array([]))
    windowed = record.get('windowed_features', pd.DataFrame())

    if len(rr_clean) < 20:
        print(f"  Record {record['record_id']}: insufficient data for stress analysis")
        return record

    # Baseline
    baseline = estimate_baseline_hr(rr_clean, times_clean, method='percentile')

    # Detect stress events
    stress_events = detect_stress_events(
        windowed, baseline,
        stress_hr_threshold=1.15,
        stress_hrv_threshold=0.70
    ) if not windowed.empty else []

    # Recovery analysis
    recovery_periods = analyze_recovery(
        rr_clean, times_clean, stress_events, baseline['baseline_hr']
    )

    # Recovery capacity
    capacity = compute_recovery_capacity(recovery_periods)

    record['stress_baseline'] = baseline
    record['stress_events'] = stress_events
    record['recovery_periods'] = recovery_periods
    record['recovery_capacity'] = capacity

    if verbose:
        print(f"  Record {record['record_id']}: baseline HR={baseline['baseline_hr']:.1f} bpm, "
              f"{len(stress_events)} stress events detected")
        print(f"  Recovery capacity: {capacity['recovery_capacity']} "
              f"(avg score: {capacity.get('avg_recovery_score', 'N/A')})")

    return record


if __name__ == "__main__":
    from load_data import generate_synthetic_record
    from preprocess import preprocess_record
    from features import compute_all_features

    rec = generate_synthetic_record(100, duration_s=600, mean_hr=75, hrv_std=40)
    rec = preprocess_record(rec, verbose=False)
    rec = compute_all_features(rec, verbose=False)
    rec = analyze_stress(rec, verbose=True)
