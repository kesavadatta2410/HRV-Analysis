"""
sleep.py - Sleep stage classification, nightmare/arousal detection, sleep quality scoring
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class SleepWindow:
    start_s: float
    end_s: float
    stage: str  # 'Wake', 'Light', 'Deep', 'REM'
    mean_hr: float
    sdnn: float
    lf_hf: Optional[float]
    confidence: float  # 0-1


@dataclass
class ArousalEvent:
    start_s: float
    end_s: float
    peak_hr: float
    baseline_hr: float
    hr_increase_pct: float
    event_type: str  # 'Arousal', 'Nightmare', 'Brief_Awakening'
    severity: str  # 'Mild', 'Moderate', 'Severe'


# ─────────────────────────────────────────────
# Sleep Stage Classification
# ─────────────────────────────────────────────

# HRV thresholds by stage (based on literature)
SLEEP_STAGE_THRESHOLDS = {
    'Deep':  {'min_rr': 900, 'max_hr': 58, 'min_rmssd': 40, 'min_hf_nu': 55},
    'REM':   {'hr_range': (60, 75), 'rmssd_range': (20, 45), 'lf_hf_max': 3.0},
    'Light': {'hr_range': (60, 75), 'rmssd_range': (25, 50)},
    'Wake':  {'min_hr': 68, 'max_rmssd': 30},
}


def classify_sleep_stage(mean_hr: float, sdnn: float, rmssd: float,
                          lf_hf: Optional[float] = None,
                          mean_rr_ms: float = 900.0) -> tuple:
    """
    Rule-based sleep stage classification using HRV features.
    Returns (stage: str, confidence: float)

    Based on:
    - Deep NREM: lowest HR, highest HRV (vagal tone)
    - REM: HR similar to Light, but irregular
    - Light NREM: moderate HR and HRV
    - Wake: highest HR, lowest HRV
    """
    stage_scores = {}

    # ── Wake ──
    wake_score = 0
    if mean_hr > 72:
        wake_score += min(1.0, (mean_hr - 72) / 20)
    if rmssd < 25:
        wake_score += min(0.8, (25 - rmssd) / 25)
    if sdnn < 30:
        wake_score += min(0.5, (30 - sdnn) / 30)
    stage_scores['Wake'] = wake_score

    # ── Deep Sleep ──
    deep_score = 0
    if mean_hr < 58:
        deep_score += min(1.5, (58 - mean_hr) / 10)
    if rmssd > 45:
        deep_score += min(1.0, (rmssd - 45) / 30)
    if lf_hf is not None and lf_hf < 1.5:
        deep_score += 0.5
    stage_scores['Deep'] = deep_score

    # ── REM ──
    rem_score = 0
    if 60 <= mean_hr <= 74:
        rem_score += 1.0
    if 18 <= rmssd <= 40:
        rem_score += 0.8
    if lf_hf is not None and 1.0 <= lf_hf <= 3.5:
        rem_score += 0.5
    # REM has higher variability (irregular HR)
    if sdnn > 35:
        rem_score += 0.3
    stage_scores['REM'] = rem_score

    # ── Light Sleep ──
    light_score = 0
    if 60 <= mean_hr <= 70:
        light_score += 0.8
    if 25 <= rmssd <= 45:
        light_score += 0.8
    if sdnn > 30:
        light_score += 0.5
    stage_scores['Light'] = light_score

    # Pick highest scoring stage
    best_stage = max(stage_scores, key=stage_scores.get)
    total = sum(stage_scores.values())
    confidence = stage_scores[best_stage] / total if total > 0 else 0.25

    return best_stage, round(min(confidence, 0.95), 3)


def classify_sleep_stages(windowed_features: pd.DataFrame,
                           sleep_start_s: Optional[float] = None,
                           sleep_end_s: Optional[float] = None) -> List[SleepWindow]:
    """
    Classify sleep stages for all windows.
    Optionally filter to sleep period [sleep_start_s, sleep_end_s].
    """
    if windowed_features.empty:
        return []

    df = windowed_features.copy()

    # Filter to sleep period if specified
    if sleep_start_s is not None:
        df = df[df['window_start_s'] >= sleep_start_s]
    if sleep_end_s is not None:
        df = df[df['window_end_s'] <= sleep_end_s]

    windows = []
    for _, row in df.iterrows():
        rmssd = row.get('rmssd', 30)
        lf_hf = row.get('lf_hf_ratio', None)

        stage, conf = classify_sleep_stage(
            mean_hr=row.get('mean_hr', 70),
            sdnn=row.get('sdnn', 40),
            rmssd=rmssd,
            lf_hf=lf_hf,
            mean_rr_ms=row.get('mean_rr', 857),
        )

        windows.append(SleepWindow(
            start_s=row['window_start_s'],
            end_s=row['window_end_s'],
            stage=stage,
            mean_hr=row.get('mean_hr', 70),
            sdnn=row.get('sdnn', 40),
            lf_hf=lf_hf,
            confidence=conf,
        ))

    return windows


# ─────────────────────────────────────────────
# Arousal / Nightmare Detection
# ─────────────────────────────────────────────

def detect_arousal_events(rr_ms: np.ndarray, rr_times: np.ndarray,
                           sleep_windows: List[SleepWindow],
                           baseline_hr: float,
                           arousal_threshold_pct: float = 15.0,
                           nightmare_threshold_pct: float = 30.0) -> List[ArousalEvent]:
    """
    Detect arousal and nightmare events during sleep.

    Arousal: HR spike ≥ arousal_threshold_pct% above local mean
    Nightmare: HR spike ≥ nightmare_threshold_pct% during REM sleep
    """
    if len(rr_ms) < 10 or not sleep_windows:
        return []

    hr = 60000.0 / rr_ms
    events = []

    # Get sleep period boundaries
    sleep_start = min(w.start_s for w in sleep_windows)
    sleep_end = max(w.end_s for w in sleep_windows)

    # Map windows to stage
    def get_stage_at_time(t):
        for w in sleep_windows:
            if w.start_s <= t < w.end_s:
                return w.stage
        return 'Unknown'

    # Only analyze sleep period
    sleep_mask = (rr_times >= sleep_start) & (rr_times <= sleep_end)
    hr_sleep = hr[sleep_mask]
    times_sleep = rr_times[sleep_mask]

    if len(hr_sleep) < 10:
        return []

    # Sliding window (30-beat) baseline
    window_size = 30
    for i in range(window_size, len(hr_sleep) - window_size):
        local_baseline = np.mean(hr_sleep[max(0, i - window_size):i])
        current_hr = hr_sleep[i]
        increase_pct = (current_hr - local_baseline) / (local_baseline + 1e-6) * 100

        if increase_pct >= arousal_threshold_pct:
            t = times_sleep[i]
            stage = get_stage_at_time(t)

            # Find event duration
            j = i
            while j < len(hr_sleep) and (hr_sleep[j] - local_baseline) / (local_baseline + 1e-6) * 100 >= 5:
                j += 1

            event_end_t = times_sleep[min(j, len(times_sleep) - 1)]

            # Classify event
            if increase_pct >= nightmare_threshold_pct and stage == 'REM':
                event_type = 'Nightmare'
                severity = 'Severe' if increase_pct >= 50 else 'Moderate'
            elif increase_pct >= nightmare_threshold_pct:
                event_type = 'Arousal'
                severity = 'Moderate'
            else:
                event_type = 'Brief_Awakening'
                severity = 'Mild'

            events.append(ArousalEvent(
                start_s=t,
                end_s=event_end_t,
                peak_hr=max(hr_sleep[i:j + 1]) if j > i else current_hr,
                baseline_hr=local_baseline,
                hr_increase_pct=round(increase_pct, 1),
                event_type=event_type,
                severity=severity,
            ))

            # Skip past this event
            i = j + 1

    # Deduplicate nearby events (within 60s)
    if len(events) > 1:
        deduped = [events[0]]
        for ev in events[1:]:
            if ev.start_s - deduped[-1].start_s > 60:
                deduped.append(ev)
        events = deduped

    return events


# ─────────────────────────────────────────────
# Sleep Quality Scoring
# ─────────────────────────────────────────────

def compute_sleep_quality_score(sleep_windows: List[SleepWindow],
                                 arousal_events: List[ArousalEvent],
                                 total_sleep_s: float) -> dict:
    """
    Compute composite sleep quality score (0-100).

    Components:
        - Sleep architecture score (stage distribution)
        - HRV quality score
        - Arousal index score
        - Continuity score
    """
    if not sleep_windows or total_sleep_s <= 0:
        return {'sleep_quality_score': 50, 'grade': 'N/A'}

    stages = [w.stage for w in sleep_windows]
    n_windows = len(stages)

    # Stage proportions
    pct_deep = stages.count('Deep') / n_windows * 100
    pct_rem = stages.count('REM') / n_windows * 100
    pct_light = stages.count('Light') / n_windows * 100
    pct_wake = stages.count('Wake') / n_windows * 100

    # ── Architecture Score (0-35) ──
    # Ideal: 15-25% Deep, 20-25% REM, 45-55% Light, <5% Wake
    deep_score = min(35, pct_deep / 20 * 15) if pct_deep <= 30 else max(0, 35 - (pct_deep - 30))
    rem_score = min(20, pct_rem / 22 * 20) if pct_rem <= 30 else max(0, 20 - (pct_rem - 30))
    wake_score = max(0, 15 - pct_wake * 1.5)
    arch_score = min(35, deep_score + rem_score * 0.5 + wake_score * 0.5)

    # ── HRV Quality Score (0-30) ──
    mean_rmssd = np.mean([w.sdnn for w in sleep_windows if w.stage != 'Wake'])
    hrv_score = min(30, mean_rmssd / 50 * 30)

    # ── Arousal Index Score (0-25) ──
    sleep_hours = total_sleep_s / 3600
    arousal_index = len(arousal_events) / max(sleep_hours, 0.1)
    nightmares = sum(1 for e in arousal_events if e.event_type == 'Nightmare')

    if arousal_index < 5:
        arousal_score = 25
    elif arousal_index < 15:
        arousal_score = max(0, 25 - (arousal_index - 5) * 1.5)
    else:
        arousal_score = 0
    arousal_score -= nightmares * 3  # Penalty for nightmares
    arousal_score = max(0, arousal_score)

    # ── Continuity Score (0-10) ──
    # Penalise frequent stage transitions
    transitions = sum(1 for i in range(1, len(stages)) if stages[i] != stages[i - 1])
    continuity_score = max(0, 10 - transitions / n_windows * 20)

    total = arch_score + hrv_score + arousal_score + continuity_score
    total = round(min(100, max(0, total)), 1)

    if total >= 85:
        grade = 'Excellent'
    elif total >= 70:
        grade = 'Good'
    elif total >= 55:
        grade = 'Fair'
    elif total >= 40:
        grade = 'Poor'
    else:
        grade = 'Very Poor'

    # Arousal index (events per hour)
    ai = round(arousal_index, 2)

    return {
        'sleep_quality_score': total,
        'grade': grade,
        'pct_deep': round(pct_deep, 1),
        'pct_rem': round(pct_rem, 1),
        'pct_light': round(pct_light, 1),
        'pct_wake': round(pct_wake, 1),
        'arousal_index': ai,
        'n_nightmares': nightmares,
        'n_arousals': len(arousal_events),
        'architecture_score': round(arch_score, 1),
        'hrv_score': round(hrv_score, 1),
        'arousal_score': round(arousal_score, 1),
        'continuity_score': round(continuity_score, 1),
        'sleep_hours': round(sleep_hours, 2),
    }


# ─────────────────────────────────────────────
# Full Sleep Analysis Pipeline
# ─────────────────────────────────────────────

def analyze_sleep(record: dict, verbose: bool = True) -> dict:
    """Run full sleep analysis pipeline on a preprocessed record with features."""
    rr_clean = record.get('rr_clean', np.array([]))
    times_clean = record.get('times_clean', np.array([]))
    windowed = record.get('windowed_features', pd.DataFrame())
    baseline = record.get('stress_baseline', {})
    rid = record['record_id']

    if len(rr_clean) < 20 or windowed.empty:
        print(f"  Record {rid}: insufficient data for sleep analysis")
        return record

    # Sleep period: assume entire recording is sleep context for MIT-BIH
    # (Real scenario would use time-of-day; we use full recording)
    sleep_windows = classify_sleep_stages(windowed)

    # Arousal detection
    baseline_hr = baseline.get('baseline_hr', np.mean(60000.0 / rr_clean))
    arousal_events = detect_arousal_events(
        rr_clean, times_clean, sleep_windows, baseline_hr,
        arousal_threshold_pct=20.0,
        nightmare_threshold_pct=35.0
    )

    # Sleep quality score
    total_sleep_s = times_clean[-1] - times_clean[0] if len(times_clean) > 1 else 0
    quality = compute_sleep_quality_score(sleep_windows, arousal_events, total_sleep_s)

    record['sleep_windows'] = sleep_windows
    record['arousal_events'] = arousal_events
    record['sleep_quality'] = quality

    if verbose:
        print(f"  Record {rid}: Sleep Quality={quality['sleep_quality_score']}/100 "
              f"({quality['grade']}), "
              f"Deep={quality['pct_deep']}%, REM={quality['pct_rem']}%")
        print(f"  Arousals: {quality['n_arousals']} ({quality['arousal_index']}/hr), "
              f"Nightmares: {quality['n_nightmares']}")

    return record


if __name__ == "__main__":
    from load_data import generate_synthetic_record
    from preprocess import preprocess_record
    from features import compute_all_features
    from stress import analyze_stress

    rec = generate_synthetic_record(100, duration_s=600, mean_hr=65, hrv_std=45)
    rec = preprocess_record(rec, verbose=False)
    rec = compute_all_features(rec, verbose=False)
    rec = analyze_stress(rec, verbose=False)
    rec = analyze_sleep(rec, verbose=True)

    print(f"\nSleep quality breakdown: {rec['sleep_quality']}")
