"""
load_data.py - Load MIT-BIH Arrhythmia Database files (CSV + JSON format)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


# MIT-BIH record IDs (48 records)
RECORD_IDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234
]

SAMPLING_RATE = 360  # Hz


def load_ecg_csv(record_id: int, data_dir: str) -> pd.DataFrame:
    """
    Load ECG signal from CSV file.
    Expected columns: index, MLII (or V1/V2/V4/V5), symbol
    """
    filepath = os.path.join(data_dir, f"{record_id}_ekg.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ECG file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Identify the lead column (prefer MLII)
    lead_candidates = ['MLII', 'V5', 'V1', 'V2', 'V4']
    lead_col = None
    for col in lead_candidates:
        if col in df.columns:
            lead_col = col
            break

    if lead_col is None:
        # Use second column as signal
        lead_col = df.columns[1]

    df = df.rename(columns={lead_col: 'signal'})

    # Ensure index column
    if 'index' not in df.columns:
        df.insert(0, 'index', range(len(df)))

    # Add time column (seconds)
    df['time_s'] = df['index'] / SAMPLING_RATE

    # Keep symbol if present
    if 'symbol' not in df.columns:
        df['symbol'] = ''

    return df[['index', 'time_s', 'signal', 'symbol']]

def load_annotations_json(record_id: int, data_dir: str) -> pd.DataFrame:

    filepath = os.path.join(data_dir, f"{record_id}_annotations_1.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Annotations file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract fs
    fs = float(data.get("fs", SAMPLING_RATE))

    # ----------------------------
    # FIX: Handle stringified lists
    # ----------------------------
    samples = data.get("sample", [])
    symbols = data.get("symbol", [])

    # If samples is string like "[1,2,3]"
    if isinstance(samples, str):
        samples = json.loads(samples)

    if isinstance(data.get("subtype", None), str):
        data["subtype"] = json.loads(data["subtype"])

    # Build DataFrame
    ann_df = pd.DataFrame({
        "sample": samples,
        "symbol": symbols
    })

    # Convert to numeric
    ann_df["sample"] = pd.to_numeric(ann_df["sample"], errors="coerce")
    ann_df = ann_df.dropna(subset=["sample"])

    ann_df["time_s"] = ann_df["sample"] / fs

    return (
        ann_df[["sample", "time_s", "symbol"]]
        .sort_values("sample")
        .reset_index(drop=True)
    )

def load_record(record_id: int, data_dir: str) -> dict:
    """
    Load both ECG signal and annotations for a given record.
    Returns dict with 'ecg', 'annotations', 'record_id', 'fs'
    """
    ecg = load_ecg_csv(record_id, data_dir)
    try:
        annotations = load_annotations_json(record_id, data_dir)
    except FileNotFoundError:
        print(f"  [Warning] No annotations for record {record_id}, using empty DataFrame")
        annotations = pd.DataFrame(columns=['sample', 'time_s', 'symbol'])

    return {
        'record_id': record_id,
        'fs': SAMPLING_RATE,
        'ecg': ecg,
        'annotations': annotations,
        'duration_s': len(ecg) / SAMPLING_RATE,
        'n_beats': len(annotations)
    }


def load_multiple_records(record_ids: list, data_dir: str, verbose: bool = True) -> dict:
    """Load multiple records, skipping missing files."""
    records = {}
    for rid in record_ids:
        try:
            rec = load_record(rid, data_dir)
            records[rid] = rec
            if verbose:
                print(f"  Loaded record {rid}: {rec['duration_s']:.1f}s, {rec['n_beats']} beats")
        except FileNotFoundError as e:
            if verbose:
                print(f"  Skipping record {rid}: {e}")
    return records


def generate_synthetic_record(record_id: int = 100, duration_s: float = 300.0,
                               mean_hr: float = 70.0, hrv_std: float = 30.0) -> dict:
    """
    Generate synthetic ECG + annotations for testing when real data unavailable.
    Creates realistic RR intervals with HRV.
    """
    np.random.seed(record_id)
    fs = SAMPLING_RATE
    n_samples = int(duration_s * fs)

    # --- Generate RR intervals ---
    mean_rr_s = 60.0 / mean_hr
    rr_intervals_s = np.random.normal(mean_rr_s, hrv_std / 1000, size=int(duration_s / mean_rr_s))
    rr_intervals_s = np.clip(rr_intervals_s, 0.3, 1.5)

    # Compute R-peak sample positions
    r_samples = np.cumsum((rr_intervals_s * fs).astype(int))
    r_samples = r_samples[r_samples < n_samples]

    # --- Generate simplified ECG signal ---
    t = np.arange(n_samples) / fs
    # Baseline wander
    ecg_signal = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    # QRS complexes
    for r in r_samples:
        width = int(0.04 * fs)
        for k in range(-width, width + 1):
            idx = r + k
            if 0 <= idx < n_samples:
                ecg_signal[idx] += 1.2 * np.exp(-0.5 * (k / (width / 2)) ** 2)
    # Noise
    ecg_signal += 0.02 * np.random.randn(n_samples)

    # Build DataFrames
    ecg_df = pd.DataFrame({
        'index': np.arange(n_samples),
        'time_s': t,
        'signal': ecg_signal,
        'symbol': ''
    })

    ann_df = pd.DataFrame({
        'sample': r_samples,
        'time_s': r_samples / fs,
        'symbol': 'N'
    })

    return {
        'record_id': record_id,
        'fs': fs,
        'ecg': ecg_df,
        'annotations': ann_df,
        'duration_s': duration_s,
        'n_beats': len(r_samples),
        'synthetic': True
    }


def save_processed(data: dict, output_dir: str, record_id: int):
    """Save processed RR intervals and features to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    if 'rr_ms' in data:
        rr_df = pd.DataFrame({'rr_ms': data['rr_ms'], 'time_s': data.get('rr_times', range(len(data['rr_ms'])))})
        rr_df.to_csv(os.path.join(output_dir, f"{record_id}_rr.csv"), index=False)
    if 'features' in data:
        feat_df = pd.DataFrame([data['features']])
        feat_df.to_csv(os.path.join(output_dir, f"{record_id}_features.csv"), index=False)


if __name__ == "__main__":
    # Demo: generate and show synthetic record
    rec = generate_synthetic_record(record_id=100, duration_s=60)
    print(f"Synthetic record 100: {rec['duration_s']}s, {rec['n_beats']} beats")
    print(rec['ecg'].head())
    print(rec['annotations'].head())
