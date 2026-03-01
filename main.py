"""
main.py - Full HRV Analysis Pipeline Runner (MIT-BIH Only)

Usage:
    python main.py --data-dir data/raw
    python main.py --data-dir data/raw --records 100 101 102
    python main.py --data-dir data/raw --all-records
    python main.py --data-dir data/raw --no-models
"""

import os
import sys
import argparse
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from load_data import (
    RECORD_IDS,
    load_record,
    load_multiple_records,
    save_processed
)
from preprocess import preprocess_record
from features import compute_all_features
from models import compare_models, save_model_results
from stress import analyze_stress
from sleep import analyze_sleep
from visualize import (
    plot_ecg, plot_rr_tachogram, plot_poincare, plot_psd,
    plot_hrv_over_time, plot_hypnogram, plot_model_comparison,
    plot_dashboard, plot_multi_record_summary
)

# ─────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

for d in [RESULTS_DIR, FIGURES_DIR, METRICS_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# Pipeline for a Single Record
# ─────────────────────────────────────────────

def run_pipeline_for_record(record: dict, run_models: bool = True,
                            verbose: bool = True) -> dict:

    rid = record['record_id']
    print(f"\n{'='*60}")
    print(f"  Processing Record {rid}")
    print(f"{'='*60}")

    t0 = time.time()

    # ── Step 1: Preprocess ──
    print("\n[1/6] Preprocessing...")
    record = preprocess_record(record, use_annotations=True, verbose=verbose)

    if len(record.get('rr_clean', [])) < 20:
        print(f"  SKIP: Insufficient RR data for record {rid}")
        return record

    # ── Step 2: Features ──
    print("\n[2/6] Computing HRV features...")
    record = compute_all_features(record, verbose=verbose)

    # ── Step 3: Models ──
    model_results = {}
    if run_models and len(record.get('rr_clean', [])) >= 100:
        print("\n[3/6] Training forecasting models...")
        model_results = compare_models(record['rr_clean'], verbose=verbose)
        record['model_results'] = model_results
        save_model_results(model_results, METRICS_DIR, rid)
    else:
        print("\n[3/6] Skipping model training")

    # ── Step 4: Stress Analysis ──
    print("\n[4/6] Stress analysis...")
    record = analyze_stress(record, verbose=verbose)

    # ── Step 5: Sleep Analysis ──
    print("\n[5/6] Sleep analysis...")
    record = analyze_sleep(record, verbose=verbose)

    # ── Step 6: Save Results ──
    print("\n[6/6] Saving results and generating plots...")

    save_processed({
        'rr_ms': record.get('rr_clean'),
        'rr_times': record.get('times_clean'),
        'features': record.get('features', {}),
    }, PROCESSED_DIR, rid)

    win_df = record.get('windowed_features', pd.DataFrame())
    if not win_df.empty:
        win_df.to_csv(os.path.join(PROCESSED_DIR, f"{rid}_windowed.csv"), index=False)

    quality = record.get('sleep_quality', {})
    if quality:
        q_path = os.path.join(METRICS_DIR, f"{rid}_sleep_quality.json")
        with open(q_path, 'w') as f:
            json.dump({k: v for k, v in quality.items() if not isinstance(v, dict)}, f, indent=2)

    figs_record_dir = os.path.join(FIGURES_DIR, str(rid))
    os.makedirs(figs_record_dir, exist_ok=True)

    try: plot_ecg(record, save_path=os.path.join(figs_record_dir, 'ecg.png'))
    except Exception as e: print(f"[warn] ECG plot failed: {e}")

    try: plot_rr_tachogram(record, save_path=os.path.join(figs_record_dir, 'rr_tachogram.png'))
    except Exception as e: print(f"[warn] Tachogram failed: {e}")

    try: plot_poincare(record, save_path=os.path.join(figs_record_dir, 'poincare.png'))
    except Exception as e: print(f"[warn] Poincaré failed: {e}")

    try: plot_psd(record, save_path=os.path.join(figs_record_dir, 'psd.png'))
    except Exception as e: print(f"[warn] PSD failed: {e}")

    try: plot_hrv_over_time(record, save_path=os.path.join(figs_record_dir, 'hrv_over_time.png'))
    except Exception as e: print(f"[warn] HRV plot failed: {e}")

    try: plot_hypnogram(record, save_path=os.path.join(figs_record_dir, 'hypnogram.png'))
    except Exception as e: print(f"[warn] Hypnogram failed: {e}")

    if model_results:
        try:
            plot_model_comparison(model_results, rid,
                                  save_path=os.path.join(figs_record_dir, 'model_comparison.png'))
        except Exception as e:
            print(f"[warn] Model comparison failed: {e}")

    try:
        plot_dashboard(record, model_results or None,
                       save_path=os.path.join(figs_record_dir, 'dashboard.png'))
    except Exception as e:
        print(f"[warn] Dashboard failed: {e}")

    elapsed = time.time() - t0
    print(f"\n✓ Record {rid} complete in {elapsed:.1f}s")

    return record


# ─────────────────────────────────────────────
# Summary Across Records
# ─────────────────────────────────────────────

def generate_summary_report(all_records: list):

    print("\n" + "="*60)
    print("  Generating Multi-Record Summary")
    print("="*60)

    rows = []
    for rec in all_records:
        if 'features' not in rec:
            continue

        row = {'record_id': rec['record_id']}
        row.update({k: v for k, v in rec['features'].items()
                    if not isinstance(v, (np.ndarray, list))})

        rows.append(row)

    if not rows:
        print("No records with features.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows).set_index('record_id')
    summary_path = os.path.join(METRICS_DIR, 'all_records_summary.csv')
    summary_df.to_csv(summary_path)

    print(f"Summary saved: {summary_path}")
    print(f"Records analyzed: {len(summary_df)}")

    try:
        plot_multi_record_summary(
            summary_df.reset_index(),
            save_path=os.path.join(FIGURES_DIR, 'multi_record_comparison.png')
        )
    except Exception as e:
        print(f"[warn] Multi-record plot failed: {e}")

    return summary_df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='HRV Analysis Pipeline (MIT-BIH)')
    parser.add_argument('--data-dir', required=True,
                        help='Path to raw MIT-BIH data directory')
    parser.add_argument('--records', nargs='+', type=int, default=None,
                        help='Specific record IDs to process')
    parser.add_argument('--all-records', action='store_true',
                        help='Process all 48 MIT-BIH records')
    parser.add_argument('--no-models', action='store_true',
                        help='Skip ARIMA/LSTM/GRU model training')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("    HRV Analysis Pipeline — MIT-BIH Arrhythmia Database")
    print("="*60)
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Figures dir : {FIGURES_DIR}")
    print(f"Metrics dir : {METRICS_DIR}")

    run_models = not args.no_models

    # ── Determine records ──
    if args.all_records:
        record_ids = RECORD_IDS
    elif args.records:
        record_ids = args.records
    else:
        record_ids = RECORD_IDS  # Default → ALL records

    print(f"\nLoading {len(record_ids)} records from: {args.data_dir}")
    raw_records = load_multiple_records(record_ids, args.data_dir, verbose=args.verbose)
    records_to_process = list(raw_records.values())

    # ── Run Pipeline ──
    all_processed = []
    for rec in records_to_process:
        try:
            processed = run_pipeline_for_record(rec, run_models, args.verbose)
            all_processed.append(processed)
        except Exception as e:
            print(f"ERROR processing record {rec['record_id']}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    if len(all_processed) > 1:
        generate_summary_report(all_processed)

    print("\n" + "="*60)
    print("✅ Pipeline complete!")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Metrics: {METRICS_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()