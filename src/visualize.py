"""
visualize.py - All visualization functions for HRV analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import os
from typing import Optional, List


# ─── Styling ───────────────────────────────────────────────────────────────────

PALETTE = {
    'ecg': '#2196F3',
    'rpeak': '#F44336',
    'rr': '#4CAF50',
    'stress': '#FF5722',
    'sleep': '#9C27B0',
    'wake': '#FFC107',
    'deep': '#1565C0',
    'rem': '#43A047',
    'light': '#7E57C2',
    'arousal': '#E91E63',
    'poincare': '#00ACC1',
    'lf': '#FB8C00',
    'hf': '#039BE5',
}

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
})


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# ECG Plot with R-peaks
# ─────────────────────────────────────────────

def plot_ecg(record: dict, duration_s: float = 10.0, save_path: Optional[str] = None):
    """Plot ECG signal with annotated R-peaks."""
    ecg_df = record['ecg']
    fs = record['fs']
    r_peaks = record.get('r_peaks', np.array([]))
    rid = record['record_id']

    n_samples = int(duration_s * fs)
    ecg_segment = ecg_df.iloc[:n_samples]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ecg_segment['time_s'], ecg_segment['signal'],
            color=PALETTE['ecg'], lw=0.8, label='ECG (MLII)')

    # Mark R-peaks within segment
    peaks_in_seg = r_peaks[r_peaks < n_samples]
    if len(peaks_in_seg) > 0:
        ax.scatter(
            peaks_in_seg / fs,
            ecg_df['signal'].values[peaks_in_seg],
            color=PALETTE['rpeak'], s=30, zorder=5, label='R-peaks'
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(f'Record {rid} — ECG Signal with R-peaks (first {duration_s:.0f}s)')
    ax.legend(loc='upper right')

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# RR Tachogram
# ─────────────────────────────────────────────

def plot_rr_tachogram(record: dict, save_path: Optional[str] = None):
    """Plot RR interval series (tachogram) with stress events highlighted."""
    rr_clean = record.get('rr_clean', np.array([]))
    times_clean = record.get('times_clean', np.array([]))
    stress_events = record.get('stress_events', [])
    rid = record['record_id']

    if len(rr_clean) == 0:
        print(f"  No RR data for record {rid}")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # RR intervals
    ax1.plot(times_clean, rr_clean, color=PALETTE['rr'], lw=0.8)
    for ev in stress_events:
        ax1.axvspan(ev.start_s, ev.end_s, alpha=0.2, color=PALETTE['stress'], label='Stress')
    ax1.set_ylabel('RR Interval (ms)')
    ax1.set_title(f'Record {rid} — RR Tachogram')
    if stress_events:
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

    # Instantaneous HR
    hr = 60000.0 / rr_clean
    ax2.plot(times_clean, hr, color=PALETTE['stress'], lw=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heart Rate (bpm)')
    ax2.set_title('Instantaneous Heart Rate')

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Poincaré Plot
# ─────────────────────────────────────────────

def plot_poincare(record: dict, save_path: Optional[str] = None):
    """Poincaré plot (SD1/SD2 ellipse)."""
    rr_clean = record.get('rr_clean', np.array([]))
    features = record.get('features', {})
    rid = record['record_id']

    if len(rr_clean) < 3:
        return None

    rr1 = rr_clean[:-1]
    rr2 = rr_clean[1:]

    sd1 = features.get('sd1', np.std(rr2 - rr1) / np.sqrt(2))
    sd2 = features.get('sd2', np.std(rr_clean))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rr1, rr2, alpha=0.3, s=8, color=PALETTE['poincare'])

    # Identity line
    lims = [min(rr1.min(), rr2.min()) - 50, max(rr1.max(), rr2.max()) + 50]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='RR_n = RR_n+1')

    # SD1/SD2 ellipse
    center = (np.mean(rr1), np.mean(rr2))
    ellipse = Ellipse(center, width=2 * sd2, height=2 * sd1,
                      angle=45, fill=False, edgecolor='red', lw=2, label=f'SD1={sd1:.1f} SD2={sd2:.1f}')
    ax.add_patch(ellipse)

    ax.set_xlabel('RR_n (ms)')
    ax.set_ylabel('RR_n+1 (ms)')
    ax.set_title(f'Record {rid} — Poincaré Plot\nSD1={sd1:.2f} ms, SD2={sd2:.2f} ms')
    ax.legend()
    ax.set_aspect('equal')

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# PSD / Frequency Domain
# ─────────────────────────────────────────────

def plot_psd(record: dict, save_path: Optional[str] = None):
    """Plot Power Spectral Density with LF/HF bands."""
    freqs = record.get('psd_freqs')
    psd = record.get('psd_values')
    features = record.get('features', {})
    rid = record['record_id']

    if freqs is None or psd is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(freqs, psd, color='#424242', lw=1)

    # Shade bands
    vlf = (freqs >= 0.003) & (freqs < 0.04)
    lf = (freqs >= 0.04) & (freqs < 0.15)
    hf = (freqs >= 0.15) & (freqs < 0.40)

    ax.fill_between(freqs[vlf], psd[vlf], alpha=0.3, color='gray', label='VLF (0.003–0.04 Hz)')
    ax.fill_between(freqs[lf], psd[lf], alpha=0.4, color=PALETTE['lf'], label='LF (0.04–0.15 Hz)')
    ax.fill_between(freqs[hf], psd[hf], alpha=0.4, color=PALETTE['hf'], label='HF (0.15–0.40 Hz)')

    lf_hf = features.get('lf_hf_ratio', 'N/A')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (ms²/Hz)')
    ax.set_title(f'Record {rid} — HRV Power Spectral Density\nLF/HF = {lf_hf}')
    ax.set_xlim(0, 0.5)
    ax.legend()

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Windowed HRV Features
# ─────────────────────────────────────────────

def plot_hrv_over_time(record: dict, save_path: Optional[str] = None):
    """Plot key HRV features across time windows."""
    win_df = record.get('windowed_features', pd.DataFrame())
    rid = record['record_id']

    if win_df.empty:
        return None

    features_to_plot = [
        ('mean_hr', 'Mean HR (bpm)', PALETTE['stress']),
        ('rmssd', 'RMSSD (ms)', PALETTE['rr']),
        ('sdnn', 'SDNN (ms)', PALETTE['poincare']),
        ('lf_hf_ratio', 'LF/HF Ratio', PALETTE['lf']),
    ]
    features_to_plot = [(k, l, c) for k, l, c in features_to_plot if k in win_df.columns]

    n = len(features_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = win_df['window_start_s'] / 60  # Convert to minutes

    for ax, (feat, label, color) in zip(axes, features_to_plot):
        ax.plot(t, win_df[feat], color=color, lw=1.5)
        ax.fill_between(t, win_df[feat], alpha=0.15, color=color)
        ax.set_ylabel(label)
        ax.set_title(f'{label} over time')

    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Record {rid} — HRV Features over Time', fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Sleep Hypnogram
# ─────────────────────────────────────────────

def plot_hypnogram(record: dict, save_path: Optional[str] = None):
    """Plot sleep hypnogram with arousal events."""
    sleep_windows = record.get('sleep_windows', [])
    arousal_events = record.get('arousal_events', [])
    quality = record.get('sleep_quality', {})
    rid = record['record_id']

    if not sleep_windows:
        return None

    stage_order = {'Wake': 0, 'REM': 1, 'Light': 2, 'Deep': 3}
    stage_colors = {
        'Wake': PALETTE['wake'],
        'REM': PALETTE['rem'],
        'Light': PALETTE['light'],
        'Deep': PALETTE['deep'],
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})

    # Hypnogram
    for w in sleep_windows:
        y = stage_order.get(w.stage, 1)
        t_center = (w.start_s + w.end_s) / 2 / 60
        width = (w.end_s - w.start_s) / 60
        ax1.barh(y, width, left=w.start_s / 60, height=0.8,
                 color=stage_colors.get(w.stage, 'gray'), alpha=0.7)

    # Arousal markers
    for ev in arousal_events:
        color = PALETTE['arousal'] if ev.event_type == 'Nightmare' else PALETTE['wake']
        ax1.axvline(ev.start_s / 60, color=color, lw=1, alpha=0.7, linestyle='--')

    ax1.set_yticks(list(stage_order.values()))
    ax1.set_yticklabels(list(stage_order.keys()))
    ax1.set_xlabel('Time (min)')
    ax1.set_title(f'Record {rid} — Sleep Hypnogram\n'
                  f'Quality: {quality.get("sleep_quality_score", "N/A")}/100 '
                  f'({quality.get("grade", "")})')
    ax1.invert_yaxis()

    # HR overlay
    rr_clean = record.get('rr_clean', np.array([]))
    times_clean = record.get('times_clean', np.array([]))
    if len(rr_clean) > 0:
        hr = 60000.0 / rr_clean
        ax2.plot(times_clean / 60, hr, color=PALETTE['stress'], lw=0.8)
        ax2.set_ylabel('HR (bpm)')
        ax2.set_xlabel('Time (min)')
        ax2.set_title('Heart Rate')

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Model Comparison
# ─────────────────────────────────────────────

def plot_model_comparison(model_results: dict, record_id: int,
                           save_path: Optional[str] = None):
    """Plot model predictions vs actuals + metric comparison bar chart."""
    if not model_results or 'summary' not in model_results:
        return None

    summary = model_results['summary']
    hr_series = model_results.get('hr_series', np.array([]))

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # ── Predictions comparison ──
    ax_pred = fig.add_subplot(gs[0, :])
    colors = ['#1565C0', '#E53935', '#43A047']
    model_names = ['LSTM', 'GRU', 'ARIMA']

    for name, color in zip(model_names, colors):
        res = model_results.get(name, {})
        if 'predictions' in res and 'actuals' in res:
            n = min(200, len(res['actuals']))
            ax_pred.plot(res['actuals'][:n], '--', color='gray', alpha=0.5, lw=1, label='Actual')
            ax_pred.plot(res['predictions'][:n], color=color, lw=1.5, label=f'{name} pred')
            break  # Just one model for clarity

    # Add all model predictions on same axis
    for name, color in zip(model_names, colors):
        res = model_results.get(name, {})
        if 'predictions' in res:
            n = min(200, len(res['predictions']))
            ax_pred.plot(res['predictions'][:n], color=color, lw=1.2, alpha=0.8, label=f'{name}')

    ax_pred.set_title(f'Record {record_id} — Model Predictions vs Actual (first 200 steps)')
    ax_pred.set_xlabel('Time step')
    ax_pred.set_ylabel('HR (bpm)')
    ax_pred.legend()

    # ── RMSE bar chart ──
    ax_rmse = fig.add_subplot(gs[1, 0])
    if 'rmse' in summary.columns:
        summary['rmse'].plot(kind='bar', ax=ax_rmse, color=colors[:len(summary)])
        ax_rmse.set_title('RMSE Comparison')
        ax_rmse.set_ylabel('RMSE')
        ax_rmse.tick_params(axis='x', rotation=0)

    # ── MAE bar chart ──
    ax_mae = fig.add_subplot(gs[1, 1])
    if 'mae' in summary.columns:
        summary['mae'].plot(kind='bar', ax=ax_mae, color=colors[:len(summary)])
        ax_mae.set_title('MAE Comparison')
        ax_mae.set_ylabel('MAE')
        ax_mae.tick_params(axis='x', rotation=0)

    fig.suptitle(f'Record {record_id} — Model Comparison', fontsize=13)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Dashboard (All-in-One)
# ─────────────────────────────────────────────

def plot_dashboard(record: dict, model_results: dict = None,
                   save_path: Optional[str] = None):
    """Comprehensive HRV analysis dashboard."""
    rid = record['record_id']
    rr_clean = record.get('rr_clean', np.array([]))
    times_clean = record.get('times_clean', np.array([]))
    features = record.get('features', {})
    quality = record.get('sleep_quality', {})
    capacity = record.get('recovery_capacity', {})

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. ECG Snippet ──
    ax_ecg = fig.add_subplot(gs[0, :2])
    ecg_df = record['ecg']
    fs = record['fs']
    n_seg = int(10 * fs)
    ax_ecg.plot(ecg_df['time_s'].values[:n_seg], ecg_df['signal'].values[:n_seg],
                color=PALETTE['ecg'], lw=0.7)
    r_peaks = record.get('r_peaks', np.array([]))
    peaks_seg = r_peaks[r_peaks < n_seg]
    if len(peaks_seg) > 0:
        ax_ecg.scatter(peaks_seg / fs, ecg_df['signal'].values[peaks_seg],
                       color=PALETTE['rpeak'], s=20, zorder=5)
    ax_ecg.set_title('ECG Signal (10s)')
    ax_ecg.set_xlabel('Time (s)')

    # ── 2. Metrics Box ──
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_metrics.axis('off')
    metrics_text = (
        f"RECORD {rid}\n"
        f"─────────────────\n"
        f"Mean HR: {features.get('mean_hr', 'N/A')} bpm\n"
        f"RMSSD:   {features.get('rmssd', 'N/A')} ms\n"
        f"SDNN:    {features.get('sdnn', 'N/A')} ms\n"
        f"pNN50:   {features.get('pnn50', 'N/A')}%\n"
        f"LF/HF:   {features.get('lf_hf_ratio', 'N/A')}\n"
        f"SD1:     {features.get('sd1', 'N/A')} ms\n"
        f"SD2:     {features.get('sd2', 'N/A')} ms\n"
        f"─────────────────\n"
        f"Sleep:   {quality.get('sleep_quality_score', 'N/A')}/100\n"
        f"Grade:   {quality.get('grade', 'N/A')}\n"
        f"Recovery:{capacity.get('recovery_capacity', 'N/A')}"
    )
    ax_metrics.text(0.05, 0.95, metrics_text, va='top', ha='left',
                    family='monospace', fontsize=9, transform=ax_metrics.transAxes)

    # ── 3. RR Tachogram ──
    ax_rr = fig.add_subplot(gs[1, :2])
    if len(rr_clean) > 0:
        ax_rr.plot(times_clean / 60, rr_clean, color=PALETTE['rr'], lw=0.8)
        for ev in record.get('stress_events', []):
            ax_rr.axvspan(ev.start_s / 60, ev.end_s / 60, alpha=0.2, color=PALETTE['stress'])
        ax_rr.set_xlabel('Time (min)')
        ax_rr.set_ylabel('RR (ms)')
        ax_rr.set_title('RR Tachogram')

    # ── 4. Poincaré ──
    ax_p = fig.add_subplot(gs[1, 2])
    if len(rr_clean) > 3:
        rr1, rr2 = rr_clean[:-1], rr_clean[1:]
        ax_p.scatter(rr1, rr2, alpha=0.2, s=5, color=PALETTE['poincare'])
        ax_p.set_xlabel('RR_n (ms)')
        ax_p.set_ylabel('RR_n+1 (ms)')
        ax_p.set_title(f"Poincaré\nSD1={features.get('sd1','?')} SD2={features.get('sd2','?')}")
        ax_p.set_aspect('equal')

    # ── 5. PSD ──
    ax_psd = fig.add_subplot(gs[2, :2])
    freqs = record.get('psd_freqs')
    psd = record.get('psd_values')
    if freqs is not None:
        ax_psd.semilogy(freqs, psd, color='#424242', lw=0.8)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        ax_psd.fill_between(freqs[lf_mask], psd[lf_mask], alpha=0.4, color=PALETTE['lf'], label='LF')
        ax_psd.fill_between(freqs[hf_mask], psd[hf_mask], alpha=0.4, color=PALETTE['hf'], label='HF')
        ax_psd.set_xlim(0, 0.5)
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('PSD')
        ax_psd.set_title('Power Spectral Density')
        ax_psd.legend()

    # ── 6. Sleep Stage Distribution ──
    ax_sleep = fig.add_subplot(gs[2, 2])
    if quality:
        stages = ['Wake', 'Light', 'Deep', 'REM']
        pcts = [quality.get(f'pct_{s.lower()}', 0) for s in stages]
        colors = [PALETTE['wake'], PALETTE['light'], PALETTE['deep'], PALETTE['rem']]
        ax_sleep.pie(pcts, labels=stages, colors=colors, autopct='%1.0f%%',
                     startangle=90)
        ax_sleep.set_title(f"Sleep Stages\nQuality: {quality.get('sleep_quality_score','N/A')}/100")

    # ── 7. HR over time with events ──
    ax_hr = fig.add_subplot(gs[3, :])
    if len(rr_clean) > 0:
        hr = 60000.0 / rr_clean
        ax_hr.plot(times_clean / 60, hr, color=PALETTE['stress'], lw=0.7, label='HR')

        # Stress events
        for ev in record.get('stress_events', []):
            ax_hr.axvspan(ev.start_s / 60, ev.end_s / 60, alpha=0.2,
                          color=PALETTE['stress'], label='Stress')
        # Arousals
        for ev in record.get('arousal_events', []):
            c = PALETTE['arousal'] if ev.event_type == 'Nightmare' else PALETTE['wake']
            ax_hr.axvline(ev.start_s / 60, color=c, lw=1, alpha=0.7)

        ax_hr.set_xlabel('Time (min)')
        ax_hr.set_ylabel('HR (bpm)')
        ax_hr.set_title('Heart Rate Timeline (stress=orange, arousals=dashed)')

    fig.suptitle(f'HRV Analysis Dashboard — Record {rid}', fontsize=14, fontweight='bold')

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Batch Summary Plots
# ─────────────────────────────────────────────

def plot_multi_record_summary(all_features: pd.DataFrame,
                               save_path: Optional[str] = None):
    """Scatter/box plots comparing HRV features across multiple records."""
    if all_features.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    feat_pairs = [
        ('sdnn', 'rmssd', 'SDNN vs RMSSD'),
        ('lf_power', 'hf_power', 'LF vs HF Power'),
        ('sd1', 'sd2', 'SD1 vs SD2'),
        ('mean_hr', 'sdnn', 'Mean HR vs SDNN'),
        ('lf_hf_ratio', 'pnn50', 'LF/HF vs pNN50'),
        ('sample_entropy', 'dfa_alpha1', 'SampEn vs DFA-α1'),
    ]

    for ax, (x_col, y_col, title) in zip(axes, feat_pairs):
        if x_col in all_features.columns and y_col in all_features.columns:
            ax.scatter(all_features[x_col], all_features[y_col],
                       alpha=0.6, s=40, color=PALETTE['poincare'])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)

    fig.suptitle('Multi-Record HRV Feature Comparison', fontsize=13)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig


if __name__ == "__main__":
    from load_data import generate_synthetic_record
    from preprocess import preprocess_record
    from features import compute_all_features
    from stress import analyze_stress
    from sleep import analyze_sleep

    rec = generate_synthetic_record(100, duration_s=600, mean_hr=70, hrv_std=40)
    rec = preprocess_record(rec, verbose=False)
    rec = compute_all_features(rec, verbose=False)
    rec = analyze_stress(rec, verbose=False)
    rec = analyze_sleep(rec, verbose=False)

    os.makedirs('/tmp/test_figs', exist_ok=True)
    plot_ecg(rec, save_path='/tmp/test_figs/ecg.png')
    plot_rr_tachogram(rec, save_path='/tmp/test_figs/rr.png')
    plot_poincare(rec, save_path='/tmp/test_figs/poincare.png')
    plot_dashboard(rec, save_path='/tmp/test_figs/dashboard.png')
    print("Test plots saved to /tmp/test_figs/")
