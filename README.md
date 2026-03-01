# HRV Analysis Project
### MIT-BIH Arrhythmia Database — Complete Pipeline

> A production-ready Python pipeline for Heart Rate Variability (HRV) analysis. Covers signal processing, feature extraction, deep-learning forecasting, stress detection, sleep stage classification, and automated reporting.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Dataset](#dataset)
6. [Pipeline Stages](#pipeline-stages)
7. [HRV Features Reference](#hrv-features-reference)
8. [Models](#models)
9. [Stress Analysis](#stress-analysis)
10. [Sleep Analysis](#sleep-analysis)
11. [Visualization and Reports](#visualization-and-reports)
12. [Output Files](#output-files)
13. [CLI Reference](#cli-reference)
14. [Troubleshooting](#troubleshooting)
15. [References](#references)

---

## Overview

This project implements a complete end-to-end HRV analysis pipeline based on the MIT-BIH Arrhythmia Database (48 patient records, 360 Hz, ~30 minutes each). It is structured as a modular Python package that can be run on real data downloaded from Kaggle, or on built-in synthetic ECG data for testing without any downloads.

**Key capabilities:**
- R-peak detection (Pan-Tompkins algorithm + annotation-based)
- Full time-domain, frequency-domain, and nonlinear HRV feature extraction
- ARIMA, LSTM, and GRU heart rate forecasting with comparison metrics
- Rule-based stress detection with recovery capacity scoring
- Sleep stage classification (Wake / Light / Deep / REM) plus nightmare detection
- 8 automated publication-quality plots per record
- Automated PowerPoint presentation + comprehensive Word report via `generate_report.js`

---

## Project Structure

```
hrv_project/
│
├── data/
│   ├── raw/                    <- Place MIT-BIH files here
│   │   ├── 100_ekg.csv
│   │   ├── 100_annotations_1.json
│   │   └── ...
│   └── processed/              <- Auto-generated: RR intervals, features
│       ├── 100_rr.csv
│       ├── 100_features.csv
│       └── 100_windowed.csv
│
├── src/
│   ├── load_data.py            <- Load CSV/JSON + synthetic generator
│   ├── preprocess.py           <- R-peak detection, RR extraction, filtering
│   ├── features.py             <- Time/frequency/nonlinear HRV features
│   ├── models.py               <- ARIMA, LSTM, GRU forecasting
│   ├── stress.py               <- Stress detection + recovery analysis
│   ├── sleep.py                <- Sleep staging + arousal/nightmare detection
│   └── visualize.py            <- All plots + dashboard
│
├── results/
│   ├── figures/
│   │   ├── 100/
│   │   │   ├── dashboard.png
│   │   │   ├── ecg.png
│   │   │   ├── rr_tachogram.png
│   │   │   ├── poincare.png
│   │   │   ├── psd.png
│   │   │   ├── hrv_over_time.png
│   │   │   ├── hypnogram.png
│   │   │   └── model_comparison.png
│   │   └── multi_record_comparison.png
│   └── metrics/
│       ├── 100_features.csv
│       ├── 100_model_metrics.csv
│       ├── 100_sleep_quality.json
│       └── all_records_summary.csv
│
├── main.py                     <- Pipeline entry point
├── generate_report.js          <- PPT + Word report generator (Node.js)
├── requirements.txt
└── README.md
```

---

## Installation

### Python Dependencies

```bash
pip install -r requirements.txt
```

Contents of requirements.txt:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
statsmodels>=0.13.0
tensorflow>=2.8.0
```

TensorFlow is optional. If not installed, LSTM/GRU models automatically fall back to linear regression.

### Node.js Dependencies (for report generation)

```bash
npm install -g pptxgenjs docx react react-dom react-icons sharp
```

Verify installation:
```bash
node --version        # >= 16.0 required
npm list -g pptxgenjs docx react-icons sharp
```

---

## Quick Start

### Option 1 — Demo (no data required)

```bash
python main.py
```

Generates a synthetic ECG record, runs the full pipeline, and saves all plots and metrics.

### Option 2 — Real MIT-BIH Data

```bash
# Single record
python main.py --data-dir data --records 100

# Multiple records
python main.py --data-dir data --records 100 101 102 103

# All 48 records
python main.py --data-dir data --all-records

# Skip ML model training (much faster)
python main.py --data-dir data --records 100 --no-models
```

### Option 3 — Generate PPT + Report

After running main.py (which populates results/):

```bash
node generate_report.js
```

This creates:
- `results/HRV_Analysis_Report.pptx` — 15-slide PowerPoint presentation
- `results/HRV_Analysis_Report.docx` — comprehensive Word report

---

## Dataset

### MIT-BIH Arrhythmia Database

| Property | Value |
|----------|-------|
| Source | PhysioNet / Kaggle |
| Records | 48 (IDs: 100-234) |
| Sampling Rate | 360 Hz |
| Duration | ~30 min per record (~650,000 samples) |
| Leads | MLII (primary), V1, V2, V4, V5 |
| Format | CSV (signal) + JSON (annotations) |

### File Naming Convention

| File | Description |
|------|-------------|
| `{id}_ekg.csv` | ECG signal with columns: index, MLII, symbol |
| `{id}_annotations_1.json` | Beat annotations with keys: sample, symbol, description, fs |

### Beat Annotation Symbols

| Symbol | Meaning |
|--------|---------|
| N | Normal beat |
| L | Left bundle branch block |
| R | Right bundle branch block |
| A | Atrial premature beat |
| V | Premature ventricular contraction |
| / | Paced beat |
| E | Ventricular escape beat |

### All 48 Record IDs

```
100  101  102  103  104  105  106  107  108  109
111  112  113  114  115  116  117  118  119  121
122  123  124  200  201  202  203  205  207  208
209  210  212  213  214  215  217  219  220  221
222  223  228  230  231  232  233  234
```

---

## Pipeline Stages

```
Raw ECG CSV/JSON
      |
      v
[1] load_data.py        -> Load signal + annotations
      |
      v
[2] preprocess.py       -> R-peak detection -> RR intervals -> outlier removal -> resampling
      |
      v
[3] features.py         -> Time-domain + Freq-domain + Nonlinear HRV features (windowed)
      |
      v
[4] models.py           -> ARIMA + LSTM + GRU training + comparison
      |
      v
[5] stress.py           -> Baseline estimation -> stress event detection -> recovery scoring
      |
      v
[6] sleep.py            -> Stage classification -> arousal/nightmare detection -> quality score
      |
      v
[7] visualize.py        -> 8 plots per record + dashboard
      |
      v
[8] generate_report.js  -> PowerPoint + Word report
```

---

## HRV Features Reference

### Time-Domain

| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| Mean RR | mean(RR) ms | Average beat-to-beat interval |
| SDNN | std(RR) ms | Overall HRV — sympathetic + parasympathetic |
| RMSSD | sqrt(mean(dRR^2)) ms | Short-term HRV — parasympathetic (vagal) |
| NN50 | count(|dRR| > 50ms) | Number of large successive differences |
| pNN50 | NN50/N x 100% | Percentage — parasympathetic activity |
| Mean HR | 60000/mean(RR) bpm | Average heart rate |
| HR Range | max(HR) - min(HR) | Autonomic responsiveness |

Interpretation: High RMSSD indicates strong vagal tone (healthy, relaxed). Low SDNN may suggest reduced cardiac autonomic function.

### Frequency-Domain (Welch PSD, 4 Hz resampling)

| Band | Range | Meaning |
|------|-------|---------|
| VLF | 0.003-0.04 Hz | Thermoregulation, renin-angiotensin |
| LF | 0.04-0.15 Hz | Mixed sympathetic + parasympathetic (baroreceptor) |
| HF | 0.15-0.40 Hz | Parasympathetic / respiratory sinus arrhythmia |
| LF/HF Ratio | — | Sympathovagal balance (>2.0 = stress/sympathetic dominance) |
| LF nu / HF nu | normalized | Percentage of total LF+HF power |

### Nonlinear

| Feature | Description | Healthy Range |
|---------|-------------|--------------|
| SD1 | Poincare: short-term RR variability | 20-50 ms |
| SD2 | Poincare: long-term RR variability | 40-90 ms |
| SD1/SD2 | Balance of short vs long-term | ~0.5-0.7 |
| CSI | Cardiac Sympathetic Index (SD2/SD1) | Lower = more parasympathetic |
| CVI | Cardiac Vagal Index log10(SD1*SD2) | Higher = better vagal tone |
| Sample Entropy | Signal complexity/irregularity | Higher = healthier |
| DFA alpha1 | Short-term fractal correlation (4-16 beats) | ~1.0 in healthy |
| DFA alpha2 | Long-term fractal correlation (>16 beats) | ~0.8-1.0 |

---

## Models

### ARIMA
- Auto-regressive Integrated Moving Average
- Order selected automatically (p=0-3, q=0-3, d=1)
- One-step ahead rolling forecast on test set
- Fast, interpretable, no GPU required

### LSTM (Long Short-Term Memory)
- 2-layer LSTM (64 to 32 units) + Dropout (0.2)
- Look-back window: 20 beats
- Early stopping on validation loss
- Trained on 80% of HR series, tested on 20%

### GRU (Gated Recurrent Unit)
- 2-layer GRU (64 to 32 units) + Dropout (0.2)
- Same architecture as LSTM, fewer parameters
- Typically faster training, comparable accuracy

### Comparison Metrics

| Metric | Formula | Lower = Better |
|--------|---------|---------------|
| RMSE | sqrt(mean((y-yhat)^2)) | Yes |
| MAE | mean(|y-yhat|) | Yes |
| MAPE | mean(|y-yhat|/y)*100 | Yes |

---

## Stress Analysis

### Baseline Estimation
Resting HR is estimated as the 10th percentile of all instantaneous HR values. This represents the individual's minimum typical HR (excluding extreme outliers).

### Stress Detection Criteria

A window is flagged as stress if either condition holds:

| Condition | Threshold | Physiological Basis |
|-----------|-----------|---------------------|
| HR elevation | >= 115% of baseline HR | Sympathetic activation |
| HRV reduction | SDNN <= 70% of baseline SDNN | Loss of vagal modulation |

Minimum 2 consecutive flagged windows required to register a stress event.

### Stress Score (0-100)
```
Stress Score = HR Component   (0-40)
             + HRV Component  (0-40)
             + LF/HF Component (0-20)
```

### Recovery Classification

| Category | Recovery Time | Score Range |
|----------|--------------|-------------|
| Fast | < 60 seconds | 85-100 |
| Normal | 60-180 seconds | 60-85 |
| Slow | > 180 seconds | 10-60 |
| Incomplete | No recovery within window | 10 |

### Recovery Capacity

| Rating | Avg Recovery Score |
|--------|--------------------|
| Excellent | >= 75 |
| Good | 55-74 |
| Fair | 35-54 |
| Poor | < 35 |

---

## Sleep Analysis

### Stage Classification

Rule-based classifier applied to each 5-minute HRV window:

| Stage | HR | RMSSD | LF/HF | Meaning |
|-------|----|-------|-------|---------|
| Deep (N3) | < 58 bpm | > 45 ms | < 1.5 | Slow-wave sleep, highest vagal tone |
| REM | 60-74 bpm | 18-40 ms | 1-3.5 | Dreaming, mixed autonomic activity |
| Light (N1/N2) | 60-70 bpm | 25-45 ms | — | Transitional sleep |
| Wake | > 72 bpm | < 25 ms | — | Awake or arousal |

### Arousal Detection

| Event Type | Trigger | Severity |
|-----------|---------|---------|
| Brief Awakening | HR spike 15-30% above local baseline | Mild |
| Arousal | HR spike >= 30% above local baseline | Moderate |
| Nightmare | HR spike >= 35% above local baseline during REM | Severe |

### Sleep Quality Score (0-100)

| Component | Max Points | Measured By |
|-----------|-----------|-------------|
| Sleep Architecture | 35 | % Deep + % REM vs ideal |
| HRV Quality | 30 | Mean RMSSD during sleep |
| Arousal Index | 25 | Events per hour |
| Continuity | 10 | Stage transition frequency |

| Grade | Score |
|-------|-------|
| Excellent | >= 85 |
| Good | 70-84 |
| Fair | 55-69 |
| Poor | 40-54 |
| Very Poor | < 40 |

Ideal sleep distribution: ~20% Deep, ~22% REM, ~50% Light, < 8% Wake.

---

## Visualization and Reports

### Automated Plots (per record)

| File | Contents |
|------|---------|
| ecg.png | Raw ECG signal with R-peaks highlighted in red |
| rr_tachogram.png | RR interval series + instantaneous HR with stress periods shaded |
| poincare.png | RR_n vs RR_n+1 scatter plot with SD1/SD2 confidence ellipse |
| psd.png | Welch power spectral density with VLF/LF/HF bands shaded |
| hrv_over_time.png | RMSSD, SDNN, Mean HR, LF/HF ratio across 5-min windows |
| hypnogram.png | Sleep stage timeline + HR overlay |
| model_comparison.png | ARIMA vs LSTM vs GRU predictions + RMSE/MAE bar charts |
| dashboard.png | All-in-one 8-panel summary dashboard |

### Multi-Record Summary
| File | Contents |
|------|---------|
| multi_record_comparison.png | Scatter plots across records: SDNN vs RMSSD, LF vs HF, SD1 vs SD2, etc. |

### Report Generator (generate_report.js)

Run after main.py to generate:

| Output | Description |
|--------|-------------|
| HRV_Analysis_Report.pptx | 15-slide PowerPoint with charts, tables, sleep analysis, model metrics |
| HRV_Analysis_Report.docx | Full Word document with methodology, results, clinical interpretation |

---

## Output Files

```
results/
├── figures/
│   ├── {record_id}/
│   │   ├── dashboard.png
│   │   ├── ecg.png
│   │   ├── rr_tachogram.png
│   │   ├── poincare.png
│   │   ├── psd.png
│   │   ├── hrv_over_time.png
│   │   ├── hypnogram.png
│   │   └── model_comparison.png
│   └── multi_record_comparison.png
├── metrics/
│   ├── {id}_features.csv
│   ├── {id}_model_metrics.csv
│   ├── {id}_sleep_quality.json
│   └── all_records_summary.csv
├── HRV_Analysis_Report.pptx
└── HRV_Analysis_Report.docx

data/processed/
├── {id}_rr.csv
├── {id}_features.csv
└── {id}_windowed.csv
```

---

## CLI Reference

```
python main.py [OPTIONS]

  --data-dir PATH          Path to raw MIT-BIH data directory
  --records N [N ...]      Specific record IDs to process
  --all-records            Process all 48 MIT-BIH records
  --no-models              Skip ARIMA/LSTM/GRU training (faster)
  --synthetic-duration S   Duration of synthetic records in seconds (default: 600)
  --n-synthetic N          Number of synthetic records to generate (default: 1)
  --verbose                Enable verbose output

node generate_report.js [OPTIONS]

  --results-dir PATH       Path to results directory (default: ./results)
  --output-dir PATH        Where to save .pptx and .docx (default: ./results)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | Run: pip install -r requirements.txt |
| numpy has no attribute trapz | Already patched — uses np.trapezoid for numpy >= 2.0 |
| TensorFlow not found | Install: pip install tensorflow or use --no-models flag |
| FileNotFoundError for ECG | Ensure files are named {id}_ekg.csv in --data-dir |
| Empty RR intervals | Record may have only ectopic beats — check annotations file |
| node: command not found | Install Node.js from https://nodejs.org |
| pptxgenjs not found | Run: npm install -g pptxgenjs docx react react-dom react-icons sharp |
| Blank figures | Normal in server environments — Matplotlib backend is Agg, files saved to results/figures/ |

---

## References

1. Pan J, Tompkins WJ. A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, 1985; 32(3):230-236.

2. Task Force of the European Society of Cardiology. Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use. Circulation, 1996; 93(5):1043-1065.

3. Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 2000; 101(23):e215-e220.

4. Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 2001; 20(3):45-50.

5. Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology, 2000; 278(6):H2039-H2049.

6. Peng CK et al. Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. Chaos, 1995; 5(1):82-87.

---

MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/ and https://www.kaggle.com/datasets/taejoonpark/mit-bih-arrhythmia-database