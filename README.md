# Intelligent Data Sampling Strategies for TabPFN v2

Evaluating 5 sampling strategies as context curators for TabPFN v2's fixed 10K-row context window. TabPFN v2 is a transformer-based foundation model for tabular classification with hard limits of 10,000 rows and 500 features. This project investigates which subsampling strategy best preserves model performance when large datasets must be reduced to fit within that context window.

## Sampling Strategies

| # | Strategy | What it does |
|---|---|---|
| 1 | Random | Uniform random baseline |
| 2 | Stratified | Class-proportional random |
| 3 | k-Center | Greedy spatial coverage (coreset) |
| 4 | Nearest-Enemy (NE) | Boundary-focused: selects points closest to the decision boundary via nearest opposite-class distance |
| 5 | Stratified k-Center | Class balance + spatial coverage (hybrid) |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Dry run: one dataset (credit-g), one seed — pipeline validation
python run_experiment_1.py --dry-run

# Experiment 1: 14 datasets × 4 seeds × 5 strategies = 280 runs (fixed 10K budget)
python run_experiment_1.py

# Experiment 2: scaling curves — 11 datasets × 4 budget fractions × 5 strategies × 4 seeds
python run_experiment_2.py

# Experiment 3: diversity vs size — 14 datasets × M in [1,3,5,10] × 2 inner strategies × 4 seeds
python run_experiment_3.py
```

## Analysis

```bash
python analysis/analyze_experiment_1.py   # ranking, Wilcoxon tests, win matrix
python analysis/analyze_experiment_2.py   # scaling curves, retention, min-budget table
python analysis/analyze_experiment_3.py   # diversity-vs-size figure, best-M table
python analysis/analyze_stability_timing.py  # prediction stability + timing analysis
python analysis/visualize_sampling.py     # PCA sampling visualizations (bank-marketing)
```

## Project Structure

```
├── run_experiment_1.py          # Entry point: strategy comparison at fixed size
├── run_experiment_2.py          # Entry point: subsample size scaling
├── run_experiment_3.py          # Entry point: diversity vs size (ensemble)
├── configs/config.py            # Global constants (seeds, datasets, limits)
├── preprocessing/
│   ├── data_loader.py           # OpenML dataset loading & caching
│   └── feature_selector.py     # Correlation + MI feature reduction
├── samplers/
│   ├── base.py                  # BaseSampler interface
│   ├── random_sampler.py        # Uniform random
│   ├── stratified_sampler.py    # Class-proportional
│   ├── coreset_sampler.py       # k-Center greedy
│   ├── prototype_sampler.py     # Nearest-Enemy (NE): boundary-focused via BallTree
│   └── stratified_coreset.py   # Stratified k-Center (hybrid)
├── experiments/
│   ├── experiment_1.py          # Fixed budget: 5 strategies × 14 datasets × 4 seeds
│   ├── experiment_2.py          # Scaling: budget fractions 10/25/50/100%
│   └── experiment_3.py          # Ensemble: M members × budget//M per member
├── analysis/
│   ├── metrics.py               # AUC-ROC computation
│   ├── statistical_tests.py     # Wilcoxon signed-rank tests
│   ├── analyze_experiment_1.py  # Exp1 analysis: ranking, pairwise tests
│   ├── analyze_experiment_2.py  # Exp2 analysis: scaling curves, retention
│   ├── analyze_experiment_3.py  # Exp3 analysis: diversity vs size
│   ├── analyze_stability_timing.py  # Prediction stability + timing
│   └── visualize_sampling.py   # PCA sampling visualizations
└── results/
    ├── figures/                 # PNG figures (300 dpi)
    ├── experiment_*_results.csv # Raw per-run results
    └── experiment_*_*.csv      # Analysis tables
```
