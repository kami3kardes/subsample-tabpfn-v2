SEEDS = [1, 2, 3, 4]
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
DEFAULT_TARGET_SIZE = 10_000       # v2 max context
MAX_CONTEXT = 10_000               # v2 limit
MAX_FEATURES = 500                 # v2 limit
TABPFN_N_ESTIMATORS = 1            # disable internal ensemble
TEST_MAX_SIZE = 10_000             # cap test set for tractable CPU inference
POOL_MAX_SIZE = 500_000            # pre-filter large pools before coreset/prototype
PCA_DIMS = 50
KCENTER_BATCH_SIZE = 10_000
CORRELATION_THRESHOLD = 0.95

# Datasets excluded from main aggregate rankings (kept for future use).
EXCLUDED_FROM_MAIN = []

# OpenML dataset IDs — from OpenML-CC18 / TabPFN evaluation suite
DATASETS = {
    # Small (<10K pool) — 2 datasets
    'credit-g':     31,
    'pendigits':    32,        # 10-class balanced, ~11K rows
    # Medium (10K-50K) — 4 datasets
    'phoneme':      1489,
    'mozilla4':     1046,
    'nomao':        1486,      # high-D binary, ~34K rows, 119 features
    'bank-marketing': 1461,
    # Large (50K+) — 8 datasets
    'adult':        1590,
    'volkert':      41166,     # high-D multiclass, ~58K rows, 180 features
    'connect-4':    40668,     # 3-class moderate imbalance, ~67K rows
    'covertype':    1596,
    'higgs':        23512,
    'jannis':       41168,
    'MiniBooNE':    41150,
    'numerai28.6':  23517,
}
