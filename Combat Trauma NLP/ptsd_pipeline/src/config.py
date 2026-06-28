"""
config.py
=========
Centralised path and hyperparameter constants for the unified Project Sasha pipeline.

Every stage imports its paths and numeric constants from here so they stay consistent
across the full run_pipeline.py orchestration and when any stage is re-run standalone.

All comments and docstrings are in English.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent
PROJECT_ROOT = _SRC_DIR.parent

DATA_DIR      = PROJECT_ROOT / "data"
VISUALS_DIR   = PROJECT_ROOT / "visuals"
REPORTS_DIR   = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR      = PROJECT_ROOT / "logs"

# Ensure output directories exist whenever config is imported
for _d in (DATA_DIR, VISUALS_DIR, REPORTS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS_DIR / "pipeline_run.log"

# ---------------------------------------------------------------------------
# Stage 1 — Data Generation
# ---------------------------------------------------------------------------

DATASET_OUTPUT_PATH: Path = DATA_DIR / "dataset.json"

# ---------------------------------------------------------------------------
# Stage 3 — EDA
# ---------------------------------------------------------------------------

EDA_TABLES_PATH: Path    = ARTIFACTS_DIR / "eda_tables.json"
BASELINE_EVAL_PATH: Path = ARTIFACTS_DIR / "baseline_eval.json"  # TF-IDF+LR metrics produced during EDA

# ---------------------------------------------------------------------------
# Stage 3.5 — Validation
# ---------------------------------------------------------------------------

GOLDEN_DATASET_PATH:    Path = DATA_DIR      / "golden_dataset.json"   # user-provided authentic sentences
REALISM_TEST_PATH:      Path = ARTIFACTS_DIR / "realism_test_results.json"
ANNOTATION_EXPORT_PATH: Path = DATA_DIR      / "annotation_export.csv"
KAPPA_RESULTS_PATH:     Path = ARTIFACTS_DIR / "kappa_results.json"

# ---------------------------------------------------------------------------
# Stage 4 — Iterative Stratification Split
# ---------------------------------------------------------------------------

TRAIN_DATASET_PATH: Path   = DATA_DIR      / "train_dataset.json"
TEST_DATASET_PATH: Path    = DATA_DIR      / "test_dataset.json"
SPLIT_MANIFEST_PATH: Path  = ARTIFACTS_DIR / "split_manifest.json"

# Sanity gate tolerances for the stratified split (splitting.py §3.4).
# Per-label minimum/maximum fraction that must end up in the training fold.
SPLIT_TRAIN_SHARE_MIN: float = 0.65
SPLIT_TRAIN_SHARE_MAX: float = 0.85
# Maximum allowed Jensen-Shannon divergence between train and test label marginals.
SPLIT_JS_DIV_MAX: float = 0.05

# ---------------------------------------------------------------------------
# Stage 5 — Modeling
# ---------------------------------------------------------------------------

# Fraction of total held out for stratified test (stage 4)
STRAT_TEST_SIZE: float = 0.25

# Internal validation carve-out from the training fold (for threshold tuning)
VAL_SIZE: float = 0.15

# Batch size placeholder for future neural / LLM inference stages
BATCH_SIZE: int = 32

# TF-IDF settings tuned for Hebrew (right-to-left, no built-in stopwords)
TFIDF_CONFIG: dict[str, Any] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,       # Apply log(1+tf) scaling
    "encoding": "utf-8",
    "decode_error": "replace",
}

LR_CONFIG: dict[str, Any] = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "C": 1.0,
    "class_weight": "balanced",   # Compensates for label-frequency imbalance
    "random_state": 42,           # RANDOM_SEED value
}

EVAL_RESULTS_PATH: Path   = ARTIFACTS_DIR / "eval_results.json"
ERROR_EXPORT_PATH: Path   = ARTIFACTS_DIR / "misclassified_errors.xlsx"  # FP/FN export from fine-tuned model

# GPT-4o-mini baseline settings — OpenAI structured-output classification (no training)
GPT4OMINI_MODEL: str = "gpt-4o-mini"
GPT4OMINI_TEMPERATURE: float = 0.0
GPT4OMINI_TIMEOUT: float = 30.0
GPT4OMINI_MAX_RETRIES: int = 3

# Fine-tuned transformer settings — multilingual SBERT fine-tuned on the synthetic data
FINETUNE_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FINETUNE_EPOCHS: int = 10
FINETUNE_LR: float = 2e-5
FINETUNE_BATCH_SIZE: int = 16
FINETUNE_MAX_LEN: int = 128
FINETUNE_THRESHOLD: float = 0.3

# AlephBERT settings — native Hebrew transformer (dicta-il/alephbertgimmel-base)
# Strong Hebrew-specific baseline; trained on large Hebrew corpora.
ALEPHBERT_MODEL: str = "dicta-il/alephbertgimmel-base"
ALEPHBERT_EPOCHS: int = 10
ALEPHBERT_LR: float = 2e-5
ALEPHBERT_BATCH_SIZE: int = 16
ALEPHBERT_MAX_LEN: int = 128
ALEPHBERT_THRESHOLD: float = 0.3

# ---------------------------------------------------------------------------
# Stage 6 — Report
# ---------------------------------------------------------------------------

SLIDE3_PATH: Path     = REPORTS_DIR / "slide3_summary.md"
README_EDA_PATH: Path = REPORTS_DIR / "README_eda.md"

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42      # used by modeling internal train/val split and sklearn
STRAT_SEED: int = 1240     # used by iterative stratification (split.py) and EDA
