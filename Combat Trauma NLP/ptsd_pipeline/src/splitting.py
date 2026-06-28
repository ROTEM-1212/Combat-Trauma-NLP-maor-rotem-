"""
splitting.py — Multi-label stratified train/test split for PTSD Hebrew slang dataset.

Ported from parent split.py.
Uses Iterative Stratification (Sechidis et al., 2011) via scikit-multilearn.
All comments and docstrings are in English.
File I/O uses utf-8-sig encoding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Iterative stratification — guaranteed per-label presence in both folds
try:
    from skmultilearn.model_selection import iterative_train_test_split
    _SKMULTILEARN_AVAILABLE = True
except ImportError:
    _SKMULTILEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — paths sourced from config for pipeline consistency
# ---------------------------------------------------------------------------

from src.config import (  # noqa: E402
    DATASET_OUTPUT_PATH,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    SPLIT_MANIFEST_PATH,
    STRAT_SEED,
    STRAT_TEST_SIZE,
    SPLIT_TRAIN_SHARE_MIN,
    SPLIT_TRAIN_SHARE_MAX,
    SPLIT_JS_DIV_MAX,
)

SEED = STRAT_SEED
TEST_SIZE = STRAT_TEST_SIZE  # 75/25 split

# Sanity gate tolerances — sourced from config.py for pipeline-wide consistency
_TRAIN_SHARE_MIN = SPLIT_TRAIN_SHARE_MIN
_TRAIN_SHARE_MAX = SPLIT_TRAIN_SHARE_MAX
_JS_DIV_MAX = SPLIT_JS_DIV_MAX

# Synthetic label injected into the label matrix so that hard-negative records
# (which have no real labels) can be stratified by scikit-multilearn. The label
# is stripped from Y and from class lists immediately after splitting.
_NEG_LABEL = "__NEG__"


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence (label marginals)
# ---------------------------------------------------------------------------


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jensen-Shannon divergence between two probability vectors."""
    p = p + 1e-12
    q = q + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ---------------------------------------------------------------------------
# Fallback: random stratified split when scikit-multilearn is unavailable
# ---------------------------------------------------------------------------


def _random_stratified_split(
    indices: np.ndarray,
    Y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic random split used only if scikit-multilearn is not installed.
    Logs a warning — iterative stratification is strongly preferred.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(indices))
    n_test = max(1, int(len(indices) * test_size))
    test_idx = shuffled[:n_test]
    train_idx = shuffled[n_test:]
    return indices[train_idx], indices[test_idx]


# ---------------------------------------------------------------------------
# Sanity gates
# ---------------------------------------------------------------------------


def _run_sanity_gates(
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    classes: list[str],
) -> float:
    """
    Assert post-split quality constraints. Raises RuntimeError on any failure.
    Returns the JS divergence value so the caller can record it in the manifest.

    Gates (plan §3.4):
      1. Every label present at least once in test fold.
      2. Per-label train share in [_TRAIN_SHARE_MIN, _TRAIN_SHARE_MAX].
      3. JS divergence between train and test marginal distributions < _JS_DIV_MAX.
    """
    errors: list[str] = []

    test_sums = Y_test.sum(axis=0)
    train_sums = Y_train.sum(axis=0)

    # Gate 1: every label appears in test
    absent = [classes[i] for i in range(len(classes)) if test_sums[i] == 0]
    if absent:
        errors.append(f"GATE 1 FAIL — labels absent from test: {absent}")

    # Gate 2: per-label train share
    total = train_sums + test_sums
    for i, label in enumerate(classes):
        if total[i] == 0:
            continue
        share = train_sums[i] / total[i]
        if not (_TRAIN_SHARE_MIN <= share <= _TRAIN_SHARE_MAX):
            errors.append(
                f"GATE 2 FAIL — '{label}' train share={share:.2f} "
                f"(expected [{_TRAIN_SHARE_MIN}, {_TRAIN_SHARE_MAX}])"
            )

    # Gate 3: JS divergence
    p_train = train_sums / max(Y_train.shape[0], 1)
    p_test = test_sums / max(Y_test.shape[0], 1)
    js = _js_divergence(p_train.copy(), p_test.copy())
    if js >= _JS_DIV_MAX:
        errors.append(
            f"GATE 3 FAIL — JS divergence={js:.4f} (max allowed {_JS_DIV_MAX})"
        )

    if errors:
        msg = "\n".join(errors)
        raise RuntimeError(f"Sanity gates failed:\n{msg}")

    logger.info(
        "[SANITY OK] All gates passed. JS divergence=%.4f  Train share range=[%s]",
        js, train_sums / (train_sums + test_sums + 1e-12),
    )
    return js


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------


def run_split_pipeline(
    input_path: str | Path = DATASET_OUTPUT_PATH,
    train_path: str | Path = TRAIN_DATASET_PATH,
    test_path: str | Path = TEST_DATASET_PATH,
    manifest_path: str | Path = SPLIT_MANIFEST_PATH,
) -> dict:
    """
    Iterative-stratification 75/25 split preserving all original keys.

    Reads directly from Stage 1's dataset.json (no intermediate quality-judge stage).

    Strategy for empty-label rows (hard-negatives):
      Temporarily inject __NEG__ as a 9th class so the stratifier
      can balance the hard-negative pool. Strip it from output records.

    Returns the split_manifest dict.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{input_path}'. Run Stage 1 (src/data_generation.py) first."
        )

    with open(input_path, encoding="utf-8-sig") as f:
        dataset: list[dict] = json.load(f)

    logger.info("Loaded %d records from %s", len(dataset), input_path.name)

    # Inject __NEG__ for empty-label rows before binarization
    augmented_labels = [
        record["labels"] if record["labels"] else [_NEG_LABEL]
        for record in dataset
    ]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(augmented_labels)  # shape: (N, n_classes_with_NEG)
    classes_with_neg: list[str] = list(mlb.classes_)

    indices = np.arange(len(dataset)).reshape(-1, 1)

    if _SKMULTILEARN_AVAILABLE:
        logger.info("Using iterative_train_test_split (scikit-multilearn), seed=%d", SEED)
        np.random.seed(SEED)
        X_train_idx, Y_train, X_test_idx, Y_test = iterative_train_test_split(
            indices, Y, test_size=TEST_SIZE
        )
        train_indices = X_train_idx.flatten().tolist()
        test_indices = X_test_idx.flatten().tolist()
    else:
        logger.warning(
            "scikit-multilearn not installed. "
            "Falling back to random split — install with: pip install scikit-multilearn"
        )
        train_flat, test_flat = _random_stratified_split(
            np.arange(len(dataset)), Y, TEST_SIZE, SEED
        )
        train_indices = train_flat.tolist()
        test_indices = test_flat.tolist()
        Y_train = Y[train_indices]
        Y_test = Y[test_indices]

    # Strip __NEG__ column index from classes for gate checks
    real_class_indices = [
        i for i, c in enumerate(classes_with_neg) if c != _NEG_LABEL
    ]
    real_classes = [classes_with_neg[i] for i in real_class_indices]
    Y_train_real = Y_train[:, real_class_indices]
    Y_test_real = Y_test[:, real_class_indices]

    # Run sanity gates
    logger.info("Running sanity gates …")
    try:
        js = _run_sanity_gates(Y_train_real, Y_test_real, real_classes)
    except RuntimeError as exc:
        logger.error("Sanity gates failed: %s", exc)
        raise

    # Build output records (restore original labels — no __NEG__ in output)
    train_records = [dataset[i] for i in train_indices]
    test_records = [dataset[i] for i in test_indices]

    train_path = Path(train_path)
    test_path = Path(test_path)
    manifest_path = Path(manifest_path)

    with open(train_path, "w", encoding="utf-8-sig") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8-sig") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    # Per-label ratio report
    per_label_ratios = {}
    for i, label in enumerate(real_classes):
        total = int(Y_train_real[:, i].sum() + Y_test_real[:, i].sum())
        train_n = int(Y_train_real[:, i].sum())
        test_n = int(Y_test_real[:, i].sum())
        per_label_ratios[label] = {
            "total": total,
            "train": train_n,
            "test": test_n,
            "train_share": round(train_n / max(total, 1), 4),
        }

    try:
        from importlib.metadata import version as _pkg_version
        skmll_version = _pkg_version("scikit-multilearn")
    except Exception:
        skmll_version = "unknown"

    manifest = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_total": len(dataset),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "actual_test_ratio": round(len(test_records) / len(dataset), 4),
        "js_divergence_train_test": round(float(js) if isinstance(js, float) else 0.0, 4),
        "per_label_ratios": per_label_ratios,
        "stratifier": "iterative_train_test_split" if _SKMULTILEARN_AVAILABLE else "random_fallback",
        "skmultilearn_version": skmll_version,
        "neg_class_used_for_stratification": _NEG_LABEL,
    }

    with open(manifest_path, "w", encoding="utf-8-sig") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(
        "Done. Train=%d  Test=%d  JS=%.4f",
        len(train_records), len(test_records), manifest["js_divergence_train_test"],
    )
    logger.info("-> %s", train_path)
    logger.info("-> %s", test_path)
    logger.info("-> %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-label stratified split for PTSD Hebrew dataset"
    )
    parser.add_argument("--input", default=str(DATASET_OUTPUT_PATH))
    parser.add_argument("--train", default=str(TRAIN_DATASET_PATH))
    parser.add_argument("--test", default=str(TEST_DATASET_PATH))
    parser.add_argument("--manifest", default=str(SPLIT_MANIFEST_PATH))
    args = parser.parse_args()

    run_split_pipeline(
        input_path=args.input,
        train_path=args.train,
        test_path=args.test,
        manifest_path=args.manifest,
    )
