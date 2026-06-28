"""
eda.py — Statistical multi-label EDA & visualization for PTSD Hebrew slang dataset.

Ported from parent eda.py.
Produces 8 PNGs under visuals/ and eda_tables.json.
All comments and docstrings are in English.
File I/O uses utf-8-sig encoding.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

# RTL Hebrew rendering helpers — applied to tick/legend labels only
try:
    import arabic_reshaper
    from bidi.algorithm import get_display as bidi_display

    def rtl(text: str) -> str:
        """Reshape and apply BiDi algorithm for correct Hebrew rendering in matplotlib."""
        return bidi_display(arabic_reshaper.reshape(text))

except ImportError:
    warnings.warn(
        "arabic_reshaper or python-bidi not installed. "
        "Hebrew labels may render incorrectly. "
        "Install: pip install arabic-reshaper python-bidi",
        stacklevel=2,
    )

    def rtl(text: str) -> str:  # type: ignore[misc]
        return text


import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG export
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Shared rcParams — Hebrew-safe font fallback chain
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans Hebrew", "David CLM", "Arial", "DejaVu Sans"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    }
)

SEED = 1240

# ---------------------------------------------------------------------------
# Paths & constants — sourced from config for pipeline consistency
# ---------------------------------------------------------------------------

from src.config import (  # noqa: E402
    PROJECT_ROOT,
    VISUALS_DIR,
    DATASET_OUTPUT_PATH,
    EDA_TABLES_PATH,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    TFIDF_CONFIG,
    LR_CONFIG,
    BASELINE_EVAL_PATH,
    EVAL_RESULTS_PATH,
    RANDOM_SEED,
    VAL_SIZE,
)

BASE_DIR = PROJECT_ROOT  # preserve original name used throughout this module

# Sanity thresholds for text length (plan §2.5)
_MIN_MEDIAN_WORDS = 8
_MIN_STD_WORDS = 3


# ---------------------------------------------------------------------------
# Save helper (plan §4.1)
# ---------------------------------------------------------------------------


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure at 300 DPI into visuals/ with tight bounding box."""
    VISUALS_DIR.mkdir(exist_ok=True)
    out = VISUALS_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved -> %s", out)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load JSON dataset and binarize multi-labels.

    Returns:
        df:      raw DataFrame with all original columns
        Y:       binary label matrix (n_samples × n_labels)
        classes: ordered list of label names
    """
    with open(path, encoding="utf-8-sig") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))
    df["char_count"] = df["text"].apply(lambda t: len(str(t)))
    df["label_count"] = df["labels"].apply(len)

    # Academic-checklist aliases — persona := platform, event_type := example_type
    df["persona"] = df["platform"]
    df["event_type"] = df["example_type"]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels"])
    classes: list[str] = list(mlb.classes_)

    return df, Y, classes


# ---------------------------------------------------------------------------
# Individual chart functions
# ---------------------------------------------------------------------------


def plot_label_marginals(Y: np.ndarray, classes: list[str]) -> None:
    """Bar chart of per-label marginal frequencies (plan §2.2)."""
    counts = Y.sum(axis=0)
    order = np.argsort(counts)[::-1]
    sorted_labels = [rtl(classes[i]) for i in order]
    sorted_counts = counts[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(sorted_labels, sorted_counts, color=sns.color_palette("muted", len(classes)))
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title("Label Marginal Frequencies")
    ax.bar_label(bars, padding=3, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "01_label_marginals.png")

    # Sanity check: warn about sparse labels
    for i, c in enumerate(sorted_counts):
        if c <= 6:
            logger.warning("Sparse label '%s' — only %d examples.", classes[order[i]], c)


def plot_label_cooccurrence(Y: np.ndarray, classes: list[str]) -> None:
    """Co-occurrence heatmap of label pairs (plan §2.3)."""
    cooc = Y.T @ Y  # shape: (n_labels, n_labels)
    rtl_classes = [rtl(c) for c in classes]
    cooc_df = pd.DataFrame(cooc, index=rtl_classes, columns=rtl_classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cooc_df,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Co-occurrence count"},
    )
    ax.set_title("Label Co-occurrence Matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "02_label_cooccurrence.png")


def plot_label_correlation(Y: np.ndarray, classes: list[str]) -> None:
    """Pearson correlation matrix between binary label vectors (phi coefficient for binary data)."""
    # np.corrcoef treats rows as variables; transpose so each label is a variable
    corr = np.corrcoef(Y.T)
    rtl_classes = [rtl(c) for c in classes]
    corr_df = pd.DataFrame(corr, index=rtl_classes, columns=rtl_classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson correlation (phi)"},
    )
    ax.set_title("Label Pearson Correlation Matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "02b_label_correlation.png")


def plot_label_cardinality(df: pd.DataFrame) -> None:
    """Histogram of label-set size per record, including 0 for hard-negatives (plan §2.4)."""
    counts = Counter(df["label_count"])
    x = sorted(counts.keys())
    y = [counts[k] for k in x]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(v) for v in x], y, color=sns.color_palette("pastel")[0])
    ax.set_xlabel("Number of labels per record")
    ax.set_ylabel("Record count")
    ax.set_title("Label-Set Cardinality Distribution")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for idx, yi in enumerate(y):
        ax.text(idx, yi + 0.3, str(yi), ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "03_cardinality.png")


def plot_length_by_platform(df: pd.DataFrame) -> None:
    """Violin plot of word count distribution per platform with sanity check (plan §2.5)."""
    platforms = sorted(df["platform"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in zip(
        axes, ["word_count", "char_count"], ["Word count", "Character count"]
    ):
        sns.violinplot(
            data=df,
            x="platform",
            y=col,
            order=platforms,
            hue="platform",
            hue_order=platforms,
            palette="Set2",
            legend=False,
            ax=ax,
            inner="quartile",
        )
        ax.set_title(f"{label} by Platform")
        ax.set_xlabel("Platform")
        ax.set_ylabel(label)

    fig.tight_layout()
    save_fig(fig, "04_length_by_platform.png")

    # Sanity gate: flag uniform-5-word collapse
    logger.info("[TEXT LENGTH SANITY]")
    for platform in platforms:
        sub = df[df["platform"] == platform]["word_count"]
        median_w = sub.median()
        std_w = sub.std()
        flag = ""
        if median_w < _MIN_MEDIAN_WORDS:
            flag += f" LOW MEDIAN ({median_w:.1f} < {_MIN_MEDIAN_WORDS})"
        if std_w < _MIN_STD_WORDS:
            flag += f" LOW STD ({std_w:.1f} < {_MIN_STD_WORDS})"
        if flag:
            logger.warning("%s: median=%.1f words  std=%.1f%s", platform, median_w, std_w, flag)
        else:
            logger.info("%s: median=%.1f words  std=%.1f", platform, median_w, std_w)


def plot_platform_x_type(df: pd.DataFrame) -> None:
    """Stacked bar: platform × example_type (plan §2.6)."""
    ct = pd.crosstab(df["platform"], df["example_type"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set3", edgecolor="white")
    ax.set_title("Platform × Example Type")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Record count")
    ax.legend(title="example_type", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    save_fig(fig, "05_platform_x_type.png")


def plot_severity_x_label(Y: np.ndarray, df: pd.DataFrame, classes: list[str]) -> None:
    """Heatmap: severity × label frequency (plan §2.6)."""
    severity_order = ["mild", "medium", "strong"]
    matrix = []
    for sev in severity_order:
        mask = (df["severity"] == sev).values
        matrix.append(Y[mask].sum(axis=0))
    mat_df = pd.DataFrame(matrix, index=severity_order, columns=[rtl(c) for c in classes])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mat_df, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Severity × Label Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "06_severity_x_label.png")


def plot_explicitness_x_label(Y: np.ndarray, df: pd.DataFrame, classes: list[str]) -> None:
    """Heatmap: explicitness × label frequency (plan §2.6)."""
    exp_values = sorted(df["explicitness"].dropna().unique())
    matrix = []
    for exp in exp_values:
        mask = (df["explicitness"] == exp).values
        matrix.append(Y[mask].sum(axis=0))
    mat_df = pd.DataFrame(matrix, index=exp_values, columns=[rtl(c) for c in classes])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mat_df, annot=True, fmt="d", cmap="Greens", linewidths=0.5, ax=ax)
    ax.set_title("Explicitness × Label Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "07_explicitness_x_label.png")


def plot_slang_top20(df: pd.DataFrame) -> None:
    """Top-20 slang token frequency bar chart (plan §2.7)."""
    all_slang: list[str] = [s for row in df["slang_used"] for s in row]
    coverage = df["slang_used"].apply(lambda x: len(x) > 0).mean()
    logger.info("Slang coverage: %.1f%% of records have non-empty slang_used.", coverage * 100)

    if not all_slang:
        logger.warning("No slang tokens found — skipping chart 08.")
        return

    top = Counter(all_slang).most_common(20)
    tokens, freqs = zip(*top)
    rtl_tokens = [rtl(t) for t in tokens]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(rtl_tokens, freqs, color=sns.color_palette("rocket_r", len(tokens)))
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top-20 Military Slang Tokens (coverage={coverage:.1%})")
    ax.bar_label(ax.containers[0], padding=3, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "08_slang_top20.png")


# ---------------------------------------------------------------------------
# EDA tables
# ---------------------------------------------------------------------------


def build_eda_tables(df: pd.DataFrame, Y: np.ndarray, classes: list[str]) -> dict:
    """Compile numeric summary tables for report (plan §2.8)."""
    label_counts = {classes[i]: int(Y[:, i].sum()) for i in range(len(classes))}
    platform_counts = df["platform"].value_counts().to_dict()
    example_type_counts = df["example_type"].value_counts().to_dict()
    severity_counts = df["severity"].value_counts().to_dict()
    length_stats = (
        df[["word_count", "char_count"]]
        .describe()
        .round(2)
        .to_dict()
    )
    cardinality_dist = df["label_count"].value_counts().sort_index().to_dict()
    cardinality_dist = {int(k): int(v) for k, v in cardinality_dist.items()}

    return {
        "n_records": len(df),
        "label_marginals": {k: int(v) for k, v in sorted(label_counts.items(), key=lambda x: -x[1])},
        "platform_distribution": {str(k): int(v) for k, v in platform_counts.items()},
        "example_type_distribution": {str(k): int(v) for k, v in example_type_counts.items()},
        "severity_distribution": {str(k): int(v) for k, v in severity_counts.items()},
        "text_length_stats": {
            metric: {str(k): v for k, v in stat.items()}
            for metric, stat in length_stats.items()
        },
        "label_cardinality_distribution": cardinality_dist,
        "mean_label_cardinality": round(float(df["label_count"].mean()), 3),
        "slang_coverage_rate": round(
            float(df["slang_used"].apply(lambda x: len(x) > 0).mean()), 4
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TF-IDF + LR Baseline (moved here from modeling.py per supervisor feedback)
# ---------------------------------------------------------------------------


def run_tfidf_lr_baseline(
    train_path: Path = TRAIN_DATASET_PATH,
    test_path: Path = TEST_DATASET_PATH,
    baseline_eval_path: Path = BASELINE_EVAL_PATH,
) -> dict:
    """
    Train TF-IDF + OneVsRest(LogisticRegression) on split data.

    Evaluates on both train and test sets, saves baseline_eval.json, and
    generates charts 10 (per-label train/test F1) and 09 (model comparison).

    Silently skips if split data does not yet exist.
    """
    if not train_path.exists() or not test_path.exists():
        logger.warning(
            "Split data not found (%s / %s) — skipping TF-IDF baseline. "
            "Run Stage 4 first, then re-run Stage 3 for baseline charts.",
            train_path.name, test_path.name,
        )
        return {}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split as _split
    from sklearn.multiclass import OneVsRestClassifier

    with open(train_path, encoding="utf-8-sig") as f:
        train_data = json.load(f)
    with open(test_path, encoding="utf-8-sig") as f:
        test_data = json.load(f)

    mlb = MultiLabelBinarizer()
    texts_train = [r["text"] for r in train_data]
    y_train = mlb.fit_transform([r["labels"] for r in train_data])
    texts_test = [r["text"] for r in test_data]
    y_test = mlb.transform([r["labels"] for r in test_data])
    classes = list(mlb.classes_)

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    clf = OneVsRestClassifier(LogisticRegression(**LR_CONFIG), n_jobs=-1)
    clf.fit(X_train, y_train)

    # Threshold tuning on a validation carve-out from training data
    X_tr, X_val, y_tr, y_val = _split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    thresholds = np.full(len(classes), 0.5)
    proba_val = clf.predict_proba(X_val)
    for i in range(len(classes)):
        best_f1, best_t = -1.0, 0.5
        for t in [j * 0.1 for j in range(1, 10)]:
            pred = (proba_val[:, i] >= t).astype(int)
            score = f1_score(y_val[:, i], pred, zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, t
        thresholds[i] = best_t

    def _eval(X, y):
        proba = clf.predict_proba(X)
        pred = (proba >= thresholds).astype(int)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "f1_micro": float(f1_score(y, pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "precision_micro": float(precision_score(y, pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y, pred, average="micro", zero_division=0)),
            "per_label_f1": {
                classes[i]: float(f1_score(y[:, i], pred[:, i], zero_division=0))
                for i in range(len(classes))
            },
            "per_label_precision": {
                classes[i]: float(precision_score(y[:, i], pred[:, i], zero_division=0))
                for i in range(len(classes))
            },
            "per_label_recall": {
                classes[i]: float(recall_score(y[:, i], pred[:, i], zero_division=0))
                for i in range(len(classes))
            },
        }

    train_metrics = _eval(X_train, y_train)
    test_metrics = _eval(X_test, y_test)

    result = {
        "train": train_metrics,
        "test": test_metrics,
        "classes": classes,
        "thresholds": list(map(float, thresholds)),
    }
    with open(baseline_eval_path, "w", encoding="utf-8-sig") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("TF-IDF baseline results → %s", baseline_eval_path)

    # Generate baseline charts
    plot_tfidf_splits(train_metrics["per_label_f1"], test_metrics["per_label_f1"], classes)
    plot_model_comparison(tfidf_f1_micro=test_metrics["f1_micro"])

    return result


def plot_tfidf_splits(
    train_per_label: dict,
    test_per_label: dict,
    classes: list,
) -> None:
    """Per-label F1: TF-IDF train vs test comparison → visuals/10_tfidf_splits.png."""
    train_f1 = [train_per_label.get(c, 0.0) for c in classes]
    test_f1 = [test_per_label.get(c, 0.0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35
    palette = sns.color_palette("muted")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, train_f1, width, label="Train F1",
           color=palette[0], alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, test_f1, width, label="Test F1",
           color=palette[1], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([rtl(c) for c in classes], rotation=35, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.09)
    ax.set_title("TF-IDF + LR — Per-Label F1: Train vs Test")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    save_fig(fig, "10_tfidf_splits.png")


def plot_model_comparison(tfidf_f1_micro: float | None = None) -> None:
    """Bar chart comparing model F1-Micro scores → visuals/09_model_comparison.png.

    Reads finetuned_test from eval_results.json if available.
    Can be called with just tfidf_f1_micro before neural models run.
    """
    models: dict[str, float] = {}

    if tfidf_f1_micro is not None:
        models["TF-IDF + LR"] = tfidf_f1_micro
    elif BASELINE_EVAL_PATH.exists():
        with open(BASELINE_EVAL_PATH, encoding="utf-8-sig") as f:
            _be = json.load(f)
        models["TF-IDF + LR"] = _be.get("test", {}).get("f1_micro", 0.0)

    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, encoding="utf-8-sig") as f:
            eval_res = json.load(f)
        if eval_res.get("gpt4o_mini_test"):
            models["GPT-4o-mini"] = eval_res["gpt4o_mini_test"].get("f1_micro", 0.0)
        if eval_res.get("finetuned_test"):
            models["Fine-tuned SBERT"] = eval_res["finetuned_test"].get("f1_micro", 0.0)
        if eval_res.get("alephbert_test"):
            models["Fine-tuned AlephBERT"] = eval_res["alephbert_test"].get("f1_micro", 0.0)

    if not models:
        logger.warning("No model metrics available for chart 09 — skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.5), 4))
    palette = sns.color_palette("muted", len(models))
    bars = ax.bar(list(models.keys()), list(models.values()),
                  color=palette, edgecolor="white", width=0.55)
    ax.set_ylabel("F1 Micro (test set)")
    ax.set_ylim(0, 1.09)
    ax.set_title("Model Comparison — F1 Micro on Test Set")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    save_fig(fig, "09_model_comparison.png")


def plot_perlabel_finetuned() -> None:
    """Per-label P/R/F1 for fine-tuned model → visuals/11_perlabel_finetuned.png.

    Reads from eval_results.json (written by modeling.py after Stage 5).
    Silently skips if the file or finetuned_test key is absent.
    """
    if not EVAL_RESULTS_PATH.exists():
        logger.warning("eval_results.json not found — skipping chart 11 (run Stage 5 first).")
        return

    with open(EVAL_RESULTS_PATH, encoding="utf-8-sig") as f:
        eval_res = json.load(f)

    ft = eval_res.get("finetuned_test", {})
    per_label_f1 = ft.get("per_label_f1", {})
    per_label_p = ft.get("per_label_precision", {})
    per_label_r = ft.get("per_label_recall", {})

    if not per_label_f1:
        logger.warning("No per-label metrics in eval_results.json — skipping chart 11.")
        return

    classes = sorted(per_label_f1.keys())
    f1_vals = [per_label_f1.get(c, 0.0) for c in classes]
    p_vals = [per_label_p.get(c, 0.0) for c in classes]
    r_vals = [per_label_r.get(c, 0.0) for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    palette = sns.color_palette("muted")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, p_vals, width, label="Precision", color=palette[2], alpha=0.85, edgecolor="white")
    ax.bar(x,         r_vals, width, label="Recall",    color=palette[3], alpha=0.85, edgecolor="white")
    ax.bar(x + width, f1_vals, width, label="F1",       color=palette[0], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([rtl(c) for c in classes], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.09)
    ax.set_title("Fine-Tuned SBERT — Per-Label Precision / Recall / F1")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    save_fig(fig, "11_perlabel_finetuned.png")


def run_eda_pipeline(
    input_path: str | Path = DATASET_OUTPUT_PATH,
    tables_path: str | Path = EDA_TABLES_PATH,
) -> dict:
    """
    Full EDA pass: load data, produce all charts, persist numeric tables.

    Reads directly from Stage 1's dataset.json (no intermediate quality-judge stage).
    After the core charts, attempts to run TF-IDF baseline if split data exists.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{input_path}'. Run Stage 1 (src/data_generation.py) first."
        )

    logger.info("Loading %s …", input_path.name)
    df, Y, classes = load_dataset(input_path)
    logger.info("Records: %d | Labels: %d | Columns: %s", len(df), len(classes), list(df.columns))

    logger.info("Generating charts …")
    plot_label_marginals(Y, classes)
    plot_label_cooccurrence(Y, classes)
    plot_label_correlation(Y, classes)
    plot_label_cardinality(df)
    plot_length_by_platform(df)
    plot_platform_x_type(df)
    plot_severity_x_label(Y, df, classes)
    plot_explicitness_x_label(Y, df, classes)
    plot_slang_top20(df)

    tables = build_eda_tables(df, Y, classes)
    tables_path = Path(tables_path)
    with open(tables_path, "w", encoding="utf-8-sig") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    logger.info("Tables -> %s", tables_path)

    # TF-IDF baseline (runs only if split data from Stage 4 exists)
    run_tfidf_lr_baseline()

    # Per-label fine-tuned chart (runs only if Stage 5 eval_results.json exists)
    plot_perlabel_finetuned()

    logger.info("All charts saved to %s/", VISUALS_DIR)

    return tables


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-label EDA for PTSD Hebrew dataset")
    parser.add_argument("--input", default=str(DATASET_OUTPUT_PATH))
    parser.add_argument("--tables", default=str(BASE_DIR / "eda_tables.json"))
    args = parser.parse_args()

    run_eda_pipeline(input_path=args.input, tables_path=args.tables)
