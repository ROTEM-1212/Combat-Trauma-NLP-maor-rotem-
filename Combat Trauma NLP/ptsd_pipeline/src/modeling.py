"""
modeling.py
===========
STEP 5 — Neural Model Training & Evaluation.

Runs three models against the same test set and writes all results to
artifacts/eval_results.json:

  1. API baseline : OpenAI gpt-4o-mini via structured-output chat completion.
                     No training — zero-shot multi-label classification baseline.

  2. Fine-tuned   : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (SBERT)
                    Mean-pool + linear head, BCEWithLogitsLoss, 10 epochs.

  3. Fine-tuned   : dicta-il/alephbertgimmel-base  (AlephBERT)
                    Native Hebrew BERT, same architecture as SBERT head, 10 epochs.

TF-IDF + LR baseline is handled exclusively by src/eda.py (run_tfidf_lr_baseline).

Pipeline stages
---------------
1. load_data()                – Read train/test JSON from splitting.py
2. preprocess()               – Binarise labels, carve validation split (reserved)
3. gpt4o_mini_baseline()      – OpenAI gpt-4o-mini structured-output inference on test set
4. fine_tune_transformer()    – SBERT fine-tuning + evaluation
5. fine_tune_alephbert()      – AlephBERT fine-tuning + evaluation (delegates to 4)
6. run_modeling_pipeline()    – Orchestrator; saves eval_results.json
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# gpu_check.py lives in the project root; make it importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from gpu_check import get_device  # noqa: E402  (intentional late import)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import (
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    VAL_SIZE,
    FINETUNE_MODEL,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FINETUNE_BATCH_SIZE,
    FINETUNE_MAX_LEN,
    FINETUNE_THRESHOLD,
    ALEPHBERT_MODEL,
    ALEPHBERT_EPOCHS,
    ALEPHBERT_LR,
    ALEPHBERT_BATCH_SIZE,
    ALEPHBERT_MAX_LEN,
    ALEPHBERT_THRESHOLD,
    RANDOM_SEED,
    EVAL_RESULTS_PATH,
    ERROR_EXPORT_PATH,
    GPT4OMINI_MODEL,
    GPT4OMINI_TEMPERATURE,
    GPT4OMINI_TIMEOUT,
    GPT4OMINI_MAX_RETRIES,
)

# ---------------------------------------------------------------------------
# Module-level classifier head (CC-1: defined once, reused by SBERT & AlephBERT)
# ---------------------------------------------------------------------------
# torch/nn are imported inside fine_tune_transformer() to keep the module
# importable without PyTorch installed.  _BERTClassifier uses TYPE_CHECKING
# style annotations so the class body doesn't require torch at import time.

class _BERTClassifier:  # becomes nn.Module at runtime inside fine_tune_transformer
    """
    Mean-pool BERT backbone + linear classification head.

    Instantiated inside fine_tune_transformer() after torch is confirmed
    available.  Defined at module level so it can be unit-tested and reused
    across SBERT and AlephBERT without code duplication.

    Architecture
    ------------
    token embeddings  →  attention-mask mean pooling  →  Linear(hidden, n_labels)
    """
    # Real __init__/forward are injected at runtime (see _make_bert_classifier below).


def _make_bert_classifier(nn_module, torch_mod):
    """
    Factory that returns a concrete nn.Module subclass using the already-imported
    torch and torch.nn references.  Called once per fine_tune_transformer() call.
    """
    import torch

    class _Impl(nn_module):
        def __init__(self, encoder, hidden_size: int, n_labels: int) -> None:
            super().__init__()
            self.encoder = encoder
            self.classifier = torch_mod.Linear(hidden_size, n_labels)

        @staticmethod
        def _mean_pool(token_embeddings, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(self._mean_pool(outputs.last_hidden_state, attention_mask))

    return _Impl


# ---------------------------------------------------------------------------
# Logging helper (CC-2: replaces duplicate metric log blocks)
# ---------------------------------------------------------------------------

def _log_metrics(label: str, m: dict) -> None:
    """Log a standard micro/macro metrics table for one model."""
    sep = "=" * 62
    logger.info(sep)
    logger.info("  EVALUATION RESULTS -- %s", label)
    logger.info(sep)
    logger.info("  %-28s%10s%10s", "Metric", "Micro", "Macro")
    logger.info("  %s", "-" * 48)
    logger.info("  %-28s%10.4f", "Accuracy", m.get("accuracy", float("nan")))
    for base in ("precision", "recall", "f1"):
        logger.info(
            "  %-28s%10.4f%10.4f",
            base.capitalize(),
            m.get(f"{base}_micro", float("nan")),
            m.get(f"{base}_macro", float("nan")),
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Stage 1: Data Loading
# ---------------------------------------------------------------------------

def load_data(
    train_path: Path = TRAIN_DATASET_PATH,
    test_path: Path = TEST_DATASET_PATH,
) -> tuple[list[dict], list[dict]]:
    """
    Load the train and test JSON datasets produced by splitting.py.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(train_data, test_data)`` — each element is a sample dict with at
        minimum the keys ``"text"`` (str) and ``"labels"`` (list[str]).

    Raises
    ------
    FileNotFoundError
        If either dataset file does not exist.
    """
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{path}'. "
                "Run 'python run_pipeline.py' or src/splitting.py first to produce the split."
            )

    with train_path.open(encoding="utf-8-sig") as fh:
        train_data = json.load(fh)
    with test_path.open(encoding="utf-8-sig") as fh:
        test_data = json.load(fh)

    logger.info(
        "Loaded %d train + %d test samples from '%s' / '%s'.",
        len(train_data), len(test_data), train_path.name, test_path.name,
    )
    return train_data, test_data


# ---------------------------------------------------------------------------
# Stage 2: Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    train_data: list[dict],
    test_data: list[dict],
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[
    list[str], list[str], list[str],        # texts_train, texts_val, texts_test
    np.ndarray, np.ndarray, np.ndarray,     # y_train, y_val, y_test
    MultiLabelBinarizer,                    # fitted binariser (needed for label names)
]:
    """
    Extract texts and labels from the raw data, binarise the label lists,
    and perform a two-way train / val split on the training fold.

    The test set is already held out by splitting.py; the validation set is
    carved from the train fold here and reserved for future threshold tuning.
    The current pipeline uses the fixed FINETUNE_THRESHOLD / ALEPHBERT_THRESHOLD
    config constants (0.3) as decision boundaries; adaptive tuning on this val
    set is a planned improvement.

    Parameters
    ----------
    train_data : list[dict]
        Training samples from splitting stage.
    test_data : list[dict]
        Test samples from splitting stage.
    val_size : float
        Fraction of training samples reserved for validation (default 0.15).
    random_seed : int
        Seed for reproducible splits.

    Returns
    -------
    tuple
        ``(texts_train, texts_val, texts_test, y_train, y_val, y_test, mlb)``
    """
    texts_all_train: list[str] = [s["text"] for s in train_data]
    labels_all_train: list[list[str]] = [s["labels"] for s in train_data]
    texts_test: list[str] = [s["text"] for s in test_data]
    labels_test: list[list[str]] = [s["labels"] for s in test_data]

    # Fit binariser on training labels; transform test labels with same schema
    mlb = MultiLabelBinarizer()
    y_all_train: np.ndarray = mlb.fit_transform(labels_all_train)
    y_test: np.ndarray = mlb.transform(labels_test)

    # Carve validation set from training fold — use iterative stratification when available
    try:
        from skmultilearn.model_selection import iterative_train_test_split as _itr_split
        import numpy as _np
        _idx = _np.arange(len(texts_all_train)).reshape(-1, 1)
        _np.random.seed(random_seed)
        _tr_idx, _y_tr, _va_idx, _y_va = _itr_split(_idx, y_all_train, test_size=val_size)
        texts_train = [texts_all_train[i] for i in _tr_idx.flatten().tolist()]
        texts_val   = [texts_all_train[i] for i in _va_idx.flatten().tolist()]
        y_train, y_val = _y_tr, _y_va
        logger.info("Val split: iterative stratification (scikit-multilearn).")
    except ImportError:
        logger.warning(
            "scikit-multilearn not installed — val split is not multi-label stratified. "
            "Install with: pip install scikit-multilearn"
        )
        texts_train, texts_val, y_train, y_val = train_test_split(
            texts_all_train, y_all_train,
            test_size=val_size,
            random_state=random_seed,
        )

    logger.info(
        "Train: %d | Val: %d | Test: %d | Labels (%d): %s",
        len(texts_train), len(texts_val), len(texts_test),
        len(mlb.classes_), list(mlb.classes_),
    )
    return texts_train, texts_val, texts_test, y_train, y_val, y_test, mlb



# ---------------------------------------------------------------------------
# Fine-tuned Transformer (Multilingual SBERT)
# ---------------------------------------------------------------------------

def fine_tune_transformer(
    texts_train: list[str],
    y_train: np.ndarray,
    texts_test: list[str],
    y_test: np.ndarray,
    mlb: MultiLabelBinarizer,
    model_id: str = FINETUNE_MODEL,
    num_epochs: int = FINETUNE_EPOCHS,
    learning_rate: float = FINETUNE_LR,
    batch_size: int = FINETUNE_BATCH_SIZE,
    max_length: int = FINETUNE_MAX_LEN,
    threshold: float = FINETUNE_THRESHOLD,
    device=None,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Fine-tune a multilingual SBERT-based transformer on the synthetic training
    data for multi-label PTSD symptom classification.

    Uses ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` by
    default. A classification head is randomly initialised on top of the
    pretrained backbone and trained end-to-end with BCEWithLogitsLoss.

    Parameters
    ----------
    texts_train, y_train : training texts and binary label matrix.
    texts_test, y_test   : held-out test texts and label matrix.
    mlb                  : fitted MultiLabelBinarizer (provides label names).
    model_id             : HuggingFace model hub identifier.
    num_epochs           : number of full training passes.
    learning_rate        : AdamW learning rate.
    batch_size           : per-device training and inference batch size.
    max_length           : tokeniser truncation / padding length.
    threshold            : sigmoid score threshold for positive prediction.
    device               : torch.device from the pipeline orchestrator.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        ``(metrics, y_pred)`` where ``metrics`` holds the evaluation scores and
        ``y_pred`` is the binary prediction matrix on the test set. On import or
        model-load failure, returns ``({}, empty array)`` so callers can still
        unpack the result safely.
    """
    _empty = np.empty((0, 0), dtype=int)
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        logger.warning(
            "'transformers' or 'torch' not installed — skipping fine-tuning. "
            "Run: pip install transformers torch"
        )
        return {}, _empty

    n_labels = y_train.shape[1]
    _device = device if device is not None else torch.device("cpu")

    # ---- Tokenise --------------------------------------------------------
    logger.info("Fine-tune: tokenising with '%s' (max_length=%d) …", model_id, max_length)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        logger.error("Fine-tune: failed to load tokenizer — %s", exc)
        return {}, _empty

    def _tokenise(texts: list[str]) -> dict:
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_enc = _tokenise(texts_train)
    test_enc  = _tokenise(texts_test)

    # ---- Dataset ---------------------------------------------------------
    class _MLDataset(Dataset):
        def __init__(self, enc: dict, labels: np.ndarray) -> None:
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels         = labels.astype(float)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            return {
                "input_ids":      torch.tensor(self.input_ids[idx],      dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
                "labels":         torch.tensor(self.labels[idx],         dtype=torch.float),
            }

    train_ds = _MLDataset(train_enc, y_train)
    test_ds  = _MLDataset(test_enc,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Model -----------------------------------------------------------
    logger.info("Fine-tune: loading backbone '%s' → %d-label head …", model_id, n_labels)

    # Build the concrete nn.Module class now that torch/nn are confirmed available.
    _ClassifierCls = _make_bert_classifier(nn.Module, nn)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _encoder = AutoModel.from_pretrained(model_id)
        model = _ClassifierCls(
            encoder=_encoder,
            hidden_size=_encoder.config.hidden_size,
            n_labels=n_labels,
        )
    except Exception as exc:
        logger.error("Fine-tune: failed to load model — %s", exc)
        return {}, _empty

    model = model.to(_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps  = len(train_loader) * num_epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ---- Training loop ---------------------------------------------------
    logger.info(
        "Fine-tune: training for %d epoch(s) on %d samples …",
        num_epochs, len(texts_train),
    )
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(_device)
            attention_mask = batch["attention_mask"].to(_device)
            labels         = batch["labels"].to(_device)

            optimizer.zero_grad()
            logits  = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        logger.info(
            "  Epoch %d/%d — avg loss: %.4f",
            epoch + 1, num_epochs, epoch_loss / len(train_loader),
        )

    # ---- Inference on test set -------------------------------------------
    model.eval()
    all_logits: list = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["input_ids"].to(_device),
                attention_mask=batch["attention_mask"].to(_device),
            )
            all_logits.append(logits.cpu())

    probs  = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
    y_pred = (probs >= threshold).astype(int)

    # ---- Metrics ---------------------------------------------------------
    def _score(fn, average):
        return fn(y_test, y_pred, average=average, zero_division=0)

    label_names = list(mlb.classes_)
    metrics: dict = {
        "accuracy":        float(accuracy_score(y_test, y_pred)),
        "precision_micro": _score(precision_score, "micro"),
        "precision_macro": _score(precision_score, "macro"),
        "recall_micro":    _score(recall_score,    "micro"),
        "recall_macro":    _score(recall_score,    "macro"),
        "f1_micro":        _score(f1_score,        "micro"),
        "f1_macro":        _score(f1_score,        "macro"),
        # Per-label breakdown — used by eda.py chart 11
        "per_label_f1": {
            label_names[i]: float(f1_score(y_test[:, i], y_pred[:, i], zero_division=0))
            for i in range(len(label_names))
        },
        "per_label_precision": {
            label_names[i]: float(precision_score(y_test[:, i], y_pred[:, i], zero_division=0))
            for i in range(len(label_names))
        },
        "per_label_recall": {
            label_names[i]: float(recall_score(y_test[:, i], y_pred[:, i], zero_division=0))
            for i in range(len(label_names))
        },
    }

    model_name = f"Fine-tuned ({model_id.split('/')[-1]})"
    sep = "=" * 62
    logger.info(sep)
    logger.info("  EVALUATION RESULTS — %s", model_name)
    logger.info(sep)
    logger.info("  %-28s%10s%10s", "Metric", "Micro", "Macro")
    logger.info("  %s", "-" * 48)
    for base in ("precision", "recall", "f1"):
        logger.info(
            "  %-28s%10.4f%10.4f",
            base.capitalize(), metrics[f"{base}_micro"], metrics[f"{base}_macro"],
        )
    logger.info(sep)

    logger.info("  Per-label Classification Report:")
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0)
    for line in report.splitlines():
        logger.info("  %s", line)

    return metrics, y_pred


# ---------------------------------------------------------------------------
# AlephBERT Fine-tuned Model (Native Hebrew Baseline)
# ---------------------------------------------------------------------------

def fine_tune_alephbert(
    texts_train: list[str],
    y_train: np.ndarray,
    texts_test: list[str],
    y_test: np.ndarray,
    mlb: MultiLabelBinarizer,
    model_id: str = ALEPHBERT_MODEL,
    num_epochs: int = ALEPHBERT_EPOCHS,
    learning_rate: float = ALEPHBERT_LR,
    batch_size: int = ALEPHBERT_BATCH_SIZE,
    max_length: int = ALEPHBERT_MAX_LEN,
    threshold: float = ALEPHBERT_THRESHOLD,
    device=None,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Fine-tune dicta-il/alephbertgimmel-base on the synthetic Hebrew training data.

    AlephBERT is a native Hebrew BERT model trained on large Hebrew corpora.
    It serves as a strong Hebrew-specific baseline to compare against the
    multilingual SBERT model. Uses the same mean-pooling + linear-head
    architecture and BCEWithLogitsLoss as fine_tune_transformer().

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        ``(metrics, y_pred)`` — same schema as fine_tune_transformer().
        Returns ``({}, empty array)`` on import or model-load failure.
    """
    logger.info(
        "AlephBERT: fine-tuning '%s' for %d epoch(s) …", model_id, num_epochs
    )
    # Delegate to fine_tune_transformer() — identical architecture, different backbone.
    metrics, y_pred = fine_tune_transformer(
        texts_train=texts_train,
        y_train=y_train,
        texts_test=texts_test,
        y_test=y_test,
        mlb=mlb,
        model_id=model_id,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        threshold=threshold,
        device=device,
    )
    return metrics, y_pred


# ---------------------------------------------------------------------------
# GPT-4o-mini Baseline (OpenAI structured-output chat completion)
# ---------------------------------------------------------------------------

# Human-readable label descriptions, surfaced to the model in the system prompt
_LABEL_DESCRIPTIONS: dict[str, str] = {
    "sleep_disturbance":    "sleep problems or nightmares",
    "hypervigilance":       "hypervigilance or anxiety",
    "avoidance":            "avoidance of people or places",
    "intrusive_memories":   "intrusive memories or flashbacks",
    "anger_irritability":   "anger or irritability",
    "emotional_numbing":    "emotional numbness or detachment",
    "guilt_shame":          "guilt or shame",
    "functional_impairment": "difficulty functioning in daily life",
}

_GPT4OMINI_SYSTEM_PROMPT = (
    "You are a clinical text classifier. Given a Hebrew sentence written by an "
    "Israeli (often containing military slang), decide which of the following "
    "PTSD-symptom indicators are present. A sentence may match zero, one, or "
    "several indicators:\n"
    + "\n".join(f"- {lbl}: {desc}" for lbl, desc in _LABEL_DESCRIPTIONS.items())
    + "\n\nRespond using the provided JSON schema only — one 0/1 field per indicator."
)


def _gpt4omini_response_schema(label_names: list[str]) -> dict:
    """Strict JSON schema for OpenAI structured outputs — one 0/1 field per label."""
    return {
        "name": "ptsd_label_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {lbl: {"type": "integer", "enum": [0, 1]} for lbl in label_names},
            "required": label_names,
            "additionalProperties": False,
        },
    }


def gpt4o_mini_baseline(
    texts_test: list[str],
    y_test: np.ndarray,
    mlb: MultiLabelBinarizer,
    model_id: str = GPT4OMINI_MODEL,
    temperature: float = GPT4OMINI_TEMPERATURE,
    timeout: float = GPT4OMINI_TIMEOUT,
    max_retries: int = GPT4OMINI_MAX_RETRIES,
) -> dict:
    """
    Zero-shot multi-label classification using OpenAI's gpt-4o-mini.

    Each test text is sent as a single chat completion request constrained by a
    strict JSON schema (OpenAI structured outputs), so every response parses
    into exactly the 8 label fields expected downstream — no free-text parsing
    or score-threshold tuning needed.

    Returns metric dict (accuracy, f1_micro, f1_macro, per_label_f1, …),
    matching the schema used by the fine-tuned models for direct comparison.
    Returns {} if 'openai' isn't installed or OPENAI_API_KEY is unset, so the
    pipeline can continue non-fatally.
    """
    import os
    import time

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("'openai' package not installed — skipping GPT-4o-mini baseline.")
        return {}

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping GPT-4o-mini baseline.")
        return {}

    label_names = list(mlb.classes_)
    schema = _gpt4omini_response_schema(label_names)
    # The SDK's own max_retries already covers transient 429 / 5xx errors with backoff.
    client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    logger.info(
        "GPT-4o-mini: classifying %d test samples with '%s' …",
        len(texts_test), model_id,
    )

    y_pred = np.zeros((len(texts_test), len(label_names)), dtype=int)
    for i, text in enumerate(texts_test):
        # Outer retry loop guards against malformed/empty responses surviving
        # the SDK's own network-level retries.
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": _GPT4OMINI_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    response_format={"type": "json_schema", "json_schema": schema},
                )
                parsed = json.loads(response.choices[0].message.content)
                for k, lbl in enumerate(label_names):
                    y_pred[i, k] = 1 if int(parsed.get(lbl, 0)) else 0
                break
            except Exception as exc:
                if attempt == max_retries:
                    logger.warning(
                        "GPT-4o-mini: sample %d failed after %d retries — %s",
                        i, max_retries, exc,
                    )
                else:
                    time.sleep(2 ** attempt)

    metrics: dict = {
        "accuracy":        float(accuracy_score(y_test, y_pred)),
        "f1_micro":        float(f1_score(y_test, y_pred, average="micro",  zero_division=0)),
        "f1_macro":        float(f1_score(y_test, y_pred, average="macro",  zero_division=0)),
        "precision_micro": float(precision_score(y_test, y_pred, average="micro", zero_division=0)),
        "recall_micro":    float(recall_score(y_test, y_pred, average="micro",    zero_division=0)),
        "per_label_f1": {
            label_names[i]: float(f1_score(y_test[:, i], y_pred[:, i], zero_division=0))
            for i in range(len(label_names))
        },
    }

    logger.info(
        "GPT-4o-mini results — Accuracy: %.4f | F1-Micro: %.4f | F1-Macro: %.4f",
        metrics["accuracy"], metrics["f1_micro"], metrics["f1_macro"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Error Export (False Positives / False Negatives → Excel)
# ---------------------------------------------------------------------------


def export_errors(
    texts_test: list[str],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    mlb: MultiLabelBinarizer,
    output_path: Path = ERROR_EXPORT_PATH,
) -> None:
    """
    Export misclassified test samples to Excel for manual error analysis.

    Columns: text | true_labels | pred_labels | false_positives | false_negatives | error_type
    Only rows with at least one FP or FN are written.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("'pandas' not installed — skipping error export.")
        return

    label_names = list(mlb.classes_)
    rows = []
    for text, true_row, pred_row in zip(texts_test, y_test, y_pred):
        true_set = {l for l, v in zip(label_names, true_row) if v}
        pred_set = {l for l, v in zip(label_names, pred_row) if v}
        fp = sorted(pred_set - true_set)
        fn = sorted(true_set - pred_set)
        if not fp and not fn:
            continue
        rows.append({
            "text":            text,
            "true_labels":     ", ".join(sorted(true_set)),
            "pred_labels":     ", ".join(sorted(pred_set)),
            "false_positives": ", ".join(fp),
            "false_negatives": ", ".join(fn),
            "error_type":      "FP+FN" if fp and fn else ("FP" if fp else "FN"),
        })

    df = pd.DataFrame(rows, columns=[
        "text", "true_labels", "pred_labels", "false_positives", "false_negatives", "error_type",
    ])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="errors")
        ws = writer.sheets["errors"]
        ws.freeze_panes = "A2"
        for col_idx, width in enumerate([80, 30, 30, 30, 30, 10], start=1):
            ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = width

    logger.info(
        "Error export: %d / %d test samples misclassified → %s",
        len(df), len(texts_test), output_path,
    )


# ---------------------------------------------------------------------------
# Callable stage function for the pipeline orchestrator
# ---------------------------------------------------------------------------

def run_modeling_pipeline(device=None) -> None:
    """
    Full modeling pass: GPT-4o-mini baseline + fine-tuned SBERT + fine-tuned AlephBERT.

    Models evaluated:
      1. GPT-4o-mini baseline   — OpenAI structured-output chat completion, no training
      2. Fine-tuned SBERT       — multilingual MiniLM, mean-pool + linear head, 10 epochs
      3. Fine-tuned AlephBERT   — dicta-il/alephbertgimmel-base, native Hebrew, 10 epochs

    All three sets of metrics are saved to artifacts/eval_results.json.

    Parameters
    ----------
    device : torch.device | None
        GPU/CPU device forwarded to the fine-tuned transformer models. Resolved
        by the pipeline orchestrator before calling this function.
    """
    train_data, test_data = load_data()

    texts_train, texts_val, texts_test, y_train, y_val, y_test, mlb = preprocess(
        train_data, test_data
    )

    # ---- GPT-4o-mini baseline (Stage 4) ---------------------------------------
    logger.info("=" * 62)
    logger.info("  GPT-4o-mini: OpenAI structured-output baseline")
    logger.info("=" * 62)
    gpt4omini_metrics = gpt4o_mini_baseline(texts_test, y_test, mlb)

    # ---- Fine-tuned SBERT -------------------------------------------------------
    logger.info("=" * 62)
    logger.info("  Fine-tuned: Multilingual SBERT (%d epochs)", FINETUNE_EPOCHS)
    logger.info("=" * 62)
    finetuned_metrics, y_pred_sbert = fine_tune_transformer(
        texts_train, y_train, texts_test, y_test, mlb, device=device
    )
    _log_metrics("Fine-tuned SBERT", finetuned_metrics)

    # Release SBERT GPU memory before loading AlephBERT (C-3: prevent OOM)
    import torch as _torch
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    # ---- Fine-tuned AlephBERT (native Hebrew baseline) --------------------------
    logger.info("=" * 62)
    logger.info("  Fine-tuned: AlephBERT — dicta-il/alephbertgimmel-base (%d epochs)", ALEPHBERT_EPOCHS)
    logger.info("=" * 62)
    alephbert_metrics, y_pred_alephbert = fine_tune_alephbert(
        texts_train, y_train, texts_test, y_test, mlb, device=device
    )
    _log_metrics("Fine-tuned AlephBERT", alephbert_metrics)

    # ---- Save all results -------------------------------------------------------
    eval_output: dict = {"finetuned_test": finetuned_metrics}
    if gpt4omini_metrics:
        eval_output["gpt4o_mini_test"] = gpt4omini_metrics
    if alephbert_metrics:
        eval_output["alephbert_test"] = alephbert_metrics
    with EVAL_RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(eval_output, fh, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved to %s", EVAL_RESULTS_PATH)

    # ---- Error export (SBERT errors — primary model) ----------------------------
    try:
        export_errors(texts_test, y_test, y_pred_sbert, mlb)
    except Exception as exc:
        logger.warning("Error export failed (non-fatal): %s", exc)

    # ---- Regenerate comparison charts ----------------------------------------
    try:
        from src.eda import plot_model_comparison, plot_perlabel_finetuned
        plot_model_comparison()
        plot_perlabel_finetuned()
    except Exception as exc:
        logger.warning("EDA chart generation after modeling failed (non-fatal): %s", exc)


def main() -> None:
    """Standalone entry point -- resolves device then runs the full modeling pipeline."""
    device = get_device()
    run_modeling_pipeline(device=device)


if __name__ == "__main__":
    main()
