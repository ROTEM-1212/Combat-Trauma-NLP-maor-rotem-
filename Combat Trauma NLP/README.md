

# Hebrew Clinical NLP Pipeline for PTSD Symptom Detection

> A multi-label NLP pipeline for detecting **DSM-5-aligned PTSD symptom indicators** in informal Hebrew text (military slang), using synthetic data generation, empirical realism validation, human inter-rater annotation, and a fine-tuned AlephBERT classifier.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Usage](#4-usage)
5. [Results](#5-results)
6. [Repository Structure](#6-repository-structure)
7. [Limitations](#7-limitations)
8. [Contributors](#8-contributors)
9. [License](#9-license)

---

## 1. Overview

The project is an end-to-end pipeline for **multi-label classification of PTSD symptom indicators in Hebrew-language text**, with a focus on the informal register used by Israeli military veterans (WhatsApp messages, tweets, Reddit posts, diary entries - frequently containing military slang).

The classifier targets **8 symptom categories**, derived from DSM-5 criteria for PTSD:

| Label | Clinical Construct (DSM-5) |
|---|---|
| `intrusive_memories` | Recurrent, involuntary intrusive memories / flashbacks |
| `avoidance` | Avoidance of trauma-associated stimuli |
| `hypervigilance` | Hyperarousal - exaggerated startle, scanning for threat |
| `sleep_disturbance` | Insomnia, nightmares, disrupted sleep |
| `anger_irritability` | Irritable behavior and angry outbursts |
| `emotional_numbing` | Persistent inability to experience positive emotions |
| `guilt_shame` | Persistent negative emotional state, self-blame |
| `functional_impairment` | Clinically significant impairment in daily functioning |

Records may carry zero labels (**hard negatives** - text that superficially resembles a symptom report but does not meet criteria), which the model must learn to reject.

**Clinical significance:** informal self-disclosure of PTSD symptoms (in peer support forums, messaging apps, or personal writing) is a low-friction signal that could support **early triage and screening** at scale, particularly for populations underserved by formal clinical assessment. Because labeled real-world clinical text in Hebrew is scarce and sensitive (privacy, IRB constraints), this project develops the dataset synthetically, then validates that synthetic distribution against real-world data before training.

---

## 2. Pipeline Architecture

```mermaid
flowchart TD
    A[DSM-5 Criteria and Hebrew Slang Marker Bank] --> B[Synthetic Hebrew Review Generation]
    B --> C[Exploratory Data Analysis and TF-IDF / LR Baseline]
    C --> D[Realism Test vs. Real-World Reddit Text]
    D --> E[Annotation Export]
    E --> F[Manual Labeling by Two Independent Raters]
    F --> G[Cohen's Kappa Inter-Rater Agreement]
    G --> H[Stratified Train / Test Split]
    H --> I[Zero-Shot GPT-4o-mini Baseline]
    H --> J[Fine-Tuned SBERT]
    H --> K[Fine-Tuned AlephBERT]
    I --> L[Evaluation and Visualization]
    J --> L
    K --> L
    L --> M[Report Generation]
```

### 2.1 Synthetic Data Generation (Stage 1)

Real clinical-grade Hebrew PTSD self-disclosure data is not available in sufficient volume to train a supervised classifier, and collecting it directly would raise privacy and ethical concerns. The dataset is therefore generated synthetically:

- **Generator:** `src/data_generation.py::generate_dataset()`, backed by an LLM (OpenAI API, with an Ollama/local-model and mock fallback for offline runs).
- **Clinical anchoring:** each generated example is conditioned on a specific DSM-5 sub-criterion (`dsm5_criterion` field), and on a hand-curated bank of Hebrew clinical-language and military-slang markers per label - these are coded directly in `DSM5_CRITERIA` rather than left to the LLM to invent at generation time.
- **Controlled variation axes:** every record varies along `platform` (`whatsapp`, `tweet`, `reddit`, `diary`), `explicitness` (`explicit`, `implicit`, `behavioral`), `severity` (`mild`, `medium`, `strong`), and `example_type` (`positive_clear`, `implicit`, `hard_negative`, `ambiguous`), to ensure the resulting classifier sees realistic surface-form diversity, not just clean DSM-5 language.
- **Hard negatives:** ~25% of the dataset (`example_type = "hard_negative"`) consists of text that is superficially similar to a symptom report but does not meet clinical criteria (e.g., generic fatigue with no traumatic referent). These carry `labels: []` and are essential for training a classifier that discriminates true symptom language from surface-level lookalikes.
- **Output:** `data/dataset.json` - a flat JSON array, one record per synthetic utterance, with full label/metadata schema.

### 2.2 Realism Test (Stage 3.5)

Before any annotation or training effort is invested, the synthetic corpus is checked against real human-written text to catch distributional or stylistic drift early.

- **Method:** `src/validation.py::run_realism_test()` runs a **forced-choice discrimination task**: for each trial, one real sentence (from `data/golden_dataset.json`, scraped from a relevant Reddit community via `scrape_reddit.py`) is paired with one synthetic sentence, and an LLM judge is asked to identify which of the two is real.
- **Metric - detection rate:** the fraction of pairs the judge correctly identifies. A detection rate near chance (**0.50**) indicates the synthetic text is **indistinguishable** from real text to the judge; a detection rate approaching 1.0 indicates the synthetic text is easily flagged as artificial.
- **Decision rule:** a configurable `flag_threshold` (`REALISM_FAIL_THRESHOLD` in `src/config.py`) marks the dataset as failing realism if the detection rate exceeds it - i.e., if the synthetic text is *too easy* to tell apart from real text.
- **Output:** `artifacts/realism_test_results.json`, containing `total_pairs`, `correct_guesses`, `detection_rate`, `flagged` (boolean), and the full per-pair trace (texts, positions, judge guesses) for qualitative review.

### 2.3 Data Annotation & Inter-Rater Reliability (Stage 3.5b)

To validate that the synthetic labels assigned at generation time are actually recoverable by independent human readers (i.e., that the labeling task is well-defined and the text is not ambiguous to the point of being unlabelable):

- **Export:** `src/validation.py::export_for_annotation()` produces `data/annotation_export.csv` - a de-identified, label-stripped version of a dataset sample for blind annotation.
- **Process:** two independent annotators label the export by hand against the 8-category schema, producing `data/annotation_rater1.csv` and `data/annotation_rater2.csv`.
- **Agreement metric:** `src/validation.py::compute_cohen_kappa()` computes **Cohen's Kappa** per label (binary present/absent agreement for each of the 8 categories independently, since this is a multi-label - not multi-class - task) plus a macro-average across labels.
- **Interpretation:** Kappa corrects raw percent-agreement for the agreement expected by chance. Conventionally: `<0` no agreement, `0-0.20` slight, `0.21-0.40` fair, `0.41-0.60` moderate, `0.61-0.80` substantial, `0.81-1.00` almost perfect (Landis & Koch, 1977).
- **Output:** `artifacts/kappa_results.json` - per-label Kappa plus `macro_avg`. This is a manual, on-demand step (not run automatically by `run_pipeline.py`), since it depends on human annotation being completed first.

### 2.4 Model Fine-Tuning (Stage 5)

`src/modeling.py::run_modeling_pipeline()` trains and evaluates the classifier on the stratified split, benchmarking against two reference points:

| Model | Role | Architecture |
|---|---|---|
| **GPT-4o-mini** | Zero-shot baseline | OpenAI structured-output API call, no gradient updates, no Hebrew-specific tuning |
| **SBERT (fine-tuned)** | Multilingual reference | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, fine-tuned with a linear multi-label classification head |
| **AlephBERT (fine-tuned)** | Primary model | `dicta-il/alephbertgimmel-base` - a BERT-family encoder pretrained natively on Hebrew, fine-tuned with a sigmoid multi-label classification head |

Both fine-tuned models use a **binary cross-entropy** objective over independent per-label sigmoid outputs (standard multi-label setup), trained for `10` epochs at `lr=2e-5` (`ALEPHBERT_EPOCHS`/`ALEPHBERT_LR` in `src/config.py`), with a held-out validation split (`VAL_SIZE=0.15`) carved from the training fold for early-stopping/model selection. A simpler **TF-IDF + Logistic Regression** baseline (`class_weight="balanced"`) is additionally computed during EDA (Stage 3) as a non-neural sanity check.

The rationale for benchmarking a native-Hebrew encoder (AlephBERT) against a multilingual encoder (SBERT) and a strong general-purpose LLM (GPT-4o-mini) is to isolate how much of the performance gap is attributable to **Hebrew-specific subword tokenization and pretraining**, versus general language understanding capacity.

### 2.5 Results & Evaluation (Stage 5/6)

All three models are scored on the same held-out test split (`data/test_dataset.json`) using standard multi-label metrics - micro/macro Precision, Recall, and F1 - plus a per-label F1 breakdown to surface which symptom categories are hardest to detect. Misclassified examples are exported to `artifacts/misclassified_errors.xlsx` for qualitative error analysis. See [Section 5](#5-results) for the full results tables.

---

## 3. Installation & Setup

### Prerequisites

- Python **3.10**
- (Optional, recommended) CUDA-capable GPU for AlephBERT/SBERT fine-tuning
- An OpenAI API key (for Stage 5's GPT-4o-mini baseline and/or Stage 1's default LLM-backed generation)

### Clone the repository

```bash
git clone https://github.com/ROTEM-1212/Combat-Trauma-NLP-maor-rotem-.git
cd ptsd_pipeline
```

### Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3.10 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
# GPU machines (CUDA-enabled PyTorch)
pip install -r requirements-gpu.txt

# CPU-only machines
pip install -r requirements-cpu.txt
```

Both files extend `requirements-base.txt`, which pins the shared scientific-stack/NLP dependencies (`transformers`, `scikit-learn`, `scikit-multilearn`, `openai`, etc.) - do not install it directly.

### Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Verify GPU availability (optional)

```bash
python gpu_check.py
```

---

## 4. Usage

### Run the full pipeline end-to-end

```bash
python run_pipeline.py
```

### Run specific stages only

```bash
# e.g. re-run EDA, split, and modeling without regenerating data
python run_pipeline.py --stages 3 4 5
```

### Common flags

```bash
python run_pipeline.py --skip-generation    # dataset.json already exists
python run_pipeline.py --skip-validation    # no golden_dataset.json available
python run_pipeline.py --mock               # offline run, no LLM/API key required
python run_pipeline.py --ollama llama3       # use a local Ollama model instead of OpenAI
```

### Compute inter-rater reliability (manual step, after annotation)

```bash
python -c "from src.validation import compute_cohen_kappa; compute_cohen_kappa()"
```

### Run inference with the fine-tuned AlephBERT model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "./alephbert-ptsd"  # point this at your saved fine-tuned checkpoint
LABELS = [
    "intrusive_memories", "avoidance", "hypervigilance", "sleep_disturbance",
    "anger_irritability", "emotional_numbing", "guilt_shame", "functional_impairment",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

text = "מאז שחזרתי מהמילואים אני לא מצליח לישון לילה רצוף."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze()

predicted = [label for label, p in zip(LABELS, probs) if p > 0.5]
print(predicted)
```

---

## 5. Results

### 5.1 Realism Test (Synthetic vs. Real Text Discrimination)

| Metric | Value |
|---|---|
| Total pairs evaluated | 47 |
| Correct judge guesses | 24 |
| Detection rate | 51.06% |
| Flag threshold | 85% |
| Flagged as unrealistic? | **No** |

A detection rate of ~51% is statistically indistinguishable from chance (50%), indicating the LLM judge could not reliably tell synthetic utterances apart from genuine human-written Hebrew trauma-disclosure text - the dataset passes the realism gate.

### 5.2 Inter-Rater Reliability (Cohen's Kappa, per label)

| Symptom Label | Cohen's Kappa |
|---|---|
| `sleep_disturbance` | 0.887 |
| `anger_irritability` | 0.853 |
| `avoidance` | 0.772 |
| `hypervigilance` | 0.677 |
| `intrusive_memories` | 0.494 |
| `emotional_numbing` | 0.432 |
| `functional_impairment` | 0.200 |
| `guilt_shame` | 0.000 |
| **Macro-average** | **0.539 (moderate agreement)** |

> **Note:** agreement is strong-to-substantial for high-frequency, behaviorally explicit labels (sleep, anger, avoidance) and weak for low-frequency, more inferential labels (`guilt_shame`, `functional_impairment`) - see [Limitations](#7-limitations).

### 5.3 Model Performance (Held-Out Test Set)

| Model | Accuracy | Precision (micro) | Recall (micro) | F1 (micro) | F1 (macro) |
|---|---|---|---|---|---|
| GPT-4o-mini (zero-shot) | 0.459 | 0.555 | 0.945 | 0.700 | 0.688 |
| SBERT (fine-tuned) | 0.760 | 0.844 | 0.912 | 0.877 | 0.838 |
| **AlephBERT (fine-tuned)** | **0.897** | **0.960** | **0.931** | **0.945** | **0.931** |

![Model comparison - F1-micro on the held-out test set across all four models (TF-IDF+LR, GPT-4o-mini, SBERT, AlephBERT).](ptsd_pipeline/visuals/09_model_comparison.png)

### 5.4 AlephBERT - Per-Label F1

| Symptom Label | Precision | Recall | F1-Score |
|---|---|---|---|
| `sleep_disturbance` | 1.000 | 1.000 | 1.000 |
| `anger_irritability` | 1.000 | 0.957 | 0.978 |
| `intrusive_memories` | 0.953 | 0.976 | 0.965 |
| `hypervigilance` | 0.990 | 0.941 | 0.964 |
| `emotional_numbing` | 0.925 | 0.925 | 0.925 |
| `avoidance` | 0.912 | 0.861 | 0.886 |
| `guilt_shame` | 0.921 | 0.814 | 0.864 |
| `functional_impairment` | 0.853 | 0.879 | 0.866 |

> Note: the per-label chart below is for the **fine-tuned SBERT** model (the table above is AlephBERT).

![Fine-tuned SBERT - per-label Precision / Recall / F1 across the 8 PTSD symptom categories.](ptsd_pipeline/visuals/11_perlabel_finetuned.png)

**Headline result:** the fine-tuned, Hebrew-native AlephBERT model outperforms both the multilingual SBERT baseline (+9.3 F1-macro points) and the zero-shot GPT-4o-mini baseline (+24.3 F1-macro points), supporting the hypothesis that native Hebrew subword pretraining materially improves clinical symptom detection over multilingual or general-purpose LLM approaches in this low-resource setting.

### 5.5 Dataset & EDA Visuals

Key charts from `src/eda.py` (300 DPI, under `visuals/`). Hard-negative records (`labels=[]`) are included throughout.

| | |
|---|---|
| ![Per-label frequency](ptsd_pipeline/visuals/01_label_marginals.png) | ![Labels per record (0 = hard-negative)](ptsd_pipeline/visuals/03_cardinality.png) |
| ![Phi correlation between labels](ptsd_pipeline/visuals/02b_label_correlation.png) | ![Top-20 military slang tokens](ptsd_pipeline/visuals/08_slang_top20.png) |

---

## 6. Repository Structure

```
ptsd_pipeline/
├── run_pipeline.py        # Orchestrator - runs all stages in sequence
├── gpu_check.py           # Detects CUDA / CPU at startup
├── scrape_reddit.py       # Builds data/golden_dataset.json (real-text reference corpus)
├── src/
│   ├── config.py          # Central path + hyperparameter constants
│   ├── data_generation.py # Stage 1 - synthetic dataset generator
│   ├── eda.py              # Stage 3 - EDA charts + TF-IDF/LR baseline
│   ├── validation.py       # Stage 3.5 - realism test, annotation export, Cohen's Kappa
│   ├── splitting.py        # Stage 4 - iterative stratified split
│   ├── modeling.py         # Stage 5 - GPT-4o-mini, SBERT, AlephBERT fine-tuning + eval
│   └── report.py           # Stage 6 - report rendering
├── data/                   # Datasets, annotation exports, train/test splits
├── artifacts/              # Machine-readable outputs (metrics, manifests)
├── reports/                # Human-readable Markdown reports
├── visuals/                # EDA + model-comparison charts (300 DPI)
└── logs/                   # pipeline_run.log
```

---

## 7. Limitations

- **Fully synthetic training data.** Despite passing the realism test (Section 5.1, N=47 pairs), the training corpus contains no real clinical text; residual synthetic artifacts not captured by the realism test's sample size may still bias the model.
- **Low inter-rater agreement on inferential labels.** `guilt_shame` (κ=0.00) and `functional_impairment` (κ=0.20) show weak human agreement, suggesting these categories are conceptually harder to apply consistently - model performance on these labels should be interpreted cautiously regardless of headline F1.
- **Realism test sample size is small** (47 pairs) relative to the full dataset (2,000 records); detection-rate estimates carry non-trivial variance.
- **No external/held-out clinical validation.** Performance is reported solely on a stratified split of the same synthetic distribution used for training; no real-world deployment cohort has been evaluated.
- **Single annotator pair.** Kappa is computed between exactly two raters; it does not capture broader population-level annotator variance.

---

## 8. Contributors

- Maor Sharon
- Rotem

---

## 9. License

Released under the MIT License. See [LICENSE](ptsd_pipeline/LICENSE).
