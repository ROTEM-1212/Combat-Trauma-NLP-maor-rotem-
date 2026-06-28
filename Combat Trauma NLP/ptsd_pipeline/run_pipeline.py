"""
run_pipeline.py
===============
Unified entry point for the full Project Sasha NLP pipeline.

Sequential stages
-----------------
  1.   Data Generation   (src/data_generation.py)  → data/dataset.json
  3.   EDA               (src/eda.py)               → visuals/*.png + eda_tables.json
  3.5  Validation        (src/validation.py)        → realism_test_results.json + annotation_export.csv
  4.   Stratified Split  (src/splitting.py)         → data/train_dataset.json + test_dataset.json
  5.   Neural Modeling   (src/modeling.py)          → Fine-tuned SBERT + eval_results.json
  6.   Report            (src/report.py)            → slide3_summary.md + README_eda.md

Stage 1's output (data/dataset.json) feeds directly into Stage 3 — there is no
intermediate quality-judge filtering stage.

Usage
-----
    python run_pipeline.py                     # run all stages end-to-end
    python run_pipeline.py --skip-generation   # skip stage 1 (data already exists)
    python run_pipeline.py --skip-validation   # skip stage 3.5 (no golden dataset)
    python run_pipeline.py --stages 3 4 5      # run only EDA + split + modeling
    python run_pipeline.py --mock              # use MockLLMClient (no Ollama / API required)

All paths are centralised in src/config.py.
Logging is configured once here (console INFO + pipeline_run.log DEBUG).
Each stage failure aborts the run with a logged exception.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so gpu_check.py is importable
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Load .env file if present (allows setting OPENAI_API_KEY etc. without touching the shell)
# The .env is at the project root (one level above ptsd_pipeline/), so use _HERE.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_HERE.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — fall back to shell environment variables

from src.logging_setup import configure_logging
from src.config import LOG_PATH, DATASET_OUTPUT_PATH


_OLLAMA_DEFAULT_MODEL = "llama3"
_OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"


def _ollama_default() -> str | None:
    """Return the default Ollama model name if Ollama is reachable, else None."""
    import urllib.request
    try:
        urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=3)
        logger.info("Ollama detected — using '%s' as default LLM.", _OLLAMA_DEFAULT_MODEL)
        return _OLLAMA_DEFAULT_MODEL
    except Exception:
        return None

configure_logging(LOG_PATH)
logger = logging.getLogger(__name__)


def _banner(text: str) -> None:
    sep = "=" * 62
    logger.info(sep)
    logger.info("  %s", text)
    logger.info(sep)


def run_stage_1(mock: bool = False, ollama_model: str | None = None) -> None:
    """Stage 1 — Generate synthetic Hebrew dataset via LLM."""
    _banner("STAGE 1 — Synthetic Data Generation")
    from src.data_generation import (
        LLMProvider, LLMClient,
        create_llm_client, MockLLMClient, generate_dataset,
        DATASET_OUTPUT_PATH as _OUT,
    )
    import os

    # Resolve effective Ollama model: explicit flag or auto-detect when no cloud key
    _ollama = ollama_model or (_ollama_default() if not os.environ.get("OPENAI_API_KEY", "").strip() and not os.environ.get("OPENROUTER_API_KEY", "").strip() else None)

    if mock:
        logger.warning("--mock flag set — using MockLLMClient (deterministic fake Hebrew).")
        llm: LLMClient = MockLLMClient()
    elif os.environ.get("OPENAI_API_KEY", "").strip():
        logger.info("OPENAI_API_KEY found — using OpenAI (gpt-4o-mini) for data generation.")
        llm = create_llm_client(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            allow_paid_apis=True,
        )
    elif _ollama:
        logger.info("Using local Ollama model '%s'.", _ollama)
        llm = create_llm_client(
            provider=LLMProvider.OLLAMA,
            model_name=_ollama,
        )
    elif os.environ.get("OPENROUTER_API_KEY"):
        logger.info("Falling back to OpenRouter (free tier).")
        llm = create_llm_client(
            provider=LLMProvider.OPENROUTER,
            model_name="mistralai/mistral-7b-instruct:free",
        )
    else:
        logger.warning(
            "No LLM provider available. Falling back to MockLLMClient. "
            "Set OPENAI_API_KEY, start Ollama, or use --ollama."
        )
        llm = MockLLMClient()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    examples = generate_dataset(llm, output_path=str(_OUT))
    logger.info("Stage 1 complete — %d examples written to %s.", len(examples), _OUT)


def run_stage_3() -> None:
    """Stage 3 — EDA charts + numeric tables."""
    _banner("STAGE 3 — Exploratory Data Analysis")
    from src.eda import run_eda_pipeline
    from src.config import DATASET_OUTPUT_PATH as _INPUT, EDA_TABLES_PATH as _TABLES

    run_eda_pipeline(input_path=_INPUT, tables_path=_TABLES)
    logger.info("Stage 3 complete.")


def run_stage_4() -> None:
    """Stage 4 — Iterative stratification split."""
    _banner("STAGE 4 — Iterative Stratification Split")
    from src.splitting import run_split_pipeline
    from src.config import (
        DATASET_OUTPUT_PATH as _IN,
        TRAIN_DATASET_PATH as _TRAIN,
        TEST_DATASET_PATH as _TEST,
        SPLIT_MANIFEST_PATH as _MANIFEST,
    )

    if not _IN.exists():
        raise FileNotFoundError(
            f"Stage 4 requires '{_IN}'. Run Stage 1 first."
        )

    run_split_pipeline(
        input_path=_IN,
        train_path=_TRAIN,
        test_path=_TEST,
        manifest_path=_MANIFEST,
    )
    logger.info("Stage 4 complete.")


def run_stage_3_5(mock: bool = False, ollama_model: str | None = None) -> None:
    """Stage 3.5 — Synthetic data validation (realism test + annotation export)."""
    _banner("STAGE 3.5 — Validation Pipeline")
    from src.validation import run_realism_test, export_for_annotation
    from src.config import GOLDEN_DATASET_PATH

    if not DATASET_OUTPUT_PATH.exists():
        logger.warning(
            "Stage 3.5: dataset not found at %s — skipping. Run Stage 1 first.",
            DATASET_OUTPUT_PATH,
        )
        return

    # Annotation export requires only the dataset (no LLM)
    try:
        export_for_annotation()
    except Exception:
        logger.exception("Annotation export failed (non-fatal).")

    # Realism test requires the golden dataset AND an LLM
    if not GOLDEN_DATASET_PATH.exists():
        logger.warning(
            "Golden dataset not found at %s — skipping realism test. "
            "Provide data/golden_dataset.json to enable this check.",
            GOLDEN_DATASET_PATH,
        )
        logger.info("Stage 3.5 complete (annotation export only — no golden dataset).")
        return

    import os
    from src.data_generation import LLMProvider, LLMClient, create_llm_client, MockLLMClient

    _ollama = ollama_model or (
        _ollama_default()
        if not os.environ.get("OPENAI_API_KEY", "").strip()
        else None
    )

    if mock:
        llm: LLMClient = MockLLMClient()
    elif os.environ.get("OPENAI_API_KEY", "").strip():
        llm = create_llm_client(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            allow_paid_apis=True,
        )
    elif _ollama:
        llm = create_llm_client(provider=LLMProvider.OLLAMA, model_name=_ollama)
    else:
        logger.warning(
            "No LLM available for realism test — falling back to MockLLMClient. "
            "Results will not be meaningful."
        )
        llm = MockLLMClient()

    try:
        result = run_realism_test(llm)
        if result.get("flagged"):
            logger.warning(
                "REALISM TEST FLAGGED — detection rate %.0f%% >= threshold %.0f%%. "
                "Review data quality before proceeding to modeling.",
                result["detection_rate"] * 100,
                result["flag_threshold"] * 100,
            )
    except Exception:
        logger.exception("Realism test failed (non-fatal). Continuing pipeline.")

    logger.info("Stage 3.5 complete.")


def run_stage_5(device=None) -> None:
    """Stage 5 — Neural model fine-tuning (Multilingual SBERT)."""
    _banner("STAGE 5 — Neural Model Fine-Tuning & Evaluation")
    from src.modeling import run_modeling_pipeline

    run_modeling_pipeline(device=device)
    logger.info("Stage 5 complete.")


def run_stage_6() -> None:
    """Stage 6 — Generate Slide 3 summary and README_eda.md."""
    _banner("STAGE 6 — Report Generation")
    from src.report import run_report_pipeline

    # EDA already ran in Stage 3; pass run_eda=False to skip re-running charts
    run_report_pipeline(run_eda=False)
    logger.info("Stage 6 complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Sasha — Unified PTSD Hebrew NLP Pipeline"
    )
    parser.add_argument(
        "--stages", nargs="+", type=int, choices=[1, 3, 4, 5, 6], metavar="N",
        help="Run only specified stages (e.g. --stages 3 4 5)",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip Stage 1 (assume data/dataset.json already exists)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip Stage 3.5 (realism test + annotation export). Use when golden dataset is absent.",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockLLMClient for stages 1 and 3.5 (no Ollama / API required)",
    )
    parser.add_argument(
        "--ollama", metavar="MODEL", default=None,
        help="Use local Ollama for stage 1 (e.g. --ollama llama3). "
             "Ollama must be running at http://localhost:11434.",
    )
    args = parser.parse_args()

    # Determine which stages to run
    if args.stages:
        active_stages = set(args.stages)
    else:
        active_stages = {1, 3, 4, 5, 6}
        if args.skip_generation:
            active_stages.discard(1)

    # Hardware check — printed once at the very beginning
    _banner("HARDWARE CHECK")
    from gpu_check import get_device
    device = get_device()

    stage_fns = {
        1: lambda: run_stage_1(mock=args.mock, ollama_model=args.ollama),
        3: run_stage_3,
        4: run_stage_4,
        5: lambda: run_stage_5(device=device),
        6: run_stage_6,
    }

    for stage_num in sorted(active_stages):
        try:
            stage_fns[stage_num]()
        except Exception:
            logger.exception("Pipeline aborted at Stage %d.", stage_num)
            sys.exit(1)

        # After Stage 3 (EDA), run Stage 3.5 (Validation) automatically
        if stage_num == 3 and not getattr(args, "skip_validation", False):
            try:
                run_stage_3_5(mock=args.mock, ollama_model=args.ollama)
            except Exception:
                logger.exception(
                    "Stage 3.5 failed (non-fatal). Continuing pipeline. "
                    "Use --skip-validation to suppress."
                )

    _banner("PIPELINE COMPLETE")
    logger.info("All stages finished. Log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
