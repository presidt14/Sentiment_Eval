import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import tqdm
import tqdm.asyncio

from .config import get_prompt_strategy, load_settings
from .data_loader import load_posts, load_staging_batch
from .models import get_active_models, get_active_models_with_strategy
from .models.base import SentimentModel
from .utils import ensure_result_dict

BASE_DIR = Path(__file__).resolve().parents[1]
STAGING_DIR = BASE_DIR / "data" / "staging"
DEFAULT_STAGING_FILE = STAGING_DIR / "current_batch.csv"

# Default concurrency limit to avoid overwhelming APIs
DEFAULT_CONCURRENCY = 10


async def _classify_single_row(
    row_id: Any,
    text: str,
    row_data: Dict[str, Any],
    models: List[SentimentModel],
    semaphore: asyncio.Semaphore,
) -> tuple[Any, Dict[str, Any]]:
    """
    Classify a single row with all models asynchronously.
    Uses semaphore to limit concurrent requests.
    """
    async with semaphore:
        row_res = dict(row_data)

        # Run all models concurrently for this row
        tasks = []
        for m in models:
            tasks.append(_classify_with_model(m, text))

        model_results = await asyncio.gather(*tasks, return_exceptions=True)

        for m, result in zip(models, model_results):
            if isinstance(result, Exception):
                row_res.update(
                    {
                        f"{m.name}_sentiment": "neutral",
                        f"{m.name}_confidence": 0.0,
                        f"{m.name}_reason": f"Error: {result.__class__.__name__}. Check API key or endpoint.",
                    }
                )
            else:
                row_res.update(ensure_result_dict(m.name, result))

        return row_id, row_res


async def _classify_with_model(model: SentimentModel, text: str) -> Dict[str, Any]:
    """Wrapper to call async classify on a model."""
    return await model.aclassify(text)


async def process_batch_async(
    df: pd.DataFrame,
    models: List[SentimentModel],
    text_column: str = "text",
    id_column: str = "post_id",
    concurrency: int = DEFAULT_CONCURRENCY,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Process a batch of posts asynchronously with controlled concurrency.

    Args:
        df: DataFrame with posts to analyze.
        models: List of sentiment models to use.
        text_column: Column name containing text.
        id_column: Column name containing unique IDs.
        concurrency: Maximum concurrent requests.
        progress_callback: Optional callback(completed, total) for progress updates.

    Returns:
        Dictionary mapping row IDs to result dictionaries.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: Dict[Any, Dict[str, Any]] = {}

    # Create tasks for all rows
    tasks = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        row_id = row[id_column]
        row_data = dict(row)
        tasks.append(_classify_single_row(row_id, text, row_data, models, semaphore))

    # Process with async progress bar using tqdm.gather
    total = len(tasks)
    completed = 0

    # Use tqdm.asyncio.tqdm.gather for proper async progress tracking
    all_results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Processing", total=total)

    for row_id, row_res in all_results:
        results[row_id] = row_res
        completed += 1
        if progress_callback:
            progress_callback(completed, total)

    return results


def run_batch(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    text_column: str = "text",
    id_column: str = "post_id",
    config_path: str | Path | None = None,
    use_staging: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
    use_async: bool = True,
    prompt_strategy: str | None = None,
) -> pd.DataFrame:
    """
    Run batch sentiment analysis on input data.

    Args:
        input_path: Path to input file. If None and use_staging=True, uses staging file.
        output_path: Path to save results. Auto-generated if None.
        text_column: Column name containing text to analyze.
        id_column: Column name containing unique IDs.
        config_path: Path to configuration file.
        use_staging: If True and input_path is None, use the staging file.
        concurrency: Maximum concurrent requests for async mode.
        use_async: If True, use async processing. If False, use sync processing.
        prompt_strategy: Name of prompt strategy from prompts.yaml. If None, uses default.

    Returns:
        DataFrame with sentiment analysis results.
    """
    # Determine input path
    if input_path is None:
        if use_staging and DEFAULT_STAGING_FILE.exists():
            input_path = DEFAULT_STAGING_FILE
            print(f"Using staging file: {input_path}")
        else:
            # Fall back to sample data
            input_path = BASE_DIR / "data" / "samples" / "posts_sample.csv"
            print(f"No staging file found, using sample: {input_path}")

    input_path = Path(input_path)

    if output_path is None:
        output_path = BASE_DIR / "results" / f"results_{input_path.stem}.csv"
    else:
        output_path = Path(output_path)

    df = load_posts(input_path)
    cfg = load_settings(config_path) if config_path else load_settings()

    # Get models with prompt strategy if specified
    if prompt_strategy:
        print(f"Using prompt strategy: {prompt_strategy}")
        models = get_active_models_with_strategy(prompt_strategy, cfg)
    else:
        models = get_active_models(cfg)

    print(f"Loaded {len(df)} posts. Active models: {[m.name for m in models]}")

    start_time = time.time()

    if use_async:
        print(f"Using async processing with concurrency={concurrency}")
        results = asyncio.run(
            process_batch_async(df, models, text_column, id_column, concurrency)
        )
    else:
        print("Using synchronous processing")
        results = _run_batch_sync(df, models, text_column, id_column)

    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f}s")

    res_df = pd.DataFrame.from_dict(results, orient="index")

    # Clean up columns for export
    for k in res_df.columns:
        if k in [text_column, id_column]:
            continue
        res_df.rename(columns={k: k.replace(f"{id_column}_", "")}, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    return res_df


def _run_batch_sync(
    df: pd.DataFrame,
    models: List[SentimentModel],
    text_column: str,
    id_column: str,
) -> Dict[Any, Dict[str, Any]]:
    """
    Synchronous batch processing (legacy fallback).
    """
    results: Dict[Any, Dict[str, Any]] = {}

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        text = str(row[text_column])
        row_id = row[id_column]
        row_res = dict(row)

        for m in models:
            try:
                raw_res = m.classify(text)
                row_res.update(ensure_result_dict(m.name, raw_res))
            except Exception as e:
                print(f"\nError with model {m.name} on row {idx} ({e})")
                row_res.update(
                    {
                        f"{m.name}_sentiment": "neutral",
                        f"{m.name}_confidence": 0.0,
                        f"{m.name}_reason": f"Error: {e.__class__.__name__}. Check API key or endpoint.",
                    }
                )

        results[row_id] = row_res

    return results


if __name__ == "__main__":
    # Simple CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-model sentiment batch.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV or JSON/JSONL file with posts. If not provided, uses staging file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a configuration YAML file.",
    )
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help="Don't use staging file even if input is not provided.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent requests (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous processing instead of async.",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default=None,
        help="Prompt strategy name from prompts.yaml (e.g., 'sarcasm_detector', 'strict_compliance').",
    )
    args = parser.parse_args()
    run_batch(
        args.input,
        args.output,
        config_path=args.config,
        use_staging=not args.no_staging,
        concurrency=args.concurrency,
        use_async=not args.sync,
        prompt_strategy=args.prompt_strategy,
    )
