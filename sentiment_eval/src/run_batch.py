from pathlib import Path
from typing import Dict, Any
import pandas as pd
import tqdm

from .config import load_settings
from .data_loader import load_posts, load_staging_batch
from .models import get_active_models
from .utils import ensure_result_dict

BASE_DIR = Path(__file__).resolve().parents[1]
STAGING_DIR = BASE_DIR / "data" / "staging"
DEFAULT_STAGING_FILE = STAGING_DIR / "current_batch.csv"


def run_batch(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    text_column: str = "text",
    id_column: str = "post_id",
    config_path: str | Path | None = None,
    use_staging: bool = True,
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
    models = get_active_models(cfg)

    print(f"Loaded {len(df)} posts. Active models: {[m.name for m in models]}")

    results: Dict[int, Dict[str, Any]] = {}
    
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        text = str(row[text_column])
        row_id = row[id_column]
        row_res = dict(row)
        
        for m in models:
            try:
                raw_res = m.classify(text)
                row_res.update(ensure_result_dict(m.name, raw_res))
            except Exception as e:
                # Crucial: Ensure the process does not crash on API failure
                print(f"\nError with model {m.name} on row {idx} ({e})")
                row_res.update({
                    f"{m.name}_sentiment": "neutral",
                    f"{m.name}_confidence": 0.0,
                    f"{m.name}_reason": f"Error: {e.__class__.__name__}. Check API key or endpoint.",
                })
        
        results[row_id] = row_res

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
    args = parser.parse_args()
    run_batch(
        args.input, 
        args.output, 
        config_path=args.config,
        use_staging=not args.no_staging
    )
