from pathlib import Path
from typing import Dict, Any
import pandas as pd
import tqdm

from .config import load_settings
from .data_loader import load_posts
from .models import get_active_models
from .utils import ensure_result_dict

BASE_DIR = Path(__file__).resolve().parents[1]

def run_batch(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str = "text",
    id_column: str = "post_id",
    config_path: str | Path | None = None,
) -> pd.DataFrame:
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
        default=str(BASE_DIR / "data" / "samples" / "posts_sample.csv"),
        help="Path to CSV or JSON/JSONL file with posts.",
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
    args = parser.parse_args()
    run_batch(args.input, args.output, config_path=args.config)
