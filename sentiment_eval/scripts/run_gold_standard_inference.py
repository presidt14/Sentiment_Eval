"""
Gold Standard Inference Script

Runs sentiment model inference on gold_standard_v2.csv (210 rows)
using the compliance_risk_assessor prompt strategy.

Usage:
    # Run with mock model (for testing pipeline)
    python scripts/run_gold_standard_inference.py --mock
    
    # Run with specific model
    python scripts/run_gold_standard_inference.py --model gemma
    python scripts/run_gold_standard_inference.py --model claude
    
    # Run with all active models from settings
    python scripts/run_gold_standard_inference.py
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

# Set up path for imports - need to import as package
BASE_DIR = Path(__file__).resolve().parents[1]
# Add parent of src to allow "from src.X import Y" style imports
sys.path.insert(0, str(BASE_DIR))

# Now import from src package
from src.utils import ensure_result_dict_v2, normalise_sentiment
from src.config import get_prompt_strategy


def load_gold_standard(path: Path) -> pd.DataFrame:
    """Load gold standard dataset."""
    print(f"Loading gold standard from: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows")
    return df


def get_model_instance(model_name: str, prompt_strategy: str = "compliance_risk_assessor"):
    """Get a specific model instance with prompt strategy using the model factory."""
    # Use the centralized model factory
    from src.model_factory import get_model
    return get_model(provider_name=model_name, prompt_strategy=prompt_strategy)


def run_inference_sync(
    df: pd.DataFrame,
    model,
    text_col: str = "text",
    brand_col: str = "brand",
    id_col: str = "post_id",
) -> pd.DataFrame:
    """Run synchronous inference on all rows."""
    results = []
    
    print(f"\nRunning inference with model: {model.name}")
    print(f"  Text column: {text_col}")
    print(f"  Brand column: {brand_col}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        text = str(row[text_col])
        brand = str(row.get(brand_col, "Unknown"))
        post_id = row[id_col]
        
        try:
            # Call model with brand context
            result = model.classify(text)
            
            # Normalize result to extended schema
            # Note: Zone of Control guardrail is now applied inside the model class
            normalized = ensure_result_dict_v2(model.name, result)
            normalized[id_col] = post_id
            
        except Exception as e:
            print(f"\n  Error on row {idx}: {e}")
            normalized = {
                id_col: post_id,
                f"{model.name}_sentiment": "neutral",
                f"{model.name}_confidence": 0.0,
                f"{model.name}_reason": f"Error: {str(e)[:50]}",
                f"{model.name}_brand_relevance": None,
                f"{model.name}_negative_type": None,
            }
        
        results.append(normalized)
    
    return pd.DataFrame(results)


async def run_inference_async(
    df: pd.DataFrame,
    model,
    text_col: str = "text",
    brand_col: str = "brand",
    id_col: str = "post_id",
    concurrency: int = 5,
) -> pd.DataFrame:
    """Run async inference with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    print(f"\nRunning async inference with model: {model.name}")
    print(f"  Concurrency: {concurrency}")
    
    async def classify_row(idx: int, row: pd.Series) -> Dict[str, Any]:
        async with semaphore:
            text = str(row[text_col])
            brand = str(row.get(brand_col, "Unknown"))
            post_id = row[id_col]
            
            try:
                result = await model.aclassify(text)
                # Note: Zone of Control guardrail is now applied inside the model class
                normalized = ensure_result_dict_v2(model.name, result)
                normalized[id_col] = post_id
                return normalized
            except Exception as e:
                return {
                    id_col: post_id,
                    f"{model.name}_sentiment": "neutral",
                    f"{model.name}_confidence": 0.0,
                    f"{model.name}_reason": f"Error: {str(e)[:50]}",
                    f"{model.name}_brand_relevance": None,
                    f"{model.name}_negative_type": None,
                }
    
    tasks = [classify_row(idx, row) for idx, row in df.iterrows()]
    
    # Use tqdm for progress
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Classifying"):
        result = await coro
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run inference on gold standard")
    parser.add_argument(
        "--input",
        type=str,
        default="data/gold_standard/gold_standard_v2.csv",
        help="Path to gold standard CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results (auto-generated if not specified)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (gemma, claude, openai, gemini, deepseek, mock)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Alias for --model (gemma, claude, openai, gemini, deepseek, mock)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model for testing",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="compliance_risk_assessor",
        help="Prompt strategy from prompts.yaml",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async processing",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrency for async processing",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = BASE_DIR / args.input
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("\nRun this first to create gold_standard_v2.csv:")
        print("  python scripts/evaluate_slices.py")
        sys.exit(1)
    
    # Determine model (--provider is alias for --model)
    if args.mock:
        model_name = "mock"
    elif args.model:
        model_name = args.model
    elif args.provider:
        model_name = args.provider
    else:
        print("Error: Specify --model, --provider, or --mock")
        print("\nExamples:")
        print("  python scripts/run_gold_standard_inference.py --mock")
        print("  python scripts/run_gold_standard_inference.py --model gemma")
        print("  python scripts/run_gold_standard_inference.py --provider openai")
        sys.exit(1)
    
    # Output path
    if args.output:
        output_path = BASE_DIR / args.output
    else:
        output_path = BASE_DIR / "results" / f"results_gold_standard_{model_name}.csv"
    
    # Load data
    df = load_gold_standard(input_path)
    
    # Get model
    print(f"\nInitializing model: {model_name}")
    print(f"  Prompt strategy: {args.prompt_strategy}")
    
    try:
        model = get_model_instance(model_name, args.prompt_strategy)
    except Exception as e:
        print(f"Error initializing model: {e}")
        if model_name != "mock":
            print("\nTip: Use --mock to test the pipeline without API keys")
        sys.exit(1)
    
    # Run inference
    start_time = time.time()
    
    if args.use_async:
        results_df = asyncio.run(
            run_inference_async(df, model, concurrency=args.concurrency)
        )
    else:
        results_df = run_inference_sync(df, model)
    
    elapsed = time.time() - start_time
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"  Rows processed: {len(results_df)}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Output saved to: {output_path}")
    
    # Quick summary of predictions
    sent_col = f"{model_name}_sentiment"
    if sent_col in results_df.columns:
        print(f"\n  Prediction distribution:")
        for sent, count in results_df[sent_col].value_counts().items():
            pct = count / len(results_df) * 100
            print(f"    {sent}: {count} ({pct:.1f}%)")
    
    print(f"\n✅ Now run evaluation:")
    print(f"   python scripts/evaluate_slices.py --model-results {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
