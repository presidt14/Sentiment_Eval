"""
Baseline Comparison Script

Compares model predictions against vendor baseline (Sentiment column from CVR2024)
and human labels (gold standard dataset).

Usage:
    python -m scripts.run_baseline_comparison --input data/gold_standard/labeled_200.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add src to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from utils import normalise_sentiment


def load_labeled_data(path: Path) -> pd.DataFrame:
    """Load labeled gold standard data."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} labeled posts from {path.name}")
    return df


def evaluate_vendor_baseline(
    df: pd.DataFrame,
    vendor_col: str = "vendor_sentiment",
    human_col: str = "human_sentiment",
) -> Dict[str, float]:
    """
    Evaluate vendor predictions against human labels.
    
    Returns dict with precision, recall, f1 for each class and overall accuracy.
    """
    # Filter to rows with both labels
    mask = df[vendor_col].notna() & df[human_col].notna()
    eval_df = df[mask].copy()
    
    if len(eval_df) == 0:
        print("Warning: No rows with both vendor and human labels")
        return {}
    
    # Normalize labels
    y_vendor = eval_df[vendor_col].apply(normalise_sentiment)
    y_human = eval_df[human_col].apply(normalise_sentiment)
    
    labels = ["negative", "neutral", "positive"]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_human, y_vendor),
        "samples": len(eval_df),
    }
    
    # Per-class metrics
    for label in labels:
        y_true_binary = (y_human == label).astype(int)
        y_pred_binary = (y_vendor == label).astype(int)
        
        metrics[f"{label}_precision"] = precision_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f"{label}_recall"] = recall_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f"{label}_f1"] = f1_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
    
    return metrics


def evaluate_model_predictions(
    df: pd.DataFrame,
    model_col: str,
    human_col: str = "human_sentiment",
) -> Dict[str, float]:
    """
    Evaluate model predictions against human labels.
    """
    mask = df[model_col].notna() & df[human_col].notna()
    eval_df = df[mask].copy()
    
    if len(eval_df) == 0:
        return {}
    
    y_model = eval_df[model_col].apply(normalise_sentiment)
    y_human = eval_df[human_col].apply(normalise_sentiment)
    
    labels = ["negative", "neutral", "positive"]
    
    metrics = {
        "accuracy": accuracy_score(y_human, y_model),
        "samples": len(eval_df),
    }
    
    for label in labels:
        y_true_binary = (y_human == label).astype(int)
        y_pred_binary = (y_model == label).astype(int)
        
        metrics[f"{label}_precision"] = precision_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f"{label}_recall"] = recall_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f"{label}_f1"] = f1_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
    
    return metrics


def print_comparison_report(
    vendor_metrics: Dict[str, float],
    model_metrics: Optional[Dict[str, float]] = None,
    model_name: str = "Model",
) -> None:
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON REPORT")
    print("=" * 70)
    
    print("\n## Vendor Baseline (BrandMentions/Dizio)")
    print(f"   Samples evaluated: {vendor_metrics.get('samples', 0)}")
    print(f"   Overall Accuracy:  {vendor_metrics.get('accuracy', 0):.1%}")
    print()
    
    print("   Per-Class Metrics:")
    print("   " + "-" * 50)
    print(f"   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("   " + "-" * 50)
    
    for label in ["negative", "neutral", "positive"]:
        p = vendor_metrics.get(f"{label}_precision", 0)
        r = vendor_metrics.get(f"{label}_recall", 0)
        f = vendor_metrics.get(f"{label}_f1", 0)
        print(f"   {label:<12} {p:<12.1%} {r:<12.1%} {f:<12.1%}")
    
    if model_metrics:
        print("\n" + "-" * 70)
        print(f"\n## {model_name}")
        print(f"   Samples evaluated: {model_metrics.get('samples', 0)}")
        print(f"   Overall Accuracy:  {model_metrics.get('accuracy', 0):.1%}")
        print()
        
        print("   Per-Class Metrics:")
        print("   " + "-" * 50)
        print(f"   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("   " + "-" * 50)
        
        for label in ["negative", "neutral", "positive"]:
            p = model_metrics.get(f"{label}_precision", 0)
            r = model_metrics.get(f"{label}_recall", 0)
            f = model_metrics.get(f"{label}_f1", 0)
            print(f"   {label:<12} {p:<12.1%} {r:<12.1%} {f:<12.1%}")
        
        # Delta comparison
        print("\n" + "-" * 70)
        print("\n## Improvement vs Vendor Baseline")
        print("   " + "-" * 50)
        
        acc_delta = model_metrics.get("accuracy", 0) - vendor_metrics.get("accuracy", 0)
        print(f"   Accuracy Delta:           {acc_delta:+.1%}")
        
        neg_p_delta = (model_metrics.get("negative_precision", 0) - 
                       vendor_metrics.get("negative_precision", 0))
        print(f"   Negative Precision Delta: {neg_p_delta:+.1%}")
        
        neg_f1_delta = (model_metrics.get("negative_f1", 0) - 
                        vendor_metrics.get("negative_f1", 0))
        print(f"   Negative F1 Delta:        {neg_f1_delta:+.1%}")
    
    print("\n" + "=" * 70)


def save_comparison_csv(
    vendor_metrics: Dict[str, float],
    model_metrics: Optional[Dict[str, float]] = None,
    output_path: Path = None,
) -> None:
    """Save comparison metrics to CSV."""
    rows = [{"source": "vendor", **vendor_metrics}]
    if model_metrics:
        rows.append({"source": "model", **model_metrics})
    
    df = pd.DataFrame(rows)
    
    if output_path is None:
        output_path = BASE_DIR / "results" / "baseline_comparison.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved comparison metrics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions against vendor baseline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/gold_standard/labeled_200.csv",
        help="Path to labeled gold standard CSV",
    )
    parser.add_argument(
        "--model-results",
        type=str,
        default=None,
        help="Path to model results CSV (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save comparison CSV",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = BASE_DIR / args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("\nTo create the gold standard dataset, run:")
        print("  python -m scripts.prepare_labeling_batch")
        print("\nThen label the posts in data/gold_standard/to_label_200.csv")
        sys.exit(1)
    
    # Load data
    df = load_labeled_data(input_path)
    
    # Check required columns
    required_cols = ["vendor_sentiment", "human_sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Check if human labels exist
    labeled_count = df["human_sentiment"].notna().sum()
    if labeled_count == 0:
        print("Error: No human labels found in the dataset")
        print("Please label the posts before running comparison")
        sys.exit(1)
    
    print(f"Found {labeled_count} posts with human labels")
    
    # Evaluate vendor baseline
    vendor_metrics = evaluate_vendor_baseline(df)
    
    # Evaluate model if results provided
    model_metrics = None
    if args.model_results:
        model_path = BASE_DIR / args.model_results
        if model_path.exists():
            model_df = pd.read_csv(model_path)
            # Merge with gold standard
            merged = df.merge(model_df, on="post_id", how="inner")
            
            # Find model sentiment column
            model_cols = [c for c in merged.columns if c.endswith("_sentiment") 
                         and c not in ["vendor_sentiment", "human_sentiment"]]
            if model_cols:
                model_col = model_cols[0]
                model_metrics = evaluate_model_predictions(merged, model_col)
    
    # Print report
    print_comparison_report(vendor_metrics, model_metrics)
    
    # Save CSV
    output_path = Path(args.output) if args.output else None
    save_comparison_csv(vendor_metrics, model_metrics, output_path)


if __name__ == "__main__":
    main()
