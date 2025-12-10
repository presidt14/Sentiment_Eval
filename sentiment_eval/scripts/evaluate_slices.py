"""
Slice-Based Evaluation Script

Combines labeled_200.csv with synthetic_edge_cases.csv and evaluates
model performance across specific data slices.

Usage:
    python scripts/evaluate_slices.py
    python scripts/evaluate_slices.py --model-results results/results_gemma.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = BASE_DIR / "data" / "gold_standard"
SAMPLES_DIR = BASE_DIR / "data" / "samples"

sys.path.insert(0, str(BASE_DIR / "src"))
from utils import normalise_sentiment


def load_and_migrate_labeled_data(path: Path) -> pd.DataFrame:
    """
    Load labeled_200.csv and migrate schema by adding edge_case_type.
    
    Backfill Logic:
    - If brand_relevance=False AND human_sentiment=NEUTRAL -> "promo_hype"
    - Otherwise -> "normal"
    """
    print(f"Loading labeled data from: {path.name}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows")
    
    # Check if edge_case_type already exists
    if "edge_case_type" not in df.columns:
        print("  Migrating schema: adding edge_case_type column")
        
        def assign_edge_case_type(row):
            brand_rel = str(row.get("brand_relevance", "")).lower()
            sentiment = normalise_sentiment(str(row.get("human_sentiment", "")))
            
            is_not_brand_relevant = brand_rel in ("false", "0", "no", "")
            is_neutral = sentiment == "neutral"
            
            if is_not_brand_relevant and is_neutral:
                return "promo_hype"
            return "normal"
        
        df["edge_case_type"] = df.apply(assign_edge_case_type, axis=1)
        
        # Count backfill results
        type_counts = df["edge_case_type"].value_counts()
        print(f"  Backfill results: {type_counts.to_dict()}")
    else:
        print("  edge_case_type column already exists")
    
    return df


def load_edge_cases(path: Path) -> pd.DataFrame:
    """Load synthetic edge cases CSV."""
    print(f"Loading edge cases from: {path.name}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Edge case types: {df['edge_case_type'].value_counts().to_dict()}")
    return df


def merge_datasets(labeled_df: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
    """Merge labeled data with edge cases, ensuring schema alignment."""
    print("\nMerging datasets...")
    
    # Ensure edge_df has all columns from labeled_df
    for col in labeled_df.columns:
        if col not in edge_df.columns:
            edge_df[col] = None
    
    # Ensure labeled_df has all columns from edge_df
    for col in edge_df.columns:
        if col not in labeled_df.columns:
            labeled_df[col] = None
    
    # Generate post_ids for edge cases if missing
    if "post_id" not in edge_df.columns or edge_df["post_id"].isna().all():
        max_id = labeled_df["post_id"].astype(str).str.extract(r'(\d+)').astype(float).max().iloc[0]
        if pd.isna(max_id):
            max_id = 10000
        edge_df["post_id"] = [f"edge_{int(max_id) + i + 1}" for i in range(len(edge_df))]
    
    # Concatenate
    combined = pd.concat([labeled_df, edge_df], ignore_index=True)
    print(f"  Combined dataset: {len(combined)} rows")
    
    return combined


def evaluate_slice_a(df: pd.DataFrame, pred_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Slice A: Noise Suppression
    Filter: edge_case_type IN ["promo_hype", "normal"] AND brand_relevance = FALSE
    Metric: % predicted as NEUTRAL (or if no predictions, % human labeled as NEUTRAL)
    """
    # Normalize brand_relevance
    df = df.copy()
    df["_brand_rel"] = df["brand_relevance"].apply(
        lambda x: str(x).lower() in ("true", "1", "yes")
    )
    
    mask = (
        df["edge_case_type"].isin(["promo_hype", "normal"]) &
        (df["_brand_rel"] == False)
    )
    slice_df = df[mask]
    
    if len(slice_df) == 0:
        return {"slice": "A - Noise Suppression", "count": 0, "metric": None}
    
    # Use predictions if available, otherwise use human labels as baseline
    if pred_col and pred_col in slice_df.columns:
        preds = slice_df[pred_col].apply(normalise_sentiment)
        neutral_rate = (preds == "neutral").mean()
        metric_name = "Predicted NEUTRAL Rate"
    else:
        labels = slice_df["human_sentiment"].apply(normalise_sentiment)
        neutral_rate = (labels == "neutral").mean()
        metric_name = "Human NEUTRAL Rate (baseline)"
    
    return {
        "slice": "A - Noise Suppression",
        "filter": "edge_case_type IN [promo_hype, normal] AND brand_relevance=FALSE",
        "count": len(slice_df),
        "metric_name": metric_name,
        "metric_value": neutral_rate,
    }


def evaluate_slice_b(df: pd.DataFrame, pred_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Slice B: True Brand Sentiment
    Filter: brand_relevance = TRUE
    Metric: Sentiment Accuracy (POS/NEG/NEU)
    """
    df = df.copy()
    df["_brand_rel"] = df["brand_relevance"].apply(
        lambda x: str(x).lower() in ("true", "1", "yes")
    )
    
    slice_df = df[df["_brand_rel"] == True]
    
    if len(slice_df) == 0:
        return {"slice": "B - True Brand Sentiment", "count": 0, "metric": None}
    
    labels = slice_df["human_sentiment"].apply(normalise_sentiment)
    
    if pred_col and pred_col in slice_df.columns:
        preds = slice_df[pred_col].apply(normalise_sentiment)
        accuracy = (preds == labels).mean()
        metric_name = "Sentiment Accuracy"
        
        # Per-class breakdown
        breakdown = {}
        for sent in ["positive", "negative", "neutral"]:
            mask = labels == sent
            if mask.sum() > 0:
                breakdown[sent] = (preds[mask] == labels[mask]).mean()
    else:
        accuracy = None
        metric_name = "Sentiment Accuracy (no predictions)"
        breakdown = labels.value_counts().to_dict()
    
    return {
        "slice": "B - True Brand Sentiment",
        "filter": "brand_relevance=TRUE",
        "count": len(slice_df),
        "metric_name": metric_name,
        "metric_value": accuracy,
        "breakdown": breakdown if 'breakdown' in dir() else None,
    }


def evaluate_slice_c(df: pd.DataFrame, pred_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Slice C: Sarcasm Detection - Brand
    Filter: edge_case_type = "sarcasm_brand"
    Metric: Sarcasm Capture Rate (% predicted as NEGATIVE)
    """
    slice_df = df[df["edge_case_type"] == "sarcasm_brand"]
    
    if len(slice_df) == 0:
        return {"slice": "C - Sarcasm Detection (Brand)", "count": 0, "metric": None}
    
    if pred_col and pred_col in slice_df.columns:
        preds = slice_df[pred_col].apply(normalise_sentiment)
        capture_rate = (preds == "negative").mean()
        metric_name = "Sarcasm Capture Rate (% NEGATIVE)"
        
        # Find failures
        failures = slice_df[preds != "negative"][["post_id", "text", pred_col]].to_dict("records")
    else:
        labels = slice_df["human_sentiment"].apply(normalise_sentiment)
        capture_rate = (labels == "negative").mean()
        metric_name = "Expected NEGATIVE Rate (baseline)"
        failures = []
    
    return {
        "slice": "C - Sarcasm Detection (Brand)",
        "filter": "edge_case_type=sarcasm_brand",
        "count": len(slice_df),
        "metric_name": metric_name,
        "metric_value": capture_rate,
        "failures": failures,
    }


def evaluate_slice_d(df: pd.DataFrame, pred_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Slice D: Sarcasm Rejection - Sport
    Filter: edge_case_type = "sarcasm_sport"
    Metric: False Positive Rate (% incorrectly predicted as NEGATIVE)
    """
    slice_df = df[df["edge_case_type"] == "sarcasm_sport"]
    
    if len(slice_df) == 0:
        return {"slice": "D - Sarcasm Rejection (Sport)", "count": 0, "metric": None}
    
    if pred_col and pred_col in slice_df.columns:
        preds = slice_df[pred_col].apply(normalise_sentiment)
        # FP rate = predicted NEGATIVE when should be NEUTRAL
        labels = slice_df["human_sentiment"].apply(normalise_sentiment)
        fp_rate = ((preds == "negative") & (labels != "negative")).mean()
        metric_name = "False Positive Rate (incorrectly NEGATIVE)"
    else:
        labels = slice_df["human_sentiment"].apply(normalise_sentiment)
        fp_rate = None
        metric_name = "Human label distribution (baseline)"
    
    return {
        "slice": "D - Sarcasm Rejection (Sport)",
        "filter": "edge_case_type=sarcasm_sport",
        "count": len(slice_df),
        "metric_name": metric_name,
        "metric_value": fp_rate,
        "label_dist": slice_df["human_sentiment"].value_counts().to_dict() if fp_rate is None else None,
    }


def evaluate_slice_e(df: pd.DataFrame, pred_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Slice E: Adversarial Cases
    Filter: edge_case_type contains adversarial patterns (mixed_sentiment, implied_causality, etc.)
    Metric: Overall accuracy on complex hybrid cases
    """
    adversarial_types = [
        "mixed_sentiment", "sport_frustration_brand_compliment", "ambiguous_blame",
        "self_deprecation", "implied_causality", "conspiracy_accusation",
        "sport_blame_deflection", "mixed_blame_sport", "mixed_sentiment_balanced",
        "sarcastic_self_blame", "positive_brand_self_deprecation", "sport_loss_brand_compliment"
    ]
    
    slice_df = df[df["edge_case_type"].isin(adversarial_types)]
    
    if len(slice_df) == 0:
        return {"slice": "E - Adversarial Cases", "count": 0, "metric_value": None}
    
    failures = []
    
    if pred_col and pred_col in slice_df.columns:
        preds = slice_df[pred_col].apply(normalise_sentiment)
        labels = slice_df["human_sentiment"].apply(normalise_sentiment)
        
        # Calculate accuracy
        accuracy = (preds == labels).mean()
        metric_name = "Overall Accuracy"
        
        # Find failures
        mask = preds != labels
        for idx, row in slice_df[mask].iterrows():
            failures.append({
                "post_id": row.get("post_id", idx),
                "text": str(row.get("text", ""))[:60],
                "edge_case_type": row.get("edge_case_type"),
                "expected": normalise_sentiment(str(row.get("human_sentiment", ""))),
                "predicted": normalise_sentiment(str(row.get(pred_col, ""))),
            })
    else:
        accuracy = None
        metric_name = "Human label distribution (baseline)"
    
    return {
        "slice": "E - Adversarial Cases",
        "filter": "edge_case_type in adversarial_types",
        "count": len(slice_df),
        "metric_name": metric_name,
        "metric_value": accuracy,
        "failures": failures,
        "label_dist": slice_df["human_sentiment"].value_counts().to_dict() if accuracy is None else None,
    }


def print_summary_table(results: list) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("SLICE-BASED EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Slice':<35} {'Count':>8} {'Metric':>35}")
    print("-" * 80)
    
    for r in results:
        slice_name = r["slice"]
        count = r["count"]
        
        if r.get("metric_value") is not None:
            metric_str = f"{r['metric_name']}: {r['metric_value']:.1%}"
        elif r.get("label_dist"):
            metric_str = f"Labels: {r['label_dist']}"
        elif r.get("breakdown"):
            metric_str = f"Breakdown: {r['breakdown']}"
        else:
            metric_str = "N/A"
        
        print(f"{slice_name:<35} {count:>8} {metric_str:>35}")
    
    print("-" * 80)
    
    # Print failures for Slice C if any
    for r in results:
        if r["slice"].startswith("C") and r.get("failures"):
            print(f"\n⚠️  Slice C Failures ({len(r['failures'])} cases):")
            for f in r["failures"]:
                text_preview = str(f.get("text", ""))[:60] + "..."
                print(f"  - ID {f.get('post_id')}: {text_preview}")
    
    # Print failures for Slice E if any
    for r in results:
        if r["slice"].startswith("E") and r.get("failures"):
            print(f"\n⚠️  Slice E Failures ({len(r['failures'])} cases):")
            for f in r["failures"]:
                text_preview = str(f.get("text", ""))[:60] + "..."
                print(f"  - [{f.get('edge_case_type')}] Expected: {f.get('expected')}, Got: {f.get('predicted')}")
                print(f"    Text: {text_preview}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Slice-based evaluation")
    parser.add_argument(
        "--labeled",
        type=str,
        default="data/gold_standard/labeled_200.csv",
        help="Path to labeled data CSV",
    )
    parser.add_argument(
        "--edge-cases",
        type=str,
        default="data/samples/synthetic_edge_cases.csv",
        help="Path to synthetic edge cases CSV",
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
        default="data/gold_standard/gold_standard_v2.csv",
        help="Path to save combined dataset",
    )
    parser.add_argument(
        "--adversarial",
        type=str,
        default="data/gold_standard/adversarial_cases.csv",
        help="Path to adversarial cases CSV (optional)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    labeled_path = BASE_DIR / args.labeled
    edge_path = BASE_DIR / args.edge_cases
    output_path = BASE_DIR / args.output
    
    # Check files exist
    if not labeled_path.exists():
        print(f"Error: Labeled data not found: {labeled_path}")
        sys.exit(1)
    if not edge_path.exists():
        print(f"Error: Edge cases not found: {edge_path}")
        sys.exit(1)
    
    # Task 1: Schema Migration
    print("\n" + "=" * 60)
    print("TASK 1: Schema Migration")
    print("=" * 60)
    labeled_df = load_and_migrate_labeled_data(labeled_path)
    
    # Task 2: Dataset Merging
    print("\n" + "=" * 60)
    print("TASK 2: Dataset Merging")
    print("=" * 60)
    edge_df = load_edge_cases(edge_path)
    combined_df = merge_datasets(labeled_df, edge_df)
    
    # Load adversarial cases if they exist
    adversarial_path = BASE_DIR / args.adversarial
    if adversarial_path.exists():
        print(f"\nLoading adversarial cases from: {adversarial_path.name}")
        adversarial_df = pd.read_csv(adversarial_path)
        print(f"  Loaded {len(adversarial_df)} adversarial cases")
        print(f"  Types: {adversarial_df['edge_case_type'].value_counts().to_dict()}")
        
        # Add post_id if missing
        if "post_id" not in adversarial_df.columns:
            max_id = combined_df["post_id"].astype(str).str.extract(r'(\d+)').astype(float).max().values[0]
            adversarial_df["post_id"] = [f"adv_{int(max_id) + i + 1}" for i in range(len(adversarial_df))]
        
        # Merge adversarial cases
        combined_df = pd.concat([combined_df, adversarial_df], ignore_index=True)
        print(f"  Combined dataset now: {len(combined_df)} rows")
    
    # Determine prediction column if model results provided
    pred_col = None
    if args.model_results:
        model_path = BASE_DIR / args.model_results
        if model_path.exists():
            print(f"\nLoading model results from: {model_path.name}")
            model_df = pd.read_csv(model_path)
            # Merge with combined
            combined_df = combined_df.merge(model_df, on="post_id", how="left", suffixes=("", "_pred"))
            # Find prediction column
            pred_cols = [c for c in combined_df.columns if c.endswith("_sentiment") 
                        and c not in ["human_sentiment", "vendor_sentiment"]]
            if pred_cols:
                pred_col = pred_cols[0]
                print(f"  Using prediction column: {pred_col}")
    
    # Task 3: Evaluation by Slice
    print("\n" + "=" * 60)
    print("TASK 3: Evaluation by Slice")
    print("=" * 60)
    
    results = [
        evaluate_slice_a(combined_df, pred_col),
        evaluate_slice_b(combined_df, pred_col),
        evaluate_slice_c(combined_df, pred_col),
        evaluate_slice_d(combined_df, pred_col),
        evaluate_slice_e(combined_df, pred_col),
    ]
    
    print_summary_table(results)
    
    # Save combined dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"✅ Saved combined dataset to: {output_path}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Edge case type distribution:")
    for ect, count in combined_df["edge_case_type"].value_counts().items():
        print(f"     {ect}: {count}")


if __name__ == "__main__":
    main()
