from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .data_loader import load_posts
from .utils import normalise_sentiment, normalise_negative_type

BASE_DIR = Path(__file__).resolve().parents[1]


# =============================================================================
# Extended Evaluation for Gemma-First Architecture
# =============================================================================

def evaluate_extended(
    results_path: str | Path,
    labels_path: str | Path,
    human_sentiment_col: str = "human_sentiment",
    human_brand_rel_col: str = "brand_relevance",
    human_neg_type_col: str = "negative_type",
    id_col: str = "post_id",
    model_prefix: str = None,
) -> Dict[str, Any]:
    """
    Evaluate model predictions against human labels for extended schema.
    
    Evaluates:
    - Sentiment accuracy (3-class)
    - Brand relevance accuracy (binary)
    - Negative type accuracy (4-class, only for negative posts)
    
    Args:
        results_path: Path to model results CSV
        labels_path: Path to human-labeled gold standard CSV
        human_sentiment_col: Column name for human sentiment labels
        human_brand_rel_col: Column name for human brand_relevance labels
        human_neg_type_col: Column name for human negative_type labels
        id_col: Column name for post IDs
        model_prefix: Model name prefix (e.g., "gemma" for "gemma_sentiment")
        
    Returns:
        Dictionary with comprehensive metrics
    """
    results_path = Path(results_path)
    labels_path = Path(labels_path)
    
    print(f"--- Extended Evaluation ---")
    print(f"Results: {results_path.name}")
    print(f"Labels: {labels_path.name}")
    
    try:
        res_df = load_posts(results_path)
        lab_df = load_posts(labels_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        return {}
    
    # Determine which columns to use from results
    if model_prefix:
        sentiment_col = f"{model_prefix}_sentiment"
        brand_rel_col = f"{model_prefix}_brand_relevance"
        neg_type_col = f"{model_prefix}_negative_type"
    else:
        # Auto-detect model columns
        sentiment_cols = [c for c in res_df.columns if c.endswith("_sentiment")]
        if sentiment_cols:
            model_prefix = sentiment_cols[0].replace("_sentiment", "")
            sentiment_col = sentiment_cols[0]
            brand_rel_col = f"{model_prefix}_brand_relevance"
            neg_type_col = f"{model_prefix}_negative_type"
        else:
            print("Error: No model sentiment columns found")
            return {}
    
    # Merge on ID
    label_cols = [id_col, human_sentiment_col]
    if human_brand_rel_col in lab_df.columns:
        label_cols.append(human_brand_rel_col)
    if human_neg_type_col in lab_df.columns:
        label_cols.append(human_neg_type_col)
    
    merged = res_df.merge(lab_df[label_cols], on=id_col, how="inner")
    print(f"Merged {len(merged)} rows for evaluation")
    
    metrics = {
        "model": model_prefix,
        "total_samples": len(merged),
    }
    
    # === Sentiment Evaluation ===
    if sentiment_col in merged.columns and human_sentiment_col in merged.columns:
        y_pred = merged[sentiment_col].apply(normalise_sentiment)
        y_true = merged[human_sentiment_col].apply(normalise_sentiment)
        
        labels = ["negative", "neutral", "positive"]
        
        metrics["sentiment_accuracy"] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics for sentiment
        for label in labels:
            y_true_bin = (y_true == label).astype(int)
            y_pred_bin = (y_pred == label).astype(int)
            
            metrics[f"sentiment_{label}_precision"] = precision_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            metrics[f"sentiment_{label}_recall"] = recall_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            metrics[f"sentiment_{label}_f1"] = f1_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
        
        print(f"\nSentiment Classification Report:")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    # === Brand Relevance Evaluation ===
    if brand_rel_col in merged.columns and human_brand_rel_col in merged.columns:
        # Normalize to boolean
        def to_bool(x):
            if pd.isna(x):
                return None
            if isinstance(x, bool):
                return x
            return str(x).lower() in ("true", "yes", "1")
        
        y_pred_br = merged[brand_rel_col].apply(to_bool)
        y_true_br = merged[human_brand_rel_col].apply(to_bool)
        
        # Filter out None values
        mask = y_pred_br.notna() & y_true_br.notna()
        if mask.sum() > 0:
            y_pred_br = y_pred_br[mask].astype(bool)
            y_true_br = y_true_br[mask].astype(bool)
            
            metrics["brand_relevance_accuracy"] = accuracy_score(y_true_br, y_pred_br)
            metrics["brand_relevance_precision"] = precision_score(
                y_true_br, y_pred_br, zero_division=0
            )
            metrics["brand_relevance_recall"] = recall_score(
                y_true_br, y_pred_br, zero_division=0
            )
            metrics["brand_relevance_f1"] = f1_score(
                y_true_br, y_pred_br, zero_division=0
            )
            metrics["brand_relevance_samples"] = mask.sum()
            
            print(f"\nBrand Relevance: Accuracy={metrics['brand_relevance_accuracy']:.1%}")
    
    # === Negative Type Evaluation (only for negative posts) ===
    if neg_type_col in merged.columns and human_neg_type_col in merged.columns:
        # Filter to negative sentiment posts only
        neg_mask = merged[human_sentiment_col].apply(normalise_sentiment) == "negative"
        neg_df = merged[neg_mask]
        
        if len(neg_df) > 0:
            y_pred_nt = neg_df[neg_type_col].apply(normalise_negative_type)
            y_true_nt = neg_df[human_neg_type_col].apply(normalise_negative_type)
            
            # Filter out None values for accuracy calc
            mask = y_pred_nt.notna() & y_true_nt.notna()
            if mask.sum() > 0:
                y_pred_nt = y_pred_nt[mask]
                y_true_nt = y_true_nt[mask]
                
                metrics["negative_type_accuracy"] = accuracy_score(y_true_nt, y_pred_nt)
                metrics["negative_type_samples"] = mask.sum()
                
                print(f"\nNegative Type: Accuracy={metrics['negative_type_accuracy']:.1%} ({mask.sum()} samples)")
    
    return metrics


def evaluate_vendor_vs_model(
    gold_standard_path: str | Path,
    model_results_path: Optional[str | Path] = None,
    vendor_col: str = "vendor_sentiment",
    human_col: str = "human_sentiment",
    id_col: str = "post_id",
) -> pd.DataFrame:
    """
    Compare vendor baseline and model predictions against human labels.
    
    Returns DataFrame with side-by-side metrics.
    """
    gold_path = Path(gold_standard_path)
    
    print(f"--- Vendor vs Model Comparison ---")
    
    try:
        gold_df = load_posts(gold_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    
    results = []
    labels = ["negative", "neutral", "positive"]
    
    # Evaluate vendor
    if vendor_col in gold_df.columns and human_col in gold_df.columns:
        mask = gold_df[vendor_col].notna() & gold_df[human_col].notna()
        eval_df = gold_df[mask]
        
        if len(eval_df) > 0:
            y_vendor = eval_df[vendor_col].apply(normalise_sentiment)
            y_human = eval_df[human_col].apply(normalise_sentiment)
            
            vendor_metrics = {
                "source": "vendor",
                "accuracy": accuracy_score(y_human, y_vendor),
                "samples": len(eval_df),
            }
            
            for label in labels:
                y_true_bin = (y_human == label).astype(int)
                y_pred_bin = (y_vendor == label).astype(int)
                vendor_metrics[f"{label}_precision"] = precision_score(
                    y_true_bin, y_pred_bin, zero_division=0
                )
                vendor_metrics[f"{label}_recall"] = recall_score(
                    y_true_bin, y_pred_bin, zero_division=0
                )
                vendor_metrics[f"{label}_f1"] = f1_score(
                    y_true_bin, y_pred_bin, zero_division=0
                )
            
            results.append(vendor_metrics)
            print(f"\nVendor: Accuracy={vendor_metrics['accuracy']:.1%}")
    
    # Evaluate model if results provided
    if model_results_path:
        model_path = Path(model_results_path)
        if model_path.exists():
            model_df = load_posts(model_path)
            merged = gold_df.merge(model_df, on=id_col, how="inner", suffixes=("", "_model"))
            
            # Find model sentiment column
            model_cols = [c for c in merged.columns if c.endswith("_sentiment") 
                         and c not in [vendor_col, human_col]]
            
            for model_col in model_cols:
                model_name = model_col.replace("_sentiment", "")
                mask = merged[model_col].notna() & merged[human_col].notna()
                eval_df = merged[mask]
                
                if len(eval_df) > 0:
                    y_model = eval_df[model_col].apply(normalise_sentiment)
                    y_human = eval_df[human_col].apply(normalise_sentiment)
                    
                    model_metrics = {
                        "source": model_name,
                        "accuracy": accuracy_score(y_human, y_model),
                        "samples": len(eval_df),
                    }
                    
                    for label in labels:
                        y_true_bin = (y_human == label).astype(int)
                        y_pred_bin = (y_model == label).astype(int)
                        model_metrics[f"{label}_precision"] = precision_score(
                            y_true_bin, y_pred_bin, zero_division=0
                        )
                        model_metrics[f"{label}_recall"] = recall_score(
                            y_true_bin, y_pred_bin, zero_division=0
                        )
                        model_metrics[f"{label}_f1"] = f1_score(
                            y_true_bin, y_pred_bin, zero_division=0
                        )
                    
                    results.append(model_metrics)
                    print(f"\n{model_name}: Accuracy={model_metrics['accuracy']:.1%}")
    
    return pd.DataFrame(results)


def evaluate(
    results_path: str | Path,
    labels_path: str | Path,
    human_col: str = "human_sentiment",
    id_col: str = "post_id",
    model_cols: List[str] | None = None,
) -> pd.DataFrame:
    results_path = Path(results_path)
    labels_path = Path(labels_path)

    print(f"--- Loading results from: {results_path.name} ---")

    try:
        res_df = load_posts(results_path)
        lab_df = load_posts(labels_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        return pd.DataFrame()

    # Merge on ID to ensure we compare the correct rows
    merged = res_df.merge(lab_df[[id_col, human_col]], on=id_col, how="inner")

    # Normalize ground truth
    merged[human_col] = merged[human_col].apply(normalise_sentiment)

    if model_cols is None:
        # Infer all *_sentiment columns except the human/supplier columns
        model_cols = [
            c for c in merged.columns if c.endswith("_sentiment") and c != human_col
        ]

    records = []
    labels = ["negative", "neutral", "positive"]

    for col in model_cols:
        if col == human_col:
            continue

        model_name = col.replace("_sentiment", "")
        print(f"\nEvaluating Model: {model_name.upper()}")

        # Normalize predictions
        y_pred = merged[col].apply(normalise_sentiment)
        y_true = merged[human_col]

        # Calculate basic accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate detailed metrics
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        print(report)

        # Confusion Matrix (optional print)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Confusion Matrix (labels: {labels}):\n{cm}")

        records.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "total": len(y_true),
                "correct": int((y_pred == y_true).sum()),
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    res_path = BASE_DIR / "results" / "results_posts_sample.csv"
    labels_path = BASE_DIR / "data" / "samples" / "labels_sample.csv"

    if res_path.exists() and labels_path.exists():
        df = evaluate(res_path, labels_path)
        print("\n--- Summary DataFrame ---")
        print(df)
    else:
        print("Results or labels file not found. Please run 'src.run_batch' first.")
