from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_loader import load_posts
from .utils import normalise_sentiment

BASE_DIR = Path(__file__).resolve().parents[1]

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
        model_cols = [c for c in merged.columns if c.endswith("_sentiment") and c != human_col]

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

        records.append({
            "model": model_name,
            "accuracy": accuracy,
            "total": len(y_true),
            "correct": int((y_pred == y_true).sum()),
        })

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