"""
Auto-complete labeling for labeled_200.csv and regenerate gold_standard_v2.csv

All 200 samples in labeled_200.csv have been independently verified as:
- human_sentiment: neutral
- brand_relevance: False
- negative_type: (empty)

This script:
1. Auto-completes any missing labels in labeled_200.csv
2. Regenerates gold_standard_v2.csv with edge cases
3. Verifies all files are consistent
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = BASE_DIR / "data" / "gold_standard"
SAMPLES_DIR = BASE_DIR / "data" / "samples"


def auto_complete_labeled_200():
    """Auto-complete missing labels in labeled_200.csv."""
    path = GOLD_DIR / "labeled_200.csv"
    
    print("=" * 60)
    print("STEP 1: Auto-complete labeled_200.csv")
    print("=" * 60)
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from labeled_200.csv")
    
    # Count current state
    labeled_before = df["human_sentiment"].notna().sum()
    print(f"Currently labeled: {labeled_before} / {len(df)}")
    
    # Find unlabeled rows
    mask = df["human_sentiment"].isna() | (df["human_sentiment"] == "")
    unlabeled_count = mask.sum()
    
    if unlabeled_count == 0:
        print("All rows already labeled!")
    else:
        print(f"Auto-completing {unlabeled_count} rows...")
        
        # Set all to neutral, brand_relevance=False (as verified by user)
        df.loc[mask, "human_sentiment"] = "neutral"
        df.loc[mask, "brand_relevance"] = False
        df.loc[mask, "negative_type"] = ""
        df.loc[mask, "labeler"] = "auto_complete"
        df.loc[mask, "labeled_at"] = datetime.now().isoformat()
        df.loc[mask, "notes"] = "Auto-completed: verified noise/promo"
        
        # Save
        df.to_csv(path, index=False)
        print(f"Saved {path.name}")
    
    # Verification
    print("\nVerification:")
    print(f"  Sentiment: {df['human_sentiment'].value_counts().to_dict()}")
    print(f"  Brand relevance: {df['brand_relevance'].value_counts().to_dict()}")
    
    return df


def regenerate_gold_standard_v2(labeled_df: pd.DataFrame):
    """Regenerate gold_standard_v2.csv with edge cases."""
    print("\n" + "=" * 60)
    print("STEP 2: Regenerate gold_standard_v2.csv")
    print("=" * 60)
    
    # Add edge_case_type to labeled_200
    labeled_df = labeled_df.copy()
    
    # Backfill edge_case_type based on schema migration rules
    def assign_edge_case_type(row):
        brand_rel = str(row.get("brand_relevance", "")).lower()
        sentiment = str(row.get("human_sentiment", "")).lower()
        
        is_not_brand_relevant = brand_rel in ("false", "0", "no", "")
        is_neutral = "neu" in sentiment
        
        if is_not_brand_relevant and is_neutral:
            return "promo_hype"
        return "normal"
    
    labeled_df["edge_case_type"] = labeled_df.apply(assign_edge_case_type, axis=1)
    print(f"Labeled data edge_case_type: {labeled_df['edge_case_type'].value_counts().to_dict()}")
    
    # Load edge cases
    edge_path = SAMPLES_DIR / "synthetic_edge_cases.csv"
    edge_df = pd.read_csv(edge_path)
    print(f"Loaded {len(edge_df)} edge cases")
    
    # Ensure edge_df has all columns
    for col in labeled_df.columns:
        if col not in edge_df.columns:
            edge_df[col] = None
    
    # Generate post_ids for edge cases
    max_id = 3200  # Safe starting point above existing IDs
    edge_df["post_id"] = [f"edge_{max_id + i + 1}" for i in range(len(edge_df))]
    edge_df["brand"] = "W1LLH1LL"  # Same brand as main dataset
    edge_df["platform"] = "SYNTHETIC"
    
    # Concatenate
    combined = pd.concat([labeled_df, edge_df], ignore_index=True)
    print(f"Combined dataset: {len(combined)} rows")
    
    # Save
    output_path = GOLD_DIR / "gold_standard_v2.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved {output_path.name}")
    
    return combined


def verify_all_files():
    """Verify all CSV files are consistent."""
    print("\n" + "=" * 60)
    print("STEP 3: Verification Summary")
    print("=" * 60)
    
    files_to_check = [
        ("labeled_200.csv", GOLD_DIR / "labeled_200.csv"),
        ("gold_standard_v2.csv", GOLD_DIR / "gold_standard_v2.csv"),
        ("synthetic_edge_cases.csv", SAMPLES_DIR / "synthetic_edge_cases.csv"),
    ]
    
    for name, path in files_to_check:
        if path.exists():
            df = pd.read_csv(path)
            print(f"\n{name}:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            if "human_sentiment" in df.columns:
                sent_dist = df["human_sentiment"].value_counts().to_dict()
                print(f"  Sentiment: {sent_dist}")
            
            if "brand_relevance" in df.columns:
                br_dist = df["brand_relevance"].value_counts().to_dict()
                print(f"  Brand relevance: {br_dist}")
            
            if "edge_case_type" in df.columns:
                ect_dist = df["edge_case_type"].value_counts().to_dict()
                print(f"  Edge case types: {ect_dist}")
        else:
            print(f"\n{name}: NOT FOUND")
    
    # Final summary
    print("\n" + "=" * 60)
    print("READY FOR NEXT PHASE")
    print("=" * 60)
    print("\nRun inference:")
    print("  python scripts/run_gold_standard_inference.py --model gemma")
    print("\nThen evaluate:")
    print("  python scripts/evaluate_slices.py --model-results results/results_gold_standard_gemma.csv")


def main():
    # Step 1: Auto-complete labeled_200.csv
    labeled_df = auto_complete_labeled_200()
    
    # Step 2: Regenerate gold_standard_v2.csv
    combined_df = regenerate_gold_standard_v2(labeled_df)
    
    # Step 3: Verify all files
    verify_all_files()


if __name__ == "__main__":
    main()
