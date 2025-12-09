"""
Prepare Labeling Batch Script

Extracts a sample of posts from the CVR2024 dataset for human labeling.
Prioritizes diversity across platforms, brands, and vendor sentiment values.
"""

import pandas as pd
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
GOLD_DIR = DATA_DIR / "gold_standard"


def load_source_data() -> pd.DataFrame:
    """Load the CVR2024 compliance data."""
    source_file = SAMPLES_DIR / "rightlan_data_compliance_CVR2024.csv"
    df = pd.read_csv(source_file)
    print(f"Loaded {len(df)} rows from {source_file.name}")
    return df


def filter_valid_posts(df: pd.DataFrame, min_text_length: int = 20) -> pd.DataFrame:
    """Filter to posts with sufficient text content."""
    # Remove rows with missing text
    df = df[df["Text"].notna()].copy()
    
    # Filter by text length
    df["text_length"] = df["Text"].str.len()
    df = df[df["text_length"] >= min_text_length]
    
    print(f"After filtering: {len(df)} posts with text >= {min_text_length} chars")
    return df


def stratified_sample(
    df: pd.DataFrame,
    n_samples: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample prioritizing:
    1. Diversity of vendor sentiment values
    2. Diversity of platforms
    3. Diversity of brands/clients
    """
    random.seed(seed)
    
    # Check available sentiment values
    sentiment_counts = df["Sentiment"].value_counts()
    print(f"\nVendor sentiment distribution:\n{sentiment_counts}")
    
    platform_counts = df["Platform"].value_counts()
    print(f"\nPlatform distribution:\n{platform_counts}")
    
    # Group by sentiment and sample proportionally
    samples = []
    
    # Get unique sentiments
    sentiments = df["Sentiment"].dropna().unique()
    
    if len(sentiments) == 0:
        # No sentiment data - random sample
        print("Warning: No vendor sentiment data found, using random sample")
        return df.sample(n=min(n_samples, len(df)), random_state=seed)
    
    # Calculate samples per sentiment (aim for balance but respect availability)
    per_sentiment = n_samples // len(sentiments)
    remainder = n_samples % len(sentiments)
    
    for i, sentiment in enumerate(sentiments):
        sentiment_df = df[df["Sentiment"] == sentiment]
        n = per_sentiment + (1 if i < remainder else 0)
        n = min(n, len(sentiment_df))
        
        if n > 0:
            sample = sentiment_df.sample(n=n, random_state=seed + i)
            samples.append(sample)
            print(f"Sampled {n} posts with sentiment='{sentiment}'")
    
    # Also include posts with null sentiment if we need more
    null_sentiment = df[df["Sentiment"].isna()]
    if len(null_sentiment) > 0 and sum(len(s) for s in samples) < n_samples:
        needed = n_samples - sum(len(s) for s in samples)
        n = min(needed, len(null_sentiment))
        sample = null_sentiment.sample(n=n, random_state=seed)
        samples.append(sample)
        print(f"Sampled {n} posts with null sentiment")
    
    result = pd.concat(samples, ignore_index=True)
    return result


def prepare_labeling_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to labeling template format."""
    labeling_df = pd.DataFrame({
        "post_id": df["Id"].astype(str),
        "text": df["Text"],
        "brand": df["Client"],
        "platform": df["Platform"],
        "vendor_sentiment": df["Sentiment"],
        "human_sentiment": "",  # To be filled by labeler
        "brand_relevance": "",  # To be filled by labeler
        "negative_type": "",    # To be filled by labeler
        "labeler": "",
        "labeled_at": "",
        "notes": ""
    })
    return labeling_df


def main():
    """Main entry point."""
    print("=" * 60)
    print("Preparing Labeling Batch for Gold Standard Dataset")
    print("=" * 60)
    
    # Load and filter data
    df = load_source_data()
    df = filter_valid_posts(df, min_text_length=20)
    
    # Create stratified sample
    sample_df = stratified_sample(df, n_samples=200, seed=42)
    
    # Convert to labeling format
    labeling_df = prepare_labeling_format(sample_df)
    
    # Save to gold_standard directory
    output_path = GOLD_DIR / "to_label_200.csv"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    labeling_df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Saved {len(labeling_df)} posts to: {output_path}")
    print(f"{'=' * 60}")
    
    # Show sample
    print("\nSample of prepared data:")
    print(labeling_df[["post_id", "brand", "platform", "vendor_sentiment"]].head(10))
    
    # Show text length distribution
    print(f"\nText length stats:")
    print(labeling_df["text"].str.len().describe())


if __name__ == "__main__":
    main()
