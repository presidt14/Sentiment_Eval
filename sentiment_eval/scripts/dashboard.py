"""
Sentiment Eval Dashboard - Failure Explorer & Dataset Editor

A Streamlit app for visualizing evaluation results and curating the gold standard dataset.

Usage:
    streamlit run scripts/dashboard.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st

# Set up path for imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils import normalise_sentiment

# =============================================================================
# Configuration
# =============================================================================

GOLD_STANDARD_PATH = BASE_DIR / "data" / "gold_standard" / "gold_standard_v3.csv"
MODEL_RESULTS_PATH = BASE_DIR / "results" / "results_gold_standard_v3_gemma.csv"

# Thresholds (mirroring evaluate_slices.py)
SLICE_THRESHOLDS = {
    "A": 1.00,  # Noise Suppression: 100%
    "B": 0.90,  # True Brand Sentiment: 90%
    "C": 0.90,  # Sarcasm Detection (Brand): 90%
    "D": 0.90,  # Sarcasm Rejection (Sport): 90% (inverted - low FP rate)
    "E": 0.90,  # Adversarial Cases: 90%
    "F": 1.00,  # Brand Relevance Health: 100%
}

# Adversarial edge case types (for Slice E)
ADVERSARIAL_TYPES = [
    "mixed_sentiment", "sport_frustration_brand_compliment", "ambiguous_blame",
    "self_deprecation", "implied_causality", "conspiracy_accusation",
    "sport_blame_deflection", "mixed_blame_sport", "mixed_sentiment_balanced",
    "sarcastic_self_blame", "positive_brand_self_deprecation", "sport_loss_brand_compliment"
]


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_brand_rel(val) -> Optional[bool]:
    """Normalize brand_relevance to boolean."""
    if pd.isna(val):
        return None
    if isinstance(val, bool):
        return val
    val_str = str(val).lower().strip()
    if val_str in ("true", "1", "yes"):
        return True
    elif val_str in ("false", "0", "no"):
        return False
    return None


def load_and_merge_data() -> Optional[pd.DataFrame]:
    """Load gold standard and model results, merge them."""
    if not GOLD_STANDARD_PATH.exists():
        st.error(f"❌ Gold standard file not found: {GOLD_STANDARD_PATH}")
        return None
    
    # Load gold standard
    gold_df = pd.read_csv(GOLD_STANDARD_PATH)
    
    # Check for model results
    if not MODEL_RESULTS_PATH.exists():
        st.warning(f"⚠️ Model results not found: {MODEL_RESULTS_PATH}")
        st.info("Showing gold standard data only. Run inference to generate model results.")
        gold_df["has_predictions"] = False
        return gold_df
    
    # Load model results
    results_df = pd.read_csv(MODEL_RESULTS_PATH)
    
    # Standardize column names to match Dashboard expectations
    # Map common model output names to expected column names
    results_df = results_df.rename(columns={
        "brand_relevant": "pred_brand_relevance",
        "relevance": "pred_brand_relevance",
        "is_relevant": "pred_brand_relevance",
        "sentiment": "pred_sentiment"
    })
    
    # Merge on post_id
    merged = gold_df.merge(results_df, on="post_id", how="left", suffixes=("", "_pred"))
    
    # Find prediction columns
    pred_col = None
    brand_rel_pred_col = None
    for col in merged.columns:
        if col.endswith("_sentiment") and col not in ["human_sentiment", "vendor_sentiment"]:
            pred_col = col
            model_prefix = col.replace("_sentiment", "")
            brand_rel_pred_col = f"{model_prefix}_brand_relevance"
            break
    
    if pred_col:
        # Normalize sentiments (case-insensitive comparison)
        merged["pred_sentiment"] = merged[pred_col].apply(normalise_sentiment)
        merged["human_sentiment_norm"] = merged["human_sentiment"].apply(normalise_sentiment)
        
        # Get brand relevance predictions
        if brand_rel_pred_col and brand_rel_pred_col in merged.columns:
            merged["pred_brand_relevance"] = merged[brand_rel_pred_col].apply(normalize_brand_rel)
        else:
            merged["pred_brand_relevance"] = None
        merged["human_brand_relevance"] = merged["brand_relevance"].apply(normalize_brand_rel)
        
        # Track which rows have predictions vs missing
        merged["has_sentiment_pred"] = merged["pred_sentiment"].notna()
        merged["has_brand_rel_pred"] = merged["pred_brand_relevance"].notna()
        merged["has_predictions"] = merged["has_sentiment_pred"]  # At minimum need sentiment
        
        # Compute correctness - only for rows WITH predictions
        sentiment_match = merged["pred_sentiment"] == merged["human_sentiment_norm"]
        
        # Brand relevance: match if both present, or ignore if prediction missing
        brand_rel_pred_missing = merged["pred_brand_relevance"].isna()
        brand_rel_match = (merged["pred_brand_relevance"] == merged["human_brand_relevance"]) | brand_rel_pred_missing
        
        # is_correct: True if matches, False if wrong, None if no prediction
        merged["is_correct"] = None  # Default
        has_pred_mask = merged["has_sentiment_pred"]
        merged.loc[has_pred_mask, "is_correct"] = (
            sentiment_match[has_pred_mask] & brand_rel_match[has_pred_mask]
        )
        
        # Separate status column for clarity
        def get_status(row):
            if not row["has_sentiment_pred"]:
                return "No Prediction"
            elif row["is_correct"] == True:
                return "Correct"
            else:
                return "Incorrect"
        merged["status"] = merged.apply(get_status, axis=1)
    else:
        merged["is_correct"] = None
        merged["has_predictions"] = False
        merged["status"] = "No Prediction"
    
    return merged


def assign_slice(row: pd.Series) -> str:
    """Assign a slice letter to each row based on evaluation logic."""
    edge_case_type = str(row.get("edge_case_type", "")).lower()
    brand_rel = normalize_brand_rel(row.get("brand_relevance"))
    
    # Slice A: Noise Suppression
    if edge_case_type in ["promo_hype", "normal"] and brand_rel == False:
        return "A"
    
    # Slice C: Sarcasm Detection (Brand)
    if edge_case_type == "sarcasm_brand":
        return "C"
    
    # Slice D: Sarcasm Rejection (Sport)
    if edge_case_type == "sarcasm_sport":
        return "D"
    
    # Slice E: Adversarial Cases
    if edge_case_type in [t.lower() for t in ADVERSARIAL_TYPES]:
        return "E"
    
    # Slice B: True Brand Sentiment (brand_relevance = TRUE)
    if brand_rel == True:
        return "B"
    
    # Default: Slice F (Brand Relevance Health) - all rows with brand_relevance
    if brand_rel is not None:
        return "F"
    
    return "Unknown"


def calculate_slice_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculate metrics for each slice, mirroring evaluate_slices.py logic."""
    metrics = {}
    
    if not df["has_predictions"].any():
        return metrics
    
    # Slice A: Noise Suppression
    slice_a = df[(df["edge_case_type"].isin(["promo_hype", "normal"])) & 
                 (df["human_brand_relevance"] == False)]
    if len(slice_a) > 0:
        neutral_rate = (slice_a["pred_sentiment"] == "neutral").mean()
        metrics["A"] = {
            "name": "Noise Suppression",
            "count": len(slice_a),
            "value": neutral_rate,
            "threshold": SLICE_THRESHOLDS["A"],
            "passed": neutral_rate >= SLICE_THRESHOLDS["A"],
            "metric_type": "rate"
        }
    
    # Slice B: True Brand Sentiment
    slice_b = df[df["human_brand_relevance"] == True]
    if len(slice_b) > 0:
        accuracy = (slice_b["pred_sentiment"] == slice_b["human_sentiment_norm"]).mean()
        metrics["B"] = {
            "name": "True Brand Sentiment",
            "count": len(slice_b),
            "value": accuracy,
            "threshold": SLICE_THRESHOLDS["B"],
            "passed": accuracy >= SLICE_THRESHOLDS["B"],
            "metric_type": "accuracy"
        }
    
    # Slice C: Sarcasm Detection (Brand)
    slice_c = df[df["edge_case_type"] == "sarcasm_brand"]
    if len(slice_c) > 0:
        capture_rate = (slice_c["pred_sentiment"] == "negative").mean()
        metrics["C"] = {
            "name": "Sarcasm Detection (Brand)",
            "count": len(slice_c),
            "value": capture_rate,
            "threshold": SLICE_THRESHOLDS["C"],
            "passed": capture_rate >= SLICE_THRESHOLDS["C"],
            "metric_type": "rate"
        }
    
    # Slice D: Sarcasm Rejection (Sport) - INVERTED (FP rate should be LOW)
    slice_d = df[df["edge_case_type"] == "sarcasm_sport"]
    if len(slice_d) > 0:
        # FP rate = predicted NEGATIVE when should NOT be NEGATIVE
        fp_rate = ((slice_d["pred_sentiment"] == "negative") & 
                   (slice_d["human_sentiment_norm"] != "negative")).mean()
        max_fp_rate = 1.0 - SLICE_THRESHOLDS["D"]  # 0.10 for 90% threshold
        metrics["D"] = {
            "name": "Sarcasm Rejection (Sport)",
            "count": len(slice_d),
            "value": fp_rate,
            "threshold": max_fp_rate,
            "passed": fp_rate <= max_fp_rate,
            "metric_type": "fp_rate",
            "inverted": True
        }
    
    # Slice E: Adversarial Cases
    slice_e = df[df["edge_case_type"].isin(ADVERSARIAL_TYPES)]
    if len(slice_e) > 0:
        accuracy = (slice_e["pred_sentiment"] == slice_e["human_sentiment_norm"]).mean()
        metrics["E"] = {
            "name": "Adversarial Cases",
            "count": len(slice_e),
            "value": accuracy,
            "threshold": SLICE_THRESHOLDS["E"],
            "passed": accuracy >= SLICE_THRESHOLDS["E"],
            "metric_type": "accuracy"
        }
    
    # Slice F: Brand Relevance Health
    slice_f = df[df["human_brand_relevance"].notna() & df["pred_brand_relevance"].notna()]
    if len(slice_f) > 0:
        accuracy = (slice_f["human_brand_relevance"] == slice_f["pred_brand_relevance"]).mean()
        metrics["F"] = {
            "name": "Brand Relevance Health",
            "count": len(slice_f),
            "value": accuracy,
            "threshold": SLICE_THRESHOLDS["F"],
            "passed": accuracy >= SLICE_THRESHOLDS["F"],
            "metric_type": "accuracy"
        }
    
    return metrics


def save_gold_standard(df: pd.DataFrame) -> bool:
    """Save the gold standard with a timestamped backup."""
    try:
        # Create backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = GOLD_STANDARD_PATH.parent / f"gold_standard_v3_backup_{timestamp}.csv"
        
        # Read original and save as backup
        if GOLD_STANDARD_PATH.exists():
            original = pd.read_csv(GOLD_STANDARD_PATH)
            original.to_csv(backup_path, index=False)
        
        # Save columns that belong to gold standard (exclude computed/prediction columns)
        gold_cols = [
            "post_id", "text", "brand", "platform", "vendor_sentiment",
            "human_sentiment", "brand_relevance", "negative_type", "labeler",
            "labeled_at", "notes", "edge_case_type"
        ]
        save_cols = [c for c in gold_cols if c in df.columns]
        df[save_cols].to_csv(GOLD_STANDARD_PATH, index=False)
        
        return True
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return False


# =============================================================================
# Streamlit App
# =============================================================================

def main():
    st.set_page_config(
        page_title="Sentiment Eval Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Sentiment Eval Dashboard")
    st.caption("Failure Explorer & Dataset Editor for Compliance Model Evaluation")
    
    # Load data
    df = load_and_merge_data()
    if df is None:
        return
    
    # Assign slices
    df["slice"] = df.apply(assign_slice, axis=1)
    
    # ==========================================================================
    # Sidebar: Metrics Scorecard
    # ==========================================================================
    with st.sidebar:
        st.header("📈 Quality Scorecard")
        
        if df["has_predictions"].any():
            metrics = calculate_slice_metrics(df)
            
            for slice_letter in ["A", "B", "C", "D", "E", "F"]:
                if slice_letter in metrics:
                    m = metrics[slice_letter]
                    
                    # Format value and delta
                    if m.get("inverted"):
                        # For FP rate, lower is better
                        delta = m["threshold"] - m["value"]
                        delta_str = f"{delta:+.1%}"
                        value_str = f"{m['value']:.1%} FP"
                    else:
                        delta = m["value"] - m["threshold"]
                        delta_str = f"{delta:+.1%}"
                        value_str = f"{m['value']:.1%}"
                    
                    st.metric(
                        label=f"Slice {slice_letter}: {m['name']} (n={m['count']})",
                        value=value_str,
                        delta=delta_str,
                        delta_color="normal" if m["passed"] else "inverse"
                    )
                else:
                    st.metric(
                        label=f"Slice {slice_letter}",
                        value="N/A",
                        delta="No data"
                    )
            
            # Overall summary
            st.divider()
            passed_count = sum(1 for m in metrics.values() if m["passed"])
            total_count = len(metrics)
            
            if passed_count == total_count:
                st.success(f"✅ Quality Gate: {passed_count}/{total_count} Passed")
            else:
                st.error(f"❌ Quality Gate: {passed_count}/{total_count} Passed")
        else:
            st.warning("No predictions available. Run inference first.")
        
        st.divider()
        st.header("🔍 Filters")
        
        # Slice filter
        slice_options = ["All"] + sorted(df["slice"].unique().tolist())
        selected_slice = st.selectbox("Slice", slice_options)
        
        # Edge case type filter
        edge_types = ["All"] + sorted(df["edge_case_type"].dropna().unique().tolist())
        selected_edge_type = st.selectbox("Edge Case Type", edge_types)
        
        # Status filter (distinguishes No Prediction from Incorrect)
        status_options = ["All", "Incorrect", "Correct", "No Prediction"]
        selected_status = st.selectbox("Status", status_options)
    
    # ==========================================================================
    # Main Content: Interactive Explorer
    # ==========================================================================
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_slice != "All":
        filtered_df = filtered_df[filtered_df["slice"] == selected_slice]
    
    if selected_edge_type != "All":
        filtered_df = filtered_df[filtered_df["edge_case_type"] == selected_edge_type]
    
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["status"] == selected_status]
    
    # Display stats - now with 5 columns to show No Prediction separately
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", len(filtered_df))
    with col2:
        correct = (filtered_df["status"] == "Correct").sum()
        st.metric("✅ Correct", correct)
    with col3:
        failures = (filtered_df["status"] == "Incorrect").sum()
        st.metric("❌ Incorrect", failures)
    with col4:
        no_pred = (filtered_df["status"] == "No Prediction").sum()
        st.metric("⚠️ No Prediction", no_pred)
    with col5:
        # Accuracy only on rows WITH predictions
        has_pred = filtered_df["status"].isin(["Correct", "Incorrect"])
        if has_pred.sum() > 0:
            acc = (filtered_df.loc[has_pred, "status"] == "Correct").mean()
            st.metric("Accuracy (w/ pred)", f"{acc:.1%}")
        else:
            st.metric("Accuracy (w/ pred)", "N/A")
    
    st.divider()
    
    # ==========================================================================
    # Tabs: View vs Edit
    # ==========================================================================
    tab1, tab2 = st.tabs(["📋 View Data", "✏️ Edit Labels"])
    
    with tab1:
        st.subheader("Data Explorer")
        
        # Select columns to display
        display_cols = [
            "post_id", "slice", "edge_case_type", "status", "text",
            "human_sentiment", "pred_sentiment",
            "brand_relevance", "pred_brand_relevance"
        ]
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Style function for highlighting by status
        def highlight_by_status(row):
            status = row.get("status", "")
            if status == "Incorrect":
                return ["background-color: #ffcccc"] * len(row)  # Red
            elif status == "No Prediction":
                return ["background-color: #fff3cd"] * len(row)  # Yellow
            elif status == "Correct":
                return ["background-color: #d4edda"] * len(row)  # Green
            return [""] * len(row)
        
        if len(filtered_df) > 0:
            styled_df = filtered_df[display_cols].style.apply(highlight_by_status, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=500)
        else:
            st.info("No data matches the current filters.")
    
    with tab2:
        st.subheader("Label Curation")
        st.info("Edit the gold standard labels below. Changes will be saved with a backup.")
        
        # Editable columns
        edit_cols = [
            "post_id", "text", "human_sentiment", "brand_relevance", 
            "edge_case_type", "negative_type", "notes"
        ]
        edit_cols = [c for c in edit_cols if c in filtered_df.columns]
        
        # Column config for data editor
        column_config = {
            "post_id": st.column_config.TextColumn("Post ID", disabled=True),
            "text": st.column_config.TextColumn("Text", width="large", disabled=True),
            "human_sentiment": st.column_config.SelectboxColumn(
                "Human Sentiment",
                options=["positive", "negative", "neutral"],
                required=True
            ),
            "brand_relevance": st.column_config.SelectboxColumn(
                "Brand Relevance",
                options=["True", "False"],
                required=True
            ),
            "edge_case_type": st.column_config.TextColumn("Edge Case Type"),
            "negative_type": st.column_config.TextColumn("Negative Type"),
            "notes": st.column_config.TextColumn("Notes"),
        }
        
        if len(filtered_df) > 0:
            # Convert columns with potential NaN/float to string for editing
            edit_data = filtered_df[edit_cols].copy()
            for col in ["negative_type", "notes", "edge_case_type"]:
                if col in edit_data.columns:
                    edit_data[col] = edit_data[col].fillna("").astype(str)
            
            edited_df = st.data_editor(
                edit_data,
                column_config=column_config,
                use_container_width=True,
                height=500,
                num_rows="fixed"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("💾 Save Changes", type="primary"):
                    # Update the original dataframe with edits
                    for col in ["human_sentiment", "brand_relevance", "edge_case_type", "negative_type", "notes"]:
                        if col in edited_df.columns:
                            df.loc[filtered_df.index, col] = edited_df[col].values
                    
                    if save_gold_standard(df):
                        st.success("✅ Changes saved! Backup created.")
                        st.rerun()
            
            with col2:
                st.caption("⚠️ A timestamped backup will be created before saving.")
        else:
            st.info("No data matches the current filters.")


if __name__ == "__main__":
    main()
