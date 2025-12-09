"""
Gold Standard Labeling Page

Interactive UI for labeling posts with extended schema:
- sentiment (positive/neutral/negative)
- brand_relevance (true/false)
- negative_type (enum or null)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

GOLD_DIR = BASE_DIR / "data" / "gold_standard"
LABELING_FILE = GOLD_DIR / "to_label_200.csv"
OUTPUT_FILE = GOLD_DIR / "labeled_200.csv"

# Schema options
SENTIMENT_OPTIONS = ["", "positive", "neutral", "negative"]
NEGATIVE_TYPE_OPTIONS = [
    "",
    "customer_dissatisfaction",
    "scam_accusation",
    "regulatory_criticism",
    "general_negativity",
]

st.set_page_config(
    page_title="Gold Standard Labeling",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Gold Standard Labeling")
st.markdown("""
Label posts for the gold standard evaluation dataset.
Each post needs: **sentiment**, **brand_relevance**, and **negative_type** (if negative).
""")


def load_labeling_data() -> pd.DataFrame:
    """Load the data to be labeled."""
    if OUTPUT_FILE.exists():
        # Load existing progress
        return pd.read_csv(OUTPUT_FILE)
    elif LABELING_FILE.exists():
        return pd.read_csv(LABELING_FILE)
    else:
        return None


def save_labeling_data(df: pd.DataFrame) -> None:
    """Save labeling progress."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)


def get_labeling_progress(df: pd.DataFrame) -> dict:
    """Calculate labeling progress statistics."""
    total = len(df)
    labeled = df["human_sentiment"].notna().sum()
    
    # Count by sentiment
    sentiment_counts = df["human_sentiment"].value_counts().to_dict()
    
    return {
        "total": total,
        "labeled": labeled,
        "remaining": total - labeled,
        "progress_pct": (labeled / total * 100) if total > 0 else 0,
        "sentiment_counts": sentiment_counts,
    }


# Load data
df = load_labeling_data()

if df is None:
    st.error("No labeling data found!")
    st.info("""
    To create the labeling dataset, run:
    ```
    python -m scripts.prepare_labeling_batch
    ```
    This will create `data/gold_standard/to_label_200.csv`
    """)
    st.stop()

# Store in session state
if "labeling_df" not in st.session_state:
    st.session_state.labeling_df = df.copy()
    st.session_state.current_index = 0

df = st.session_state.labeling_df

# Sidebar - Progress and Navigation
st.sidebar.header("📊 Progress")
progress = get_labeling_progress(df)

st.sidebar.metric("Total Posts", progress["total"])
st.sidebar.metric("Labeled", progress["labeled"])
st.sidebar.metric("Remaining", progress["remaining"])
st.sidebar.progress(progress["progress_pct"] / 100)

if progress["sentiment_counts"]:
    st.sidebar.subheader("Label Distribution")
    for sentiment, count in progress["sentiment_counts"].items():
        st.sidebar.write(f"- {sentiment}: {count}")

st.sidebar.divider()

# Navigation
st.sidebar.subheader("🧭 Navigation")

# Jump to specific post
jump_to = st.sidebar.number_input(
    "Jump to post #",
    min_value=1,
    max_value=len(df),
    value=st.session_state.current_index + 1,
)
if st.sidebar.button("Go"):
    st.session_state.current_index = jump_to - 1
    st.rerun()

# Filter options
show_unlabeled = st.sidebar.checkbox("Show only unlabeled", value=False)
if show_unlabeled:
    unlabeled_indices = df[df["human_sentiment"].isna()].index.tolist()
    if unlabeled_indices:
        st.sidebar.write(f"Found {len(unlabeled_indices)} unlabeled posts")
    else:
        st.sidebar.success("All posts labeled!")

st.sidebar.divider()

# Save button
if st.sidebar.button("💾 Save Progress", type="primary"):
    save_labeling_data(df)
    st.sidebar.success("Progress saved!")

# Export button
if st.sidebar.button("📤 Export Labeled Data"):
    save_labeling_data(df)
    labeled_df = df[df["human_sentiment"].notna()]
    st.sidebar.download_button(
        "Download CSV",
        labeled_df.to_csv(index=False),
        file_name="labeled_gold_standard.csv",
        mime="text/csv",
    )

# Main content - Current post
st.divider()

current_idx = st.session_state.current_index
if current_idx >= len(df):
    current_idx = len(df) - 1
    st.session_state.current_index = current_idx

row = df.iloc[current_idx]

# Navigation buttons
col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
with col1:
    if st.button("⬅️ Previous", disabled=current_idx == 0):
        st.session_state.current_index -= 1
        st.rerun()
with col2:
    if st.button("Next ➡️", disabled=current_idx >= len(df) - 1):
        st.session_state.current_index += 1
        st.rerun()
with col3:
    st.write(f"**Post {current_idx + 1} of {len(df)}**")
with col4:
    # Quick skip to next unlabeled
    if st.button("Skip to Unlabeled"):
        unlabeled = df[df["human_sentiment"].isna()].index.tolist()
        if unlabeled:
            st.session_state.current_index = unlabeled[0]
            st.rerun()

st.divider()

# Post details
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Post Content")
    
    # Post metadata
    st.markdown(f"**Post ID:** `{row['post_id']}`")
    st.markdown(f"**Brand:** `{row['brand']}`")
    st.markdown(f"**Platform:** `{row['platform']}`")
    
    # Vendor sentiment
    vendor_sent = row.get("vendor_sentiment", "N/A")
    st.markdown(f"**Vendor Sentiment:** `{vendor_sent}`")
    
    st.divider()
    
    # Post text in a box
    st.markdown("**Text:**")
    st.text_area(
        "Post text (read-only)",
        value=str(row["text"]),
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

with col2:
    st.subheader("🏷️ Labels")
    
    # Current labels
    current_sentiment = row.get("human_sentiment", "")
    current_brand_rel = row.get("brand_relevance", "")
    current_neg_type = row.get("negative_type", "")
    
    # Sentiment selection
    sentiment_idx = SENTIMENT_OPTIONS.index(current_sentiment) if current_sentiment in SENTIMENT_OPTIONS else 0
    sentiment = st.selectbox(
        "Sentiment",
        options=SENTIMENT_OPTIONS,
        index=sentiment_idx,
        help="What is the TRUE sentiment of this post?",
    )
    
    # Brand relevance
    brand_rel_options = ["", "true", "false"]
    brand_rel_idx = 0
    if str(current_brand_rel).lower() == "true":
        brand_rel_idx = 1
    elif str(current_brand_rel).lower() == "false":
        brand_rel_idx = 2
    
    brand_relevance = st.selectbox(
        "Brand Relevance",
        options=brand_rel_options,
        index=brand_rel_idx,
        help="Is this post specifically about the brand?",
    )
    
    # Negative type (only if sentiment is negative)
    neg_type_idx = NEGATIVE_TYPE_OPTIONS.index(current_neg_type) if current_neg_type in NEGATIVE_TYPE_OPTIONS else 0
    
    if sentiment == "negative":
        negative_type = st.selectbox(
            "Negative Type",
            options=NEGATIVE_TYPE_OPTIONS,
            index=neg_type_idx,
            help="What type of negativity is expressed?",
        )
    else:
        negative_type = ""
        st.selectbox(
            "Negative Type",
            options=["N/A (sentiment not negative)"],
            disabled=True,
        )
    
    # Notes
    current_notes = row.get("notes", "")
    notes = st.text_area(
        "Notes (optional)",
        value=str(current_notes) if pd.notna(current_notes) else "",
        height=100,
    )
    
    st.divider()
    
    # Save label button
    if st.button("✅ Save Label", type="primary", use_container_width=True):
        if not sentiment:
            st.error("Please select a sentiment")
        elif not brand_relevance:
            st.error("Please select brand relevance")
        elif sentiment == "negative" and not negative_type:
            st.error("Please select negative type for negative sentiment")
        else:
            # Update dataframe
            df.at[current_idx, "human_sentiment"] = sentiment
            df.at[current_idx, "brand_relevance"] = brand_relevance
            df.at[current_idx, "negative_type"] = negative_type if sentiment == "negative" else ""
            df.at[current_idx, "notes"] = notes
            df.at[current_idx, "labeler"] = st.session_state.get("labeler_name", "unknown")
            df.at[current_idx, "labeled_at"] = datetime.now().isoformat()
            
            st.session_state.labeling_df = df
            save_labeling_data(df)
            
            st.success("Label saved!")
            
            # Stay on current post - user can use Next or Skip to Unlabeled to navigate
            st.rerun()

# Labeler name input (in sidebar)
st.sidebar.divider()
st.sidebar.subheader("👤 Labeler")
labeler_name = st.sidebar.text_input(
    "Your name/initials",
    value=st.session_state.get("labeler_name", ""),
)
if labeler_name:
    st.session_state.labeler_name = labeler_name

# Guidelines in expander
with st.expander("📖 Labeling Guidelines"):
    st.markdown("""
    ### Sentiment
    - **positive**: Genuine praise, satisfaction, recommendation
    - **neutral**: Factual statements, promotional content, no opinion
    - **negative**: Complaints, criticism, dissatisfaction (including sarcasm!)
    
    ### Brand Relevance
    - **true**: Post is specifically about the brand (complaint, praise, direct mention)
    - **false**: Brand mentioned in passing, promotional/affiliate content, unrelated
    
    ### Negative Type (only for negative sentiment)
    - **customer_dissatisfaction**: UX issues, slow site, login problems, withdrawal delays
    - **scam_accusation**: Claims of rigged games, stolen money, fraud
    - **regulatory_criticism**: Underage gambling concerns, KYC failures
    - **general_negativity**: Vague complaints, "I hate this"
    
    ### Tips
    - Consider sarcasm: "Great job breaking the site again" = NEGATIVE
    - Affiliate promotional posts are typically NEUTRAL with brand_relevance=FALSE
    - When in doubt, add a note explaining your reasoning
    """)
