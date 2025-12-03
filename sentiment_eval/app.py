import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CORRECTED_PATH = RESULTS_DIR / "corrected_results.csv"

# Add src to path for imports
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from config import list_prompt_strategies, load_prompts

    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure human_sentiment column exists
    if "human_sentiment" not in df.columns:
        df["human_sentiment"] = ""
    return df


def colorize_sentiment(sentiment: str) -> str:
    """Return Streamlit markdown with color-coded sentiment."""
    s = str(sentiment).lower().strip()
    if s == "positive":
        return ":green[positive]"
    elif s == "negative":
        return ":red[negative]"
    else:
        return ":gray[neutral]"


def discover_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.csv"))


st.title("Sentiment Labeling Tool")

result_files = discover_result_files()

# Prompt Strategy Info Section
if PROMPTS_AVAILABLE:
    st.sidebar.header("🎯 Prompt Strategies")
    try:
        strategies = list_prompt_strategies()
        st.sidebar.caption(f"**{len(strategies)}** strategies available")
        with st.sidebar.expander("View Strategies"):
            for key, name in strategies.items():
                st.markdown(f"- **{name}**")
        st.sidebar.info(
            "Use **🚀 Run Batch** page to analyze with different strategies"
        )
    except Exception:
        pass

st.sidebar.divider()
st.sidebar.header("Results Source")

if not result_files:
    st.sidebar.info("No CSV outputs found. Run `make mock` to generate demo data.")
    st.warning("No result files detected. Run the batch script first.")
    st.stop()

file_options = {f.name: f for f in result_files}
selected_label = st.sidebar.selectbox(
    "Available result files", list(file_options.keys())
)
selected_path = file_options[selected_label]
st.sidebar.caption(f"Loading **{selected_label}**")

# Load data into session state for in-memory editing
state_key = f"df_{selected_label}"
if state_key not in st.session_state:
    st.session_state[state_key] = load_results(selected_path)

df_full = st.session_state[state_key]
sentiment_cols = [
    c for c in df_full.columns if c.endswith("_sentiment") and c != "human_sentiment"
]

# Compute verification metrics
verified_count = (df_full["human_sentiment"].astype(str).str.strip() != "").sum()
total_rows = len(df_full)

st.sidebar.header("Human Verification")
st.sidebar.metric("Verified", f"{verified_count} / {total_rows}")

# Save button
if st.sidebar.button("💾 Save Corrections"):
    df_full.to_csv(CORRECTED_PATH, index=False)
    st.sidebar.success(f"Saved to {CORRECTED_PATH.name}")

st.sidebar.header("Filters")

# Work on a filtered view for display
df = df_full.copy()

# Filter 1: Show only disagreements
only_disagree = st.sidebar.checkbox("Show only rows where models disagree", value=False)
if only_disagree:
    disagree = df[sentiment_cols].nunique(axis=1) > 1
    df = df[disagree]

# Filter 2: Search
search = st.sidebar.text_input("Search text contains", "")
if search:
    df = df[df["text"].str.contains(search, case=False, na=False)]

if len(df) == 0:
    st.info("Showing 0 rows")
else:
    # Initialize pagination state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Clamp current_index to valid range for filtered df
    max_idx = len(df) - 1
    st.session_state.current_index = max(
        0, min(st.session_state.current_index, max_idx)
    )

    # Navigation UI
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

    with nav_col1:
        if st.button(
            "⬅️ Previous",
            use_container_width=True,
            disabled=st.session_state.current_index == 0,
        ):
            st.session_state.current_index -= 1
            st.rerun()

    with nav_col2:
        # Number input for direct jump (1-indexed for user-friendliness)
        new_row = st.number_input(
            "Row #",
            min_value=1,
            max_value=len(df),
            value=st.session_state.current_index + 1,
            step=1,
            key="row_number_input",
        )
        # Update index if user changed the number input
        if new_row - 1 != st.session_state.current_index:
            st.session_state.current_index = new_row - 1
            st.rerun()

    with nav_col3:
        if st.button(
            "Next ➡️",
            use_container_width=True,
            disabled=st.session_state.current_index >= max_idx,
        ):
            st.session_state.current_index += 1
            st.rerun()

    # Progress indicator
    st.markdown(f"**Post {st.session_state.current_index + 1} of {len(df)}**")

    # Get current row from filtered dataframe
    row_index = df.index[st.session_state.current_index]
    row = df.loc[row_index]

    # Verified badge
    human_label = str(df_full.at[row_index, "human_sentiment"]).strip()
    verified = human_label != ""

    st.subheader("Post")
    if verified:
        st.markdown(f"✅ **Verified as {colorize_sentiment(human_label)}**")
    st.write(row["text"])

    st.subheader("Sentiment by model")
    for col in sentiment_cols:
        model_name = col.replace("_sentiment", "")
        sentiment_value = row[col]
        colored_sentiment = colorize_sentiment(sentiment_value)
        confidence = row.get(f"{model_name}_confidence", "N/A")
        reason = row.get(f"{model_name}_reason", "No reason provided")

        st.markdown(
            f"**{model_name.upper()}** — {colored_sentiment} (Conf: {confidence})"
        )
        st.caption(f"_{reason}_")

    # Human correction controls
    st.subheader("Human Correction")
    st.caption("Click a button to label this post (auto-advances to next):")

    col1, col2, col3, col4 = st.columns(4)

    def advance_to_next():
        """Auto-advance to next row after verification, if possible."""
        if st.session_state.current_index < len(df) - 1:
            st.session_state.current_index += 1

    with col1:
        if st.button("✓ Correct", key=f"correct_{row_index}"):
            first_model_sentiment = (
                row[sentiment_cols[0]] if sentiment_cols else "neutral"
            )
            df_full.at[row_index, "human_sentiment"] = first_model_sentiment
            advance_to_next()
            st.rerun()
    with col2:
        if st.button("🟢 Positive", key=f"pos_{row_index}"):
            df_full.at[row_index, "human_sentiment"] = "positive"
            advance_to_next()
            st.rerun()
    with col3:
        if st.button("🔴 Negative", key=f"neg_{row_index}"):
            df_full.at[row_index, "human_sentiment"] = "negative"
            advance_to_next()
            st.rerun()
    with col4:
        if st.button("⚪ Neutral", key=f"neu_{row_index}"):
            df_full.at[row_index, "human_sentiment"] = "neutral"
            advance_to_next()
            st.rerun()
