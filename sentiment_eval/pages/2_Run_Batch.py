"""
Run Batch Page - Batch Sentiment Analysis with Prompt Strategy Selection

Allows users to run batch sentiment analysis on staged data,
with the ability to select different prompt strategies.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import asyncio
from datetime import datetime

# Add src to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from config import load_settings, list_prompt_strategies, get_prompt_strategy, load_prompts
from models import get_active_models_with_strategy
from run_batch import process_batch_async, DEFAULT_CONCURRENCY
from data_loader import load_posts

STAGING_DIR = BASE_DIR / "data" / "staging"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_STAGING_FILE = STAGING_DIR / "current_batch.csv"


st.set_page_config(page_title="Run Batch", page_icon="🚀", layout="wide")
st.title("🚀 Batch Sentiment Analysis")
st.markdown("""
Run sentiment analysis on your staged data using AI models.
Select a prompt strategy to customize how the AI interprets and classifies sentiment.
""")

# Sidebar - Prompt Strategy Selection
st.sidebar.header("🎯 Prompt Strategy")

# Load available strategies
try:
    strategies = list_prompt_strategies()
    prompts_data = load_prompts()
except Exception as e:
    st.sidebar.error(f"Error loading prompts: {e}")
    strategies = {"default_sentiment": "Default Sentiment Classifier"}
    prompts_data = {}

# Strategy selector
strategy_names = list(strategies.keys())
strategy_labels = [f"{strategies[k]}" for k in strategy_names]

selected_idx = st.sidebar.selectbox(
    "Select Strategy",
    range(len(strategy_names)),
    format_func=lambda i: strategy_labels[i],
    help="Choose a prompt strategy that defines how the AI will analyze sentiment"
)
selected_strategy = strategy_names[selected_idx]

# Show strategy details
if selected_strategy in prompts_data:
    strategy_info = prompts_data[selected_strategy]
    with st.sidebar.expander("📋 Strategy Details", expanded=True):
        st.markdown(f"**{strategy_info.get('name', selected_strategy)}**")
        st.caption(strategy_info.get('description', 'No description available'))
        
        st.markdown("**System Prompt Preview:**")
        system_preview = strategy_info.get('system', '')[:200]
        if len(strategy_info.get('system', '')) > 200:
            system_preview += "..."
        st.code(system_preview, language=None)

# Sidebar - Processing Options
st.sidebar.header("⚙️ Processing Options")
concurrency = st.sidebar.slider(
    "Concurrency",
    min_value=1,
    max_value=50,
    value=DEFAULT_CONCURRENCY,
    help="Number of concurrent API requests"
)

# Check for staging file
if not DEFAULT_STAGING_FILE.exists():
    st.warning("⚠️ No staging file found. Please upload data first using the Data Upload page.")
    st.info("Go to **📤 Data Upload** to prepare your data for analysis.")
    st.stop()

# Load staging data
try:
    df = load_posts(DEFAULT_STAGING_FILE)
    st.success(f"✅ Loaded {len(df)} posts from staging")
except Exception as e:
    st.error(f"Error loading staging file: {e}")
    st.stop()

# Data preview
st.subheader("📊 Data Preview")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df.head(5), use_container_width=True)
with col2:
    st.metric("Total Posts", len(df))
    st.metric("Strategy", strategies.get(selected_strategy, selected_strategy))

# Run Analysis Section
st.divider()
st.subheader("🔬 Run Analysis")

# Load settings to show active models
cfg = load_settings()
active_models = cfg.get("active_models", [])
st.info(f"**Active Models:** {', '.join(active_models) if active_models else 'None configured'}")

# Run button
if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
    if not active_models:
        st.error("No active models configured. Check your settings.yaml file.")
        st.stop()
    
    # Get models with selected strategy
    try:
        models = get_active_models_with_strategy(selected_strategy, cfg)
    except KeyError as e:
        st.error(f"Error loading prompt strategy: {e}")
        st.stop()
    
    st.info(f"Using **{selected_strategy}** strategy with models: {[m.name for m in models]}")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run async processing
    start_time = datetime.now()
    status_text.text("Starting batch processing...")
    
    try:
        # Run the async batch processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            process_batch_async(
                df,
                models,
                text_column="text",
                id_column="post_id",
                concurrency=concurrency,
            )
        )
        loop.close()
        
        progress_bar.progress(100)
        elapsed = (datetime.now() - start_time).total_seconds()
        status_text.text(f"✅ Completed in {elapsed:.2f}s")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient="index")
        
        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"results_{selected_strategy}_{timestamp}.csv"
        output_path = RESULTS_DIR / output_filename
        results_df.to_csv(output_path, index=False)
        
        st.success(f"✅ Results saved to `{output_path.relative_to(BASE_DIR)}`")
        
        # Show results summary
        st.subheader("📈 Results Summary")
        
        # Find sentiment columns
        sentiment_cols = [c for c in results_df.columns if c.endswith("_sentiment")]
        
        if sentiment_cols:
            # Create summary metrics
            cols = st.columns(len(sentiment_cols))
            for i, col_name in enumerate(sentiment_cols):
                model_name = col_name.replace("_sentiment", "").upper()
                with cols[i]:
                    st.markdown(f"**{model_name}**")
                    value_counts = results_df[col_name].value_counts()
                    for sentiment, count in value_counts.items():
                        pct = (count / len(results_df)) * 100
                        if sentiment == "positive":
                            st.markdown(f":green[{sentiment}]: {count} ({pct:.1f}%)")
                        elif sentiment == "negative":
                            st.markdown(f":red[{sentiment}]: {count} ({pct:.1f}%)")
                        else:
                            st.markdown(f":gray[{sentiment}]: {count} ({pct:.1f}%)")
        
        # Show sample results
        st.subheader("📄 Sample Results")
        display_cols = ["post_id", "text"] + sentiment_cols
        display_cols = [c for c in display_cols if c in results_df.columns]
        st.dataframe(results_df[display_cols].head(10), use_container_width=True)
        
        # Store in session state for viewing
        st.session_state["last_results"] = results_df
        st.session_state["last_strategy"] = selected_strategy
        
    except Exception as e:
        st.error(f"Error during batch processing: {e}")
        import traceback
        st.code(traceback.format_exc())

# Show previous results if available
if "last_results" in st.session_state:
    st.divider()
    with st.expander("📊 View Last Results", expanded=False):
        st.caption(f"Strategy used: **{st.session_state.get('last_strategy', 'Unknown')}**")
        st.dataframe(st.session_state["last_results"], use_container_width=True)

# Sidebar info
st.sidebar.divider()
st.sidebar.header("ℹ️ About Strategies")
st.sidebar.markdown("""
**Prompt strategies** customize how AI models interpret and classify sentiment:

- **Default**: Standard sentiment classification
- **Sarcasm Detector**: Accounts for irony and sarcasm
- **Strict Compliance**: Conservative, risk-focused
- **Customer Feedback**: Optimized for reviews
- **Multilingual**: Handles mixed languages

Edit `config/prompts.yaml` to add custom strategies.
""")
