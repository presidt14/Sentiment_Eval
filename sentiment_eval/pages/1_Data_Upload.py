"""
Data Upload Page - Smart Data Loader UI

Allows users to upload messy CSV/JSON files, preview data,
map columns to the internal format, and save to staging.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import sys

# Add src to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from ingest import detect_columns, standardize_dataframe, get_data_preview, get_data_stats
from data_loader import load_posts

STAGING_DIR = BASE_DIR / "data" / "staging"


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load an uploaded file into a DataFrame."""
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    if file_ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif file_ext == ".json":
        try:
            data = json.load(uploaded_file)
            # Handle nested JSON
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        return pd.json_normalize(value, sep="_")
                return pd.json_normalize(data, sep="_")
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    has_nested = any(
                        isinstance(v, (dict, list))
                        for item in data[:5]
                        for v in item.values()
                    )
                    if has_nested:
                        return pd.json_normalize(data, sep="_")
                return pd.DataFrame(data)
            else:
                return pd.DataFrame([data])
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return None
    elif file_ext == ".jsonl":
        lines = uploaded_file.read().decode("utf-8").strip().split("\n")
        data = [json.loads(line) for line in lines if line.strip()]
        return pd.DataFrame(data)
    else:
        st.error(f"Unsupported file type: {file_ext}")
        return None


st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")
st.title("📤 Smart Data Loader")
st.markdown("""
Upload your messy CSV or JSON files. The system will:
- Auto-detect text and ID columns
- Handle missing values and duplicates
- Normalize nested JSON structures
- Standardize to the internal format for batch processing
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "json", "jsonl"],
    help="Upload a CSV, JSON, or JSONL file"
)

if uploaded_file is not None:
    # Load and store in session state
    if "uploaded_df" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
        with st.spinner("Loading file..."):
            df = load_uploaded_file(uploaded_file)
            if df is not None:
                st.session_state.uploaded_df = df
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.column_detection = detect_columns(df)
    
    if "uploaded_df" in st.session_state:
        df = st.session_state.uploaded_df
        detection = st.session_state.column_detection
        
        # Data Statistics
        st.subheader("📊 Data Overview")
        stats = get_data_stats(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", stats["total_rows"])
        with col2:
            st.metric("Total Columns", stats["total_columns"])
        with col3:
            missing_total = sum(stats["missing_values"].values())
            st.metric("Missing Values", missing_total)
        
        # Raw Data Preview
        st.subheader("👀 Raw Data Preview")
        preview_rows = st.slider("Preview rows", 3, 20, 5)
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Column info
        with st.expander("📋 Column Details"):
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(df[c].dtype) for c in df.columns],
                "Non-Null": [df[c].notna().sum() for c in df.columns],
                "Sample": [str(df[c].dropna().iloc[0])[:50] + "..." if len(df[c].dropna()) > 0 else "N/A" for c in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Column Mapping Section
        st.subheader("🔗 Column Mapping")
        st.markdown("Map your columns to the internal format. Auto-detected values are pre-selected.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text column selection
            text_options = list(df.columns)
            default_text_idx = 0
            if detection["best_text"] and detection["best_text"] in text_options:
                default_text_idx = text_options.index(detection["best_text"])
            
            text_column = st.selectbox(
                "📝 Select Text Column",
                options=text_options,
                index=default_text_idx,
                help=f"Auto-detected candidates: {', '.join(detection['text_columns'][:3]) or 'None'}"
            )
            
            if detection["text_columns"]:
                st.caption(f"💡 Detected text columns: {', '.join(detection['text_columns'][:5])}")
        
        with col2:
            # ID column selection
            id_options = list(df.columns)
            default_id_idx = 0
            if detection["best_id"] and detection["best_id"] in id_options:
                default_id_idx = id_options.index(detection["best_id"])
            
            id_column = st.selectbox(
                "🔑 Select ID Column",
                options=id_options,
                index=default_id_idx,
                help=f"Auto-detected candidates: {', '.join(detection['id_columns'][:3]) or 'None'}"
            )
            
            if detection["id_columns"]:
                st.caption(f"💡 Detected ID columns: {', '.join(detection['id_columns'][:5])}")
        
        # Data Cleaning Options
        st.subheader("🧹 Data Cleaning Options")
        col1, col2 = st.columns(2)
        
        with col1:
            fill_missing = st.checkbox("Fill missing text values with empty string", value=True)
        with col2:
            drop_duplicates = st.checkbox("Drop duplicate IDs (keep first)", value=True)
        
        # Preview standardized output
        st.subheader("✨ Standardized Output Preview")
        
        try:
            standardized_df = standardize_dataframe(
                df,
                text_column=text_column,
                id_column=id_column,
                fill_missing=fill_missing,
                drop_duplicates=drop_duplicates
            )
            
            st.dataframe(standardized_df.head(preview_rows), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Output Rows", len(standardized_df))
            with col2:
                dropped = len(df) - len(standardized_df)
                st.metric("Rows Dropped", dropped)
            with col3:
                empty_text = (standardized_df["text"].str.strip() == "").sum()
                st.metric("Empty Text Rows", empty_text)
            
            # Process & Load Button
            st.divider()
            
            if st.button("🚀 Process & Load", type="primary", use_container_width=True):
                with st.spinner("Processing and saving..."):
                    # Ensure staging directory exists
                    STAGING_DIR.mkdir(parents=True, exist_ok=True)
                    
                    # Save to staging
                    output_path = STAGING_DIR / "current_batch.csv"
                    standardized_df.to_csv(output_path, index=False)
                    
                    st.success(f"✅ Successfully saved {len(standardized_df)} rows to `{output_path.relative_to(BASE_DIR)}`")
                    st.info("You can now run batch processing on this data using the 'Run Batch' workflow.")
                    
                    # Show sample of saved data
                    st.subheader("📄 Saved Data Sample")
                    st.dataframe(standardized_df.head(3), use_container_width=True)
                    
                    # Clear session state for fresh upload
                    if st.button("Upload Another File"):
                        del st.session_state.uploaded_df
                        del st.session_state.uploaded_filename
                        del st.session_state.column_detection
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error standardizing data: {e}")
            st.info("Please check your column selections and try again.")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Upload a file to get started")
    
    with st.expander("📖 Supported Formats"):
        st.markdown("""
        **CSV Files**
        - Standard comma-separated values
        - Any column names supported
        
        **JSON Files**
        - Array of objects: `[{"id": 1, "text": "..."}, ...]`
        - Nested objects: `{"data": [{"id": 1, "text": "..."}]}`
        - Nested fields are flattened with underscore separators
        
        **JSONL Files**
        - One JSON object per line
        """)
    
    with st.expander("🎯 Column Detection"):
        st.markdown("""
        The system auto-detects columns based on:
        
        **Text Columns** (keywords: text, body, message, content, feedback, review, etc.)
        - Also considers average string length
        
        **ID Columns** (keywords: id, uuid, key, code, tracking, reference, etc.)
        - Also considers uniqueness ratio
        """)

# Sidebar info
st.sidebar.header("📤 Data Upload")
st.sidebar.markdown("""
This tool helps you prepare messy real-world data for sentiment analysis.

**Workflow:**
1. Upload your file
2. Review the data preview
3. Map columns to internal format
4. Process & Load to staging
5. Run batch analysis
""")

# Show current staging status
staging_file = STAGING_DIR / "current_batch.csv"
if staging_file.exists():
    staging_df = pd.read_csv(staging_file)
    st.sidebar.success(f"✅ Staging file ready: {len(staging_df)} rows")
else:
    st.sidebar.info("No staging file yet")
