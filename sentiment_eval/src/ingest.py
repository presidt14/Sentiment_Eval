"""
Smart Data Ingestion Module

Provides utilities for detecting and mapping columns from messy real-world data
to the standardized internal format (post_id, text).
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import json


# Keywords that suggest a column contains text content
TEXT_KEYWORDS = [
    "text", "body", "msg", "message", "content", "comment", "review",
    "feedback", "description", "post", "tweet", "note", "summary",
    "title", "subject", "question", "answer", "reply", "response"
]

# Keywords that suggest a column is an ID
ID_KEYWORDS = [
    "id", "uuid", "key", "code", "number", "num", "index", "idx",
    "tracking", "reference", "ref", "serial", "record"
]


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Analyze a DataFrame and return best guesses for text and ID columns.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        dict with keys:
            - 'text_columns': list of potential text column names (ranked by confidence)
            - 'id_columns': list of potential ID column names (ranked by confidence)
            - 'best_text': single best guess for text column (or None)
            - 'best_id': single best guess for ID column (or None)
    """
    text_candidates = []
    id_candidates = []
    
    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        
        # Check for text column indicators
        text_score = 0
        for keyword in TEXT_KEYWORDS:
            if keyword in col_lower:
                text_score += 1
        
        # Also check if column contains long strings (likely text)
        if df[col].dtype == object:
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                avg_len = sample.astype(str).str.len().mean()
                if avg_len > 50:  # Long strings suggest text content
                    text_score += 2
                elif avg_len > 20:
                    text_score += 1
        
        if text_score > 0:
            text_candidates.append((col, text_score))
        
        # Check for ID column indicators
        id_score = 0
        for keyword in ID_KEYWORDS:
            if keyword in col_lower:
                id_score += 1
        
        # Check if column has unique values (suggests ID)
        if len(df) > 0:
            uniqueness = df[col].nunique() / len(df)
            if uniqueness > 0.9:  # High uniqueness suggests ID
                id_score += 1
        
        if id_score > 0:
            id_candidates.append((col, id_score))
    
    # Sort by score (descending)
    text_candidates.sort(key=lambda x: x[1], reverse=True)
    id_candidates.sort(key=lambda x: x[1], reverse=True)
    
    text_columns = [c[0] for c in text_candidates]
    id_columns = [c[0] for c in id_candidates]
    
    # If no text columns found, fall back to any string column
    if not text_columns:
        for col in df.columns:
            if df[col].dtype == object:
                text_columns.append(col)
    
    # If no ID columns found, fall back to first column or index
    if not id_columns:
        id_columns = [df.columns[0]] if len(df.columns) > 0 else []
    
    return {
        "text_columns": text_columns,
        "id_columns": id_columns,
        "best_text": text_columns[0] if text_columns else None,
        "best_id": id_columns[0] if id_columns else None,
    }


def normalize_nested_json(data: list | dict) -> pd.DataFrame:
    """
    Attempt to normalize nested JSON data into a flat DataFrame.
    
    Args:
        data: JSON data (list of dicts or nested dict)
        
    Returns:
        Flattened DataFrame
    """
    if isinstance(data, dict):
        # Check if it's a dict with a single key containing a list
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                return pd.json_normalize(value, sep="_")
        # Otherwise normalize the dict itself
        return pd.json_normalize(data, sep="_")
    elif isinstance(data, list):
        return pd.json_normalize(data, sep="_")
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")


def standardize_dataframe(
    df: pd.DataFrame,
    text_column: str,
    id_column: str,
    fill_missing: bool = True,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    Standardize a DataFrame to the internal format (post_id, text).
    
    Args:
        df: Input DataFrame
        text_column: Name of the column containing text
        id_column: Name of the column containing IDs
        fill_missing: Whether to fill missing text values with ""
        drop_duplicates: Whether to drop duplicate IDs
        
    Returns:
        Standardized DataFrame with columns: post_id, text
    """
    # Create standardized dataframe
    result = pd.DataFrame({
        "post_id": df[id_column],
        "text": df[text_column]
    })
    
    # Handle missing values
    if fill_missing:
        result["text"] = result["text"].fillna("")
        result["post_id"] = result["post_id"].fillna("").astype(str)
    
    # Handle duplicate IDs
    if drop_duplicates:
        original_len = len(result)
        result = result.drop_duplicates(subset=["post_id"], keep="first")
        dropped = original_len - len(result)
        if dropped > 0:
            print(f"Dropped {dropped} duplicate IDs")
    
    # Reset index
    result = result.reset_index(drop=True)
    
    return result


def save_to_staging(df: pd.DataFrame, staging_dir: Path) -> Path:
    """
    Save a standardized DataFrame to the staging directory.
    
    Args:
        df: Standardized DataFrame
        staging_dir: Path to staging directory
        
    Returns:
        Path to the saved file
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_path = staging_dir / "current_batch.csv"
    df.to_csv(output_path, index=False)
    return output_path


def get_data_preview(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Get a preview of the DataFrame for display.
    
    Args:
        df: Input DataFrame
        n_rows: Number of rows to preview
        
    Returns:
        Preview DataFrame
    """
    return df.head(n_rows)


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict with statistics
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
