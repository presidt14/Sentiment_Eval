"""
Data Loader Module

Handles loading of various data formats with support for:
- Missing value handling
- Duplicate ID removal
- Nested JSON normalization
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _normalize_nested_json(data: list | dict) -> pd.DataFrame:
    """
    Attempt to normalize nested JSON data into a flat DataFrame.

    Args:
        data: JSON data (list of dicts or nested dict)

    Returns:
        Flattened DataFrame
    """
    if isinstance(data, dict):
        # Check if it's a dict with a single key containing a list (common pattern)
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                return pd.json_normalize(value, sep="_")
        # Otherwise normalize the dict itself
        return pd.json_normalize(data, sep="_")
    elif isinstance(data, list):
        return pd.json_normalize(data, sep="_")
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")


def load_posts(
    path: str | Path,
    fill_missing: bool = True,
    drop_duplicate_ids: bool = False,
    id_column: Optional[str] = None,
    text_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load posts from a file with optional data cleaning.

    Args:
        path: Path to the data file (CSV, JSON, or JSONL)
        fill_missing: Whether to fill missing text values with ""
        drop_duplicate_ids: Whether to drop duplicate IDs
        id_column: Column name for IDs (for deduplication)
        text_column: Column name for text (for missing value handling)

    Returns:
        Loaded and optionally cleaned DataFrame
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        # Try to handle nested JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if it's a simple list of flat dicts
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # Check if any values are nested
                    has_nested = any(
                        isinstance(v, (dict, list))
                        for item in data[:5]
                        for v in item.values()
                    )
                    if has_nested:
                        df = _normalize_nested_json(data)
                    else:
                        df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = _normalize_nested_json(data)
            else:
                df = pd.DataFrame([data])
        except json.JSONDecodeError:
            # Fall back to pandas read_json
            df = pd.read_json(path)
    elif path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Handle missing values
    if fill_missing:
        # Fill missing values in text column if specified
        if text_column and text_column in df.columns:
            df[text_column] = df[text_column].fillna("")
        # Also fill common text column names
        for col in ["text", "body", "content", "message"]:
            if col in df.columns:
                df[col] = df[col].fillna("")

    # Handle duplicate IDs
    if drop_duplicate_ids and id_column and id_column in df.columns:
        original_len = len(df)
        df = df.drop_duplicates(subset=[id_column], keep="first")
        dropped = original_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} duplicate IDs from column '{id_column}'")

    return df


def load_labels(path: str | Path) -> pd.DataFrame:
    """Load label data from a file."""
    return load_posts(path, fill_missing=False, drop_duplicate_ids=False)


def load_staging_batch(staging_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the current staging batch if it exists.

    Args:
        staging_dir: Path to the staging directory

    Returns:
        DataFrame if staging file exists, None otherwise
    """
    staging_file = staging_dir / "current_batch.csv"
    if staging_file.exists():
        return pd.read_csv(staging_file)
    return None
