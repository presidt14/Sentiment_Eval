"""
Tests for the Smart Data Ingestion module.
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_posts
from ingest import (detect_columns, get_data_preview, get_data_stats,
                    normalize_nested_json, standardize_dataframe)


class TestDetectColumns:
    """Tests for column detection."""

    def test_detect_text_column_by_name(self):
        """Should detect columns with text-related keywords."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "customer_feedback": ["good", "bad", "ok"],
                "rating": [5, 1, 3],
            }
        )
        result = detect_columns(df)
        assert "customer_feedback" in result["text_columns"]
        assert result["best_text"] == "customer_feedback"

    def test_detect_id_column_by_name(self):
        """Should detect columns with ID-related keywords."""
        df = pd.DataFrame(
            {
                "tracking_code": ["TRK-001", "TRK-002", "TRK-003"],
                "message": ["hello", "world", "test"],
            }
        )
        result = detect_columns(df)
        assert "tracking_code" in result["id_columns"]
        assert result["best_id"] == "tracking_code"

    def test_detect_text_by_length(self):
        """Should detect text columns by average string length."""
        df = pd.DataFrame(
            {
                "short_col": ["a", "b", "c"],
                "long_col": ["This is a very long text that should be detected"] * 3,
            }
        )
        result = detect_columns(df)
        # long_col should rank higher due to length
        assert "long_col" in result["text_columns"]

    def test_detect_id_by_uniqueness(self):
        """Should detect ID columns by high uniqueness."""
        df = pd.DataFrame(
            {
                "unique_col": ["A", "B", "C", "D", "E"],
                "repeated_col": ["X", "X", "Y", "Y", "Z"],
            }
        )
        result = detect_columns(df)
        # unique_col should be detected as potential ID
        assert "unique_col" in result["id_columns"]

    def test_fallback_to_string_columns(self):
        """Should fall back to any string column if no text keywords found."""
        df = pd.DataFrame(
            {
                "col_a": ["hello", "world", "test"],
                "col_b": [1, 2, 3],
            }
        )
        result = detect_columns(df)
        assert "col_a" in result["text_columns"]


class TestNormalizeNestedJson:
    """Tests for nested JSON normalization."""

    def test_normalize_list_of_dicts(self):
        """Should normalize a simple list of dicts."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        df = normalize_nested_json(data)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_normalize_nested_dict(self):
        """Should normalize nested dict with list value."""
        data = {"records": [{"id": 1}, {"id": 2}]}
        df = normalize_nested_json(data)
        assert len(df) == 2
        assert "id" in df.columns

    def test_normalize_deeply_nested(self):
        """Should flatten deeply nested structures."""
        data = [{"user": {"name": "John", "email": "john@test.com"}}]
        df = normalize_nested_json(data)
        assert "user_name" in df.columns
        assert "user_email" in df.columns


class TestStandardizeDataframe:
    """Tests for DataFrame standardization."""

    def test_basic_standardization(self):
        """Should create standard post_id and text columns."""
        df = pd.DataFrame(
            {
                "tracking_code": ["TRK-001", "TRK-002"],
                "customer_feedback": ["good", "bad"],
            }
        )
        result = standardize_dataframe(df, "customer_feedback", "tracking_code")
        assert list(result.columns) == ["post_id", "text"]
        assert result["post_id"].tolist() == ["TRK-001", "TRK-002"]
        assert result["text"].tolist() == ["good", "bad"]

    def test_fill_missing_values(self):
        """Should fill missing text values with empty string."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "text": ["hello", None, "world"],
            }
        )
        result = standardize_dataframe(df, "text", "id", fill_missing=True)
        assert result["text"].tolist() == ["hello", "", "world"]

    def test_drop_duplicates(self):
        """Should drop duplicate IDs, keeping first."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 1, 3],
                "text": ["first", "second", "duplicate", "third"],
            }
        )
        result = standardize_dataframe(df, "text", "id", drop_duplicates=True)
        assert len(result) == 3
        assert result[result["post_id"] == "1"]["text"].iloc[0] == "first"

    def test_no_fill_missing(self):
        """Should preserve NaN when fill_missing=False."""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "text": ["hello", None],
            }
        )
        result = standardize_dataframe(df, "text", "id", fill_missing=False)
        assert pd.isna(result["text"].iloc[1])


class TestDataLoader:
    """Tests for the updated data loader."""

    def test_load_csv(self, tmp_path):
        """Should load CSV files."""
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": [1, 2], "text": ["a", "b"]}).to_csv(csv_path, index=False)

        df = load_posts(csv_path)
        assert len(df) == 2

    def test_load_json(self, tmp_path):
        """Should load JSON files."""
        json_path = tmp_path / "test.json"
        with open(json_path, "w") as f:
            json.dump([{"id": 1, "text": "hello"}], f)

        df = load_posts(json_path)
        assert len(df) == 1

    def test_load_nested_json(self, tmp_path):
        """Should normalize nested JSON."""
        json_path = tmp_path / "nested.json"
        data = {"data": [{"id": 1, "user": {"name": "John"}}]}
        with open(json_path, "w") as f:
            json.dump(data, f)

        df = load_posts(json_path)
        assert "user_name" in df.columns

    def test_fill_missing_in_loader(self, tmp_path):
        """Should fill missing values when loading."""
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": [1, 2], "text": ["a", None]}).to_csv(csv_path, index=False)

        df = load_posts(csv_path, fill_missing=True, text_column="text")
        assert df["text"].iloc[1] == ""


class TestGetDataStats:
    """Tests for data statistics."""

    def test_basic_stats(self):
        """Should return correct statistics."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "text": ["a", None, "c"],
            }
        )
        stats = get_data_stats(df)
        assert stats["total_rows"] == 3
        assert stats["total_columns"] == 2
        assert stats["missing_values"]["text"] == 1


class TestMessyDataIntegration:
    """Integration tests with messy data files."""

    @pytest.fixture
    def messy_data_path(self):
        return (
            Path(__file__).resolve().parents[1] / "data" / "samples" / "messy_data.json"
        )

    @pytest.fixture
    def nested_data_path(self):
        return (
            Path(__file__).resolve().parents[1]
            / "data"
            / "samples"
            / "nested_data.json"
        )

    def test_load_messy_data(self, messy_data_path):
        """Should load messy_data.json correctly."""
        if not messy_data_path.exists():
            pytest.skip("messy_data.json not found")

        df = load_posts(messy_data_path)
        assert len(df) > 0
        assert "customer_feedback" in df.columns
        assert "tracking_code" in df.columns

    def test_detect_messy_columns(self, messy_data_path):
        """Should detect columns in messy data."""
        if not messy_data_path.exists():
            pytest.skip("messy_data.json not found")

        df = load_posts(messy_data_path)
        result = detect_columns(df)

        # Should detect customer_feedback as text
        assert "customer_feedback" in result["text_columns"]
        # Should detect tracking_code as ID
        assert "tracking_code" in result["id_columns"]

    def test_standardize_messy_data(self, messy_data_path):
        """Should standardize messy data correctly."""
        if not messy_data_path.exists():
            pytest.skip("messy_data.json not found")

        df = load_posts(messy_data_path)
        result = standardize_dataframe(
            df,
            text_column="customer_feedback",
            id_column="tracking_code",
            fill_missing=True,
            drop_duplicates=True,
        )

        # Should have standard columns
        assert list(result.columns) == ["post_id", "text"]
        # Should have dropped duplicate TRK-001
        assert len(result[result["post_id"] == "TRK-001"]) == 1
        # Should have filled null values
        assert "" in result["text"].values

    def test_load_nested_data(self, nested_data_path):
        """Should load and normalize nested_data.json."""
        if not nested_data_path.exists():
            pytest.skip("nested_data.json not found")

        df = load_posts(nested_data_path)
        assert len(df) > 0
        # Should have flattened nested fields
        assert "user_name" in df.columns or "record_id" in df.columns
