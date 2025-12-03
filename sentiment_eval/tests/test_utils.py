import pytest

from src.utils import normalise_sentiment


@pytest.mark.parametrize(
    "raw_label, expected",
    [
        ("Positive!", "positive"),
        ("Very NEGATIVE", "negative"),
        (None, "neutral"),
        ("", "neutral"),
    ],
)
def test_normalise_sentiment(raw_label, expected):
    assert normalise_sentiment(raw_label) == expected
