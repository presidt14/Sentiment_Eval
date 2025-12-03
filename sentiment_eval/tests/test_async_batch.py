"""
Test async batch processing performance.
Compares sync vs async processing with 100 simulated items.
"""
import asyncio
import time
import pandas as pd
import pytest
from pathlib import Path
import tempfile

import sys
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_batch import run_batch, process_batch_async, _run_batch_sync
from src.models.mock import MockSentimentModel
from src.config import load_settings


def generate_test_data(n: int = 100) -> pd.DataFrame:
    """Generate n test posts with varied sentiment keywords."""
    texts = []
    for i in range(n):
        if i % 3 == 0:
            texts.append(f"Post {i}: This is a great product, I love it!")
        elif i % 3 == 1:
            texts.append(f"Post {i}: This is terrible, worst experience ever.")
        else:
            texts.append(f"Post {i}: Just a regular update about things.")
    
    return pd.DataFrame({
        "post_id": list(range(1, n + 1)),
        "text": texts,
    })


@pytest.fixture
def mock_models():
    """Create mock models for testing."""
    return [MockSentimentModel(seed=42)]


@pytest.fixture
def test_df():
    """Generate 100 test posts."""
    return generate_test_data(100)


def test_async_vs_sync_performance(test_df, mock_models, tmp_path):
    """
    Test that async processing handles concurrent requests efficiently.
    
    Note: The sync version doesn't have simulated delay (instant),
    while async has 100ms simulated network latency per request.
    
    This test verifies:
    1. Both produce correct number of results
    2. Async with concurrency=10 processes 100 items in ~1s (not 10s)
    """
    # Test sync processing (no delay - instant)
    sync_start = time.time()
    sync_results = _run_batch_sync(test_df, mock_models, "text", "post_id")
    sync_elapsed = time.time() - sync_start
    
    # Test async processing (100ms delay per item, but concurrent)
    async_start = time.time()
    async_results = asyncio.run(
        process_batch_async(test_df, mock_models, "text", "post_id", concurrency=10)
    )
    async_elapsed = time.time() - async_start
    
    print(f"\nSync processing (no delay): {sync_elapsed:.2f}s")
    print(f"Async processing (100ms simulated delay): {async_elapsed:.2f}s")
    
    # Verify results are equivalent
    assert len(sync_results) == len(async_results) == 100
    
    # Async should complete 100 items with 100ms delay each in ~1-2 seconds
    # (not 10 seconds which would be sequential)
    # With concurrency=10: 100 items / 10 concurrent = 10 batches * 0.1s = 1s
    assert async_elapsed < 3.0, f"Async took {async_elapsed:.2f}s, expected < 3s"
    
    # Verify concurrency benefit: if it were sequential, it would take ~10s
    # We expect it to be much faster due to parallelism
    expected_sequential_time = 100 * 0.1  # 10 seconds
    assert async_elapsed < expected_sequential_time / 3, \
        f"Async should be at least 3x faster than sequential ({expected_sequential_time}s)"


def test_async_batch_with_file(tmp_path):
    """Test full run_batch with async processing."""
    # Create test input file
    df = generate_test_data(20)
    input_path = tmp_path / "test_input.csv"
    output_path = tmp_path / "test_output.csv"
    df.to_csv(input_path, index=False)
    
    # Create mock config
    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.mock.yaml"
    
    # Run async batch
    result_df = run_batch(
        input_path=input_path,
        output_path=output_path,
        config_path=config_path,
        use_async=True,
        concurrency=5,
    )
    
    assert len(result_df) == 20
    assert output_path.exists()
    assert "mock_sentiment" in result_df.columns


def test_semaphore_limits_concurrency(test_df, mock_models):
    """Test that semaphore properly limits concurrent requests."""
    # Track concurrent executions
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()
    
    original_aclassify = mock_models[0].aclassify
    
    async def tracked_aclassify(text):
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        
        result = await original_aclassify(text)
        
        async with lock:
            current_concurrent -= 1
        
        return result
    
    mock_models[0].aclassify = tracked_aclassify
    
    # Run with concurrency limit of 5
    asyncio.run(
        process_batch_async(test_df[:20], mock_models, "text", "post_id", concurrency=5)
    )
    
    # Max concurrent should not exceed the semaphore limit
    assert max_concurrent <= 5, f"Max concurrent {max_concurrent} exceeded limit of 5"
    
    # Restore original method
    mock_models[0].aclassify = original_aclassify


if __name__ == "__main__":
    # Quick manual test
    print("Generating 100 test items...")
    df = generate_test_data(100)
    models = [MockSentimentModel(seed=42)]
    
    # For fair comparison, add delay to sync classify too
    import time as time_module
    original_classify = models[0].classify
    def delayed_classify(text):
        time_module.sleep(0.1)  # Same 100ms delay as aclassify
        return original_classify(text)
    
    print("\n--- Sync Processing (with 100ms delay per item) ---")
    models[0].classify = delayed_classify
    sync_start = time.time()
    sync_results = _run_batch_sync(df, models, "text", "post_id")
    sync_elapsed = time.time() - sync_start
    print(f"Sync completed in {sync_elapsed:.2f}s")
    
    # Restore original for async test
    models[0].classify = original_classify
    
    print("\n--- Async Processing (with 100ms simulated network lag) ---")
    async_start = time.time()
    async_results = asyncio.run(
        process_batch_async(df, models, "text", "post_id", concurrency=10)
    )
    async_elapsed = time.time() - async_start
    print(f"Async completed in {async_elapsed:.2f}s")
    
    speedup = sync_elapsed / async_elapsed if async_elapsed > 0 else float('inf')
    print(f"\n=== Speedup: {speedup:.1f}x ===")
    print(f"Expected: ~10x (100 items / 10 concurrent = 10 batches)")
    print(f"Sync: 100 * 0.1s = 10s sequential")
    print(f"Async: 10 batches * 0.1s = 1s parallel")
