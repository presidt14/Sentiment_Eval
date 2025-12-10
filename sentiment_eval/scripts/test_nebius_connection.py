"""
Test Nebius TokenFactory API connection with Gemma model.

Usage:
    python scripts/test_nebius_connection.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.models.gemma import GemmaSentimentModel
from src.config import get_prompt_strategy


def main():
    print("=" * 60)
    print("Testing Nebius TokenFactory API Connection")
    print("=" * 60)
    
    # Initialize model with compliance prompt strategy
    print("\n1. Initializing GemmaSentimentModel...")
    prompt_config = get_prompt_strategy("compliance_risk_assessor")
    model = GemmaSentimentModel(prompt_config=prompt_config)
    
    print(f"   Model: {model.model}")
    print(f"   Base URL: {model.base_url}")
    print(f"   API Key set: {bool(model.api_key)}")
    
    if not model.api_key:
        print("\n❌ ERROR: NEBIUS_API_KEY not set!")
        print("   Create a .env file in the project root with:")
        print("   NEBIUS_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Test with a simple example
    print("\n2. Testing classification...")
    test_text = "I love using William Hill, the app is fantastic and withdrawals are super fast!"
    
    print(f"   Test text: {test_text[:60]}...")
    
    result = model.classify(test_text)
    
    print("\n3. Result:")
    print(f"   Sentiment: {result.get('sentiment')}")
    print(f"   Confidence: {result.get('confidence')}")
    print(f"   Brand Relevance: {result.get('brand_relevance')}")
    print(f"   Negative Type: {result.get('negative_type')}")
    print(f"   Reason: {result.get('reason', '')[:100]}...")
    
    if "_raw_response" in result:
        print(f"\n   Raw response preview: {result['_raw_response'][:200]}...")
    
    # Test sarcasm detection
    print("\n4. Testing sarcasm detection...")
    sarcasm_text = "Oh wow, another 'technical issue' right when I try to cash out. Great timing as always William Hill 🤡"
    
    print(f"   Test text: {sarcasm_text[:60]}...")
    
    result2 = model.classify(sarcasm_text)
    
    print(f"   Sentiment: {result2.get('sentiment')} (expected: negative)")
    print(f"   Brand Relevance: {result2.get('brand_relevance')} (expected: True)")
    
    print("\n" + "=" * 60)
    print("✅ Connection test complete!")
    print("=" * 60)
    
    if result2.get('sentiment') == 'negative':
        print("🎉 Sarcasm detected correctly!")
    else:
        print("⚠️  Sarcasm not detected - model may need prompt tuning")


if __name__ == "__main__":
    main()
