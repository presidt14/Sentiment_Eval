"""
Test sarcasm distinction: brand sarcasm vs sport sarcasm
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.models.gemma import GemmaSentimentModel
from src.config import get_prompt_strategy


def main():
    print("=" * 60)
    print("Testing Sarcasm Distinction")
    print("=" * 60)
    
    prompt_config = get_prompt_strategy("compliance_risk_assessor")
    model = GemmaSentimentModel(prompt_config=prompt_config)
    
    test_cases = [
        # Brand sarcasm - should be NEGATIVE
        {
            "text": "Oh wow, another 'technical issue' right when I try to cash out. Great timing as always William Hill 🤡",
            "expected_sentiment": "negative",
            "expected_brand_rel": True,
            "type": "BRAND SARCASM"
        },
        {
            "text": "Love how they limit my stakes to 50p after I win once. Best bookie ever! NOT.",
            "expected_sentiment": "negative", 
            "expected_brand_rel": True,
            "type": "BRAND SARCASM"
        },
        # Sport sarcasm - should be NEUTRAL
        {
            "text": "Thanks William Hill for the boosted odds on that horse. It's still running backwards 🙄",
            "expected_sentiment": "neutral",
            "expected_brand_rel": False,
            "type": "SPORT SARCASM"
        },
        {
            "text": "Genius bet by me using the boost. Lost in 5 seconds. #clown",
            "expected_sentiment": "neutral",
            "expected_brand_rel": False,
            "type": "SPORT SARCASM"
        },
    ]
    
    results = []
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {tc['type']} ---")
        print(f"Text: {tc['text'][:60]}...")
        
        result = model.classify(tc['text'])
        
        sent_match = result.get('sentiment', '').lower() == tc['expected_sentiment']
        
        print(f"Expected: sentiment={tc['expected_sentiment']}, brand_rel={tc['expected_brand_rel']}")
        print(f"Got:      sentiment={result.get('sentiment')}, brand_rel={result.get('brand_relevance')}")
        print(f"Match: {'✅' if sent_match else '❌'}")
        
        results.append(sent_match)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} correct")
    print("=" * 60)
    
    if all(results):
        print("🎉 All sarcasm types correctly distinguished!")
    else:
        print("⚠️ Some cases need further refinement")


if __name__ == "__main__":
    main()
