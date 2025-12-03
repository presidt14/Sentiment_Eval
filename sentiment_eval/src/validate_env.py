from pathlib import Path
import sys
from .models import get_active_models

def validate_environment():
    print("--- Validating Environment & API Keys ---\n")
    
    try:
        models = get_active_models()
    except Exception as e:
        print(f"CRITICAL: Failed to load models or configuration. Error: {e}")
        sys.exit(1)

    if not models:
        print("WARNING: No models configured in 'active_models' in settings.yaml.")
        return

    print(f"Found {len(models)} active models: {[m.name for m in models]}")
    
    test_text = "This is a connection test."
    success_count = 0

    for model in models:
        print(f"\nTesting connection for: {model.name}...")
        try:
            # Attempt a minimal classification
            result = model.classify(test_text)
            
            # Basic validation of the response structure
            if "sentiment" in result:
                print(f"✅ SUCCESS: {model.name} is connected.")
                success_count += 1
            else:
                print(f"⚠️  WARNING: {model.name} responded but missing 'sentiment' key.")
                print(f"   Response: {result}")
                
        except Exception as e:
            print(f"❌ FAILED: {model.name} encountered an error.")
            print(f"   Error details: {e}")
            # Hint at common auth errors
            error_str = str(e).lower()
            if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                print("   -> Likely an invalid API Key. Check your .env file.")

    print(f"\n--- Validation Complete: {success_count}/{len(models)} models operational ---")

if __name__ == "__main__":
    validate_environment()