from typing import Dict, Any

SENTIMENT_SCHEMA = {"negative", "neutral", "positive"}

def normalise_sentiment(label: str) -> str:
    if not label:
        return "neutral"
    
    l = label.strip().lower()
    if "pos" in l:
        return "positive"
    elif "neg" in l:
        return "negative"
    elif "neu" in l:
        return "neutral"
    else:
        return "neutral"

def ensure_result_dict(provider: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise provider result into a fixed schema.
    Expected keys in row: sentiment, confidence, reason.
    """
    
    sentiment = normalise_sentiment(str(row.get("sentiment", "")))
    conf = row.get("confidence", 0.0)
    
    try:
        confidence = float(conf)
    except Exception:
        confidence = 0.0
        
    reason = str(row.get("reason", ""))
    
    return {
        f"{provider}_sentiment": sentiment,
        f"{provider}_confidence": confidence,
        f"{provider}_reason": reason,
    }
