from typing import Any, Dict, Optional
import pandas as pd

SENTIMENT_SCHEMA = {"negative", "neutral", "positive"}

NEGATIVE_TYPE_SCHEMA = {
    "customer_dissatisfaction",
    "scam_accusation",
    "regulatory_criticism",
    "general_negativity",
    None,
}

# Keywords that should always trigger escalation to fallback model
ESCALATION_KEYWORDS = {
    "scam",
    "rigged",
    "fraud",
    "stolen",
    "illegal",
    "underage",
    "money laundering",
    "cheat",
    "rip off",
    "ripoff",
    "con",
    "theft",
}


def normalise_sentiment(label: str) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return "neutral"
    if not label:
        return "neutral"

    l = str(label).strip().lower()
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


def normalise_negative_type(value: Any) -> Optional[str]:
    """Normalise negative_type to valid enum value or None."""
    if value is None or value == "" or str(value).lower() in ("null", "none", "na", "n/a"):
        return None
    
    value_str = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    
    valid_types = {
        "customer_dissatisfaction": "customer_dissatisfaction",
        "dissatisfaction": "customer_dissatisfaction",
        "ux_issue": "customer_dissatisfaction",
        "service_issue": "customer_dissatisfaction",
        "scam_accusation": "scam_accusation",
        "scam": "scam_accusation",
        "fraud": "scam_accusation",
        "rigged": "scam_accusation",
        "regulatory_criticism": "regulatory_criticism",
        "regulatory": "regulatory_criticism",
        "compliance": "regulatory_criticism",
        "general_negativity": "general_negativity",
        "general": "general_negativity",
        "other": "general_negativity",
    }
    
    return valid_types.get(value_str, None)


def ensure_result_dict_v2(provider: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise provider result into extended schema.
    
    Expected keys in row:
    - sentiment: positive/neutral/negative
    - confidence: 0.0-1.0
    - reason/rationale: explanation string
    - brand_relevance: boolean
    - negative_type: enum or null
    
    Returns dict with provider-prefixed keys for all fields.
    """
    sentiment = normalise_sentiment(str(row.get("sentiment", "")))
    
    # Handle confidence
    conf = row.get("confidence", 0.0)
    try:
        confidence = float(conf)
    except (ValueError, TypeError):
        confidence = 0.0
    
    # Handle reason (support both 'reason' and 'rationale' keys)
    reason = str(row.get("reason", row.get("rationale", "")))
    
    # Handle brand_relevance (boolean)
    brand_rel = row.get("brand_relevance", None)
    if isinstance(brand_rel, bool):
        brand_relevance = brand_rel
    elif isinstance(brand_rel, str):
        brand_relevance = brand_rel.lower() in ("true", "yes", "1")
    else:
        brand_relevance = None
    
    # Handle negative_type
    negative_type = normalise_negative_type(row.get("negative_type", None))
    
    result = {
        f"{provider}_sentiment": sentiment,
        f"{provider}_confidence": confidence,
        f"{provider}_reason": reason,
        f"{provider}_brand_relevance": brand_relevance,
        f"{provider}_negative_type": negative_type,
    }
    
    return result


def has_escalation_keywords(text: str) -> bool:
    """Check if text contains any escalation keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ESCALATION_KEYWORDS)


def is_actionable(result: Dict[str, Any], provider: str = "") -> bool:
    """
    Determine if a classification result should trigger an action.
    
    Actionable = brand_relevance AND sentiment=negative AND negative_type != general_negativity
    """
    prefix = f"{provider}_" if provider else ""
    
    brand_relevance = result.get(f"{prefix}brand_relevance", False)
    sentiment = result.get(f"{prefix}sentiment", "neutral")
    negative_type = result.get(f"{prefix}negative_type", None)
    
    return (
        brand_relevance is True and
        sentiment == "negative" and
        negative_type is not None and
        negative_type != "general_negativity"
    )
