"""
Hybrid Router Module for Gemma-First Architecture

Routes classification requests between fast internal model (Gemma) and
external fallback models (Claude/DeepSeek) based on confidence and risk.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .models.base import SentimentModel
from .utils import ESCALATION_KEYWORDS, has_escalation_keywords
from .filters import should_skip_post


@dataclass
class RouterConfig:
    """Configuration for the hybrid router."""
    
    # Confidence threshold for routing to fallback
    confidence_threshold: float = 0.85
    
    # Latency threshold in ms - if fast model exceeds this, log warning
    latency_warning_ms: float = 500.0
    
    # Negative types that always trigger fallback
    escalate_negative_types: List[str] = field(default_factory=lambda: [
        "scam_accusation",
        "regulatory_criticism",
    ])
    
    # Whether to use fallback for low confidence
    use_fallback_for_low_confidence: bool = True
    
    # Whether to use fallback for high-risk negative types
    use_fallback_for_high_risk: bool = True
    
    # Whether to apply hard filters before routing
    apply_hard_filters: bool = True
    
    # Minimum tokens for hard filter
    min_tokens: int = 4


@dataclass
class RouterResult:
    """Result from the hybrid router."""
    
    # The classification result
    result: Dict[str, Any]
    
    # Which model was used: "fast", "fallback", "skipped"
    model_used: str
    
    # Latency in milliseconds
    latency_ms: float
    
    # Reason for routing decision
    routing_reason: str
    
    # Whether fallback was triggered
    used_fallback: bool = False
    
    # Skip reason if post was filtered
    skip_reason: Optional[str] = None


class HybridRouter:
    """
    Routes classification between fast (internal) and fallback (external) models.
    
    Routing Logic:
    1. Apply hard filters (skip promotional, link-only, too short)
    2. Check for escalation keywords (always use fallback)
    3. Run fast model
    4. If confidence < threshold OR high-risk negative_type: use fallback
    5. Return result with routing metadata
    """
    
    def __init__(
        self,
        fast_model: SentimentModel,
        fallback_model: Optional[SentimentModel] = None,
        config: Optional[RouterConfig] = None,
    ):
        """
        Initialize the hybrid router.
        
        Args:
            fast_model: The fast internal model (e.g., Gemma)
            fallback_model: The fallback external model (e.g., Claude)
            config: Router configuration
        """
        self.fast_model = fast_model
        self.fallback_model = fallback_model
        self.config = config or RouterConfig()
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "fast_path": 0,
            "fallback_path": 0,
            "skipped": 0,
            "total_latency_ms": 0.0,
        }
    
    def _should_escalate_to_fallback(
        self,
        text: str,
        result: Dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Determine if result should be escalated to fallback model.
        
        Returns:
            Tuple of (should_escalate, reason)
        """
        # Check for escalation keywords in text
        if has_escalation_keywords(text):
            return True, "escalation_keywords"
        
        # Check confidence threshold
        confidence = result.get("confidence", 0.0)
        if (self.config.use_fallback_for_low_confidence and 
            confidence < self.config.confidence_threshold):
            return True, f"low_confidence_{confidence:.2f}"
        
        # Check for high-risk negative types
        negative_type = result.get("negative_type")
        if (self.config.use_fallback_for_high_risk and
            negative_type in self.config.escalate_negative_types):
            return True, f"high_risk_{negative_type}"
        
        return False, ""
    
    def classify(
        self,
        text: str,
        brand_name: str = "Unknown",
    ) -> RouterResult:
        """
        Synchronous classification with routing.
        
        Args:
            text: Post text to classify
            brand_name: Brand name for context
            
        Returns:
            RouterResult with classification and routing metadata
        """
        self._stats["total_requests"] += 1
        start_time = time.time()
        
        # Apply hard filters
        if self.config.apply_hard_filters:
            should_skip, skip_reason = should_skip_post(
                text, 
                min_tokens=self.config.min_tokens
            )
            if should_skip:
                self._stats["skipped"] += 1
                return RouterResult(
                    result={
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "reason": f"Filtered: {skip_reason}",
                        "brand_relevance": False,
                        "negative_type": None,
                    },
                    model_used="skipped",
                    latency_ms=0.0,
                    routing_reason=f"hard_filter:{skip_reason}",
                    skip_reason=skip_reason,
                )
        
        # Check for escalation keywords before running fast model
        if has_escalation_keywords(text) and self.fallback_model:
            # Go directly to fallback for high-risk content
            result = self.fallback_model.classify(text)
            latency_ms = (time.time() - start_time) * 1000
            self._stats["fallback_path"] += 1
            self._stats["total_latency_ms"] += latency_ms
            
            return RouterResult(
                result=result,
                model_used="fallback",
                latency_ms=latency_ms,
                routing_reason="escalation_keywords_direct",
                used_fallback=True,
            )
        
        # Run fast model
        fast_result = self.fast_model.classify(text)
        fast_latency = (time.time() - start_time) * 1000
        
        # Check if we should escalate to fallback
        should_escalate, escalate_reason = self._should_escalate_to_fallback(
            text, fast_result
        )
        
        if should_escalate and self.fallback_model:
            # Run fallback model
            fallback_result = self.fallback_model.classify(text)
            total_latency = (time.time() - start_time) * 1000
            self._stats["fallback_path"] += 1
            self._stats["total_latency_ms"] += total_latency
            
            return RouterResult(
                result=fallback_result,
                model_used="fallback",
                latency_ms=total_latency,
                routing_reason=escalate_reason,
                used_fallback=True,
            )
        
        # Use fast model result
        self._stats["fast_path"] += 1
        self._stats["total_latency_ms"] += fast_latency
        
        # Log warning if latency exceeded threshold
        if fast_latency > self.config.latency_warning_ms:
            print(f"Warning: Fast model latency {fast_latency:.0f}ms exceeds threshold")
        
        return RouterResult(
            result=fast_result,
            model_used="fast",
            latency_ms=fast_latency,
            routing_reason="fast_path",
            used_fallback=False,
        )
    
    async def aclassify(
        self,
        text: str,
        brand_name: str = "Unknown",
    ) -> RouterResult:
        """
        Asynchronous classification with routing.
        
        Args:
            text: Post text to classify
            brand_name: Brand name for context
            
        Returns:
            RouterResult with classification and routing metadata
        """
        self._stats["total_requests"] += 1
        start_time = time.time()
        
        # Apply hard filters
        if self.config.apply_hard_filters:
            should_skip, skip_reason = should_skip_post(
                text,
                min_tokens=self.config.min_tokens
            )
            if should_skip:
                self._stats["skipped"] += 1
                return RouterResult(
                    result={
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "reason": f"Filtered: {skip_reason}",
                        "brand_relevance": False,
                        "negative_type": None,
                    },
                    model_used="skipped",
                    latency_ms=0.0,
                    routing_reason=f"hard_filter:{skip_reason}",
                    skip_reason=skip_reason,
                )
        
        # Check for escalation keywords before running fast model
        if has_escalation_keywords(text) and self.fallback_model:
            result = await self.fallback_model.aclassify(text)
            latency_ms = (time.time() - start_time) * 1000
            self._stats["fallback_path"] += 1
            self._stats["total_latency_ms"] += latency_ms
            
            return RouterResult(
                result=result,
                model_used="fallback",
                latency_ms=latency_ms,
                routing_reason="escalation_keywords_direct",
                used_fallback=True,
            )
        
        # Run fast model
        fast_result = await self.fast_model.aclassify(text)
        fast_latency = (time.time() - start_time) * 1000
        
        # Check if we should escalate to fallback
        should_escalate, escalate_reason = self._should_escalate_to_fallback(
            text, fast_result
        )
        
        if should_escalate and self.fallback_model:
            fallback_result = await self.fallback_model.aclassify(text)
            total_latency = (time.time() - start_time) * 1000
            self._stats["fallback_path"] += 1
            self._stats["total_latency_ms"] += total_latency
            
            return RouterResult(
                result=fallback_result,
                model_used="fallback",
                latency_ms=total_latency,
                routing_reason=escalate_reason,
                used_fallback=True,
            )
        
        self._stats["fast_path"] += 1
        self._stats["total_latency_ms"] += fast_latency
        
        return RouterResult(
            result=fast_result,
            model_used="fast",
            latency_ms=fast_latency,
            routing_reason="fast_path",
            used_fallback=False,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self._stats["total_requests"]
        if total == 0:
            return self._stats.copy()
        
        stats = self._stats.copy()
        stats["fast_path_pct"] = (self._stats["fast_path"] / total) * 100
        stats["fallback_path_pct"] = (self._stats["fallback_path"] / total) * 100
        stats["skipped_pct"] = (self._stats["skipped"] / total) * 100
        stats["avg_latency_ms"] = self._stats["total_latency_ms"] / total
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = {
            "total_requests": 0,
            "fast_path": 0,
            "fallback_path": 0,
            "skipped": 0,
            "total_latency_ms": 0.0,
        }
