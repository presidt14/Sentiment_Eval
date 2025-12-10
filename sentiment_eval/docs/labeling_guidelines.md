# Labeling Guidelines: Edge Case Taxonomy

> **Purpose**: This document codifies the labeling definitions for edge cases to prevent concept drift in future labeling efforts. All labelers must follow these guidelines to ensure consistency.

---

## Core Principles

1. **Zone of Control**: Only flag sentiment as NEGATIVE if the issue is within the brand's control (app, service, withdrawals, customer support).
2. **Sport Outcomes Are Not Brand Issues**: Losing a bet due to match results is NOT a brand complaint.
3. **Mixed Sentiment Rule**: If a post contains ANY negative brand-controlled element, flag as NEGATIVE.
4. **Sarcasm Detection**: Irony about brand services = NEGATIVE. Irony about sport results = NEUTRAL.

---

## Edge Case Taxonomy

### 1. `promo_hype`
**Definition**: Affiliate noise, promotional content, or odds-related posts with no genuine sentiment signal.

**Label**: `NEUTRAL` | **Brand Relevance**: `False`

**Canonical Example**:
> "🔥 Best odds on Man City vs Arsenal! Bet now at William Hill! #ad"

**Why**: This is marketing noise, not customer feedback. No actionable insight.

---

### 2. `sarcasm_brand`
**Definition**: Ironic or sarcastic statements about brand-controlled services (app, limits, withdrawals, support).

**Label**: `NEGATIVE` | **Brand Relevance**: `True`

**Canonical Example**:
> "Love how the app crashes every time I try to cash out. Really great experience 👍"

**Why**: The sarcasm masks genuine frustration about a brand-controlled issue. This is actionable.

---

### 3. `sarcasm_sport`
**Definition**: Ironic or sarcastic statements about match results or team performance.

**Label**: `NEUTRAL` | **Brand Relevance**: `False`

**Canonical Example**:
> "Thanks William Hill for making Arsenal lose that penalty 🙄"

**Why**: The user is sarcastically blaming the brand for a sport outcome. The brand cannot control match results.

---

### 4. `mixed_sentiment`
**Definition**: Posts containing both positive and negative elements, where at least one negative element is brand-controlled.

**Label**: `NEGATIVE` | **Brand Relevance**: `True`

**Canonical Example**:
> "Great odds on the app, but customer service is non-existent."

**Why**: The positive (odds) does not negate the negative (CS). Brand-controlled issues take precedence.

---

### 5. `ambiguous_blame`
**Definition**: Posts with unclear causality where it's not obvious if the brand or external factors are responsible.

**Label**: `NEGATIVE` (if no sport context) | **Brand Relevance**: `True` (default)

**Canonical Example**:
> "This site is a joke."

**Why**: Without explicit sport context, assume the user is criticizing the brand. Err on the side of flagging.

---

### 6. `self_deprecation`
**Definition**: User blaming themselves for gambling losses, not the brand.

**Label**: `NEUTRAL` | **Brand Relevance**: `False`

**Canonical Example**:
> "I am the worst gambler in history."

**Why**: The user is not criticizing the brand. This is personal reflection, not actionable feedback.

---

### 7. `implied_causality`
**Definition**: Sarcastic or conspiratorial implications that the brand is sabotaging the user.

**Label**: `NEGATIVE` | **Brand Relevance**: `True`

**Canonical Example**:
> "Funny how the server only goes down when I'm winning."

**Why**: Even if irrational, this is a service complaint. The user perceives a brand failure.

---

### 8. `conspiracy_accusation`
**Definition**: Explicit accusations that the brand is rigging odds or scamming users.

**Label**: `NEGATIVE` | **Brand Relevance**: `True`

**Canonical Example**:
> "Rigged odds. Every time I bet big, I lose."

**Why**: Scam accusations are high-severity compliance risks, regardless of validity.

---

### 9. `sport_blame_deflection`
**Definition**: User correctly attributes their loss to sport outcomes, not the brand.

**Label**: `NEUTRAL` | **Brand Relevance**: `False`

**Canonical Example**:
> "William Hill didn't make Arsenal miss that penalty lol"

**Why**: The user explicitly acknowledges the brand is not at fault.

---

### 10. `tech_failure_consequence`
**Definition**: Technical failure (app crash, timeout) that caused the user to miss a bet or lose money.

**Label**: `NEGATIVE` | **Brand Relevance**: `True`

**Canonical Example**:
> "App crashed mid-bet and I missed the winner. Fuming."

**Why**: App reliability is brand-controlled. This is a valid service complaint.

---

### 11. `mixed_sentiment_balanced`
**Definition**: Posts with roughly equal positive and negative brand elements.

**Label**: `NEGATIVE` | **Brand Relevance**: `True`

**Canonical Example**:
> "Verification took 2 weeks but support was helpful."

**Why**: A 2-week delay is a material compliance/churn risk. Polite support does not negate service failure.

---

### 12. `positive_brand_self_deprecation`
**Definition**: All brand aspects are positive, but the user lost due to their own gambling choices.

**Label**: `NEUTRAL` | **Brand Relevance**: `False`

**Canonical Example**:
> "The odds were great, the app was smooth, and I still lost. Classic me."

**Why**: The user praises the brand but blames themselves for the loss. No brand issue.

---

## Labeling Checklist

Before assigning a label, ask:

1. ✅ Is the complaint about something the brand controls? (app, service, withdrawals, support)
2. ✅ Is there ANY negative brand-controlled element? → Flag as NEGATIVE
3. ✅ Is the complaint about sport outcomes or user's own choices? → NEUTRAL, brand_relevance=False
4. ✅ Is sarcasm present? Determine if it's about brand (NEGATIVE) or sport (NEUTRAL)
5. ✅ When in doubt, flag as NEGATIVE (err on the side of caution)

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-10 | ML Team | Initial taxonomy with 12 edge case types |
