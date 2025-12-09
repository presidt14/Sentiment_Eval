# Gold Standard Labeling Dataset

## Purpose
Human-labeled ground truth for evaluating sentiment models against vendor baseline.

## Labeling Schema

| Field | Type | Values |
|-------|------|--------|
| `post_id` | string | From Id column |
| `text` | string | Post content |
| `brand` | string | Client name |
| `platform` | string | TWITTER, FACEBOOK |
| `vendor_sentiment` | string | Original vendor output |
| `human_sentiment` | enum | positive, neutral, negative |
| `brand_relevance` | boolean | true, false |
| `negative_type` | enum | See below |
| `labeler` | string | Labeler initials |
| `labeled_at` | datetime | ISO format |
| `notes` | string | Optional notes |

## Negative Type Values

- `customer_dissatisfaction` - UX issues, site problems, login failures
- `scam_accusation` - Rigged games, stolen money, fraud claims
- `regulatory_criticism` - Underage gambling, KYC failures
- `general_negativity` - Vague complaints
- `null` - Use when sentiment is not negative

## Labeling Guidelines

1. **Brand Relevance**: Is the user specifically discussing the brand?
   - Promotional posts BY affiliates = false (not user sentiment)
   - Complaints ABOUT the brand = true
   - Passing mention = false

2. **Sarcasm**: Flag sarcastic positive statements as NEGATIVE

3. **Promotional Noise**: Affiliate marketing posts are typically neutral
