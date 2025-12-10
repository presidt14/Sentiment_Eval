# Compliance Sentiment Engine 🛡️

A production-grade sentiment analysis system designed for the betting industry. It distinguishes between actionable **Brand Risk** (e.g., "Withdrawal failed") and ignorable **Sport Sarcasm** (e.g., "My horse ran backwards").

## 🚀 Key Features

* **Zone of Control Logic:** Automatically filters out complaints about match results, referees, or bad luck.
* **Safety Guardrails:** Hard-coded logic (`src/models/base.py`) forces `Sentiment=NEUTRAL` if `BrandRelevance=FALSE`.
* **Hard-Gate CI/CD:** GitHub Actions pipeline blocks any code change that drops accuracy below 100% on critical slices.
* **Multi-Model Support:** Swap between Gemma, Claude, OpenAI, and DeepSeek with a single CLI flag.
* **Observability:** Streamlit Dashboard for failure analysis and data curation.

## 📂 Project Structure

```text
├── .github/workflows/       # CI/CD Quality Gate
├── data/
│   └── gold_standard/       # v3 Gold Standard (The Source of Truth)
├── docs/
│   └── labeling_guidelines.md  # Taxonomy & Rulebook
├── results/                 # Cached Golden Results
├── scripts/
│   ├── dashboard.py         # Streamlit Analysis Tool
│   └── evaluate_slices.py   # The Examiner (Slices A-F)
└── src/
    ├── models/              # Model Implementations (Gemma, OpenAI, Claude, DeepSeek)
    └── model_factory.py     # Factory Pattern for Model Selection
```

## 🛠️ Usage

### 1. Run the Dashboard

Visualize results and fix labels interactively.

```bash
streamlit run scripts/dashboard.py
```

### 2. Run Evaluation (CLI)

Test the model against the Gold Standard.

```bash
python scripts/evaluate_slices.py --model-results results/results_gold_standard_v3_gemma.csv --print-failures
```

### 3. Run Inference

```bash
# Default (Gemma)
python scripts/run_gold_standard_inference.py --model gemma

# Switch providers
python scripts/run_gold_standard_inference.py --model claude
python scripts/run_gold_standard_inference.py --model openai
python scripts/run_gold_standard_inference.py --model deepseek

# Mock mode (no API keys needed)
python scripts/run_gold_standard_inference.py --mock
```

## 🔐 Environment Variables

Create a `.env` file with your API keys:

```bash
NEBIUS_API_KEY=your_nebius_key      # For Gemma
ANTHROPIC_API_KEY=your_anthropic_key # For Claude
OPENAI_API_KEY=your_openai_key       # For GPT-4
DEEPSEEK_API_KEY=your_deepseek_key   # For DeepSeek
```

## 🐳 Docker

```bash
# Build
docker build -t sentiment-engine .

# Run Dashboard
docker run -p 8501:8501 sentiment-engine

# Run Inference (with API keys)
docker run -e NEBIUS_API_KEY=xxx sentiment-engine python scripts/run_gold_standard_inference.py --model gemma
```

## 📊 Evaluation Slices

| Slice | Description | Threshold |
|-------|-------------|-----------|
| A | Noise Suppression (promo/irrelevant) | 100% |
| B | True Brand Sentiment | 90% |
| C | Sarcasm Detection (Brand) | 90% |
| D | Sarcasm Rejection (Sport) | 90% |
| E | Adversarial Cases | 90% |
| F | Brand Relevance Health | 100% |

## 🔒 Zone of Control Guardrail

The safety guardrail is enforced in `src/models/base.py` and inherited by ALL models:

```python
if brand_relevance == False:
    sentiment = "neutral"  # Cannot be negative about things outside brand control
```

This ensures consistent behavior regardless of which LLM provider is used.
