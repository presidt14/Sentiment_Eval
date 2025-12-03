# Multi-Model Sentiment Evaluation Starter Kit

Runs social posts through multiple LLM sentiment providers (Gemma, Claude, Gemini, DeepSeek), compares them to human labels, and generates reports.

## Quick Start

```bash
# Create venv and install
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Validate environment
python -m src.validate_env
```

## Usage

### Batch Processing
```bash
python -m src.run_batch --input data/samples/posts_sample.csv
```

### Evaluate & Report
```bash
python -m src.evaluate    # Accuracy, F1-Score, Confusion Matrix
python -m src.report      # Generate markdown summary
```

### Interactive UI
```bash
streamlit run app.py
```

---

## Advanced Features

### Mock Mode (No API Keys Required)
Test the full pipeline without API calls using deterministic mock responses:
```bash
# Via Makefile
make mock

# Or directly
python -m src.run_batch --input data/samples/posts_sample.csv --config config/settings.mock.yaml
```
Mock mode uses `seed: 42` for reproducible results.

### Async Processing
For large batches, enable concurrent API calls:
```bash
python -m src.run_batch --input data/samples/posts_sample.csv --async
```

### Prompt Strategies
Customize LLM behavior via `config/prompts.yaml`. Available strategies:

| Strategy | Use Case |
|----------|----------|
| `default_sentiment` | Standard i-gaming/compliance classification |
| `sarcasm_detector` | Detects irony and hidden sentiment |
| `strict_compliance` | Conservative risk-focused classification |
| `customer_feedback` | Customer review analysis |
| `multilingual` | Mixed-language and code-switching support |

### Human Labeling Workflow
1. Upload data via **Data Upload** page in Streamlit UI
2. Label posts manually in the interface
3. Export labeled data for evaluation

---

## Automation

```bash
make setup   # Install dependencies
make test    # Run pytest
make mock    # Run mock batch
make ui      # Launch Streamlit
make clean   # Remove cache/temp files
```

Windows users: Use `run_demo.bat` for a guided demo.
