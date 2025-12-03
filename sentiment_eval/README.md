Multi-Model Sentiment Evaluation Starter Kit

This repo runs social posts through multiple LLM sentiment providers (Gemma, Claude, Gemini, DeepSeek), compares them to human labels, and generates reports.

Setup

Create venv and install dependencies:

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt


Config:

cp .env.example .env
# Open .env and fill in your API keys


Validate Environment (New):
Run this quick check to ensure your keys are working before processing large batches.

python -m src.validate_env


Run Batch Processing:

python -m src.run_batch --input data/samples/posts_sample.csv


Evaluate Results:
Calculate metrics (Accuracy, F1-Score, Confusion Matrix).

python -m src.evaluate


Generate Report:
Create a markdown summary.

python -m src.report


Browse Results:
Launch the interactive UI.

streamlit run app.py
