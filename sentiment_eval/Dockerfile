# Compliance Sentiment Engine
# Production Docker image for inference and dashboard

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Default command: Run the Dashboard
# Override with: docker run sentiment-engine python scripts/run_gold_standard_inference.py --model gemma
CMD ["streamlit", "run", "scripts/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
