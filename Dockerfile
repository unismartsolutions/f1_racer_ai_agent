# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for NLP libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-download NLTK data to avoid runtime downloads
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('vader_lexicon', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    nltk.download('maxent_ne_chunker', quiet=True); \
    nltk.download('words', quiet=True)"

# Copy application files
COPY f1_agent.py .
COPY run_agent.py .

# Make run_agent.py executable
RUN chmod +x run_agent.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for future web interface)
EXPOSE 8000

# Default command - interactive mode
CMD ["python", "run_agent.py"]

# Alternative commands for different use cases:
# For demo mode: CMD ["python", "f1_agent.py"]
# For API mode: CMD ["python", "-c", "from f1_agent import F1RacerAgent; print('F1 Agent ready for import')"]
