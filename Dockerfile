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

# Copy the required application files
COPY requirements.txt .
COPY setup_nltk.py .
COPY f1_agent.py .
COPY run_agent.py .
COPY test_agent.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the NLTK setup script
RUN chmod +x setup_nltk.py
RUN python setup_nltk.py

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Make run_agent.py executable
RUN chmod +x run_agent.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose a suitable for the application 
EXPOSE 8000

# Run the agent script 
CMD ["python", "run_agent.py"]
 