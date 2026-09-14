# Dockerfile for Module 3 (support_assistant) -- Zepto Data & AI Platform
#
# Build context is the REPO ROOT (not this folder), so this image installs
# from the project's single consolidated root requirements.txt instead of
# duplicating a second requirements file inside this module. Build/run from
# the repo root:
#   docker build -t zepto-support -f support_assistant/Dockerfile .
#   docker run -p 7860:7860 zepto-support
#
# NOTE: This Dockerfile was written to be correct but has NOT been built or
# run in this development environment because Docker is not installed on
# this machine ("docker: command not found"). See README.md for details.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by torch / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching).
# requirements.txt here is the project's single root requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the policy document corpus
COPY support_assistant/ingest.py .
COPY support_assistant/prompt_template.py .
COPY support_assistant/graph.py .
COPY support_assistant/main.py .
COPY support_assistant/docs/ ./docs/

# Build the ChromaDB collection at image-build time so the container starts
# ready to serve retrieval requests without a first-request ingestion delay.
RUN python ingest.py

EXPOSE 7860

ENV MOCK_LLM=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
