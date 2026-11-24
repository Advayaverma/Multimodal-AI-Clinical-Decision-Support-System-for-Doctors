# ═══════════════════════════════════════════════════════════════
# Dockerfile — AI-CDSS Backend (Root single-service / Cloud PaaS)
# Compatible with Docker Compose, Render, Fly.io, Railway, HF Spaces
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim

# System dependencies for OCR, OpenCV, and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Cache dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend, machine learning weights, and clinical RAG index
COPY backend/ ./backend/
COPY ml/models/ ./ml/models/
COPY ml/ecg_classification/ ./ml/ecg_classification/
COPY ml/scripts/ ./ml/scripts/
COPY rag/ ./rag/

RUN mkdir -p ./backend/uploads ./backend/data

ENV PYTHONPATH="/app/backend:/app"
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend
EXPOSE 8000
EXPOSE 10000

CMD ["python", "run_backend.py"]
