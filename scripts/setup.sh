#!/bin/bash
# =============================================================================
# scripts/setup.sh
# One-time setup for the Health AI service on your Linux server.
# Run: chmod +x scripts/setup.sh && ./scripts/setup.sh
# =============================================================================
set -e

echo "=== Health AI Service Setup ==="

# 1. Copy env file
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — please edit it with your secrets before starting."
fi

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download spaCy model (needed by presidio)
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# 4. Download faster-whisper model
echo "Downloading faster-whisper model (this may take a few minutes)..."
python -c "
from faster_whisper import WhisperModel
import os
from app.core.config import get_settings
s = get_settings()
WhisperModel(s.WHISPER_MODEL_SIZE, device='cpu', compute_type='int8',
             download_root=os.path.join(s.MODELS_DIR, 'whisper'))
print('Whisper model downloaded.')
"

# 5. Export HuggingFace models to ONNX
echo "Exporting NLP models to ONNX (first run only)..."
python scripts/export_onnx.py

# 6. Build FAISS index from demo content
echo "Building FAISS recommendation index..."
python scripts/build_faiss_index.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Start the full stack:"
echo "  docker compose up --build -d"
echo ""
echo "Or start services individually:"
echo "  # FastAPI"
echo "  gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8001"
echo ""
echo "  # Celery workers (run each in a separate terminal/screen)"
echo "  celery -A app.core.celery_app worker --queues emergency  --concurrency 4 --hostname emergency@%h"
echo "  celery -A app.core.celery_app worker --queues realtime   --concurrency 2 --hostname realtime@%h"
echo "  celery -A app.core.celery_app worker --queues normal     --concurrency 8 --hostname normal@%h"
echo "  celery -A app.core.celery_app worker --queues batch      --concurrency 4 --hostname batch@%h"
echo ""
echo "  # Celery beat (periodic tasks)"
echo "  celery -A app.core.celery_app beat --loglevel info"
echo ""
echo "  # Flower (monitoring UI at http://localhost:5555)"
echo "  celery -A app.core.celery_app flower --port=5555"
