# Equity Intelligence Platform - one image for the pipeline, API and dashboard.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# libgomp: OpenMP runtime needed by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image ~2 GB smaller than the CUDA build
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .
RUN pip install --no-deps -e . \
    && useradd --create-home appuser \
    && mkdir -p data models results \
    && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/health || exit 1

# Default: the REST API. docker-compose overrides the command for the other services.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
