# ArchiGen 3D - Production Multi-Stage Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# System libraries for image / rendering packages
# (libgl1 replaces libgl1-mesa-glx, which no longer exists on current Debian/Ubuntu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch first, from the PyTorch CPU index: keeps the image small enough for the
# GitHub runner (the default PyPI build pulls several GB of CUDA libraries). The rest of
# requirements.txt then reuses this torch/torchvision instead of downloading another build.
RUN pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir --user -r requirements.txt

# Final production stage
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy project files
COPY src/ /app/src/
# Trained weights are not in Git (DVC / Kaggle): the folder is created empty and can be
# mounted at runtime (docker-compose mounts ./models). The API loads models lazily.
RUN mkdir -p /app/models
COPY params.yaml /app/params.yaml
COPY dvc.yaml /app/dvc.yaml
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md

EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI via Uvicorn with 2 workers
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
