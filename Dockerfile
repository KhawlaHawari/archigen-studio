# ArchiGen 3D - Production Multi-Stage Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV/image/rendering libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final production stage
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy project files
COPY src/ /app/src/
COPY models/ /app/models/
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
