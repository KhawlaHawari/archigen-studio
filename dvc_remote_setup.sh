#!/usr/bin/env bash
# ArchiGen 3D - DVC Remote Storage Setup (MLOps Level 2+ / Level 3)
# TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
# Encadrant : Haythem Ghazouani
# Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari
#
# Configures DVC remote storage for full data & model versioning.
# Supported backends:
#   1. MinIO (Self-hosted S3-compatible, via Docker Compose)
#   2. AWS S3 (Cloud object storage)
#   3. Local Directory (Offline fallback for testing)
#
# Usage:
#   # For local MinIO (default):
#   bash dvc_remote_setup.sh minio
#
#   # For AWS S3:
#   export AWS_ACCESS_KEY_ID="AKIA..."
#   export AWS_SECRET_ACCESS_KEY="..."
#   bash dvc_remote_setup.sh s3 s3://my-archigen-bucket
#
#   # For Local Directory fallback:
#   bash dvc_remote_setup.sh local /path/to/dvc_storage
#
set -euo pipefail

BACKEND="${1:-minio}"
ARG2="${2:-}"

echo "======================================================"
echo " 🏛️ ArchiGen 3D - Configuration DVC Remote Storage"
echo " Backend sélectionné : ${BACKEND}"
echo "======================================================"

# Ensure DVC is initialized
if [[ ! -d ".dvc" ]]; then
  echo "==> Initializing DVC repository..."
  dvc init
fi

case "${BACKEND}" in
  minio)
    REMOTE_NAME="minio-storage"
    BUCKET_URL="${ARG2:-s3://archigen-dvc}"
    ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:9000}"
    ACCESS_KEY="${AWS_ACCESS_KEY_ID:-minioadmin}"
    SECRET_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"

    echo "==> Configuring MinIO Remote:"
    echo "    Remote Name : ${REMOTE_NAME}"
    echo "    Bucket URL  : ${BUCKET_URL}"
    echo "    Endpoint    : ${ENDPOINT_URL}"

    dvc remote add -d -f "${REMOTE_NAME}" "${BUCKET_URL}"
    dvc remote modify "${REMOTE_NAME}" endpointurl "${ENDPOINT_URL}"
    dvc remote modify --local "${REMOTE_NAME}" access_key_id "${ACCESS_KEY}"
    dvc remote modify --local "${REMOTE_NAME}" secret_access_key "${SECRET_KEY}"
    ;;

  s3)
    REMOTE_NAME="s3-cloud"
    BUCKET_URL="${ARG2:-s3://archigen-3d-mlops}"

    echo "==> Configuring AWS S3 Remote:"
    echo "    Remote Name : ${REMOTE_NAME}"
    echo "    Bucket URL  : ${BUCKET_URL}"

    dvc remote add -d -f "${REMOTE_NAME}" "${BUCKET_URL}"
    if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
      dvc remote modify --local "${REMOTE_NAME}" access_key_id "${AWS_ACCESS_KEY_ID}"
      dvc remote modify --local "${REMOTE_NAME}" secret_access_key "${AWS_SECRET_ACCESS_KEY}"
    fi
    ;;

  local)
    REMOTE_NAME="local-storage"
    STORAGE_PATH="${ARG2:-../dvc_storage}"

    echo "==> Configuring Local Directory Remote:"
    echo "    Remote Name : ${REMOTE_NAME}"
    echo "    Local Path  : ${STORAGE_PATH}"

    mkdir -p "${STORAGE_PATH}"
    dvc remote add -d -f "${REMOTE_NAME}" "${STORAGE_PATH}"
    ;;

  *)
    echo "Unknown backend: ${BACKEND}. Use: minio | s3 | local"
    exit 1
    ;;
esac

echo ""
echo "==> Configured DVC Remotes:"
dvc remote list

echo ""
echo "==> Status of tracked data/models:"
dvc status || true

echo ""
echo "✅ DVC Remote setup completed successfully!"
echo "   Push to remote : dvc push"
echo "   Pull from remote: dvc pull"
