# ArchiGen 3D - DVC Remote Storage Setup (PowerShell for Windows)
# TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
# Encadrant : Haythem Ghazouani
# Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari
#
# Configures DVC remote storage for versioning model weights and datasets.
#
# Usage:
#   .\dvc_remote_setup.ps1 -Backend minio
#   .\dvc_remote_setup.ps1 -Backend s3 -BucketUrl "s3://my-archigen-bucket"
#   .\dvc_remote_setup.ps1 -Backend local -StoragePath "..\dvc_storage"

param (
    [ValidateSet("minio", "s3", "local")]
    [string]$Backend = "minio",
    [string]$BucketUrl = "",
    [string]$EndpointUrl = "http://localhost:9000",
    [string]$AccessKey = "minioadmin",
    [string]$SecretKey = "minioadmin",
    [string]$StoragePath = "..\dvc_storage"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " 🏛️ ArchiGen 3D - Configuration DVC Remote Storage" -ForegroundColor Cyan
Write-Host " Backend sélectionné : $Backend" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Check DVC initialization
if (-not (Test-Path ".dvc")) {
    Write-Host "==> Initializing DVC repository..." -ForegroundColor Yellow
    python -m dvc init
}

switch ($Backend) {
    "minio" {
        $remoteName = "minio-storage"
        if ([string]::IsNullOrEmpty($BucketUrl)) { $BucketUrl = "s3://archigen-dvc" }

        Write-Host "==> Configuring MinIO Remote:" -ForegroundColor Green
        Write-Host "    Remote Name : $remoteName"
        Write-Host "    Bucket URL  : $BucketUrl"
        Write-Host "    Endpoint    : $EndpointUrl"

        python -m dvc remote add -d -f $remoteName $BucketUrl
        python -m dvc remote modify $remoteName endpointurl $EndpointUrl
        python -m dvc remote modify --local $remoteName access_key_id $AccessKey
        python -m dvc remote modify --local $remoteName secret_access_key $SecretKey
    }
    "s3" {
        $remoteName = "s3-cloud"
        if ([string]::IsNullOrEmpty($BucketUrl)) { $BucketUrl = "s3://archigen-3d-mlops" }

        Write-Host "==> Configuring AWS S3 Remote:" -ForegroundColor Green
        Write-Host "    Remote Name : $remoteName"
        Write-Host "    Bucket URL  : $BucketUrl"

        python -m dvc remote add -d -f $remoteName $BucketUrl
        if ($env:AWS_ACCESS_KEY_ID) {
            python -m dvc remote modify --local $remoteName access_key_id $env:AWS_ACCESS_KEY_ID
            python -m dvc remote modify --local $remoteName secret_access_key $env:AWS_SECRET_ACCESS_KEY
        }
    }
    "local" {
        $remoteName = "local-storage"
        Write-Host "==> Configuring Local Directory Remote:" -ForegroundColor Green
        Write-Host "    Remote Name : $remoteName"
        Write-Host "    Path        : $StoragePath"

        if (-not (Test-Path $StoragePath)) {
            New-Item -ItemType Directory -Path $StoragePath -Force | Out-Null
        }
        python -m dvc remote add -d -f $remoteName $StoragePath
    }
}

Write-Host "`n==> Configured DVC Remotes:" -ForegroundColor Yellow
python -m dvc remote list

Write-Host "`n==> Checking DVC status:" -ForegroundColor Yellow
python -m dvc status

Write-Host "`n✅ DVC Remote configuration completed successfully!" -ForegroundColor Green
Write-Host "   Pour pousser les données/modèles : python -m dvc push"
Write-Host "   Pour récupérer depuis le remote : python -m dvc pull"
