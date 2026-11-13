# 🏛️ ArchiGen 3D — From 2D floor plan to 3D building, and from layout to facade

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.x-005ced.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React%2018-Vite-61dafb.svg)](https://vitejs.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg)](https://threejs.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945dd6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194e2.svg)](https://mlflow.org/)

> **ArchiGen** turns a **2D house floor plan into a 3D model** (multi-task UNet++ semantic segmentation, CubiCasa5K)
> and a **colour-coded facade layout into a photorealistic elevation** (Pix2Pix conditional GAN, CMP Facades),
> inside a cinematic web studio (React + Three.js) backed by a FastAPI service.

---


## 🧭 Overview

```mermaid
flowchart LR
    subgraph Plan["Plan → 3D"]
        A["2D raster floor plan"] --> B["Multi-task UNet++<br/>(ResNet-34)"]
        B --> C1["Structure: exterior/interior walls, doors, windows"]
        B --> C2["Rooms: 12 classes"]
        C1 & C2 --> D["Vectorisation (Shapely)<br/>+ 3D extrusion (Three.js)"]
    end
    subgraph Facade["Facade"]
        E["CMP semantic layout<br/>(256×256)"] --> F["Pix2Pix: U-Net + PatchGAN"]
        F --> G["Photorealistic facade"]
    end
    D & G --> H["FastAPI service (ONNX Runtime / PyTorch)"]
    H --> I["React + Three.js web studio"]
```

---

## 📊 Results

> Metrics are **never typed in by hand**: the training and evaluation scripts write them to `models/metrics/*.json`,
> and they are served by the API (`GET /metrics/summary`) and shown in the UI.
> **The models in this version are currently being trained**; this section will be filled with the measured values.

### Evaluation protocol (facades)
- Official splits **400 / 100 / 106**, **256×256** resolution, identical data for the baseline and Pix2Pix.
- Checkpoint selection on **validation only** (L1); the test split is evaluated once, after the protocol is frozen.
- Test metrics **averaged per image**: L1, PSNR, SSIM, LPIPS (AlexNet).
- Target: Pix2Pix validation L1 **at least 5 % better** than the U-Net baseline — a miss is reported as is.

### Evaluation protocol (floor plans)
- **Official** CubiCasa5K splits **4,200 / 400 / 400**.
- Per-class IoU (structure and rooms), mIoU, pixel accuracy; room pixels of unknown type (label 255) are excluded.

### Data audit (measured 13 Nov 2026 — `python -m src.data.audit_data`)
| Dataset | Findings |
| :--- | :--- |
| **CMP Facades** | 606 pairs (66.01 / 16.50 / 17.49 %), 0 corrupted files, 0 duplicates (files, photos, label maps) |
| **CubiCasa5K** | 5,000 plans, 0 missing files, 0 rasterisation failures; **18 groups of identical images, 5 of them across train and val** (minor leakage, documented) |
| **CubiCasa rooms** | ≈ 8.5 % of room pixels are `Undefined` in the original annotations → ignored by the loss and the metrics |

---

## 🗂️ Data

Datasets are **not** stored in Git (see `.gitignore`); they are tracked with DVC or downloaded.

| Dataset | Expected location | Source / licence |
| :--- | :--- | :--- |
| CMP Facades (pix2pix archive) | `data/raw/facades/{train,val,test}/*.jpg` (`dvc pull data/raw/facades.dvc`) | [CMP Facade DB](https://cmp.felk.cvut.cz/~tylecr1/facade/) — CC BY-SA |
| CubiCasa5K | `data/raw/cubicasa5k/` (folder containing `train.txt`, `val.txt`, `test.txt`) | [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) — CC BY-NC 4.0 |

Preparation:
```bash
python -m src.data.rasterize_cubicasa   # SVG -> 512×512 NPZ, official splits -> data/processed/cubicasa_cache/{train,val,test}
python -m src.data.audit_data           # integrity, SHA-256 duplicates, class frequencies -> data/processed/data_audit_summary.json
```

---

## 🏋️ Training & evaluation

```bash
# Facades (core)
python -m src.training.train_facade --mode l1_baseline     # U-Net baseline, L1 only
python -m src.training.train_facade --mode pix2pix         # U-Net + PatchGAN (LSGAN + L1 + feature matching)

# Floor plans (extension) — GPU recommended
python -m src.training.train_floorplan                     # 512×512
python -m src.training.train_floorplan --image-size 256 --amp --time-budget-min 50   # reduced, documented budget

# Final test evaluation (refuses any checkpoint not trained on real data)
python -m src.evaluation.evaluate facade    --checkpoint models/checkpoints/facade_pix2pix_best.pt
python -m src.evaluation.evaluate facade    --checkpoint models/checkpoints/facade_unet_l1_best.pt
python -m src.evaluation.evaluate floorplan --checkpoint models/checkpoints/floorplan_unetpp_best.pt
```

- Hyperparameters: `params.yaml` (`training_facade`, `training_floorplan`).
- Every checkpoint records its data source (`data_source`), hyperparameters and seed;
  ONNX export and evaluation **refuse** untrained weights or weights trained on synthetic data.
- MLflow tracking (`mlflow.db`); per-epoch sample grids in `models/samples/`.

### Training on Kaggle (free GPU)
Measured on CPU (12 threads): facades ≈ 8 h (L1) and ≈ 12 h (Pix2Pix) for 50 epochs; UNet++ at 512² ≈ 35 h **per epoch**.
A ready-made notebook trains everything in ≈ 1 h on **GPU T4 ×2**:

1. `python kaggle/make_bundle.py --part facades` → `dist/archigen_facades.zip` (≈ 31 MB, code + facades); upload it as a Kaggle *Dataset*.
2. Import `kaggle/archigen_kaggle.ipynb` and add the dataset (for floor plans, also add the public *CubiCasa5k* dataset — it is rasterised automatically in the notebook).
3. *Accelerator*: **GPU T4 ×2**, *Internet*: **On**, then **Save Version → Save & Run All**.
4. Download `archigen_results.zip` (checkpoints, val/test metrics, sample grids, logs) from the *Output* tab.

---

## 🖥️ Running the app

**Backend (API)** — from the repository root:
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --port 8000
```
Interactive docs: http://localhost:8000/docs

**Frontend (web studio)** — Node.js ≥ 18:
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (the Vite proxy forwards /predict, /health, /metrics… to port 8000)
```

| Page | What it shows |
| :--- | :--- |
| **Home** | Cinematic landing page with video, presenting *Plan → 3D* and the *Facade generator* |
| **Floorplan workspace** | Plan upload, 2D / 3D / split views, detected rooms, furnishing |
| **Generative studio** | *Furniture* (image → 3D) and *Facades* tabs |

Containerised deployment: `docker compose up --build -d`.

---

## 🔄 DVC pipeline

`dvc repro` runs, on real data only:

| Stage | Command | Output |
| :--- | :--- | :--- |
| `rasterize_cubicasa` | `python -m src.data.rasterize_cubicasa` | `data/processed/cubicasa_cache` |
| `audit_data` | `python -m src.data.audit_data` | `data/processed/data_audit_summary.json` |
| `train_facade_baseline` | `train_facade --mode l1_baseline` | `models/checkpoints/facade_unet_l1_best.pt` |
| `train_facade` | `train_facade --mode pix2pix` | `models/checkpoints/facade_pix2pix_best.pt` |
| `train_floorplan` | `train_floorplan` | `models/checkpoints/floorplan_unetpp_best.pt` |
| `evaluate_test` | `src.evaluation.evaluate` (×3) | `models/metrics/*_best_test.json` |
| `export_onnx_models` | `src.mlops.export_onnx` | `models/onnx/*.onnx` |
| `verify_onnx_parity` | `src.mlops.onnx_verifier` | `models/onnx/parity_report.json` |
| `benchmark_inference_latency` | `src.mlops.benchmark_latency` | `models/onnx/latency_benchmark.json` |
| `mlflow_tracking_and_registry` | `src.mlops.model_registry` | `models/mlflow_registry_summary.json` |
| `drift_baseline_and_monitor` | `src.mlops.drift_detector --mode baseline` | `data/monitoring/reference_profile.json` |

DVC remote: MinIO (`docker compose up minio minio-init -d`), S3 or a local folder — see `dvc_remote_setup.sh` / `dvc_remote_setup.ps1`.

---

## 📈 MLOps

- **GitHub Actions CI** (`.github/workflows/mlops-ci.yml`): unit and API tests, ONNX export, parity, latency, Docker build, Locust load test.
  CI has neither the datasets nor the weights: it validates the **technical pipeline** on weights explicitly labelled *untrained* (`ARCHIGEN_CI_UNTRAINED=1`).
- **Drift monitoring** (`src/mlops/drift_detector.py`): PSI, Kolmogorov-Smirnov, Wasserstein and Jensen-Shannon on inputs and outputs;
  the reference profile is built from the real *train* splits. Endpoints under `/monitoring/drift/*`, dashboard at `/monitoring/drift/dashboard`.
- **Continuous training** (`src/mlops/continuous_retraining.py`), triggered by a drift threshold.
- **MLflow model registry** with Staging → Production promotion (`src/mlops/model_registry.py`).

---

## 📁 Project structure

```
archigen-3d/
├── src/
│   ├── config.py                 # paths, classes, colours
│   ├── data/                     # facade_dataset, rasterize_cubicasa, audit_data, svg_preprocessor
│   ├── models/                   # unet_plus_plus (multi-task UNet++), pix2pix_cgan
│   ├── training/                 # trainer_base, train_facade, train_floorplan, checkpoints
│   ├── evaluation/               # metrics (L1, PSNR, SSIM, LPIPS, IoU), evaluate (final test)
│   ├── geometry/                 # vectorisation and 3D extrusion
│   ├── mlops/                    # export_onnx, onnx_verifier, benchmark_latency, mlflow_tracker,
│   │                             # model_registry, drift_detector, continuous_retraining, dvc_remote
│   └── api/                      # FastAPI: main, service, monitoring_endpoints, static/
├── frontend/                     # React + Vite + Three.js
│   ├── src/                      # App and components (Viewer3D, Floorplan2DCanvas, GenerativeStudio…)
│   └── public/landing/           # home page (video in public/showcase/media/)
├── kaggle/                       # make_bundle.py + archigen_kaggle.ipynb
├── tests/                        # models, ONNX parity, API, drift, real-data pipeline
├── data/                         # raw/ (DVC), processed/ (generated), monitoring/
├── models/                       # checkpoints/, onnx/, metrics/ (generated)
├── dvc.yaml · params.yaml · requirements.txt · Dockerfile · docker-compose.yml
```

Tests: `pytest tests/ -v`

---

## ⚠️ Known limitations

- **Models are still being trained**: without checkpoints the API starts, but its predictions are not meaningful.
- The frontend calls `/design`, `/facade/generate` and `/furniture/*`, which are **not implemented** in this version of the backend:
  generating facades and furniture from the studio currently returns an error.
- The CMP label maps in the pix2pix archive are JPEG-compressed, so exact per-class pixel counts for facades are not possible.
- The 5 train/val duplicate groups in CubiCasa5K are kept to respect the official splits; they are documented by the audit.

---

## 📚 References

- P. Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (Pix2Pix), CVPR 2017.
- R. Tyleček, R. Šára, *Spatial Pattern Templates for Recognition of Objects with Regular Structure*, GCPR 2013 — CMP Facade Database.
- A. Kalervo et al., *CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis*, SCIA 2019.
- Z. Zhou et al., *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*, DLMIA 2018.
- R. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS), CVPR 2018.
