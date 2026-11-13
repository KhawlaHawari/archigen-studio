"""
ArchiGen 3D - MLOps Level 3 Drift Monitoring & Retraining Engine
TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
Encadrant : Haythem Ghazouani
Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari

Implements:
  1. Data Drift Detection (Input Image Statistics: brightness, contrast, edge density, aspect ratio).
  2. Prediction / Concept Drift Detection (Predicted structure/room distributions, segmentation entropy, area).
  3. Facade Quality & Style Drift (CMP semantic class distributions, generated sharpness, color profiles).
  4. Statistical Drift Metrics:
     - Population Stability Index (PSI): PSI < 0.1 (OK), 0.1-0.2 (Warning), >= 0.2 (Critical Drift)
     - Kolmogorov-Smirnov 2-Sample Test (KS test) with p-value significance
     - Wasserstein Distance (Earth Mover's Distance)
     - Jensen-Shannon Divergence / Chi-Square on class distributions
  5. Continuous Training (CT) Trigger:
     - Automated Retraining Recommendation when Drift Score >= threshold or critical PSI >= 0.20
  6. MLflow Integration: Logs drift metrics, alerts, and reports to MLflow Tracking Server.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image
from scipy import stats
from scipy.spatial.distance import jensenshannon
import mlflow

from src.config import (
    BASE_DIR,
    DATA_DIR,
    STRUCTURE_CLASSES,
    ROOM_CLASSES,
    FACADE_CLASSES,
    MLOPS_CFG
)

# Monitoring directories
MONITORING_DIR = DATA_DIR / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_PROFILE_PATH = MONITORING_DIR / "reference_profile.json"
INFERENCE_BUFFER_PATH = MONITORING_DIR / "inference_buffer.jsonl"
DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"

# Default SLA / Drift Thresholds (aligns with params.yaml)
DEFAULT_DRIFT_THRESHOLDS = {
    "psi_warning": 0.10,
    "psi_critical": 0.20,
    "ks_alpha": 0.05,
    "wasserstein_threshold": 0.15,
    "composite_drift_threshold": 0.35,
    "min_samples_required": 5
}


# =============================================================================
# STATISTICAL DRIFT UTILITIES
# =============================================================================

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and current distributions.
    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Determine quantile bin edges from baseline
    percentiles = np.linspace(0, 100, num_bins + 1)
    try:
        bin_edges = np.percentile(baseline, percentiles)
    except Exception:
        bin_edges = np.linspace(min(baseline.min(), current.min()), max(baseline.max(), current.max()), num_bins + 1)

    # Ensure unique and strictly increasing edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate counts and proportions
    b_counts, _ = np.histogram(baseline, bins=bin_edges)
    c_counts, _ = np.histogram(current, bins=bin_edges)

    b_prop = (b_counts + epsilon) / (len(baseline) + epsilon * len(b_counts))
    c_prop = (c_counts + epsilon) / (len(current) + epsilon * len(c_counts))

    psi_value = np.sum((c_prop - b_prop) * np.log(c_prop / b_prop))
    return float(np.clip(psi_value, 0.0, 10.0))


def calculate_ks_test(baseline: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Computes Two-sample Kolmogorov-Smirnov test."""
    b = np.asarray(baseline, dtype=float).ravel()
    c = np.asarray(current, dtype=float).ravel()

    if len(b) < 2 or len(c) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    res = stats.ks_2samp(b, c)
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "drift_detected": bool(p_val < DEFAULT_DRIFT_THRESHOLDS["ks_alpha"])
    }


def calculate_wasserstein(baseline: np.ndarray, current: np.ndarray) -> float:
    """Computes Earth Mover's / Wasserstein Distance normalized by standard deviation."""
    b = np.asarray(baseline, dtype=float).ravel()
    c = np.asarray(current, dtype=float).ravel()

    if len(b) == 0 or len(c) == 0:
        return 0.0

    scale = float(np.std(b)) if np.std(b) > 1e-6 else 1.0
    dist = stats.wasserstein_distance(b, c) / scale
    return float(round(dist, 4))


def calculate_categorical_divergence(baseline_dist: np.ndarray, current_dist: np.ndarray) -> float:
    """Computes Jensen-Shannon Divergence between two normalized categorical probability distributions."""
    b = np.asarray(baseline_dist, dtype=float) + 1e-6
    c = np.asarray(current_dist, dtype=float) + 1e-6
    b /= b.sum()
    c /= c.sum()
    return float(round(jensenshannon(b, c), 4))


# =============================================================================
# FEATURE EXTRACTION ENGINE
# =============================================================================

def extract_floorplan_features(
    image_np: np.ndarray,
    struct_mask: Optional[np.ndarray] = None,
    room_mask: Optional[np.ndarray] = None,
    total_area_m2: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extracts statistical input and prediction features from a floorplan image and masks.
    """
    # 1. Grayscale luminance & dynamic range
    if image_np.ndim == 3:
        gray = np.mean(image_np, axis=2).astype(np.float32)
    else:
        gray = image_np.astype(np.float32)

    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    background_ratio = float(np.mean(gray > 240))  # Ratio of white blueprint paper

    # 2. Edge / high-frequency density (approximation of blueprint complexity)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_density = float(np.mean(grad_x > 25) + np.mean(grad_y > 25)) / 2.0

    # 3. Geometric aspect ratio
    h, w = gray.shape[:2]
    aspect_ratio = round(float(w) / float(h), 3)

    # 4. Predictions distribution (if masks provided)
    struct_ratios = [0.0] * len(STRUCTURE_CLASSES)
    if struct_mask is not None:
        total_px = struct_mask.size
        for c in range(len(STRUCTURE_CLASSES)):
            struct_ratios[c] = round(float(np.sum(struct_mask == c)) / total_px, 4)

    room_ratios = [0.0] * len(ROOM_CLASSES)
    num_rooms = 0
    if room_mask is not None:
        total_px = room_mask.size
        for c in range(len(ROOM_CLASSES)):
            room_ratios[c] = round(float(np.sum(room_mask == c)) / total_px, 4)
        # Approximate number of rooms by non-zero classes present
        num_rooms = int(np.sum(np.array(room_ratios[1:]) > 0.005))

    return {
        "mean_brightness": round(mean_brightness, 2),
        "std_brightness": round(std_brightness, 2),
        "background_ratio": round(background_ratio, 4),
        "edge_density": round(edge_density, 4),
        "aspect_ratio": aspect_ratio,
        "total_area_m2": round(total_area_m2 or (num_rooms * 14.5 + 25.0), 2),
        "num_rooms": num_rooms,
        "struct_wall_ratio": round(struct_ratios[1] + struct_ratios[2], 4) if len(struct_ratios) > 2 else 0.0,
        "struct_door_ratio": round(struct_ratios[3], 4) if len(struct_ratios) > 3 else 0.0,
        "struct_window_ratio": round(struct_ratios[4], 4) if len(struct_ratios) > 4 else 0.0,
        "room_living_ratio": round(room_ratios[1], 4) if len(room_ratios) > 1 else 0.0,
        "room_bedroom_ratio": round(room_ratios[2], 4) if len(room_ratios) > 2 else 0.0,
        "timestamp": time.time()
    }


def extract_facade_features(
    mask_np: np.ndarray,
    generated_np: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extracts statistical input and output features from a facade mask and generated image.
    """
    h, w = mask_np.shape[:2]
    # Class distribution approximation from mask colors
    is_blue_wall = np.mean((mask_np[:, :, 2] > 180) & (mask_np[:, :, 0] < 50))
    is_cyan_window = np.mean((mask_np[:, :, 1] > 180) & (mask_np[:, :, 2] > 180))
    is_yellow_door = np.mean((mask_np[:, :, 0] > 180) & (mask_np[:, :, 1] > 180))

    gen_brightness = 128.0
    gen_contrast = 50.0
    gen_sharpness = 100.0

    if generated_np is not None:
        gen_gray = np.mean(generated_np, axis=2).astype(np.float32)
        gen_brightness = float(np.mean(gen_gray))
        gen_contrast = float(np.std(gen_gray))
        # Perceptual gradient variance as sharpness metric
        var_x = float(np.var(np.diff(gen_gray, axis=1))) if gen_gray.shape[1] > 1 else 0.0
        var_y = float(np.var(np.diff(gen_gray, axis=0))) if gen_gray.shape[0] > 1 else 0.0
        gen_sharpness = var_x + var_y

    return {
        "facade_wall_ratio": round(float(is_blue_wall), 4),
        "facade_window_ratio": round(float(is_cyan_window), 4),
        "facade_door_ratio": round(float(is_yellow_door), 4),
        "gen_brightness": round(gen_brightness, 2),
        "gen_contrast": round(gen_contrast, 2),
        "gen_sharpness": round(gen_sharpness, 2),
        "timestamp": time.time()
    }


# =============================================================================
# REFERENCE BASELINE PROFILE GENERATION
# =============================================================================

def build_reference_baseline_profile(sample_count: int = 40) -> Dict[str, Any]:
    """
    Constructs the canonical training/validation reference distribution profile
    from cached CubiCasa5K floorplans and CMP Facades.
    """
    from src.data.svg_preprocessor import (
        generate_synthetic_cubicasa_sample,
        generate_synthetic_facade_sample
    )

    floorplan_samples = []
    facade_samples = []

    # 1. Gather floorplan features from cache or synthetic reference generator
    cubicasa_cache_dir = DATA_DIR / "processed" / "cubicasa_cache"
    # Reference = TRAIN split only (rasterized from the official CubiCasa5K split).
    npz_files = sorted((cubicasa_cache_dir / "train").glob("*.npz")) if cubicasa_cache_dir.exists() else []

    if npz_files:
        for f in npz_files[:sample_count]:
            try:
                data = np.load(f)
                img = data.get("image", None)
                struct = data.get("structure_mask", None)
                room = data.get("room_mask", None)
                if img is not None:
                    feat = extract_floorplan_features(img, struct, room)
                    floorplan_samples.append(feat)
            except Exception:
                continue

    floorplan_source = "real:cubicasa5k/train" if floorplan_samples else "synthetic (no real cache found)"
    # Synthetic samples are used ONLY when no real data is available (e.g. CI without DVC pull).
    while not floorplan_samples or (floorplan_source.startswith("synthetic") and len(floorplan_samples) < sample_count):
        img, struct, room = generate_synthetic_cubicasa_sample(512, 512)
        feat = extract_floorplan_features(img, struct, room)
        floorplan_samples.append(feat)

    # 2. Gather facade features from cache or synthetic reference generator
    # Reference = real CMP Facades TRAIN pairs (semantic map + photo).
    try:
        from src.data.facade_dataset import list_split, load_pair
        for f in list_split("train")[:sample_count]:
            mask, photo = load_pair(f)
            facade_samples.append(extract_facade_features(mask, photo))
    except FileNotFoundError:
        pass

    facade_source = "real:cmp_facades/train" if facade_samples else "synthetic (no real data found)"
    while not facade_samples or (facade_source.startswith("synthetic") and len(facade_samples) < sample_count):
        mask, photo = generate_synthetic_facade_sample(256, 256)
        feat = extract_facade_features(mask, photo)
        facade_samples.append(feat)

    # 3. Assemble distribution summaries
    profile = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_floorplan_samples": len(floorplan_samples),
            "num_facade_samples": len(facade_samples),
            "floorplan_data_source": floorplan_source,
            "facade_data_source": facade_source,
            "version": "1.0.0",
            "mlops_tier": "Niveau 3 - CT & Drift Monitoring"
        },
        "floorplan_reference": {
            k: [s[k] for s in floorplan_samples if k in s]
            for k in floorplan_samples[0].keys() if k != "timestamp"
        },
        "facade_reference": {
            k: [s[k] for s in facade_samples if k in s]
            for k in facade_samples[0].keys() if k != "timestamp"
        }
    }

    # Save to disk
    with open(REFERENCE_PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    return profile


def load_reference_profile() -> Dict[str, Any]:
    """Loads baseline reference profile or generates it if not found."""
    if not REFERENCE_PROFILE_PATH.exists():
        return build_reference_baseline_profile()
    try:
        with open(REFERENCE_PROFILE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return build_reference_baseline_profile()


# =============================================================================
# PRODUCTION TELEMETRY LOGGING
# =============================================================================

def log_inference_telemetry(domain: str, features: Dict[str, Any]):
    """
    Appends single inference telemetry record to sliding production window.
    """
    entry = {
        "domain": domain,
        "timestamp": time.time(),
        "features": features
    }
    try:
        with open(INFERENCE_BUFFER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[DriftMonitor] Telemetry log failed: {e}")


def get_recent_telemetry(domain: str = "floorplan", max_entries: int = 100) -> List[Dict[str, Any]]:
    """Reads latest production inference samples."""
    if not INFERENCE_BUFFER_PATH.exists():
        return []
    records = []
    try:
        with open(INFERENCE_BUFFER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("domain") == domain:
                    records.append(obj["features"])
    except Exception:
        return []
    return records[-max_entries:]


# =============================================================================
# DRIFT EVALUATION & CONTINUOUS TRAINING (CT) TRIGGER
# =============================================================================

class DriftDetector:
    """
    Statistical Drift Detector comparing production inference distribution
    against the verified baseline reference profile.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or DEFAULT_DRIFT_THRESHOLDS
        self.profile = load_reference_profile()

    def evaluate_features_drift(
        self,
        reference_data: Dict[str, List[float]],
        current_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Runs comprehensive statistical tests (PSI, KS test, Wasserstein)
        on all continuous and categorical features.
        """
        feature_reports = {}
        psi_scores = []
        ks_drift_flags = []

        for feat_name, ref_values in reference_data.items():
            curr_values = [s[feat_name] for s in current_samples if feat_name in s]
            if len(curr_values) < self.thresholds["min_samples_required"]:
                continue

            ref_arr = np.array(ref_values, dtype=float)
            curr_arr = np.array(curr_values, dtype=float)

            psi = calculate_psi(ref_arr, curr_arr)
            ks_res = calculate_ks_test(ref_arr, curr_arr)
            w_dist = calculate_wasserstein(ref_arr, curr_arr)

            # PSI status: 0: No Drift, 1: Moderate Drift, 2: Critical Drift
            if psi >= self.thresholds["psi_critical"]:
                status = "CRITICAL_DRIFT"
            elif psi >= self.thresholds["psi_warning"]:
                status = "MODERATE_DRIFT"
            else:
                status = "STABLE"

            psi_scores.append(psi)
            ks_drift_flags.append(ks_res["drift_detected"])

            feature_reports[feat_name] = {
                "psi": round(psi, 4),
                "psi_status": status,
                "ks_statistic": ks_res["statistic"],
                "ks_p_value": ks_res["p_value"],
                "ks_drift_detected": ks_res["drift_detected"],
                "wasserstein_distance": w_dist,
                "baseline_mean": round(float(np.mean(ref_arr)), 3),
                "current_mean": round(float(np.mean(curr_arr)), 3)
            }

        if not psi_scores:
            return {
                "status": "INSUFFICIENT_DATA",
                "drift_score": 0.0,
                "drift_detected": False,
                "retraining_recommended": False,
                "feature_reports": {}
            }

        mean_psi = float(np.mean(psi_scores))
        max_psi = float(np.max(psi_scores))
        ks_drift_ratio = float(np.mean(ks_drift_flags))

        # Composite Drift Score [0.0 - 1.0]
        drift_score = round(min(1.0, 0.5 * min(1.0, max_psi / 0.3) + 0.5 * ks_drift_ratio), 3)
        drift_detected = (max_psi >= self.thresholds["psi_critical"]) or (drift_score >= self.thresholds["composite_drift_threshold"])
        retraining_recommended = bool(drift_detected)

        # Overall Status
        if drift_detected:
            overall_status = "DRIFT_DETECTED"
        elif max_psi >= self.thresholds["psi_warning"]:
            overall_status = "WARNING_DRIFT"
        else:
            overall_status = "HEALTHY"

        return {
            "status": overall_status,
            "drift_score": drift_score,
            "drift_detected": drift_detected,
            "retraining_recommended": retraining_recommended,
            "mean_psi": round(mean_psi, 4),
            "max_psi": round(max_psi, 4),
            "ks_features_drifted_pct": round(ks_drift_ratio * 100, 1),
            "num_features_evaluated": len(feature_reports),
            "feature_reports": feature_reports
        }

    def detect_drift_on_buffer(self, domain: str = "floorplan") -> Dict[str, Any]:
        """Evaluates drift over recent production inference samples."""
        current_samples = get_recent_telemetry(domain=domain, max_entries=100)
        ref_key = f"{domain}_reference"
        ref_data = self.profile.get(ref_key, {})

        if len(current_samples) < self.thresholds["min_samples_required"]:
            return {
                "domain": domain,
                "status": "INSUFFICIENT_DATA",
                "message": f"Requires at least {self.thresholds['min_samples_required']} inference samples (current: {len(current_samples)}).",
                "current_samples_count": len(current_samples),
                "drift_detected": False,
                "retraining_recommended": False
            }

        eval_result = self.evaluate_features_drift(ref_data, current_samples)
        eval_result["domain"] = domain
        eval_result["current_samples_count"] = len(current_samples)
        eval_result["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save report
        with open(DRIFT_REPORT_PATH, "w") as f:
            json.dump(eval_result, f, indent=2)

        return eval_result

    def log_drift_to_mlflow(self, report: Dict[str, Any]):
        """Logs drift metrics and alerts to MLflow Tracking Server."""
        try:
            mlflow.set_tracking_uri(MLOPS_CFG.tracking_uri)
            mlflow.set_experiment("ArchiGen3D-Drift-Monitoring")

            with mlflow.start_run(run_name=f"drift-audit-{report.get('domain', 'global')}-{int(time.time())}"):
                mlflow.log_param("mlops_maturity_level", "Level 3 - Continuous Monitoring & Retraining")
                mlflow.log_param("domain", report.get("domain", "all"))
                mlflow.log_metric("drift_score", report.get("drift_score", 0.0))
                mlflow.log_metric("max_psi", report.get("max_psi", 0.0))
                mlflow.log_metric("drift_detected", 1.0 if report.get("drift_detected") else 0.0)
                mlflow.log_metric("retraining_recommended", 1.0 if report.get("retraining_recommended") else 0.0)

                # Log top features PSI
                for feat, data in report.get("feature_reports", {}).items():
                    mlflow.log_metric(f"psi_{feat}", data.get("psi", 0.0))

                # Log full report as artifact
                if DRIFT_REPORT_PATH.exists():
                    mlflow.log_artifact(str(DRIFT_REPORT_PATH))
        except Exception as e:
            print(f"[DriftMonitor] MLflow logging warning: {e}")


# =============================================================================
# SYNTHETIC DRIFT GENERATOR (FOR TESTS & LIVE DEMOS)
# =============================================================================

def simulate_production_drift(domain: str = "floorplan", drift_type: str = "noise_and_shift", n_samples: int = 15) -> List[Dict[str, Any]]:
    """
    Generates synthetic drifted production samples (e.g. inverted brightness, heavy noise,
    abnormal blueprints, or altered facade styles) to test automated drift detection.
    """
    from src.data.svg_preprocessor import (
        generate_synthetic_cubicasa_sample,
        generate_synthetic_facade_sample
    )

    drifted_records = []

    if domain == "floorplan":
        for _ in range(n_samples):
            img, struct, room = generate_synthetic_cubicasa_sample(512, 512)

            if drift_type == "inverted_dark_mode":
                # Dark blueprints (white lines on dark blue background)
                img = 255 - img
                feat = extract_floorplan_features(img, struct, room, total_area_m2=180.0)
            elif drift_type == "heavy_noise":
                # Low quality fax/scan with intense speckle noise
                noise = np.random.normal(0, 50, img.shape).astype(np.int16)
                noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                feat = extract_floorplan_features(noisy_img, struct, room, total_area_m2=45.0)
            else:  # "noise_and_shift"
                img = (img * 0.5).astype(np.uint8)  # severe brightness shift
                feat = extract_floorplan_features(img, struct, room, total_area_m2=220.0)

            drifted_records.append(feat)
            log_inference_telemetry(domain="floorplan", features=feat)

    else:  # facade
        for _ in range(n_samples):
            mask, photo = generate_synthetic_facade_sample(256, 256)
            if drift_type == "inverted_dark_mode":
                photo = 255 - photo
            else:
                photo = (photo * 0.4).astype(np.uint8)

            feat = extract_facade_features(mask, photo)
            drifted_records.append(feat)
            log_inference_telemetry(domain="facade", features=feat)

    return drifted_records


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ArchiGen 3D MLOps Level 3 Drift Monitoring Engine")
    parser.add_argument("--mode", choices=["baseline", "evaluate", "simulate"], default="evaluate",
                        help="Action to execute: baseline, evaluate, or simulate")
    parser.add_argument("--domain", choices=["floorplan", "facade"], default="floorplan",
                        help="Domain to evaluate")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples for simulate/baseline")
    args = parser.parse_args()

    if args.mode == "baseline":
        print("==> Building reference baseline distribution profile...")
        prof = build_reference_baseline_profile(sample_count=args.samples)
        print(f"✅ Baseline saved to {REFERENCE_PROFILE_PATH}")
        print(f"   Floorplan reference samples: {len(prof['floorplan_reference']['mean_brightness'])}")
        print(f"   Facade reference samples:    {len(prof['facade_reference']['gen_brightness'])}")

    elif args.mode == "simulate":
        detector = DriftDetector()
        print(f"==> Simulating {args.samples} drifted {args.domain} inference requests...")
        sim_records = simulate_production_drift(domain=args.domain, n_samples=args.samples)
        print(f"✅ Injected {len(sim_records)} samples into inference buffer.")
        print("==> Running drift detection over updated buffer...")
        report = detector.detect_drift_on_buffer(domain=args.domain)
        print(json.dumps(report, indent=2))
        detector.log_drift_to_mlflow(report)

    else:  # evaluate
        detector = DriftDetector()
        print(f"==> Evaluating current production drift on domain '{args.domain}'...")
        report = detector.detect_drift_on_buffer(domain=args.domain)
        print(json.dumps(report, indent=2))
