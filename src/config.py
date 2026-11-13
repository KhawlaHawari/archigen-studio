"""
ArchiGen 3D - Central Configuration & Constants
TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
Encadrant : Haythem Ghazouani
Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ONNX_DIR = MODELS_DIR / "onnx"

# Ensure runtime directories exist
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, ONNX_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# DATASET TAXONOMY & LABELS
# -------------------------------------------------------------------------

# Modèle 1 : CubiCasa5K Structures (Tête 1 - 5 classes)
STRUCTURE_CLASSES: List[str] = [
    "background",        # 0: Hors-bâti / Arrière-plan
    "exterior_wall",     # 1: Mur Porteur / Extérieur
    "interior_wall",     # 2: Cloison Intérieure
    "door",              # 3: Porte / Passage
    "window",            # 4: Fenêtre
]

STRUCTURE_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),  # Blanc
    1: (40, 40, 40),     # Noir/Anthracite (Mur porteur)
    2: (120, 120, 120),  # Gris (Cloison)
    3: (220, 50, 50),    # Rouge (Porte)
    4: (50, 120, 220),   # Bleu (Fenêtre)
}

# Modèle 1 : CubiCasa5K Pièces (Tête 2 - 12 classes)
ROOM_CLASSES: List[str] = [
    "background",        # 0: Fond
    "living_room",       # 1: Salon / Séjour
    "bedroom",           # 2: Chambre
    "kitchen",           # 3: Cuisine
    "bathroom",          # 4: Salle de bain
    "hallway",           # 5: Couloir / Dégagement
    "closet",            # 6: Placard / Dressing
    "balcony",           # 7: Balcon / Terrasse
    "dining",            # 8: Salle à manger
    "office",            # 9: Bureau
    "storage",           # 10: Cellier / Débarras
    "technical",         # 11: Local technique / Buanderie
]

ROOM_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),   # Background
    1: (255, 204, 153),   # Living room (Pêche clair)
    2: (178, 218, 250),   # Bedroom (Bleu pastel)
    3: (255, 178, 178),   # Kitchen (Rose corail)
    4: (178, 255, 218),   # Bathroom (Vert menthe)
    5: (220, 220, 220),   # Hallway (Gris clair)
    6: (204, 178, 255),   # Closet (Lavande)
    7: (255, 255, 178),   # Balcony (Jaune doux)
    8: (255, 153, 102),   # Dining (Orange)
    9: (153, 204, 255),   # Office (Bleu ciel)
    10: (200, 200, 170),  # Storage (Kaki doux)
    11: (180, 180, 180),  # Technical (Gris moyen)
}

# Modèle 2 : CMP Facades (12 classes architecturales sémantiques)
FACADE_CLASSES: List[str] = [
    "background",        # 0: Ciel / Arrière-plan
    "facade",            # 1: Mur / Enduit / Pierre
    "window",            # 2: Fenêtre / Baie vitrée
    "door",              # 3: Porte d'entrée
    "cornice",           # 4: Corniche / Bandeau
    "sill",              # 5: Seuil / Appui de fenêtre
    "balcony",           # 6: Balcon / Garde-corps
    "blind",             # 7: Volet / Persienne
    "pillar",            # 8: Pilier / Colonne
    "deco",              # 9: Moulure / Ornement
    "roof",              # 10: Toiture
    "shop",              # 11: Vitrine commerciale
]

FACADE_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),         # Fond (Noir)
    1: (0, 0, 255),       # Façade (Bleu)
    2: (0, 255, 255),     # Fenêtre (Cyan)
    3: (255, 255, 0),     # Porte (Jaune)
    4: (255, 0, 0),       # Corniche (Rouge)
    5: (128, 0, 0),       # Seuil (Bordeaux)
    6: (0, 128, 0),       # Balcon (Vert foncé)
    7: (0, 255, 0),       # Volet (Vert clair)
    8: (128, 128, 0),     # Pilier (Olive)
    9: (255, 0, 255),     # Ornement (Magenta)
    10: (128, 0, 128),    # Toit (Violet)
    11: (0, 128, 128),    # Vitrine (Teal)
}


# -------------------------------------------------------------------------
# MODEL HYPERPARAMETERS & SETTINGS
# -------------------------------------------------------------------------

@dataclass
class FloorplanModelConfig:
    name: str = "UNetPlusPlus-ResNet34-MultiTask"
    input_size: Tuple[int, int] = (512, 512)
    in_channels: int = 3
    num_structure_classes: int = len(STRUCTURE_CLASSES)  # 5
    num_room_classes: int = len(ROOM_CLASSES)            # 12
    encoder_name: str = "resnet34"
    # Loss weights according to Cahier des Charges:
    # Arrière-plan=0.5, Mur=4.0, Cloison=4.0, Porte=25.0, Fenetre=10.0
    structure_weights: List[float] = field(
        default_factory=lambda: [0.5, 4.0, 4.0, 25.0, 10.0]
    )
    checkpoint_path: Path = CHECKPOINTS_DIR / "floorplan_unetpp_best.pt"
    onnx_path: Path = ONNX_DIR / "floorplan_unetpp.onnx"


@dataclass
class FacadeModelConfig:
    name: str = "Pix2Pix-Perceptual-SpectralNorm"
    input_size: Tuple[int, int] = (256, 256)
    in_channels: int = 3
    out_channels: int = 3
    ngf: int = 64
    ndf: int = 64
    use_spectral_norm: bool = True
    use_nearest_upsample: bool = True  # Avoid checkerboard artifacts
    lambda_l1: float = 30.0
    lambda_fm: float = 10.0
    lambda_perceptual: float = 10.0
    checkpoint_path: Path = CHECKPOINTS_DIR / "facade_pix2pix_best.pt"
    onnx_path: Path = ONNX_DIR / "facade_pix2pix.onnx"


@dataclass
class Geometry3DConfig:
    scale_meters_per_pixel: float = 0.05  # 512px = 25.6 meters
    exterior_wall_height: float = 3.0    # meters
    interior_wall_height: float = 2.8    # meters
    wall_extrusion_bevel: bool = False
    closing_kernel_size: int = 5         # morphological bridge for doors/windows
    simplify_tolerance: float = 1.2      # Douglas-Peucker epsilon in pixels


@dataclass
class MLOpsConfig:
    experiment_name: str = "ArchiGen3D-MultiTask-Facading"
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(BASE_DIR / "mlflow.db").replace("\\", "/"))
    parity_atol_threshold: float = 1e-5  # Maximum absolute difference |PyTorch - ONNX|
    latency_sla_gpu_ms: float = 400.0   # p95 GPU SLA (< 400ms)
    latency_sla_cpu_ms: float = 2000.0  # p95 CPU SLA (< 2s)
    default_onnx_opset: int = 17


# Instantiate singletons
FLOORPLAN_CFG = FloorplanModelConfig()
FACADE_CFG = FacadeModelConfig()
GEOMETRY_CFG = Geometry3DConfig()
MLOPS_CFG = MLOpsConfig()
