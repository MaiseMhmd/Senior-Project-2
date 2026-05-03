"""
================================================================================
  MULTI-OBJECT TRACKING — OccluTrack vs OAMN COMPARISON PIPELINE
================================================================================
  PIPELINE ARCHITECTURE (identical for both methods):

    ┌─────────────────────┐
    │  Trained YOLOv11    │  ← OVIS-trained weights (best.pt)
    │  (Detection Stage)  │    detects soldiers in every frame
    └────────┬────────────┘
             │ bounding boxes
    ┌────────▼────────────┐
    │  Kalman Filter      │  ← state estimation under occlusion
    │  (State Estimation) │    full occlusion  : track → OCCLUDED state
    │                     │    long-term (>30f): IoU gate bypassed,
    │                     │    appearance-only Re-ID used for recovery
    │                     │    max track lifetime: 150 frames (10 s @ 15 fps)
    └────────┬────────────┘
             │ predicted positions + occlusion state
    ┌────────▼────────────┐
    │  Re-ID Extractor    │  ← ONE of 2 methods:
    │  (Identity Recovery │    • OccluTrack  → trained on OVIS crops here
    │   after Occlusion)  │    • OAMN        → loaded from pretrained best.pth
    └────────┬────────────┘
             │ 512-d appearance embedding
    ┌────────▼────────────┐
    │  Association        │  ← combined IoU + appearance cost matrix
    │                     │    greedy matching → track ID preserved
    └─────────────────────┘

  2 COMBINATIONS EVALUATED (same YOLO + Kalman, different Re-ID):
    Run 1: Trained YOLOv11 + Kalman Filter + OAMN        (pretrained best.pth)
    Run 2: Trained YOLOv11 + Kalman Filter + OccluTrack  (trained on OVIS)

  METRICS PER COMBINATION:
    Tracking : MOTA, IDF1, ADE (normalised), RawADE (pixels), ID-Switches, MT, ML
    Detection: TP, FP, FN, TN, Precision, Recall, F1-score, Specificity

  ALL OUTPUTS → D:\\hanan\\senior-proj\\result
    ├── reid_comparison.csv           — aggregated metrics (both methods)
    ├── reid_comparison.xlsx          — full Excel workbook with all data
    ├── graphs/
    │   ├── 1_tracking_metrics.png   — MOTA, IDF1, ADE, IDsw, MT, ML bar charts
    │   ├── 2_confusion_matrix.png   — TP, FP, FN, TN grouped bar chart
    │   ├── 3_derived_metrics.png    — Precision, Recall, F1, Specificity
    │   └── 4_radar_summary.png      — spider chart: all score metrics
    ├── Video32_OAMN.mp4             ─┐
    ├── Video32_OccluTrack.mp4        │  annotated inference videos
    ├── Video33_OAMN.mp4              │  for videos 32, 33, 35, 37, 39
    ├── Video33_OccluTrack.mp4        │  (two runs each: OAMN & OccluTrack)
    ├── ...                          ─┘
    └── per_method/
        ├── OAMN/
        │   ├── per_video_metrics.csv
        │   └── aggregated_metrics.json
        └── OccluTrack/
            ├── per_video_metrics.csv
            └── aggregated_metrics.json

================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS  +  DEVICE SELECTION
# ============================================================================

import os
import sys
import json
import random
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import motmetrics as mm
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

matplotlib.use("Agg")  # Non-interactive backend — safe on any machine

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _select_device() -> str:
    """
    Detect and return the best available compute device.
    Priority: CUDA GPU → Apple MPS → CPU.
    Sets PyTorch thread count for CPU fallback.
    """
    torch.set_num_threads(os.cpu_count() or 4)
    if torch.cuda.is_available():
        dev = "cuda"
        # Enable cuDNN benchmark mode for maximum GPU throughput
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU detected — using CUDA: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        print("[Device] Apple MPS detected — using Metal GPU")
    else:
        dev = "cpu"
        print(f"[Device] No GPU detected — using CPU ({os.cpu_count()} threads)")
    return dev


DEVICE = _select_device()  # Resolved once at import time; used everywhere


# ============================================================================
# SECTION 2: CONFIGURATION  ← Edit paths here to adapt to your environment
# ============================================================================

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\hanan\senior-proj")
RESULT_DIR = BASE_DIR / "result"                      # All outputs go here
TRAIN_DIR = BASE_DIR                                  # Video folders live directly here
COMPARE_CSV = RESULT_DIR / "reid_comparison.csv"
COMPARE_XLSX = RESULT_DIR / "reid_comparison.xlsx"

# ── Methods to evaluate ───────────────────────────────────────────────────────
# OAMN  → loads pretrained weights from OAMN_WEIGHTS_PATH (no training)
# OccluTrack → trained from scratch on OVIS dataset crops
ALL_METHODS: List[str] = ["OAMN", "OccluTrack"]

# ── YOLOv11 detector ─────────────────────────────────────────────────────────
YOLO_TRAINED_PATH = r"D:\hanan\senior-proj\runs\detect\train-28\weights\best.pt"
YOLO_FALLBACK_PATHS: List[str] = ["yolo11n.pt"]
YOLO_PRETRAINED_FALLBACK = "yolo11n.pt"
YOLO_CONF = 0.50
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 640
YOLO_VERBOSE = False

# ── OAMN: use existing pretrained weights — DO NOT retrain ────────────────────
OAMN_WEIGHTS_PATH = r"D:\hanan\senior-proj\reid_experiments_2\weights\OAMN\best.pth"

# ── Inference test videos (processed with both methods) ───────────────────────
INFERENCE_VIDEO_NAMES = {"32", "33", "35", "37", "39"}

# ── Kalman filter parameters ─────────────────────────────────────────────────
VIDEO_FPS = 15
LONG_TERM_OCC_SEC = 10
LONG_TERM_MAX_AGE = VIDEO_FPS * LONG_TERM_OCC_SEC   # 150 frames
KF_MIN_HITS = 3

# ── Re-ID association thresholds ─────────────────────────────────────────────
REID_EMBED_DIM = 512
REID_IOU_GATE = 0.20        # Minimum IoU to activate appearance matching
                             # Bypassed for long-term occluded tracks
REID_MATCH_THRESH = 0.55
W_IOU = 0.40                # IoU weight in combined cost matrix
W_APP = 0.60                # Appearance weight in combined cost matrix
DET_IOU_THRESH = 0.50       # IoU threshold for TP/FP detection matching

# ── Occlusion frame thresholds ────────────────────────────────────────────────
# Short occlusion (1..REID_LT_OCC_FRAMES): IoU gate applied normally.
# Long-term occlusion (>REID_LT_OCC_FRAMES): IoU gate BYPASSED.
# After LONG_TERM_MAX_AGE consecutive missed frames → track marked LOST.
REID_LT_OCC_FRAMES = 30

# ── Training hyperparameters (GPU-optimised) ──────────────────────────────────
EPOCHS = 15
EARLY_STOP_PAT = 4
LEARNING_RATE = 3e-4
TRIPLET_MARGIN = 0.3
RANDOM_SEED = 42
TRAIN_SPLIT = 0.80          # 80% train / 20% evaluation — no overlap
BATCH_P = 8                 # Identities per triplet batch  (GPU: 8, CPU: 4)
BATCH_K = 4                 # Crops per identity per batch

# ── Per-method architecture hyperparameters ───────────────────────────────────
REID_PARAMS: Dict[str, dict] = {
    "OAMN": {
        "mask_thr": 0.50,
        "n_branches": 2,
    },
    "OccluTrack": {
        "n_stripes": 6,             # Horizontal visibility stripes
        "visibility_thr": 0.40,     # Gradient-based visibility threshold
        "n_parts": 4,               # Part-level spatial feature regions
        "gcn_layers": 2,            # Graph convolution layers for part graph
    },
}


# ============================================================================
# SECTION 3: DATA STRUCTURES
# ============================================================================

class OcclusionState(Enum):
    """Track life-cycle states."""
    TENTATIVE = "tentative"   # New track, not yet confirmed
    CONFIRMED = "confirmed"   # Seen >= KF_MIN_HITS times
    OCCLUDED  = "occluded"    # Confirmed but currently not detected
    LOST      = "lost"        # Exceeded max occlusion age → remove


@dataclass
class Detection:
    """Single detection from YOLO for one frame."""
    bbox:       np.ndarray   # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id:   int
    frame_id:   int = 0


@dataclass
class Track:
    """Active track maintained by the Kalman Filter + Re-ID pipeline."""
    id:                 int
    bbox:               np.ndarray
    class_id:           int
    confidence:         float
    state:              OcclusionState = OcclusionState.TENTATIVE
    trajectory:         List[np.ndarray] = field(default_factory=list)
    age:                int = 0
    hits:               int = 0
    time_since_update:  int = 0
    occlusion_duration: int = 0
    is_kalman_pred:     bool = False


# ============================================================================
# SECTION 4: OVIS CROP DATASET  (built once from GT annotations)
# ============================================================================

class OVISCropDataset(Dataset):
    """
    Reads OVIS-format annotations from each video folder and extracts
    per-soldier image crops for Re-ID training.

    Expected folder layout per video:
        <video_name>/
            images/default/<frame_*.png>
            annotations/instances_default.json  (COCO-style)

    Each annotation entry must contain a soldier identity key in:
        ann['attributes']['soldier_ID']  OR  ann['track_id']  OR  ann['id']
    """

    def __init__(self, video_root: Path, allowed_folders: Optional[set] = None):
        self.video_root = Path(video_root)
        self.allowed_folders = allowed_folders
        self.samples: List[tuple] = []

        # Standard ImageNet normalisation used by all MobileNetV2 backbones
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._load()
        self.num_classes = len(set(s[1] for s in self.samples))

    def _load(self):
        """Iterate video folders, parse annotations, extract and store crops."""
        logger.info(f"[Dataset] Loading crops from: {self.video_root}")
        raw_samples: List[tuple] = []

        for folder_path in self.video_root.iterdir():
            if not folder_path.is_dir():
                continue
            if self.allowed_folders is not None and folder_path.name not in self.allowed_folders:
                continue

            ann_path = folder_path / "annotations" / "instances_default.json"
            if not ann_path.exists():
                continue

            with open(ann_path, "r") as f:
                data = json.load(f)

            if "images" not in data:
                continue

            img_map = {img["id"]: img["file_name"] for img in data["images"]}

            for ann in data.get("annotations", []):
                # Resolve soldier identity — prefer the most specific key
                track_id = ann.get("attributes", {}).get(
                    "soldier_ID", ann.get("track_id", ann.get("id"))
                )
                unique_key = f"{folder_path.name}_{track_id}"

                file_name = img_map.get(ann["image_id"])
                if not file_name:
                    continue
                img_path = folder_path / "images" / "default" / file_name
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                x, y, w, h = ann["bbox"]
                crop = img[int(y):int(y + h), int(x):int(x + w)]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                raw_samples.append((Image.fromarray(crop_rgb), unique_key))

        if not raw_samples:
            logger.warning("[Dataset] No samples found — check folder structure.")
            return

        unique_ids = sorted(set(s[1] for s in raw_samples))
        id_map = {uid: i for i, uid in enumerate(unique_ids)}

        for pil_img, uid in raw_samples:
            self.samples.append((self.transform(pil_img), id_map[uid]))

        logger.info(
            f"[Dataset] Loaded {len(self.samples)} crops from "
            f"{len(unique_ids)} unique soldier identities."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ============================================================================
# SECTION 5: PK BATCH SAMPLER  (P identities × K crops per batch)
# ============================================================================

class TripletBatchSampler:
    """
    Samples exactly P identities × K crops per mini-batch, guaranteeing
    valid anchor-positive-negative triplets in every batch.

    P=8 / K=4 is the GPU-tuned default.  Reduce P to 4 if running on CPU.
    """

    def __init__(self, labels: List[int], P: int = BATCH_P, K: int = BATCH_K):
        self.P = P
        self.K = K
        self.lbl2idx: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.lbl2idx[lbl].append(i)
        # Only include identities that have at least K samples
        self.valid = [lbl for lbl, idxs in self.lbl2idx.items() if len(idxs) >= K]

    def __iter__(self):
        lbls = self.valid.copy()
        random.shuffle(lbls)
        for i in range(0, len(lbls) - self.P + 1, self.P):
            batch: List[int] = []
            for lbl in lbls[i:i + self.P]:
                batch.extend(
                    random.sample(self.lbl2idx[lbl], min(self.K, len(self.lbl2idx[lbl])))
                )
            yield batch

    def __len__(self) -> int:
        return max(1, len(self.valid) // self.P)


# ============================================================================
# SECTION 6: LOSS FUNCTIONS
# ============================================================================

class TripletLoss(nn.Module):
    """
    Batch-hard online triplet loss.
    Mines the hardest positive (max distance, same identity) and hardest
    negative (min distance, different identity) within each mini-batch.
    """

    def __init__(self, margin: float = TRIPLET_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(emb, emb, p=2)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_dist = (dist * mask_pos).max(dim=1)[0]
        neg_dist = (dist + 1e6 * mask_pos).min(dim=1)[0]
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class IDClassifier(nn.Module):
    """Auxiliary identity classification head — used only during training."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


# ============================================================================
# SECTION 7: YOLO DETECTOR
# ============================================================================

def _load_trained_yolo() -> YOLO:
    """
    Load the OVIS-trained YOLOv11 model with fallback logic:
      1. YOLO_TRAINED_PATH  (primary — your OVIS-trained weights)
      2. Each entry in YOLO_FALLBACK_PATHS
      3. YOLO_PRETRAINED_FALLBACK  (generic yolo11n.pt — last resort)
    """
    for path in [YOLO_TRAINED_PATH] + YOLO_FALLBACK_PATHS:
        if Path(path).exists():
            logger.info(f"[YOLO] Loading model: {path}")
            model = YOLO(path)
            model.to(DEVICE)
            logger.info(
                f"[YOLO] Classes ({len(model.names)}): "
                + ", ".join(f"{k}:{v}" for k, v in model.names.items())
            )
            return model

    logger.warning(
        f"[YOLO] No trained model found. Falling back to: {YOLO_PRETRAINED_FALLBACK}"
    )
    model = YOLO(YOLO_PRETRAINED_FALLBACK)
    model.to(DEVICE)
    return model


class YOLODetector:
    """Wraps the YOLO model and provides a clean detect() interface."""

    def __init__(self):
        self.model = _load_trained_yolo()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=YOLO_IMG_SIZE,
            verbose=YOLO_VERBOSE,
        )
        detections: List[Detection] = []
        for r in results:
            for i in range(len(r.boxes)):
                detections.append(
                    Detection(
                        bbox=r.boxes.xyxy[i].cpu().numpy(),
                        confidence=float(r.boxes.conf[i].cpu().numpy()),
                        class_id=int(r.boxes.cls[i].cpu().numpy()),
                    )
                )
        return detections


# ============================================================================
# SECTION 8: KALMAN PREDICTOR
# ============================================================================

class KalmanPredictor:
    """
    7-dimensional Kalman Filter for bounding-box state estimation.

    State vector: [cx, cy, s, r, vx, vy, vs]
      cx, cy  — bounding-box centre
      s       — area (w × h)
      r       — aspect ratio (w / h)
      vx, vy  — velocity in x and y
      vs      — velocity of area

    Under long-term occlusion the process noise Q is scaled up
    proportionally, allowing the filter to account for increasing
    uncertainty as the target remains hidden.
    """

    def __init__(self):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State transition matrix (constant-velocity model)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        # Observation matrix (observe cx, cy, s, r only)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.kf.R *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self._base_Q = self.kf.Q.copy()
        self._occ_frames = 0

    def initialize(self, bbox: np.ndarray):
        """Initialise Kalman state from a detection bounding box."""
        z = self._to_z(bbox)
        self.kf.x[:4] = z
        self.kf.update(z)

    def update(self, bbox: np.ndarray):
        """Correct the Kalman state with a new matched detection."""
        self.kf.update(self._to_z(bbox))
        self._occ_frames = 0
        self.kf.Q = self._base_Q.copy()

    def predict(self, is_occluded: bool = False) -> np.ndarray:
        """
        Advance the Kalman state by one time step.
        If the track is occluded, scale up process noise to model uncertainty.
        """
        if is_occluded:
            self._occ_frames += 1
            factor = 1.0 + 2.0 * min(self._occ_frames, LONG_TERM_MAX_AGE) / LONG_TERM_MAX_AGE
            self.kf.Q = self._base_Q * factor
        self.kf.predict()
        return self._to_bbox(self.kf.x)

    @staticmethod
    def _to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] → Kalman observation [cx, cy, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return np.array(
            [[bbox[0] + w / 2], [bbox[1] + h / 2], [w * h], [w / max(float(h), 1e-6)]],
            dtype=float,
        )

    @staticmethod
    def _to_bbox(x: np.ndarray) -> np.ndarray:
        """Convert Kalman state [cx, cy, s, r, ...] → [x1, y1, x2, y2]."""
        s = float(np.squeeze(x[2]))
        r = float(np.squeeze(x[3]))
        if s > 0 and r > 0:
            w = np.sqrt(s * r)
            h = s / w
        else:
            w = h = 0.0
        cx = float(np.squeeze(x[0]))
        cy = float(np.squeeze(x[1]))
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ============================================================================
# SECTION 9: RE-ID EXTRACTORS
# ============================================================================
#
#  Base class — shared backbone + crop/extract utilities.
#  Backbone: MobileNetV2 pretrained on ImageNet.
#            Last 5 blocks are unfrozen and fine-tuned.
#            Early blocks remain frozen (transfer learning).
#  Embedding: 512-d L2-normalised appearance vector.
#  Training:  Triplet loss (batch-hard) + CrossEntropy (ID classification).
#
# ============================================================================

class BaseReIDExtractor(nn.Module):
    """Shared backbone and utility methods for all Re-ID extractors."""

    BACKBONE_DIM = 1280   # MobileNetV2 final feature-map channels

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._device = DEVICE
        self._transform_eval: Optional[transforms.Compose] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract a 512-d appearance embedding for one bounding box crop."""
        crop = self._crop_frame(frame, bbox)
        if crop is None:
            return np.zeros(REID_EMBED_DIM, dtype=np.float32)
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = self._transform_eval(rgb).unsqueeze(0).to(self._device)
            self.eval()
            with torch.no_grad():
                emb = self.forward(inp).squeeze(0)
            return emb.cpu().numpy()
        except Exception:
            return np.zeros(REID_EMBED_DIM, dtype=np.float32)

    def _load_backbone(self) -> nn.Module:
        """
        Load MobileNetV2 pretrained on ImageNet.
        Unfreeze the last 5 blocks for fine-tuning; freeze earlier blocks.
        """
        import torchvision.models as tvm
        import torchvision.transforms as T

        mn = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        bb = nn.Sequential(*list(mn.features.children()))
        for child in list(bb.children())[:-5]:
            for p in child.parameters():
                p.requires_grad = False
        for child in list(bb.children())[-5:]:
            for p in child.parameters():
                p.requires_grad = True

        self._transform_eval = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return bb

    def _crop_frame(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        w: int = 128,
        h: int = 256,
    ) -> Optional[np.ndarray]:
        """Safe crop: clamps coordinates to frame boundaries before slicing."""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return cv2.resize(frame[y1:y2, x1:x2], (w, h))

    def _gradient_visibility(self, gray: np.ndarray) -> float:
        """
        Estimate the visibility of a grayscale image patch using
        the standard deviation of the Laplacian (sharpness proxy).
        Returns a value in [0, 1].
        """
        if gray.size == 0:
            return 0.0
        return min(1.0, float(np.std(cv2.Laplacian(gray, cv2.CV_64F))) / 255.0 * 8.0)

    def _gray_batch(self, x: torch.Tensor) -> List[np.ndarray]:
        """Convert a normalised batch tensor to a list of grayscale numpy arrays."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgs = np.clip(
            (x.detach().cpu().permute(0, 2, 3, 1).numpy() * std + mean) * 255,
            0,
            255,
        ).astype(np.uint8)
        return [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs]


# ── 9a. OAMN (Occlusion-Aware Mask Network) ──────────────────────────────────

class OAMNExtractor(BaseReIDExtractor):
    """
    Chen et al., ACM MM 2021 — Occlusion-Aware Mask Network.

    Architecture:
      • MobileNetV2 backbone (last 5 blocks fine-tuned)
      • Spatial mask predictor: Conv(1280→64)→ReLU→Conv(64→1)→Sigmoid
      • Confidence estimator:   AvgPool→Flatten→Linear(1280→1)→Sigmoid
      • Two-branch fusion:
          full_feat  × (1 - conf)  — global appearance
          vis_feat   × conf        — mask-weighted visible region
      • Projection head: Linear(2560→512)→BN→ReLU→Dropout→Linear(512→512)

    For this pipeline: weights are loaded from OAMN_WEIGHTS_PATH.
    The model is NOT retrained here.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.mask_thr = params.get("mask_thr", 0.50)
        self.backbone = self._load_backbone().to(self._device)

        self.spatial_mask = nn.Sequential(
            nn.Conv2d(self.BACKBONE_DIM, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        ).to(self._device)

        self.mask_conf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.BACKBONE_DIM, 1),
            nn.Sigmoid(),
        ).to(self._device)

        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)

        logger.info("[OAMN] Two-branch occlusion-aware extractor ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        B = x.size(0)
        full_feat = F.adaptive_avg_pool2d(fmap, 1).view(B, -1)
        vis_feat = F.adaptive_avg_pool2d(
            fmap * (self.spatial_mask(fmap) >= self.mask_thr).float(), 1
        ).view(B, -1)
        conf = self.mask_conf(fmap)
        fused = torch.cat([full_feat * (1 - conf), vis_feat * conf], dim=1)
        return F.normalize(self.head(fused), dim=1)


# ── 9b. OccluTrack ────────────────────────────────────────────────────────────

class OccluTrackExtractor(BaseReIDExtractor):
    """
    OccluTrack Re-ID Extractor — Trained on OVIS dataset crops.

    Architecture (multi-branch occlusion-robust design):

    Branch 1 — Stripe Visibility Weighting (inspired by Zhuo et al., ICME 2018):
      • Divides the feature map into N horizontal stripes.
      • Computes a gradient-based visibility score per stripe.
      • Stripes below the visibility threshold are suppressed.
      • Weighted aggregation produces an occlusion-robust global feature.

    Branch 2 — Part-level Graph Reasoning (inspired by Wang et al., CVPR 2020):
      • Divides the feature map into M spatial parts.
      • Builds a learnable adjacency matrix (softmax-normalised).
      • Applies graph convolution layers to propagate context across parts.
      • Aggregates part embeddings into a global part-graph feature.

    Fusion:
      • Both branch outputs are projected to 256-d, then concatenated → 512-d.
      • L2-normalised output embedding.

    Training:
      • Batch-hard triplet loss + cross-entropy identity classification.
      • Trained end-to-end on OVIS soldier crops.
      • Best weights saved to <RESULT_DIR>/weights/OccluTrack/best.pth.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.n_stripes      = params.get("n_stripes", 6)
        self.visibility_thr = params.get("visibility_thr", 0.40)
        self.n_parts        = params.get("n_parts", 4)
        self.gcn_layers     = params.get("gcn_layers", 2)

        self.backbone = self._load_backbone().to(self._device)

        # ── Branch 1: Stripe head ──────────────────────────────────────────
        self.stripe_head = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        ).to(self._device)

        # ── Branch 2: Part-graph head ──────────────────────────────────────
        # Learnable adjacency logits (softmax-normalised during forward pass)
        self.adj_logits = nn.Parameter(
            torch.ones(self.n_parts, self.n_parts, device=self._device)
        )
        self.gcn_w = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.BACKBONE_DIM, self.BACKBONE_DIM, device=self._device) * 0.01
            )
            for _ in range(self.gcn_layers)
        ])
        self.part_head = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM * self.n_parts, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        ).to(self._device)

        # ── Fusion projection (256 + 256 → 512) ───────────────────────────
        self.fusion_head = nn.Sequential(
            nn.Linear(512, REID_EMBED_DIM),
            nn.BatchNorm1d(REID_EMBED_DIM),
        ).to(self._device)

        logger.info(
            f"[OccluTrack] {self.n_stripes}-stripe + {self.n_parts}-part "
            f"{self.gcn_layers}-layer GCN extractor ready."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)                        # [B, C, H, W]
        B, C, H, W = fmap.shape
        grays = self._gray_batch(x)

        # ── Branch 1: Stripe visibility weighting ─────────────────────────
        sh = max(1, H // self.n_stripes)
        c_sh = max(1, x.shape[2] // self.n_stripes)
        agg = torch.zeros(B, C, device=self._device)
        w_sum = torch.zeros(B, device=self._device)

        for i in range(self.n_stripes):
            sf = fmap[:, :, i * sh:(i + 1) * sh, :]
            if sf.shape[2] == 0:
                continue
            sf_pool = F.adaptive_avg_pool2d(sf, 1).squeeze(-1).squeeze(-1)
            vis = torch.tensor(
                [
                    self._gradient_visibility(g[i * c_sh:(i + 1) * c_sh, :])
                    for g in grays
                ],
                device=self._device,
            )
            mask = (vis >= self.visibility_thr).float()
            agg += sf_pool * (vis * mask).unsqueeze(1)
            w_sum += vis * mask

        gpool = F.adaptive_avg_pool2d(fmap, 1).squeeze(-1).squeeze(-1)
        stripe_feat = torch.where(
            (w_sum == 0).unsqueeze(1),
            gpool,
            agg / (w_sum.unsqueeze(1) + 1e-9),
        )
        stripe_out = self.stripe_head(stripe_feat)     # [B, 256]

        # ── Branch 2: Part-level graph reasoning ──────────────────────────
        ph = max(1, H // self.n_parts)
        A = torch.softmax(self.adj_logits, dim=1)
        nodes = []
        for i in range(self.n_parts):
            sf = fmap[:, :, i * ph:(i + 1) * ph, :]
            if sf.shape[2] > 0:
                nodes.append(F.adaptive_avg_pool2d(sf, 1).view(B, C))
            else:
                nodes.append(torch.zeros(B, C, device=self._device))

        X = torch.stack(nodes, dim=1)                  # [B, n_parts, C]
        for W_gcn in self.gcn_w:
            X_nb = torch.einsum("pq,bqc->bpc", A, X)
            X_nb = F.relu(torch.einsum("bpc,cd->bpd", X_nb, W_gcn))
            X = X + X_nb
        part_out = self.part_head(X.reshape(B, -1))    # [B, 256]

        # ── Fusion ────────────────────────────────────────────────────────
        fused = torch.cat([stripe_out, part_out], dim=1)  # [B, 512]
        return F.normalize(self.fusion_head(fused), dim=1)


# ── Registry and factory ──────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {
    "OAMN": OAMNExtractor,
    "OccluTrack": OccluTrackExtractor,
}


def build_reid(name: str) -> BaseReIDExtractor:
    """Instantiate a Re-ID extractor by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown Re-ID method: '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](REID_PARAMS[name])


# ============================================================================
# SECTION 10: UNIFIED TRACKER  (Kalman + Re-ID association)
# ============================================================================

def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(cost: np.ndarray, threshold: float):
    """
    Greedy bipartite matching: select pairs with the lowest cost first.
    Returns (matched_pairs, unmatched_detections, unmatched_tracks).
    """
    matched, used_r, used_c = [], set(), set()
    entries = sorted(
        [(cost[r, c], r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1])]
    )
    for val, r, c in entries:
        if val > threshold:
            break
        if r not in used_r and c not in used_c:
            matched.append((r, c))
            used_r.add(r)
            used_c.add(c)
    unm_d = [r for r in range(cost.shape[0]) if r not in used_r]
    unm_t = [c for c in range(cost.shape[1]) if c not in used_c]
    return matched, unm_d, unm_t


class UnifiedTracker:
    """
    Multi-object tracker combining Kalman Filter predictions with
    Re-ID appearance embeddings for association.

    Occlusion handling:
      • Short occlusion (≤ REID_LT_OCC_FRAMES): IoU gate + appearance.
      • Long-term occlusion (> REID_LT_OCC_FRAMES): appearance-only matching
        (IoU gate bypassed because Kalman drift makes IoU unreliable).
      • After LONG_TERM_MAX_AGE missed frames: track marked LOST and removed.
    """

    def __init__(self, reid: BaseReIDExtractor):
        self.reid = reid
        self._tracks: List[Track] = []
        self._kf: Dict[int, KalmanPredictor] = {}
        self._gallery: Dict[int, List[np.ndarray]] = {}
        self._next_id = 1

    def reset(self):
        """Clear all tracks — call between videos."""
        self._tracks.clear()
        self._kf.clear()
        self._gallery.clear()
        self._next_id = 1

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Process one frame: predict → associate → update tracks.
        Returns the list of currently active (CONFIRMED + OCCLUDED) tracks.
        """
        # Step 1: Kalman predict for all existing tracks
        for t in self._tracks:
            is_occ = t.state == OcclusionState.OCCLUDED
            t.bbox = self._kf[t.id].predict(is_occluded=is_occ)
            t.time_since_update += 1
            t.age += 1
            if is_occ:
                t.occlusion_duration += 1

        # Step 2: Extract appearance embeddings for all detections
        det_feats = [self.reid.extract(frame, d.bbox) for d in detections]
        active = [t for t in self._tracks if t.state != OcclusionState.LOST]

        # Step 3: Build cost matrix and match detections to tracks
        if active and detections:
            cost = np.ones((len(detections), len(active)))
            for di, det in enumerate(detections):
                for ti, trk in enumerate(active):
                    iv = _iou(det.bbox, trk.bbox)
                    ic = 1.0 - iv
                    long_term = (
                        trk.state == OcclusionState.OCCLUDED
                        and trk.occlusion_duration > REID_LT_OCC_FRAMES
                    )
                    use_appearance = trk.id in self._gallery and (
                        iv >= REID_IOU_GATE or long_term
                    )
                    if use_appearance:
                        gal = np.stack(self._gallery[trk.id][-50:])
                        qf = det_feats[di]
                        nq = np.linalg.norm(qf)
                        ng = np.linalg.norm(gal, axis=1)
                        if nq > 0 and np.any(ng > 0):
                            ac = 1.0 - float((gal @ qf / (ng * nq + 1e-9)).max())
                        else:
                            ac = ic
                        # Long-term: weight appearance more heavily
                        cost[di, ti] = 0.1 * ic + 0.9 * ac if long_term else W_IOU * ic + W_APP * ac
                    else:
                        cost[di, ti] = ic

            matched, unm_d, unm_t = _greedy_match(cost, 1.0 - REID_MATCH_THRESH)
        else:
            matched = []
            unm_d = list(range(len(detections)))
            unm_t = list(range(len(active)))

        # Step 4: Update matched tracks
        for di, ti in matched:
            t = active[ti]
            t.bbox = detections[di].bbox
            t.confidence = detections[di].confidence
            t.hits += 1
            t.time_since_update = 0
            t.occlusion_duration = 0
            t.is_kalman_pred = False
            t.trajectory.append(detections[di].bbox.copy())
            self._kf[t.id].update(detections[di].bbox)
            self._gallery.setdefault(t.id, []).append(det_feats[di])
            if t.state == OcclusionState.OCCLUDED:
                t.state = OcclusionState.CONFIRMED
            elif t.state == OcclusionState.TENTATIVE and t.hits >= KF_MIN_HITS:
                t.state = OcclusionState.CONFIRMED

        # Step 5: Handle unmatched tracks
        for ti in unm_t:
            t = active[ti]
            if t.state == OcclusionState.TENTATIVE:
                t.state = OcclusionState.LOST
            elif t.state == OcclusionState.CONFIRMED:
                t.state = OcclusionState.OCCLUDED
                t.is_kalman_pred = True
            elif t.state == OcclusionState.OCCLUDED:
                if t.occlusion_duration >= LONG_TERM_MAX_AGE:
                    t.state = OcclusionState.LOST
                else:
                    t.is_kalman_pred = True

        # Step 6: Spawn new tracks for unmatched detections
        for di in unm_d:
            det = detections[di]
            tid = self._next_id
            self._next_id += 1
            kf = KalmanPredictor()
            kf.initialize(det.bbox)
            self._tracks.append(
                Track(
                    id=tid,
                    bbox=det.bbox,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    state=OcclusionState.TENTATIVE,
                    trajectory=[det.bbox.copy()],
                    hits=1,
                )
            )
            self._kf[tid] = kf
            self._gallery[tid] = [det_feats[di]]

        # Purge lost tracks
        self._tracks = [t for t in self._tracks if t.state != OcclusionState.LOST]

        return [
            t for t in self._tracks
            if t.state in (OcclusionState.CONFIRMED, OcclusionState.OCCLUDED)
        ]


# ============================================================================
# SECTION 11: GROUND-TRUTH HELPERS
# ============================================================================

def _build_gt(folder_path: Path):
    """
    Parse the COCO-style annotation file for one video folder.

    Returns:
        (video_info, gt_by_frame) where gt_by_frame is a dict mapping
        frame_index → list of {"id": int, "bbox": [x1, y1, x2, y2]}.
    """
    ann_path = folder_path / "annotations" / "instances_default.json"
    if not ann_path.exists():
        return None, {}

    with open(ann_path, "r") as f:
        data = json.load(f)

    gt: Dict[int, List[dict]] = defaultdict(list)

    if "images" not in data:
        return {}, {}

    img_map = {
        img["id"]: int(
            img["file_name"].replace("frame_", "").replace(".png", "")
        )
        for img in data["images"]
    }

    for ann in data.get("annotations", []):
        frame_idx = img_map.get(ann["image_id"])
        if frame_idx is None:
            continue
        tid = ann.get("attributes", {}).get(
            "soldier_ID", ann.get("track_id", ann.get("id"))
        )
        x, y, w, h = ann["bbox"]
        gt[frame_idx].append({"id": tid, "bbox": [x, y, x + w, y + h]})

    return data.get("info", {}), gt


# ============================================================================
# SECTION 12: METRICS
# ============================================================================

def _compute_detection_metrics(
    predictions: list,
    gt_by_frame: dict,
    vid_len: int,
    iou_thr: float = DET_IOU_THRESH,
) -> dict:
    """
    Compute per-frame detection confusion matrix over a full video.

    TP: detected box matches a GT box (IoU ≥ iou_thr)
    FP: detected box has no matching GT box
    FN: GT box was not detected
    TN: frame had zero GT objects AND zero detections
    """
    TP = FP = FN = TN = 0
    pred_lut = {p["frame_id"]: p for p in predictions}

    for fid in range(vid_len):
        gt_objs = gt_by_frame.get(fid, [])
        pi = pred_lut.get(fid)
        pred_bbs = [np.array(t[1]) for t in pi["tracks"]] if pi else []
        gt_bbs = [np.array(g["bbox"]) for g in gt_objs]
        ng, np_ = len(gt_bbs), len(pred_bbs)

        if ng == 0 and np_ == 0:
            TN += 1
            continue
        if ng == 0:
            FP += np_
            continue
        if np_ == 0:
            FN += ng
            continue

        mg: set = set()
        mp: set = set()
        pairs = sorted(
            [
                (pi2, gi, _iou(pred_bbs[pi2], gt_bbs[gi]))
                for pi2 in range(np_)
                for gi in range(ng)
            ],
            key=lambda x: -x[2],
        )
        for pi2, gi, iv in pairs:
            if pi2 in mp or gi in mg:
                continue
            if iv >= iou_thr:
                TP += 1
                mp.add(pi2)
                mg.add(gi)
        FP += np_ - len(mp)
        FN += ng - len(mg)

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


def _evaluate_video(
    predictions: list,
    folder_path: Path,
    img_w: int,
    img_h: int,
) -> Optional[dict]:
    """
    Compute full MOT + detection metrics for one video.

    Returns a dict with keys:
        mota, idf1, ade, raw_ade, id_switches, mt, ml,
        TP, FP, FN, TN, precision, recall, f1, specificity
    Returns None if no GT annotations are found.
    """
    _video_info, gt_by_frame = _build_gt(folder_path)
    if not gt_by_frame:
        return None

    vid_len = max(gt_by_frame.keys()) + 1
    img_diag = np.sqrt(img_w ** 2 + img_h ** 2) if (img_w and img_h) else 1.0

    # ── MOT metrics via motmetrics ────────────────────────────────────────────
    acc = mm.MOTAccumulator(auto_id=True)
    pred_trajs: Dict[int, list] = defaultdict(list)
    gt_trajs: Dict[int, list] = defaultdict(list)

    for pf in predictions:
        fid = pf["frame_id"]
        if fid >= vid_len:
            continue
        gt_objs = gt_by_frame.get(fid, [])
        gt_ids = [o["id"] for o in gt_objs]
        gt_bbs = [o["bbox"] for o in gt_objs]
        p_ids = [t[0] for t in pf["tracks"]]
        p_bbs = [t[1] for t in pf["tracks"]]

        for pid, pb in zip(p_ids, p_bbs):
            pred_trajs[pid].append(pb)
        for gid, gb in zip(gt_ids, gt_bbs):
            gt_trajs[gid].append(gb)

        dist = (
            np.array([
                [1.0 - _iou(np.array(gb), np.array(pb)) for pb in p_bbs]
                for gb in gt_bbs
            ])
            if gt_bbs and p_bbs
            else np.empty((len(gt_bbs), len(p_bbs)))
        )
        acc.update(gt_ids, p_ids, dist)

    mh = mm.metrics.create()
    smr = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "mostly_tracked", "mostly_lost"],
        name="acc",
    )

    def _g(col: str, default: float = 0.0) -> float:
        v = smr[col].values[0] if col in smr.columns else default
        return float(v) if not np.isnan(float(v)) else default

    # ── Average Displacement Error ────────────────────────────────────────────
    errs: List[float] = []
    for pid, pt in pred_trajs.items():
        for gid, gt in gt_trajs.items():
            n = min(len(pt), len(gt))
            if n == 0:
                continue
            pc = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in pt[:n]])
            gc = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in gt[:n]])
            errs.extend(np.linalg.norm(pc - gc, axis=1).tolist())

    raw_ade = float(np.mean(errs)) if errs else 0.0
    ade = raw_ade / img_diag if img_diag > 0 else raw_ade

    # ── Detection confusion matrix ────────────────────────────────────────────
    cfm = _compute_detection_metrics(predictions, gt_by_frame, vid_len)
    tp, fp, fn, tn = cfm["TP"], cfm["FP"], cfm["FN"], cfm["TN"]

    return {
        "mota":        max(-1.0, _g("mota")),
        "idf1":        max(0.0, _g("idf1")),
        "ade":         ade,
        "raw_ade":     raw_ade,
        "id_switches": int(_g("num_switches")),
        "mt":          int(_g("mostly_tracked")),
        "ml":          int(_g("mostly_lost")),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision":   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall":      tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1":          2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


# ============================================================================
# SECTION 13: TRAINING LOOP  (OccluTrack only — OAMN loads pretrained weights)
# ============================================================================

class ReIDTrainer:
    """
    Trains one Re-ID model on OVIS crops.
    Combined loss: batch-hard triplet + cross-entropy identity classification.
    Early stopping based on validation MOTA.
    """

    LAMBDA_TRIP = 1.0
    LAMBDA_CE = 0.5

    def __init__(
        self,
        method_name: str,
        reid: BaseReIDExtractor,
        detector: YOLODetector,
        all_videos: List[Path],
        crop_dataset: OVISCropDataset,
    ):
        self.name = method_name
        self.reid = reid
        self.detector = detector
        self.all_videos = all_videos
        self.dataset = crop_dataset
        self.device = reid._device
        self.weights_dir = RESULT_DIR / "weights" / method_name
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = crop_dataset.num_classes

        self.classifier = IDClassifier(REID_EMBED_DIM, self.num_classes).to(self.device)
        self.optimizer = optim.Adam(
            list(reid.parameters()) + list(self.classifier.parameters()),
            lr=LEARNING_RATE,
            weight_decay=5e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        self.triplet_loss = TripletLoss(TRIPLET_MARGIN)
        self.ce_loss = nn.CrossEntropyLoss()

    def run(self) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(
            f"  TRAINING [{self.name}]  "
            f"{len(self.dataset)} crops / {self.num_classes} IDs"
        )
        logger.info(f"{'='*60}")

        labels = [self.dataset.samples[i][1] for i in range(len(self.dataset))]
        # GPU: use more workers for data loading; CPU: keep at 0
        num_workers = 2 if self.device == "cuda" else 0
        sampler = TripletBatchSampler(labels, P=BATCH_P, K=BATCH_K)
        loader = DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )

        best_mota = float("-inf")
        no_improve = 0
        history = []

        for epoch in range(1, EPOCHS + 1):
            loss = self._train_epoch(loader)
            val_mota = self._val_mota(epoch)
            self.scheduler.step()
            history.append({"epoch": epoch, "loss": loss, "val_mota": val_mota})
            logger.info(
                f"  [{self.name}] Ep {epoch:02d}/{EPOCHS}  "
                f"Loss={loss:.4f}  ValMOTA={val_mota:.4f}  "
                f"LR={self.scheduler.get_last_lr()[0]:.2e}"
            )
            if val_mota > best_mota:
                best_mota = val_mota
                no_improve = 0
                self._save("best.pth")
                logger.info(f"    ✔ [{self.name}] New best ValMOTA={best_mota:.4f}")
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PAT:
                    logger.info(f"  [{self.name}] Early stopping at epoch {epoch}.")
                    break

        self._save("last.pth")
        logger.info(f"  [{self.name}] Training complete. Best ValMOTA={best_mota:.4f}")
        return {"history": history, "best_mota": best_mota}

    def load_best(self):
        """Load the best checkpoint saved during training."""
        ckpt = self.weights_dir / "best.pth"
        if ckpt.exists():
            st = torch.load(ckpt, map_location=self.device)
            self.reid.load_state_dict(st["reid_state"])
            logger.info(f"  [{self.name}] Best weights loaded ← {ckpt}")
        else:
            logger.warning(f"  [{self.name}] No best.pth found — using current weights.")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.reid.train()
        self.classifier.train()
        total_loss = 0.0
        n_batches = 0
        for imgs, labels in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            emb = self.reid(imgs)
            logits = self.classifier(emb)
            loss = (
                self.LAMBDA_TRIP * self.triplet_loss(emb, labels)
                + self.LAMBDA_CE * self.ce_loss(logits, labels)
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.reid.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    def _val_mota(self, epoch: int) -> float:
        """Quick MOTA estimate on a random sample of training videos."""
        random.seed(RANDOM_SEED + epoch)
        sample = random.sample(self.all_videos, min(5, len(self.all_videos)))
        tracker = UnifiedTracker(self.reid)
        motas: List[float] = []

        for vp in sample:
            frames = sorted((vp / "images" / "default").glob("*.png"))
            if not frames:
                continue
            first = cv2.imread(str(frames[0]))
            if first is None:
                continue
            h, w = first.shape[:2]
            tracker.reset()
            preds = []
            for fi, fp in enumerate(frames):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                dets = self.detector.detect(img)
                trks = tracker.update(dets, img)
                preds.append({
                    "frame_id": fi,
                    "tracks": [(t.id, t.bbox.tolist(), t.class_id, t.confidence) for t in trks],
                })
            m = _evaluate_video(preds, vp, w, h)
            if m:
                motas.append(m["mota"])

        return float(np.mean(motas)) if motas else 0.0

    def _save(self, filename: str):
        torch.save(
            {
                "reid_name":  self.name,
                "params":     REID_PARAMS.get(self.name, {}),
                "timestamp":  datetime.now().isoformat(),
                "reid_state": self.reid.state_dict(),
            },
            self.weights_dir / filename,
        )


# ============================================================================
# SECTION 14: EVALUATION LOOP
# ============================================================================

class ReIDEvaluator:
    """
    Runs the full tracking pipeline on a list of video folders and computes
    aggregated performance metrics.
    """

    def __init__(
        self,
        method_name: str,
        tracker: UnifiedTracker,
        detector: YOLODetector,
        all_videos: List[Path],
    ):
        self.name = method_name
        self.tracker = tracker
        self.detector = detector
        self.all_videos = all_videos
        method_dir = RESULT_DIR / "per_method" / method_name
        self.csv_out = method_dir / "per_video_metrics.csv"
        self.met_out = method_dir / "aggregated_metrics.json"
        method_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  EVALUATION [{self.name}]  {len(self.all_videos)} videos")
        logger.info(f"{'='*60}")

        rows: List[dict] = []

        for idx, vp in enumerate(self.all_videos, 1):
            frames = sorted((vp / "images" / "default").glob("*.png"))
            if not frames:
                continue
            first = cv2.imread(str(frames[0]))
            if first is None:
                continue
            h, w = first.shape[:2]
            self.tracker.reset()
            preds: List[dict] = []

            for fi, fp in enumerate(frames):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                dets = self.detector.detect(img)
                trks = self.tracker.update(dets, img)
                preds.append({
                    "frame_id": fi,
                    "tracks": [(t.id, t.bbox.tolist(), t.class_id, t.confidence) for t in trks],
                })

            m = _evaluate_video(preds, vp, w, h)
            if m:
                rows.append({"video": vp.name, **m})
            if idx % 10 == 0 or idx == len(self.all_videos):
                logger.info(f"  [{self.name}] {idx}/{len(self.all_videos)} videos evaluated.")

        pd.DataFrame(rows).to_csv(self.csv_out, index=False)
        agg = self._aggregate(rows, self.name)
        _print_method_summary(self.name, agg)

        with open(self.met_out, "w") as f:
            json.dump(agg, f, indent=2)

        return agg

    @staticmethod
    def _aggregate(rows: List[dict], name: str) -> dict:
        """Aggregate per-video metrics into a single summary dict."""
        if not rows:
            return {"reid_name": name}
        tp = int(np.sum([r["TP"] for r in rows]))
        fp = int(np.sum([r["FP"] for r in rows]))
        fn = int(np.sum([r["FN"] for r in rows]))
        tn = int(np.sum([r["TN"] for r in rows]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "reid_name":   name,
            "n_videos":    len(rows),
            "mota":        float(np.mean([r["mota"]        for r in rows])),
            "idf1":        float(np.mean([r["idf1"]        for r in rows])),
            "ade":         float(np.mean([r["ade"]         for r in rows])),
            "raw_ade":     float(np.mean([r["raw_ade"]     for r in rows])),
            "id_switches": int(np.sum([r["id_switches"] for r in rows])),
            "mt":          int(np.sum([r["mt"]          for r in rows])),
            "ml":          int(np.sum([r["ml"]          for r in rows])),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision":   round(prec, 6),
            "recall":      round(rec,  6),
            "f1":          round(f1,   6),
            "specificity": round(spec, 6),
        }


# ============================================================================
# SECTION 15: FRAME RENDERING
# ============================================================================

# Colour palette — one distinct colour per track ID
_PALETTE = [
    (255, 56, 56), (255, 157, 51), (51, 255, 255), (56, 255, 132),
    (255, 56, 210), (131, 255, 56), (56, 131, 255), (255, 210, 51),
    (51, 255, 131), (131, 56, 255), (100, 200, 100), (200, 100, 200),
    (100, 150, 255), (255, 100, 100),
]


def _render_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    """
    Annotate one frame with all active tracks.

    Detected tracks  (is_kalman_pred=False): solid coloured bounding box.
    Occluded tracks  (is_kalman_pred=True) : semi-transparent filled box
                                             labelled [OCC] so the Kalman-
                                             predicted position is visible.
    A 20-frame trajectory tail is drawn for every track.
    """
    img = frame.copy()
    H, W = img.shape[:2]

    for t in tracks:
        x1 = int(max(0,     min(t.bbox[0], W - 2)))
        y1 = int(max(0,     min(t.bbox[1], H - 2)))
        x2 = int(max(x1 + 1, min(t.bbox[2], W - 1)))
        y2 = int(max(y1 + 1, min(t.bbox[3], H - 1)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        col = _PALETTE[t.id % len(_PALETTE)]

        if t.is_kalman_pred:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            lbl = f"ID{t.id}[OCC]"
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            lbl = f"ID{t.id} {t.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw, ty), col, -1)
        cv2.putText(img, lbl, (x1, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Trajectory tail — last 20 frame centres
        for i in range(1, min(len(t.trajectory), 20)):
            p1 = (
                int((t.trajectory[-i][0] + t.trajectory[-i][2]) / 2),
                int((t.trajectory[-i][1] + t.trajectory[-i][3]) / 2),
            )
            p2 = (
                int((t.trajectory[-i - 1][0] + t.trajectory[-i - 1][2]) / 2),
                int((t.trajectory[-i - 1][1] + t.trajectory[-i - 1][3]) / 2),
            )
            cv2.line(img, p1, p2, col, 1)

    return img


# ============================================================================
# SECTION 16: RESULTS PRINTING
# ============================================================================

def _print_method_summary(name: str, agg: dict):
    """Print a formatted result summary box for one method."""
    tp = agg.get("TP", 0)
    fp = agg.get("FP", 0)
    fn = agg.get("FN", 0)
    tn = agg.get("TN", 0)
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    bar = "─" * 58
    print(f"\n  ┌{bar}┐")
    print(f"  │  RESULT SUMMARY: {name:<38s}│")
    print(f"  ├{bar}┤")
    print(f"  │  Tracking Metrics                                      │")
    print(f"  │    MOTA         : {agg.get('mota', 0):>8.4f}                          │")
    print(f"  │    IDF1         : {agg.get('idf1', 0):>8.4f}                          │")
    print(f"  │    ADE (norm.)  : {agg.get('ade', 0):>8.4f}                          │")
    print(f"  │    ID Switches  : {agg.get('id_switches', 0):>8d}                          │")
    print(f"  │    MT / ML      : {agg.get('mt', 0):>4d} / {agg.get('ml', 0):<4d}                        │")
    print(f"  ├{bar}┤")
    print(f"  │  Detection Confusion Matrix                            │")
    print(f"  │         Predicted +    Predicted -                     │")
    print(f"  │  GT +   TP={tp:>9,}   FN={fn:>9,}                  │")
    print(f"  │  GT -   FP={fp:>9,}   TN={tn:>9,}                  │")
    print(f"  ├{bar}┤")
    print(f"  │  Precision  : {agg.get('precision', 0):>8.4f}                          │")
    print(f"  │  Recall     : {agg.get('recall', 0):>8.4f}                          │")
    print(f"  │  F1-score   : {agg.get('f1', 0):>8.4f}                          │")
    print(f"  │  Specificity: {agg.get('specificity', 0):>8.4f}                          │")
    print(f"  │  Accuracy   : {acc:>8.4f}  (TP+TN) / {total}            │")
    print(f"  └{bar}┘")


def _save_comparison_table(all_results: List[dict]):
    """
    Save reid_comparison.csv and reid_comparison.xlsx, and print a
    full formatted comparison report to the console.
    """
    if not all_results:
        logger.warning("[Comparison] No results to save.")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cols = [
        "reid_name", "mota", "idf1", "ade", "raw_ade", "id_switches", "mt", "ml",
        "TP", "FP", "FN", "TN", "precision", "recall", "f1", "specificity",
    ]
    df = pd.DataFrame(all_results)
    df = df[[c for c in cols if c in df.columns]]

    # CSV
    df.to_csv(COMPARE_CSV, index=False)
    logger.info(f"[Output] Comparison CSV saved → {COMPARE_CSV}")

    # Excel
    with pd.ExcelWriter(COMPARE_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Aggregated Metrics", index=False)
        # Per-method per-video sheets
        for method_name in [r["reid_name"] for r in all_results]:
            csv_path = RESULT_DIR / "per_method" / method_name / "per_video_metrics.csv"
            if csv_path.exists():
                per_vid_df = pd.read_csv(csv_path)
                per_vid_df.to_excel(
                    writer,
                    sheet_name=f"{method_name}_per_video"[:31],
                    index=False,
                )
    logger.info(f"[Output] Comparison Excel saved → {COMPARE_XLSX}")

    # Console report
    w = 130
    print(); print("=" * w)
    print("  FINAL COMPARISON REPORT — YOLOv11 + Kalman Filter + Re-ID")
    print("=" * w)

    print(f"\n  {'─'*74}")
    print(f"  ── TRACKING METRICS ──")
    print(f"  {'Method':<16} {'MOTA':>8} {'IDF1':>8} {'ADE':>8} {'RawADE':>9} {'IDsw':>7} {'MT':>6} {'ML':>6}")
    print(f"  {'─'*74}")
    for _, row in df.iterrows():
        print(
            f"  {row['reid_name']:<16} "
            f"{row.get('mota', 0):>8.4f} {row.get('idf1', 0):>8.4f} "
            f"{row.get('ade', 0):>8.4f} {row.get('raw_ade', 0):>9.2f} "
            f"{int(row.get('id_switches', 0)):>7d} {int(row.get('mt', 0)):>6d} "
            f"{int(row.get('ml', 0)):>6d}"
        )

    print(f"\n  ── DETECTION CONFUSION MATRIX ──")
    print(f"  {'Method':<16} {'TP':>12} {'FP':>12} {'FN':>12} {'TN':>12}")
    print(f"  {'─'*56}")
    for _, row in df.iterrows():
        print(
            f"  {row['reid_name']:<16} "
            f"{int(row.get('TP', 0)):>12,} {int(row.get('FP', 0)):>12,} "
            f"{int(row.get('FN', 0)):>12,} {int(row.get('TN', 0)):>12,}"
        )

    print(f"\n  ── DERIVED DETECTION METRICS ──")
    print(f"  {'Method':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Specificity':>12} {'Accuracy':>10}")
    print(f"  {'─'*68}")
    for _, row in df.iterrows():
        tp_ = int(row.get("TP", 0))
        fp_ = int(row.get("FP", 0))
        fn_ = int(row.get("FN", 0))
        tn_ = int(row.get("TN", 0))
        tot = tp_ + fp_ + fn_ + tn_
        acc = (tp_ + tn_) / tot if tot > 0 else 0.0
        print(
            f"  {row['reid_name']:<16} "
            f"{row.get('precision', 0):>10.4f} {row.get('recall', 0):>8.4f} "
            f"{row.get('f1', 0):>8.4f} {row.get('specificity', 0):>12.4f} {acc:>10.4f}"
        )

    if len(df) > 1:
        print(f"\n  ── BEST METHOD PER METRIC ──")
        print(f"  {'─'*50}")
        for met in ["mota", "idf1", "f1", "precision", "recall", "specificity"]:
            if met in df.columns:
                idx_best = df[met].idxmax()
                best_row = df.loc[idx_best]
                print(f"    {met:<14}  →  {best_row['reid_name']:<16}  ({best_row[met]:.4f})")

    print("=" * w); print()


# ============================================================================
# SECTION 17: VISUALISATIONS
# ============================================================================

# Per-method colour assignment
_METHOD_COLORS: Dict[str, str] = {
    "OAMN":       "#1ABC9C",   # teal-green
    "OccluTrack": "#E74C3C",   # red-orange
}
_DEFAULT_COLORS = ["#3498DB", "#E67E22", "#9B59B6", "#F39C12"]


def _bar_group(
    ax,
    methods: List[str],
    values_dict: Dict[str, List[float]],
    title: str,
    ylabel: str,
    pct: bool = False,
):
    """Draw a grouped horizontal bar chart on the given axes."""
    metrics = list(values_dict.keys())
    n_metrics = len(metrics)
    n_methods = len(methods)
    bar_h = 0.8 / n_methods
    y_base = np.arange(n_metrics)

    for mi, method in enumerate(methods):
        vals = [values_dict[m][mi] for m in metrics]
        color = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        bars = ax.barh(
            y_base - mi * bar_h + (n_methods - 1) * bar_h / 2,
            vals,
            bar_h * 0.85,
            label=method,
            color=color,
            alpha=0.88,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            txt = (
                f"{val:.1f}%"
                if pct
                else (f"{val:.4f}" if val < 10 else f"{int(val):,}")
            )
            ax.text(
                bar.get_width() + max(abs(bar.get_width()) * 0.01, 0.002),
                bar.get_y() + bar.get_height() / 2,
                txt,
                va="center",
                ha="left",
                fontsize=7.5,
                color="#333333",
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.7)


def _save_comparison_graphs(all_results: List[dict]):
    """
    Generate and save four publication-quality comparison charts.

    Chart 1 — Tracking metrics   : MOTA, IDF1, ADE, RawADE, IDsw, MT, ML
    Chart 2 — Confusion matrix   : TP, FP, FN, TN (absolute counts)
    Chart 3 — Derived metrics    : Precision, Recall, F1, Specificity
    Chart 4 — Radar summary      : All score metrics in one spider chart
    """
    if not all_results:
        logger.warning("[Graphs] No results — skipping chart generation.")
        return

    graph_dir = RESULT_DIR / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    methods = [r["reid_name"] for r in all_results]
    pipeline_tag = "YOLOv11 + Kalman Filter + Re-ID method"

    def col(name: str, default: float = 0.0) -> List[float]:
        return [float(r.get(name, default)) for r in all_results]

    # ── Chart 1: Tracking metrics ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Tracking Metrics — {pipeline_tag}", fontsize=13, fontweight="bold", y=1.01)

    _bar_group(
        axes[0], methods,
        {"MOTA": col("mota"), "IDF1": col("idf1"), "ADE\n(norm.)": col("ade")},
        title="Score Metrics (higher = better)", ylabel="Score  [0–1]",
    )
    _bar_group(
        axes[1], methods,
        {
            "RawADE\n(pixels)": col("raw_ade"),
            "ID Switches":      col("id_switches"),
            "Mostly\nTracked":  col("mt"),
            "Mostly\nLost":     col("ml"),
        },
        title="Count Metrics  (↓ better for IDsw, ML, RawADE)", ylabel="Count / Pixels",
    )
    plt.tight_layout()
    out1 = graph_dir / "1_tracking_metrics.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Graphs] Saved → {out1}")

    # ── Chart 2: Confusion matrix counts ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Detection Confusion Matrix — {pipeline_tag}", fontsize=13, fontweight="bold", y=1.01
    )
    n_methods = len(methods)
    x = np.arange(4)
    bar_w = 0.8 / n_methods
    cm_labels = ["TP", "FP", "FN", "TN"]
    cm_vals = {
        m: [r.get("TP", 0), r.get("FP", 0), r.get("FN", 0), r.get("TN", 0)]
        for m, r in zip(methods, all_results)
    }
    for mi, method in enumerate(methods):
        color = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        offset = (mi - n_methods / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset, cm_vals[method], bar_w * 0.85,
            label=method, color=color, alpha=0.88, edgecolor="white",
        )
        for bar, val in zip(bars, cm_vals[method]):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{int(val):,}",
                    ha="center", va="bottom", fontsize=7, rotation=45, color="#333333",
                )
    ax.set_xticks(x)
    ax.set_xticklabels(cm_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        "TP: correct detections   FP: false alarms   FN: missed   TN: true empty frames",
        fontsize=9, color="#555555", pad=6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, framealpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    out2 = graph_dir / "2_confusion_matrix.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Graphs] Saved → {out2}")

    # ── Chart 3: Derived metrics ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Derived Detection Metrics — {pipeline_tag}", fontsize=13, fontweight="bold", y=1.01
    )
    der_labels = ["Precision", "Recall", "F1-score", "Specificity"]
    der_vals = {
        m: [r.get("precision", 0), r.get("recall", 0), r.get("f1", 0), r.get("specificity", 0)]
        for m, r in zip(methods, all_results)
    }
    for mi, method in enumerate(methods):
        color = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        offset = (mi - n_methods / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset, der_vals[method], bar_w * 0.85,
            label=method, color=color, alpha=0.88, edgecolor="white",
        )
        for bar, val in zip(bars, der_vals[method]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45, color="#333333",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(der_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score  [0–1]", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, framealpha=0.7)
    ax.axhline(1.0, color="#AAAAAA", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    out3 = graph_dir / "3_derived_metrics.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Graphs] Saved → {out3}")

    # ── Chart 4: Radar / spider chart ─────────────────────────────────────────
    try:
        radar_metrics = ["MOTA", "IDF1", "Precision", "Recall", "F1", "Specificity"]
        radar_keys    = ["mota", "idf1", "precision", "recall", "f1", "specificity"]
        N = len(radar_metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]   # Close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.suptitle(f"Overall Score Radar — {pipeline_tag}", fontsize=12, fontweight="bold", y=1.02)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="#888888")
        ax.grid(color="#CCCCCC", linestyle="--", linewidth=0.6)

        for mi, (method, r) in enumerate(zip(methods, all_results)):
            vals = [float(r.get(k, 0)) for k in radar_keys]
            vals += vals[:1]
            color = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
            ax.plot(angles, vals, linewidth=1.8, linestyle="solid", label=method, color=color)
            ax.fill(angles, vals, alpha=0.10, color=color)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.8)
        plt.tight_layout()
        out4 = graph_dir / "4_radar_summary.png"
        fig.savefig(out4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Graphs] Saved → {out4}")
    except Exception as e:
        logger.warning(f"[Graphs] Radar chart skipped: {e}")

    logger.info(f"[Graphs] All charts saved to → {graph_dir}")
    print(f"\n  Charts saved → {graph_dir}")
    print("    1_tracking_metrics.png  — MOTA, IDF1, ADE, IDsw, MT, ML")
    print("    2_confusion_matrix.png  — TP, FP, FN, TN")
    print("    3_derived_metrics.png   — Precision, Recall, F1, Specificity")
    print("    4_radar_summary.png     — All score metrics (radar)")


# ============================================================================
# SECTION 18: INFERENCE VIDEO EXPORT
# (Videos 32, 33, 35, 37, 39 — two runs each: OAMN and OccluTrack)
# ============================================================================

def _run_inference_videos(
    method_name: str,
    reid: BaseReIDExtractor,
    detector: YOLODetector,
):
    """
    Run inference on the 5 benchmark videos using the given Re-ID method.
    Saves annotated .mp4 files named Video{N}_{method}.mp4 to RESULT_DIR.
    """
    logger.info(f"\n[Inference] Running benchmark videos with [{method_name}]")
    tracker = UnifiedTracker(reid)

    # Locate benchmark video folders inside BASE_DIR
    target_folders = [
        folder
        for folder in BASE_DIR.iterdir()
        if folder.is_dir() and folder.name in INFERENCE_VIDEO_NAMES
    ]

    if not target_folders:
        logger.warning(
            f"[Inference] None of the benchmark video folders "
            f"{INFERENCE_VIDEO_NAMES} found in {BASE_DIR}"
        )
        return

    for vp in sorted(target_folders):
        frames = sorted((vp / "images" / "default").glob("*.png"))
        if not frames:
            logger.warning(f"[Inference] No frames found in {vp}")
            continue
        first = cv2.imread(str(frames[0]))
        if first is None:
            continue
        h, w = first.shape[:2]

        out_path = RESULT_DIR / f"Video{vp.name}_{method_name}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (w, h),
        )
        tracker.reset()

        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            dets = detector.detect(img)
            trks = tracker.update(dets, img)
            writer.write(_render_tracks(img, trks))

        writer.release()
        logger.info(f"[Inference] Saved → {out_path}")

    print(f"\n  Inference videos ({method_name}) → {RESULT_DIR}")


# ============================================================================
# SECTION 19: SEED + MAIN
# ============================================================================

def set_seed(seed: int):
    """Ensure reproducible training across CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(RANDOM_SEED)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(); print("=" * 70)
    print("  MOT COMPARISON — OccluTrack  vs  OAMN")
    dev_label = (
        f"GPU · {torch.cuda.get_device_name(0)}" if DEVICE == "cuda"
        else "Apple MPS" if DEVICE == "mps"
        else f"CPU ({os.cpu_count()} threads)"
    )
    print(f"  Device   : {dev_label}")
    print(f"  Methods  : {', '.join(ALL_METHODS)}")
    print(f"  Data     : {TRAIN_DIR}")
    print(f"  Results  : {RESULT_DIR}")
    print(f"  Epochs   : {EPOCHS}  (early-stop patience={EARLY_STOP_PAT})")
    print(f"  Split    : {int(TRAIN_SPLIT * 100)}% train / {100 - int(TRAIN_SPLIT * 100)}% eval")
    print(f"  Inference: videos {sorted(INFERENCE_VIDEO_NAMES)}")
    print("=" * 70)

    # ── Validate paths ────────────────────────────────────────────────────────
    if not TRAIN_DIR.exists():
        sys.exit(f"[ERROR] Training directory not found: {TRAIN_DIR}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Train / eval split ────────────────────────────────────────────────────
    # Benchmark videos (32, 33, 35, 37, 39) are always in the eval set.
    all_folders = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    if not all_folders:
        sys.exit("[ERROR] No video folders found in the data directory.")

    forced_eval = [f for f in all_folders if f.name in INFERENCE_VIDEO_NAMES]
    remaining = [f for f in all_folders if f.name not in INFERENCE_VIDEO_NAMES]
    random.shuffle(remaining)

    target_eval_count = int(len(all_folders) * (1.0 - TRAIN_SPLIT))
    extra_needed = max(0, target_eval_count - len(forced_eval))
    eval_vids = forced_eval + remaining[:extra_needed]
    train_vids = remaining[extra_needed:]

    logger.info(
        f"Total: {len(all_folders)} | Train: {len(train_vids)} | Eval: {len(eval_vids)}"
    )

    # ── Build crop dataset ONCE (shared by all methods that need training) ────
    print(f"\n{'─'*70}")
    print("  STEP 1 / 4 — Building OVIS crop dataset for training")
    print(f"{'─'*70}")
    dataset = OVISCropDataset(
        video_root=TRAIN_DIR,
        allowed_folders={v.name for v in train_vids},
    )
    if dataset.num_classes < 2:
        sys.exit(
            "[ERROR] Fewer than 2 unique soldier identities found. "
            "Check TRAIN_DIR and annotation files."
        )

    # ── Load YOLO detector ONCE ───────────────────────────────────────────────
    detector = YOLODetector()

    # ── Train / evaluate / benchmark each method ──────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEP 2 / 4 — Training & evaluating {len(ALL_METHODS)} Re-ID methods")
    print(f"{'─'*70}")

    all_results: List[dict] = []

    for method_name in ALL_METHODS:
        print()
        print(f"  ╔{'═'*58}╗")
        print(f"  ║  METHOD: {method_name:<50s}║")
        print(f"  ╚{'═'*58}╝")
        set_seed(RANDOM_SEED)

        reid = build_reid(method_name)
        reid.to(reid._device)

        if method_name == "OAMN":
            # ── OAMN: load pretrained weights — DO NOT retrain ─────────────
            oamn_pth = Path(OAMN_WEIGHTS_PATH)
            if oamn_pth.exists():
                try:
                    checkpoint = torch.load(str(oamn_pth), map_location=reid._device)
                    # Handle both raw state_dict and wrapped checkpoint formats
                    state = checkpoint.get("reid_state", checkpoint)
                    reid.load_state_dict(state, strict=False)
                    logger.info(f"  [OAMN] Pretrained weights loaded ← {oamn_pth}")
                except Exception as e:
                    logger.warning(f"  [OAMN] Could not load weights: {e}. Using random init.")
            else:
                logger.warning(
                    f"  [OAMN] Weights not found at {oamn_pth}. "
                    "Using randomly initialised weights."
                )
        else:
            # ── OccluTrack: train on OVIS crops ────────────────────────────
            trainer = ReIDTrainer(
                method_name=method_name,
                reid=reid,
                detector=detector,
                all_videos=train_vids,
                crop_dataset=dataset,
            )
            trainer.run()
            trainer.load_best()

        # ── Evaluate on held-out videos ────────────────────────────────────
        tracker = UnifiedTracker(reid)
        evaluator = ReIDEvaluator(
            method_name=method_name,
            tracker=tracker,
            detector=detector,
            all_videos=eval_vids,
        )
        agg = evaluator.run()

        if agg:
            all_results.append(agg)
            logger.info(
                f"  [{method_name}] DONE  "
                f"MOTA={agg.get('mota', 0):.4f}  "
                f"IDF1={agg.get('idf1', 0):.4f}  "
                f"F1={agg.get('f1', 0):.4f}"
            )
        else:
            logger.warning(f"  [{method_name}] No metrics returned.")

        # ── Run benchmark inference videos ────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  STEP 3 / 4 — Inference on benchmark videos [{method_name}]")
        print(f"{'─'*70}")
        _run_inference_videos(method_name, reid, detector)

        # Free GPU/RAM before processing the next method
        del tracker, evaluator
        if method_name != "OAMN":
            del trainer

    # ── Comparison table + charts ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 4 / 4 — Generating comparison report and charts")
    print(f"{'─'*70}")
    _save_comparison_table(all_results)
    _save_comparison_graphs(all_results)

    # ── Final output summary ──────────────────────────────────────────────────
    print()
    print("  ╔" + "═" * 68 + "╗")
    print("  ║  ALL OUTPUTS SAVED TO:                                           ║")
    print(f"  ║  {str(RESULT_DIR):<68s}║")
    print("  ╠" + "═" * 68 + "╣")
    print("  ║  reid_comparison.csv / .xlsx   — aggregated metrics             ║")
    print("  ║  graphs/                        — 4 comparison charts            ║")
    print("  ║  per_method/<name>/             — per-video metrics + JSON       ║")
    print("  ║  Video{N}_{method}.mp4          — 10 annotated benchmark videos  ║")
    print("  ╚" + "═" * 68 + "╝")
    print()

    logger.info(f"Pipeline complete. All outputs → {RESULT_DIR}")


if __name__ == "__main__":
    main()


# ============================================================================
# PROPOSALS FOR FUTURE IMPROVEMENTS
# (Not implemented — listed here as requested)
# ============================================================================
#
# 1. Hungarian Algorithm instead of Greedy Matching:
#    Replace _greedy_match() with scipy.optimize.linear_sum_assignment
#    for globally optimal bipartite matching.  Greedy matching can miss
#    globally optimal assignments in crowded scenes.
#
# 2. Automatic Mixed Precision (AMP) Training:
#    Wrap the training loop with torch.cuda.amp.autocast() and
#    GradScaler for ~2× GPU throughput with no accuracy loss.
#
# 3. Test-Time Augmentation (TTA) for Re-ID:
#    Average embeddings extracted from horizontally flipped and
#    colour-jittered crops to improve gallery robustness.
#
# 4. ByteTrack-style Two-Stage Association:
#    A second matching pass using only IoU (no appearance) for low-
#    confidence detections can recover occluded tracks missed in stage 1.
#
# 5. Learning Rate Warmup:
#    Add a linear LR warmup phase (first 2 epochs) before cosine decay.
#    Beneficial when fine-tuning a pretrained backbone with a large batch.
#
# 6. ONNX / TorchScript Export:
#    Export the trained Re-ID extractor to ONNX for deployment on edge
#    devices without requiring a full PyTorch runtime.
#
# 7. YAML-based Configuration:
#    Move all constants from SECTION 2 into a config.yaml file and load
#    them at startup, allowing parameter sweeps without editing source.
#
# ============================================================================
