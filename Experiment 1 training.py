"""
================================================================================
  MULTI-OBJECT TRACKING — Re-ID COMPARISON PIPELINE
================================================================================
  PIPELINE ARCHITECTURE (fixed for all 6 combinations):
    ┌─────────────────────┐
    │  Trained YOLO11     │  ← your OVIS-trained model (best.pt)
    │  (Detection)        │    detects objects in every frame
    └────────┬────────────┘
             │ bounding boxes
    ┌────────▼────────────┐
    │  Kalman Filter      │  ← position estimation under occlusion
    │  (State Estimation) │    full occlusion  : track → OCCLUDED state
    │                     │    long-term (>30f): IoU gate bypassed,
    │                     │    appearance-only Re-ID used for recovery
    │                     │    max lifetime    : 150 frames (10 s @ 15 fps)
    └────────┬────────────┘
             │ predicted positions + occlusion state
    ┌────────▼────────────┐
    │  Re-ID Extractor    │  ← ONE of 6 methods trained on OVIS crops
    │  (Identity after    │    OccludedReID · PGFA · HOReID
    │   occlusion)        │    PAT · TransReID · OAMN
    └────────┬────────────┘
             │ 512-d appearance embedding
    ┌────────▼────────────┐
    │  Association        │  ← combined IoU + appearance cost matrix
    │                     │    greedy matching → track ID preserved
    └─────────────────────┘

  6 COMBINATIONS EVALUATED (same YOLO + Kalman, different Re-ID):
    1. Trained YOLO11 + Kalman + OccludedReID
    2. Trained YOLO11 + Kalman + PGFA
    3. Trained YOLO11 + Kalman + HOReID
    4. Trained YOLO11 + Kalman + PAT
    5. Trained YOLO11 + Kalman + TransReID
    6. Trained YOLO11 + Kalman + OAMN

  METRICS PER COMBINATION:
    Tracking : MOTA, IDF1, ADE, RawADE, ID-Switches, MT, ML
    Detection: TP, FP, FN, TN, Precision, Recall, F1, Specificity

  OUTPUTS:
    • reid_comparison.csv          — aggregated metrics for all 6 combinations
    • graphs/1_tracking_metrics    — MOTA, IDF1, ADE, IDsw, MT, ML bar charts
    • graphs/2_confusion_matrix    — TP, FP, FN, TN bar charts
    • graphs/3_derived_metrics     — Precision, Recall, F1, Specificity bar charts
    • graphs/4_radar_summary       — spider chart of all score metrics
    • outputs/<method>/videos/     — 20 annotated .mp4 files per combination
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS  +  FORCE CPU
# ============================================================================

import os, sys, json, cv2, pickle, random, logging, warnings
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

# ── Device selection: CPU + any available GPU (CUDA or Apple MPS) ────────────
import torch

def _select_device() -> str:
    """
    Priority:
      1. CUDA  — NVIDIA discrete or integrated GPU (if driver + CUDA toolkit present)
      2. MPS   — Apple Silicon integrated GPU
      3. CPU   — fallback (also used alongside GPU for non-tensor work)
    For Intel/AMD integrated GPUs on Windows: PyTorch routes through CUDA if the
    driver exposes a CUDA-compatible interface; otherwise CPU is used automatically.
    CPU threads are maximised regardless so non-GPU ops are as fast as possible.
    """
    torch.set_num_threads(os.cpu_count() or 4)   # always maximise CPU threads
    if torch.cuda.is_available():
        dev = "cuda"
        name = torch.cuda.get_device_name(0)
        logger.info(f"[Device] GPU detected — using CUDA: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        logger.info("[Device] Apple MPS (integrated GPU) detected — using MPS")
    else:
        dev = "cpu"

        print("[Device] No GPU detected — using CPU")
    return dev

DEVICE = _select_device()   # resolved once at import time, used everywhere

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import motmetrics as mm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 2: CONFIGURATION  ← only edit this section
# ============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(r"C:\Users\hp\Documents\Senior project 2026\code")
EXPERIMENT_OUT   = BASE_DIR / "reid_experiments_2"
TRAIN_DIR        = BASE_DIR / "Train"
ANNOTATIONS_FILE = BASE_DIR / "annotations_train.json"
CROP_CACHE       = EXPERIMENT_OUT / "crop_cache"
COMPARE_CSV      = EXPERIMENT_OUT / "results" / "reid_comparison.csv"

# ── Methods to run ────────────────────────────────────────────────────────────
ALL_METHODS = ["OccludedReID", "PGFA", "HOReID", "PAT", "TransReID", "OAMN"]

# ── YOLOv11  (trained model — fixed, shared across all 6 Re-ID methods) ──────
# Primary path: your OVIS-trained model.
# The loader tries every path in YOLO_FALLBACK_PATHS in order, then falls back
# to the generic pretrained yolo11n.pt if nothing is found.
YOLO_TRAINED_PATH   = "yolo11_dataset/train/weights/best.pt"
YOLO_FALLBACK_PATHS = [
    "yolo11_dataset/train/weights/best.pt",
    "yolo11_dataset/train/weights/last.pt",
    "yolo11_dataset/train2/weights/best.pt",
    "yolo11_dataset/train3/weights/best.pt",
    "runs/detect/train/weights/best.pt",
    "runs/detect/train/weights/last.pt",
    "yolo11_ovis_trained/train/weights/best.pt",
]
YOLO_PRETRAINED_FALLBACK = "yolo11n.pt"   # last-resort if no trained model found
YOLO_CONF     = 0.50
YOLO_IOU      = 0.45
YOLO_IMG_SIZE = 640
YOLO_VERBOSE  = False

# ── Kalman filter (fixed) ─────────────────────────────────────────────────────
VIDEO_FPS         = 15
LONG_TERM_OCC_SEC = 10
LONG_TERM_MAX_AGE = VIDEO_FPS * LONG_TERM_OCC_SEC    # 150 frames
KF_MIN_HITS       = 3

# ── Re-ID association ─────────────────────────────────────────────────────────
REID_EMBED_DIM    = 512
REID_IOU_GATE     = 0.20   # minimum IoU to activate appearance matching
                            # bypassed for long-term occluded tracks (see below)
REID_MATCH_THRESH = 0.55
W_IOU             = 0.40
W_APP             = 0.60
DET_IOU_THRESH    = 0.50

# ── Occlusion thresholds ──────────────────────────────────────────────────────
# Full occlusion   : object disappears entirely for any duration.
#                    On the first missed frame a CONFIRMED track → OCCLUDED.
#                    Kalman keeps predicting; Re-ID gallery is preserved.
# Short occlusion  : 1 .. REID_LT_OCC_FRAMES frames.
#                    IoU gate applied normally (Kalman bbox still near true pos).
# Long-term occ.   : > REID_LT_OCC_FRAMES frames (default 30 = 2 s at 15 fps).
#                    IoU gate BYPASSED; appearance-only cost used instead.
#                    This is critical: after 30+ frames Kalman drift is large,
#                    so IoU with the reappearing detection is near zero — without
#                    this bypass the Re-ID appearance branch never fires and the
#                    track can never be recovered.
# Track lifetime   : up to LONG_TERM_MAX_AGE = 150 frames (10 s at 15 fps).
#                    This intentionally exceeds the 120-frame requirement.
#                    After 150 consecutive missed frames → LOST (new ID on return).
REID_LT_OCC_FRAMES = 30   # frame threshold: below = IoU gate active,
                           #                 above = appearance-only matching

# ── Training  (CPU-tuned) ─────────────────────────────────────────────────────
EPOCHS           = 15       # was 5
EARLY_STOP_PAT   = 4        # was 3
LEARNING_RATE    = 3e-4
TRIPLET_MARGIN   = 0.3
RANDOM_SEED      = 42
MIN_CROPS_PER_ID = 4
MAX_CROPS_PER_ID = 50       # was 200; keeps epoch time manageable on CPU
VISUAL_VIDEOS    = 20       # output .mp4 videos saved per method
TRAIN_SPLIT      = 0.80     # 80% train / 20% evaluation — no overlap

# ── Per-method architecture hyper-parameters (CPU-friendly sizes) ─────────────
REID_PARAMS: Dict[str, dict] = {
    "OccludedReID": {"n_stripes": 6,          "visibility_thr": 0.40},
    "PGFA":         {"n_parts":   5,          "visibility_thr": 0.30},
    "HOReID":       {"n_parts":   4,          "gcn_layers":     2},
    "PAT":          {"n_part_tokens": 4,      "n_heads": 4,
                     "n_layers":      2,      "d_model": 128},    # d_model 256→128
    "TransReID":    {"patch_size": 16,        "n_heads": 4,       # n_heads 8→4
                     "n_layers":   2,         "d_model": 128,     # n_layers 6→2, d_model 384→128
                     "jigsaw_k":   2},
    "OAMN":         {"mask_thr":  0.50,       "n_branches": 2},
}


# ============================================================================
# SECTION 3: DATA STRUCTURES
# ============================================================================

class OcclusionState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    OCCLUDED  = "occluded"
    LOST      = "lost"


@dataclass
class Detection:
    bbox:       np.ndarray
    confidence: float
    class_id:   int
    frame_id:   int = 0


@dataclass
class Track:
    id:                 int
    bbox:               np.ndarray
    class_id:           int
    confidence:         float
    state:              OcclusionState   = OcclusionState.TENTATIVE
    trajectory:         List[np.ndarray] = field(default_factory=list)
    age:                int  = 0
    hits:               int  = 0
    time_since_update:  int  = 0
    occlusion_duration: int  = 0
    is_kalman_pred:     bool = False


# ============================================================================
# SECTION 4: OVIS CROP DATASET  (built once from GT annotations, then cached)
# ============================================================================

class OVISCropDataset(Dataset):
    """
    Reads every annotation in annotations_train.json.
    For each bounding box, reads the corresponding video frame,
    crops and resizes to 128×256, assigns a contiguous identity label.

    On the first run this scans all videos (slow).
    Result is cached to CROP_CACHE/crop_index.pkl — all 6 method runs
    share the same cache so extraction happens only ONCE.
    """

    def __init__(self, video_root: Path, annotations_file: Path,
                 cache_dir: Optional[Path] = None,
                 allowed_folders: Optional[set] = None):
        """
        allowed_folders : set of folder name strings to include (train split).
                          If None, all videos are used (legacy behaviour).
        """
        import torchvision.transforms as T
        self.cache_dir       = cache_dir
        self.allowed_folders = allowed_folders
        self.samples:    List[Tuple[np.ndarray, int]] = []
        self.num_classes: int = 0
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._load(video_root, annotations_file)

    # ── build / cache ─────────────────────────────────────────────────────────

    def _load(self, video_root: Path, annotations_file: Path):
        # Use a distinct cache file when a train split is applied
        if self.cache_dir:
            suffix = ("_train80" if self.allowed_folders is not None
                      else "_all")
            cache_file = self.cache_dir / f"crop_index{suffix}.pkl"
        else:
            cache_file = None

        if cache_file and cache_file.exists():
            logger.info(f"[Dataset] Loading crop cache ← {cache_file}")
            with open(cache_file, "rb") as f:
                self.samples, self.num_classes = pickle.load(f)
            logger.info(f"[Dataset] {len(self.samples)} crops / "
                        f"{self.num_classes} identities  (cached)")
            return

        split_desc = (f"{len(self.allowed_folders)} train folders"
                      if self.allowed_folders is not None else "all folders")
        logger.info(f"[Dataset] Extracting crops from OVIS annotations "
                    f"({split_desc}) — first time only, will be cached…")
        with open(annotations_file) as f:
            ann_data = json.load(f)

        vid_list = ann_data.get("videos", [])
        vid_map: Dict[int, str] = {}
        for v in vid_list:
            names = v.get("file_names", [])
            if names:
                folder = names[0].replace("\\", "/").split("/")[0]
                vid_map[v["id"]] = folder

        identity_crops: Dict[int, List[np.ndarray]] = defaultdict(list)

        for ann in ann_data.get("annotations", []):
            vid_id = ann["video_id"]
            folder = vid_map.get(vid_id)
            if folder is None:
                continue
            # ── 80/20 split filter ─────────────────────────────────────────
            if self.allowed_folders is not None and folder not in self.allowed_folders:
                continue
            # ──────────────────────────────────────────────────────────────
            vid_path = video_root / folder
            if not vid_path.is_dir():
                continue

            track_id  = (ann.get("instance_id") or
                         ann.get("track_id")    or
                         ann["id"])
            global_id = vid_id * 100_000 + track_id

            frame_files = sorted(vid_path.glob("*.jpg"))
            if not frame_files:
                continue

            for fi, bbox in enumerate(ann.get("bboxes", [])):
                if bbox is None or len(bbox) != 4:
                    continue
                x, y, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    continue
                if fi >= len(frame_files):
                    continue
                if len(identity_crops[global_id]) >= MAX_CROPS_PER_ID:
                    continue

                img = cv2.imread(str(frame_files[fi]))
                if img is None:
                    continue

                x1 = max(0, int(x));         y1 = max(0, int(y))
                x2 = min(img.shape[1], int(x + bw))
                y2 = min(img.shape[0], int(y + bh))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = cv2.resize(img[y1:y2, x1:x2], (128, 256))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                identity_crops[global_id].append(crop)

        valid_ids  = sorted([gid for gid, crops in identity_crops.items()
                              if len(crops) >= MIN_CROPS_PER_ID])
        label_map  = {gid: idx for idx, gid in enumerate(valid_ids)}
        for gid in valid_ids:
            for crop in identity_crops[gid]:
                self.samples.append((crop, label_map[gid]))

        self.num_classes = len(valid_ids)
        logger.info(f"[Dataset] {len(self.samples)} crops  |  "
                    f"{self.num_classes} identities")

        if cache_file:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump((self.samples, self.num_classes), f)
            logger.info(f"[Dataset] Crops cached → {cache_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, label = self.samples[idx]
        return self.transform(crop), label


# ============================================================================
# SECTION 5: PK BATCH SAMPLER  (P=4 for CPU)
# ============================================================================

class TripletBatchSampler:
    """
    Samples P identities × K crops per batch.
    Every batch is guaranteed to contain valid anchor-positive-negative triplets.
    P is reduced to 4 (was 8) for faster CPU iteration.
    """

    def __init__(self, labels: List[int], P: int = 4, K: int = 4):
        self.P = P; self.K = K
        self.lbl2idx: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.lbl2idx[lbl].append(i)
        self.valid = [l for l, idxs in self.lbl2idx.items() if len(idxs) >= K]

    def __iter__(self):
        lbls = self.valid.copy(); random.shuffle(lbls)
        for i in range(0, len(lbls) - self.P + 1, self.P):
            batch = []
            for lbl in lbls[i:i + self.P]:
                batch.extend(random.sample(self.lbl2idx[lbl],
                                           min(self.K, len(self.lbl2idx[lbl]))))
            yield batch

    def __len__(self):
        return max(1, len(self.valid) // self.P)


# ============================================================================
# SECTION 6: LOSS FUNCTIONS
# ============================================================================

class TripletLoss(nn.Module):
    """Batch-hard online triplet loss — mines hardest pos/neg in every batch."""

    def __init__(self, margin: float = TRIPLET_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist     = torch.cdist(emb, emb, p=2)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_dist = (dist * mask_pos).max(dim=1)[0]
        neg_dist = (dist + 1e6 * mask_pos).min(dim=1)[0]
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class IDClassifier(nn.Module):
    """Identity classification head — used only during training."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


# ============================================================================
# SECTION 7: YOLO DETECTOR  — loads your trained OVIS model
# ============================================================================

def _load_trained_yolo() -> YOLO:
    """
    Tries to load the OVIS-trained YOLO model in this order:
      1. YOLO_TRAINED_PATH  (primary)
      2. Each entry in YOLO_FALLBACK_PATHS
      3. YOLO_PRETRAINED_FALLBACK  (generic yolo11n.pt — last resort)

    Mirrors the load_trained_yolo() logic from the OccluTrack pipeline so
    both scripts use exactly the same detector.
    """
    # 1. Primary path
    if Path(YOLO_TRAINED_PATH).exists():
        logger.info(f"[YOLO] ✓ Trained model found: {YOLO_TRAINED_PATH}")
        model = YOLO(YOLO_TRAINED_PATH)
        model.to(DEVICE)
        logger.info(f"[YOLO] Classes ({len(model.names)}): "
                    + ", ".join(f"{k}:{v}" for k,v in model.names.items()))
        return model

    logger.warning(f"[YOLO] ⚠ Primary path not found: {YOLO_TRAINED_PATH}")
    logger.info("[YOLO] Checking fallback paths…")

    # 2. Fallback paths
    for alt in YOLO_FALLBACK_PATHS:
        if alt != YOLO_TRAINED_PATH and Path(alt).exists():
            logger.info(f"[YOLO] ✓ Trained model found at fallback: {alt}")
            model = YOLO(alt)
            model.to(DEVICE)
            logger.info(f"[YOLO] Classes ({len(model.names)}): "
                        + ", ".join(f"{k}:{v}" for k,v in model.names.items()))
            return model

    # 3. Last resort
    logger.warning("[YOLO] ❌ No trained model found in any path.")
    logger.warning(f"[YOLO] ⚠ Falling back to pretrained: {YOLO_PRETRAINED_FALLBACK}")
    logger.warning("[YOLO]   Results will be weaker — use your OVIS-trained model "
                   "for best performance.")
    model = YOLO(YOLO_PRETRAINED_FALLBACK)
    model.to(DEVICE)
    return model


class YOLODetector:
    def __init__(self):
        self.device = DEVICE
        self.model  = _load_trained_yolo()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                             imgsz=YOLO_IMG_SIZE, verbose=YOLO_VERBOSE)
        dets: List[Detection] = []
        for r in results:
            for i in range(len(r.boxes)):
                dets.append(Detection(
                    bbox       = r.boxes.xyxy[i].cpu().numpy(),
                    confidence = float(r.boxes.conf[i].cpu().numpy()),
                    class_id   = int(r.boxes.cls[i].cpu().numpy()),
                ))
        return dets


# ============================================================================
# SECTION 8: KALMAN PREDICTOR  (fixed)
# ============================================================================

class KalmanPredictor:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]], dtype=float)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], dtype=float)
        self.kf.R    *= 10.0
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P    *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self._base_Q  = self.kf.Q.copy()
        self._occ_frames = 0

    def initialize(self, b: np.ndarray):
        z = self._to_z(b); self.kf.x[:4] = z; self.kf.update(z)

    def update(self, b: np.ndarray):
        self.kf.update(self._to_z(b))
        self._occ_frames = 0; self.kf.Q = self._base_Q.copy()

    def predict(self, is_occluded: bool = False) -> np.ndarray:
        if is_occluded:
            self._occ_frames += 1
            factor = (1.0 + 2.0 * min(self._occ_frames, LONG_TERM_MAX_AGE)
                      / LONG_TERM_MAX_AGE)
            self.kf.Q = self._base_Q * factor
        self.kf.predict()
        return self._to_bbox(self.kf.x)

    @staticmethod
    def _to_z(b):
        w = b[2]-b[0]; h = b[3]-b[1]
        return np.array([[b[0]+w/2], [b[1]+h/2],
                          [w*h], [w/max(float(h),1e-6)]], dtype=float)

    @staticmethod
    def _to_bbox(x):
        s, r = float(x[2]), float(x[3])
        if s > 0 and r > 0:
            w = np.sqrt(s*r); h = s/w
        else:
            w = h = 0.0
        cx, cy = float(x[0]), float(x[1])
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


# ============================================================================
# SECTION 9: TRAINABLE RE-ID EXTRACTORS
# ============================================================================
#
#  Every extractor is an nn.Module.
#  Backbone = MobileNetV2 pretrained on ImageNet.
#            Last 5 blocks are UNFROZEN and fine-tuned.
#            Early blocks stay frozen (transfer learning).
#  Head     = Method-specific LEARNABLE layers → REID_EMBED_DIM.
#  Training = Triplet loss (batch-hard) + CrossEntropy (ID classification).
#
#  FIXES vs original:
#    • TransReID: now uses MobileNetV2 backbone (was training ViT from scratch)
#    • All 6 methods share the same pretrained backbone + fine-tuning strategy
#    • device resolved at startup via _select_device() — uses GPU if available
#
# ============================================================================

class BaseReIDExtractor(nn.Module):
    BACKBONE_DIM = 1280   # MobileNetV2 final feature channels

    def __init__(self, params: dict):
        super().__init__()
        self.params  = params
        self._device = DEVICE   # cuda / mps / cpu — resolved at startup
        self._transform_eval = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
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
        import torchvision.models as tvm, torchvision.transforms as T
        mn = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        bb = nn.Sequential(*list(mn.features.children()))
        for child in list(bb.children())[:-5]:
            for p in child.parameters(): p.requires_grad = False
        for child in list(bb.children())[-5:]:
            for p in child.parameters(): p.requires_grad = True
        self._transform_eval = T.Compose([
            T.ToPILImage(), T.Resize((256, 128)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        return bb

    def _crop_frame(self, frame: np.ndarray, bbox: np.ndarray,
                    w: int = 128, h: int = 256) -> Optional[np.ndarray]:
        x1,y1,x2,y2 = map(int, bbox)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
        if x2<=x1 or y2<=y1: return None
        return cv2.resize(frame[y1:y2,x1:x2], (w,h))

    def _gradient_visibility(self, gray: np.ndarray) -> float:
        if gray.size == 0: return 0.0
        return min(1.0, float(np.std(cv2.Laplacian(gray,cv2.CV_64F)))/255.0*8.0)

    def _gray_batch(self, x: torch.Tensor) -> List[np.ndarray]:
        mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
        imgs = np.clip((x.detach().cpu().permute(0,2,3,1).numpy()*std+mean)*255,
                       0, 255).astype(np.uint8)
        return [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs]


# ── 9a. OccludedReID ──────────────────────────────────────────────────────────

class OccludedReIDExtractor(BaseReIDExtractor):
    """
    Zhuo et al. ICME 2018 — horizontal stripe visibility weighting.
    Backbone last-5-blocks fine-tuned via backprop.
    Head (Linear→BN→ReLU→Dropout→Linear) trained end-to-end.
    """
    def __init__(self, params):
        super().__init__(params)
        self.n_stripes      = params.get("n_stripes", 6)
        self.visibility_thr = params.get("visibility_thr", 0.40)
        self.backbone = self._load_backbone().to(self._device)
        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)
        logger.info(f"[OccludedReID] {self.n_stripes}-stripe trainable extractor ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        B, C, H, W = fmap.shape
        grays = self._gray_batch(x)
        sh = max(1, H // self.n_stripes)
        c_sh = max(1, x.shape[2] // self.n_stripes)
        agg   = torch.zeros(B, C, device=self._device)
        w_sum = torch.zeros(B,    device=self._device)
        for i in range(self.n_stripes):
            sf = fmap[:, :, i*sh:(i+1)*sh, :]
            if sf.shape[2] == 0: continue
            sf_pool = F.adaptive_avg_pool2d(sf, 1).squeeze(-1).squeeze(-1)
            vis = torch.tensor(
                [self._gradient_visibility(g[i*c_sh:(i+1)*c_sh,:]) for g in grays],
                device=self._device)
            mask = (vis >= self.visibility_thr).float()
            agg   += sf_pool * (vis * mask).unsqueeze(1)
            w_sum += vis * mask
        gpool = F.adaptive_avg_pool2d(fmap, 1).squeeze(-1).squeeze(-1)
        agg = torch.where((w_sum == 0).unsqueeze(1), gpool,
                          agg / (w_sum.unsqueeze(1) + 1e-9))
        return F.normalize(self.head(agg), dim=1)


# ── 9b. PGFA ──────────────────────────────────────────────────────────────────

class PGFAExtractor(BaseReIDExtractor):
    """
    Miao et al. ICCV 2019 — pose-guided feature alignment.
    Per-part projection layers trained end-to-end.
    """
    def __init__(self, params):
        super().__init__(params)
        self.n_parts        = params.get("n_parts", 5)
        self.visibility_thr = params.get("visibility_thr", 0.30)
        self.backbone = self._load_backbone().to(self._device)
        part_dim = self.BACKBONE_DIM // self.n_parts
        self.part_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(self.BACKBONE_DIM, part_dim), nn.ReLU(inplace=True))
            for _ in range(self.n_parts)
        ]).to(self._device)
        self.final_head = nn.Sequential(
            nn.Linear(part_dim * self.n_parts, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)
        logger.info(f"[PGFA] {self.n_parts}-part trainable extractor ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        B, C, H, W = fmap.shape
        grays = self._gray_batch(x)
        sh    = max(1, H // self.n_parts)
        c_sh  = max(1, x.shape[2] // self.n_parts)
        p_dim = self.BACKBONE_DIM // self.n_parts
        parts = []
        for i in range(self.n_parts):
            sf = fmap[:, :, i*sh:(i+1)*sh, :]
            if sf.shape[2] == 0:
                parts.append(torch.zeros(B, p_dim, device=self._device)); continue
            pf  = F.adaptive_avg_pool2d(sf, 1).squeeze(-1).squeeze(-1)
            emb = self.part_proj[i](pf)
            vis = torch.tensor(
                [self._gradient_visibility(g[i*c_sh:(i+1)*c_sh,:]) for g in grays],
                device=self._device).unsqueeze(1)
            emb = emb * (vis >= self.visibility_thr).float()
            parts.append(emb)
        return F.normalize(self.final_head(torch.cat(parts, dim=1)), dim=1)


# ── 9c. HOReID ────────────────────────────────────────────────────────────────

class HOReIDExtractor(BaseReIDExtractor):
    """
    Wang et al. CVPR 2020 — high-order GCN over body-part nodes.
    Adjacency logits and GCN weight matrices are learnable parameters.
    """
    def __init__(self, params):
        super().__init__(params)
        self.n_parts    = params.get("n_parts",    4)
        self.gcn_layers = params.get("gcn_layers", 2)
        self.backbone   = self._load_backbone().to(self._device)
        self.adj_logits = nn.Parameter(torch.ones(self.n_parts, self.n_parts))
        self.gcn_w = nn.ParameterList([
            nn.Parameter(torch.randn(self.BACKBONE_DIM, self.BACKBONE_DIM) * 0.01)
            for _ in range(self.gcn_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM * self.n_parts, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)
        logger.info(f"[HOReID] {self.n_parts}-node {self.gcn_layers}-layer "
                    "trainable GCN ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        B, C, H, W = fmap.shape
        sh = max(1, H // self.n_parts)
        A  = torch.softmax(self.adj_logits, dim=1)
        nodes = []
        for i in range(self.n_parts):
            sf = fmap[:, :, i*sh:(i+1)*sh, :]
            nodes.append(F.adaptive_avg_pool2d(sf, 1).view(B, C)
                         if sf.shape[2] > 0
                         else torch.zeros(B, C, device=self._device))
        X = torch.stack(nodes, dim=1)   # [B, n_parts, C]
        for W_gcn in self.gcn_w:
            X_nb = torch.einsum("pq,bqc->bpc", A, X)
            X_nb = F.relu(torch.einsum("bpc,cd->bpd", X_nb, W_gcn))
            X    = X + X_nb
        return F.normalize(self.head(X.reshape(B, -1)), dim=1)


# ── 9d. PAT ───────────────────────────────────────────────────────────────────

class PATExtractor(BaseReIDExtractor):
    """
    Li et al. 2021 — learnable part tokens in a Transformer encoder.
    Part token parameters are learnable (nn.Parameter).
    All components trained end-to-end on OVIS crops.
    """
    def __init__(self, params):
        super().__init__(params)
        self.n_part_tokens = params.get("n_part_tokens", 4)
        self.n_heads       = params.get("n_heads", 4)
        self.n_layers      = params.get("n_layers", 2)
        dm = params.get("d_model", 128)
        self.backbone    = self._load_backbone().to(self._device)
        self.feat_proj   = nn.Linear(self.BACKBONE_DIM, dm).to(self._device)
        self.part_tokens = nn.Parameter(torch.randn(1, self.n_part_tokens, dm) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dm, nhead=self.n_heads, dim_feedforward=dm*4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, self.n_layers).to(self._device)
        self.head = nn.Sequential(
            nn.Linear(dm * self.n_part_tokens, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)
        logger.info(f"[PAT] {self.n_part_tokens}-part transformer trainable ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap    = self.backbone(x)
        B, C, H, W = fmap.shape
        patches = self.feat_proj(fmap.view(B, C, -1).permute(0,2,1))
        tok     = self.part_tokens.expand(B, -1, -1)
        enc     = self.transformer(torch.cat([tok, patches], dim=1))
        flat    = enc[:, :self.n_part_tokens, :].reshape(B, -1)
        return F.normalize(self.head(flat), dim=1)


# ── 9e. TransReID — FIXED: now uses MobileNetV2 backbone ─────────────────────

class TransReIDExtractor(BaseReIDExtractor):
    """
    He et al. ICCV 2021 — ViT with Jigsaw Patch Module.

    FIX vs original code: MobileNetV2 backbone (pretrained ImageNet, last 5
    blocks unfrozen) replaces the original from-scratch ViT patch embedding.
    This makes the comparison fair — all 6 methods start from the same
    pretrained backbone. The Jigsaw shift and Transformer encoder layers
    are unchanged and still learnable.
    """
    def __init__(self, params):
        super().__init__(params)
        self.n_heads  = params.get("n_heads",  4)
        self.n_layers = params.get("n_layers", 2)
        dm            = params.get("d_model", 128)
        self.jigsaw_k = params.get("jigsaw_k", 2)

        # ✅ Pretrained MobileNetV2 backbone — same as all other 5 methods
        self.backbone  = self._load_backbone().to(self._device)

        # Project backbone features into Transformer model dimension
        self.feat_proj = nn.Linear(self.BACKBONE_DIM, dm).to(self._device)

        # Learnable [CLS] token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        # 50 positions > any realistic H'×W' from 256×128 image → safe clip below
        self.pos_emb   = nn.Parameter(torch.randn(1, 50, dm) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dm, nhead=self.n_heads, dim_feedforward=dm * 4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, self.n_layers).to(self._device)

        self.head = nn.Sequential(
            nn.Linear(dm, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)

        logger.info(f"[TransReID] MobileNetV2+Transformer d={dm} "
                    f"h={self.n_heads} l={self.n_layers} trainable ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap    = self.backbone(x)                                  # [B,1280,H',W']
        B, C, H, W = fmap.shape
        # Project spatial features → [B, H'*W', dm]
        patches = self.feat_proj(fmap.view(B, C, -1).permute(0, 2, 1))
        # Jigsaw Patch Module: cyclic roll for spatial robustness
        patches = torch.roll(patches, shifts=self.jigsaw_k, dims=1)
        cls     = self.cls_token.expand(B, -1, -1)
        seq     = torch.cat([cls, patches], dim=1)                  # [B, 1+H'W', dm]
        n       = seq.shape[1]
        seq     = seq + self.pos_emb[:, :n, :]                      # clip pos_emb to seq len
        out     = self.transformer(seq)
        return F.normalize(self.head(out[:, 0, :]), dim=1)          # use [CLS] token


# ── 9f. OAMN ──────────────────────────────────────────────────────────────────

class OAMNExtractor(BaseReIDExtractor):
    """
    Chen et al. ACM MM 2021 — occlusion-aware mask network.
    Spatial mask predictor and confidence estimator are both trained Conv layers.
    Fuses visible-only + full-image branches end-to-end.
    """
    def __init__(self, params):
        super().__init__(params)
        self.mask_thr  = params.get("mask_thr", 0.50)
        self.backbone  = self._load_backbone().to(self._device)
        self.spatial_mask = nn.Sequential(
            nn.Conv2d(self.BACKBONE_DIM, 64, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1), nn.Sigmoid(),
        ).to(self._device)
        self.mask_conf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.BACKBONE_DIM, 1), nn.Sigmoid(),
        ).to(self._device)
        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM * 2, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)
        logger.info("[OAMN] Two-branch trainable extractor ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap      = self.backbone(x)
        B         = x.size(0)
        full_feat = F.adaptive_avg_pool2d(fmap, 1).view(B, -1)
        vis_feat  = F.adaptive_avg_pool2d(
            fmap * (self.spatial_mask(fmap) >= self.mask_thr).float(), 1).view(B, -1)
        conf      = self.mask_conf(fmap)
        fused     = torch.cat([full_feat*(1-conf), vis_feat*conf], dim=1)
        return F.normalize(self.head(fused), dim=1)


# ── Factory ───────────────────────────────────────────────────────────────────

_REGISTRY = {
    "OccludedReID": OccludedReIDExtractor,
    "PGFA":         PGFAExtractor,
    "HOReID":       HOReIDExtractor,
    "PAT":          PATExtractor,
    "TransReID":    TransReIDExtractor,
    "OAMN":         OAMNExtractor,
}

def build_reid(name: str) -> BaseReIDExtractor:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[name](REID_PARAMS[name])


# ============================================================================
# SECTION 10: UNIFIED TRACKER
# ============================================================================

def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0],b2[0]); y1 = max(b1[1],b2[1])
    x2 = min(b1[2],b2[2]); y2 = min(b1[3],b2[3])
    inter = max(0.,x2-x1) * max(0.,y2-y1)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter/u if u > 0 else 0.0


def _greedy_match(cost: np.ndarray, threshold: float):
    matched, used_r, used_c = [], set(), set()
    for val, r, c in sorted(
            [(cost[r,c],r,c) for r in range(cost.shape[0])
             for c in range(cost.shape[1])]):
        if val > threshold: break
        if r not in used_r and c not in used_c:
            matched.append((r,c)); used_r.add(r); used_c.add(c)
    return (matched,
            [r for r in range(cost.shape[0]) if r not in used_r],
            [c for c in range(cost.shape[1]) if c not in used_c])


class UnifiedTracker:
    def __init__(self, reid: BaseReIDExtractor):
        self.reid = reid
        self._tracks:  List[Track]                 = []
        self._kf:      Dict[int, KalmanPredictor]  = {}
        self._gallery: Dict[int, List[np.ndarray]] = {}
        self._next_id  = 1

    def reset(self):
        self._tracks.clear(); self._kf.clear()
        self._gallery.clear(); self._next_id = 1

    def update(self, detections: List[Detection],
               frame: np.ndarray) -> List[Track]:
        for t in self._tracks:
            is_occ = (t.state == OcclusionState.OCCLUDED)
            t.bbox = self._kf[t.id].predict(is_occluded=is_occ)
            t.time_since_update += 1; t.age += 1
            if is_occ: t.occlusion_duration += 1

        det_feats = [self.reid.extract(frame, d.bbox) for d in detections]
        active    = [t for t in self._tracks if t.state != OcclusionState.LOST]

        if active and detections:
            cost = np.ones((len(detections), len(active)))
            for di, det in enumerate(detections):
                for ti, trk in enumerate(active):
                    iv = _iou(det.bbox, trk.bbox)
                    ic = 1.0 - iv

                    # Long-term occlusion: Kalman has drifted far, so IoU with
                    # the reappearing detection can be near zero even when it IS
                    # the correct track.  Bypass the IoU gate and use appearance-
                    # only cost so Re-ID still fires and can recover the track.
                    long_term = (trk.state == OcclusionState.OCCLUDED and
                                 trk.occlusion_duration > REID_LT_OCC_FRAMES)

                    use_appearance = (trk.id in self._gallery and
                                      (iv >= REID_IOU_GATE or long_term))

                    if use_appearance:
                        gal = np.stack(self._gallery[trk.id][-50:])
                        qf  = det_feats[di]
                        nq  = np.linalg.norm(qf)
                        ng  = np.linalg.norm(gal, axis=1)
                        if nq > 0 and np.any(ng > 0):
                            ac = 1.0 - float((gal @ qf / (ng * nq + 1e-9)).max())
                        else:
                            ac = ic
                        # For long-term occlusion weight appearance more heavily
                        # since IoU is unreliable after heavy Kalman drift
                        if long_term:
                            cost[di, ti] = 0.1 * ic + 0.9 * ac
                        else:
                            cost[di, ti] = W_IOU * ic + W_APP * ac
                    else:
                        cost[di, ti] = W_IOU * ic + W_APP * ic   # = ic
            matched, unm_d, unm_t = _greedy_match(cost, 1.0-REID_MATCH_THRESH)
        else:
            matched=[]; unm_d=list(range(len(detections)))
            unm_t=list(range(len(active)))

        for di, ti in matched:
            t = active[ti]
            t.bbox=detections[di].bbox; t.confidence=detections[di].confidence
            t.hits+=1; t.time_since_update=0; t.occlusion_duration=0
            t.is_kalman_pred=False
            t.trajectory.append(detections[di].bbox.copy())
            self._kf[t.id].update(detections[di].bbox)
            self._gallery.setdefault(t.id,[]).append(det_feats[di])
            if t.state == OcclusionState.OCCLUDED:
                t.state = OcclusionState.CONFIRMED
            elif t.state == OcclusionState.TENTATIVE and t.hits >= KF_MIN_HITS:
                t.state = OcclusionState.CONFIRMED

        for ti in unm_t:
            t = active[ti]
            if   t.state == OcclusionState.TENTATIVE:  t.state = OcclusionState.LOST
            elif t.state == OcclusionState.CONFIRMED:
                t.state = OcclusionState.OCCLUDED; t.is_kalman_pred = True
            elif t.state == OcclusionState.OCCLUDED:
                if t.occlusion_duration >= LONG_TERM_MAX_AGE:
                    t.state = OcclusionState.LOST
                else: t.is_kalman_pred = True

        for di in unm_d:
            det=detections[di]; tid=self._next_id; self._next_id+=1
            kf=KalmanPredictor(); kf.initialize(det.bbox)
            self._tracks.append(Track(
                id=tid, bbox=det.bbox, class_id=det.class_id,
                confidence=det.confidence, state=OcclusionState.TENTATIVE,
                trajectory=[det.bbox.copy()], hits=1))
            self._kf[tid]=kf; self._gallery[tid]=[det_feats[di]]

        self._tracks=[t for t in self._tracks if t.state != OcclusionState.LOST]
        return [t for t in self._tracks
                if t.state in (OcclusionState.CONFIRMED, OcclusionState.OCCLUDED)]


# ============================================================================
# SECTION 11: GROUND-TRUTH HELPERS
# ============================================================================

def _build_gt(annotations: dict, video_name: str):
    video_info = None
    for v in annotations.get("videos", []):
        folder = v.get("file_names",[""])[0].replace("\\","/").split("/")[0]
        if folder == video_name: video_info = v; break
    if video_info is None: return None, {}
    vid_id  = video_info["id"]
    vid_len = video_info.get("length", len(video_info.get("file_names",[])))
    gt: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations.get("annotations", []):
        if ann["video_id"] != vid_id: continue
        tid = ann.get("instance_id") or ann.get("track_id") or ann["id"]
        for fi, bbox in enumerate(ann.get("bboxes",[])):
            if fi >= vid_len or not bbox or len(bbox) != 4: continue
            x,y,bw,bh = bbox
            if bw > 0 and bh > 0:
                gt[fi].append({"id": tid, "bbox": [x,y,x+bw,y+bh]})
    return video_info, gt


# ============================================================================
# SECTION 12: METRICS  — TP / TN / FP / FN are computed at every level
# ============================================================================

def _compute_detection_metrics(predictions, gt_by_frame, vid_len,
                                iou_thr=DET_IOU_THRESH):
    """
    Returns a dict with TP, FP, FN, TN per video.

    TP : detected bbox matches a GT bbox (IoU ≥ iou_thr)
    FP : detected bbox has no matching GT bbox
    FN : GT bbox was not detected
    TN : frame had no GT objects AND no detections (true empty frame)
    """
    TP=FP=FN=TN=0
    pred_lut = {p["frame_id"]: p for p in predictions}
    for fid in range(vid_len):
        gt_objs  = gt_by_frame.get(fid,[])
        pi       = pred_lut.get(fid)
        pred_bbs = [np.array(t[1]) for t in pi["tracks"]] if pi else []
        gt_bbs   = [np.array(g["bbox"]) for g in gt_objs]
        ng,np_   = len(gt_bbs), len(pred_bbs)
        if ng==0 and np_==0: TN+=1; continue
        if ng==0: FP+=np_; continue
        if np_==0: FN+=ng; continue
        mg=set(); mp=set()
        for pi2,gi,iv in sorted(
                [(pi2,gi,_iou(pred_bbs[pi2],gt_bbs[gi]))
                 for pi2 in range(np_) for gi in range(ng)],
                key=lambda x:-x[2]):
            if pi2 in mp or gi in mg: continue
            if iv >= iou_thr: TP+=1; mp.add(pi2); mg.add(gi)
        FP+=np_-len(mp); FN+=ng-len(mg)
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN}


def _evaluate_video(predictions, annotations, video_name, img_w, img_h):
    video_info, gt_by_frame = _build_gt(annotations, video_name)
    if video_info is None: return None
    vid_len  = video_info.get("length", 0)
    img_diag = np.sqrt(img_w**2+img_h**2) if (img_w and img_h) else 1.0
    acc = mm.MOTAccumulator(auto_id=True)
    pred_trajs: Dict[int,list] = defaultdict(list)
    gt_trajs:   Dict[int,list] = defaultdict(list)
    for pf in predictions:
        fid = pf["frame_id"]
        if fid >= vid_len: continue
        gt_objs = gt_by_frame.get(fid,[])
        gt_ids=[o["id"] for o in gt_objs]; gt_bbs=[o["bbox"] for o in gt_objs]
        p_ids=[t[0] for t in pf["tracks"]]; p_bbs=[t[1] for t in pf["tracks"]]
        for pid,pb in zip(p_ids,p_bbs): pred_trajs[pid].append(pb)
        for gid,gb in zip(gt_ids,gt_bbs): gt_trajs[gid].append(gb)
        dist = (np.array([[1.-_iou(np.array(gb),np.array(pb))
                            for pb in p_bbs] for gb in gt_bbs])
                if gt_bbs and p_bbs else np.empty((len(gt_bbs),len(p_bbs))))
        acc.update(gt_ids, p_ids, dist)
    mh  = mm.metrics.create()
    smr = mh.compute(acc,
                     metrics=["mota","idf1","num_switches",
                               "mostly_tracked","mostly_lost"],
                     name="acc")
    def _g(col, d=0.0):
        v = smr[col].values[0] if col in smr.columns else d
        return float(v) if not np.isnan(v) else d
    errs=[]
    for pt in pred_trajs.values():
        for gt in gt_trajs.values():
            n=min(len(pt),len(gt))
            if n==0: continue
            pc=np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in pt[:n]])
            gc=np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in gt[:n]])
            errs.extend(np.linalg.norm(pc-gc,axis=1))
    raw_ade=float(np.mean(errs)) if errs else 0.0
    ade=raw_ade/img_diag if img_diag>0 else raw_ade
    cfm=_compute_detection_metrics(predictions, gt_by_frame, vid_len)
    tp,fp,fn,tn=cfm["TP"],cfm["FP"],cfm["FN"],cfm["TN"]
    prec=tp/(tp+fp) if (tp+fp)>0 else 0.
    rec =tp/(tp+fn) if (tp+fn)>0 else 0.
    f1  =2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.
    spec=tn/(tn+fp) if (tn+fp)>0 else 0.
    return {
        "mota": max(-1.,_g("mota")), "idf1": max(0.,_g("idf1")),
        "ade": ade, "raw_ade": raw_ade,
        "id_switches": int(_g("num_switches")),
        "mt": int(_g("mostly_tracked")), "ml": int(_g("mostly_lost")),
        "TP":tp, "FP":fp, "FN":fn, "TN":tn,
        "precision":prec,"recall":rec,"f1":f1,"specificity":spec,
    }


# ============================================================================
# SECTION 13: TRAINING LOOP
# ============================================================================

class ReIDTrainer:
    """
    Trains one Re-ID model on OVIS crops with combined metric learning loss.

    Per-epoch steps:
      1. Forward all PK batches → embeddings
      2. L = TripletLoss(batch-hard) + 0.5 * CrossEntropyLoss(ID)
      3. Backward + Adam + gradient clipping
      4. Val MOTA on 5 random videos (actual tracking, no cheating)
         — reduced from 30 for CPU speed
      5. Save best.pth on improvement; early-stop after patience epochs
    """
    LAMBDA_TRIP = 1.0
    LAMBDA_CE   = 0.5

    def __init__(self, method_name: str,
                 reid: BaseReIDExtractor,
                 detector: YOLODetector,
                 all_videos: List[Path],
                 annotations: dict,
                 crop_dataset: OVISCropDataset):
        self.name         = method_name
        self.reid         = reid
        self.detector     = detector
        self.all_videos   = all_videos
        self.annotations  = annotations
        self.dataset      = crop_dataset
        self.device       = reid._device
        self.weights_dir  = EXPERIMENT_OUT / "weights" / method_name
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes  = crop_dataset.num_classes
        self.classifier   = IDClassifier(REID_EMBED_DIM,
                                         self.num_classes).to(self.device)
        self.optimizer    = optim.Adam(
            list(reid.parameters()) + list(self.classifier.parameters()),
            lr=LEARNING_RATE, weight_decay=5e-4)
        self.scheduler    = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=1e-6)
        self.triplet_loss = TripletLoss(TRIPLET_MARGIN)
        self.ce_loss      = nn.CrossEntropyLoss()

    def run(self) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  TRAINING [{self.name}]  "
                    f"{len(self.dataset)} crops / {self.num_classes} IDs  "
                    f"/ {len(self.all_videos)} videos (all used for crop extraction)")
        logger.info(f"{'='*60}")
        labels  = [self.dataset.samples[i][1] for i in range(len(self.dataset))]
        sampler = TripletBatchSampler(labels, P=4, K=4)  # P=4 for CPU (was 8)
        loader  = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0)
        best_mota=float("-inf"); no_improve=0; history=[]
        for epoch in range(1, EPOCHS+1):
            loss    = self._train_epoch(loader)
            val_m   = self._val_mota(epoch)
            self.scheduler.step()
            history.append({"epoch":epoch,"loss":loss,"val_mota":val_m})
            logger.info(f"  [{self.name}] Ep {epoch:02d}/{EPOCHS}  "
                        f"Loss={loss:.4f}  ValMOTA={val_m:.4f}  "
                        f"LR={self.scheduler.get_last_lr()[0]:.2e}")
            if val_m > best_mota:
                best_mota=val_m; no_improve=0; self._save("best.pth")
                logger.info(f"    ✔ [{self.name}] Best ValMOTA={best_mota:.4f}")
            else:
                no_improve+=1
                if no_improve >= EARLY_STOP_PAT:
                    logger.info(f"  [{self.name}] Early stop at epoch {epoch}.")
                    break
        self._save("last.pth")
        logger.info(f"  [{self.name}] Training done. Best={best_mota:.4f}")
        return {"history":history,"best_mota":best_mota}

    def load_best(self):
        ckpt = self.weights_dir / "best.pth"
        if ckpt.exists():
            st = torch.load(ckpt, map_location=self.device)
            self.reid.load_state_dict(st["reid_state"])
            logger.info(f"  [{self.name}] Loaded best weights ← {ckpt}")
        else:
            logger.warning(f"  [{self.name}] No best.pth — using last weights.")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.reid.train(); self.classifier.train()
        total=0.; n=0
        for imgs, labels in loader:
            imgs=imgs.to(self.device); labels=labels.to(self.device)
            emb=self.reid(imgs); logits=self.classifier(emb)
            loss=(self.LAMBDA_TRIP*self.triplet_loss(emb,labels)
                  +self.LAMBDA_CE*self.ce_loss(logits,labels))
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.reid.parameters(), 5.0)
            self.optimizer.step()
            total+=loss.item(); n+=1
        return total/max(1,n)

    def _val_mota(self, epoch: int) -> float:
        random.seed(RANDOM_SEED + epoch)
        # 5 videos per val (was 30) — keeps CPU validation fast
        sample  = random.sample(self.all_videos, min(5, len(self.all_videos)))
        tracker = UnifiedTracker(self.reid)
        motas   = []
        for vp in sample:
            frames = sorted(vp.glob("*.jpg"))
            if not frames: continue
            first  = cv2.imread(str(frames[0]))
            if first is None: continue
            h, w   = first.shape[:2]
            tracker.reset(); preds=[]
            for fi, fp in enumerate(frames):
                img=cv2.imread(str(fp))
                if img is None: continue
                dets=self.detector.detect(img)
                trks=tracker.update(dets,img)
                preds.append({"frame_id":fi,
                              "tracks":[(t.id,t.bbox.tolist(),
                                         t.class_id,t.confidence)
                                        for t in trks]})
            m=_evaluate_video(preds,self.annotations,vp.name,w,h)
            if m: motas.append(m["mota"])
        return float(np.mean(motas)) if motas else 0.0

    def _save(self, filename: str):
        torch.save({
            "reid_name":  self.name,
            "params":     REID_PARAMS.get(self.name, {}),
            "timestamp":  datetime.now().isoformat(),
            "reid_state": self.reid.state_dict(),
        }, self.weights_dir / filename)


# ============================================================================
# SECTION 14: EVALUATION LOOP
# ============================================================================

class ReIDEvaluator:
    """
    Runs one trained method over the held-out EVAL split (20% of videos).
    These videos are never used during training or crop extraction.
    TP / FP / FN / TN are aggregated across all eval videos and printed once.
    Output .mp4 videos are saved for the first VISUAL_VIDEOS eval videos.
    """

    def __init__(self, method_name: str,
                 tracker: UnifiedTracker,
                 detector: YOLODetector,
                 all_videos: List[Path],
                 annotations: dict):
        self.name        = method_name
        self.tracker     = tracker
        self.detector    = detector
        self.all_videos  = all_videos
        self.annotations = annotations
        self.vid_out = EXPERIMENT_OUT/"outputs"/method_name/"videos"
        self.met_out = EXPERIMENT_OUT/"outputs"/method_name/"metrics"
        self.csv_out = EXPERIMENT_OUT/"outputs"/method_name/"csv"

    def run(self) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  EVALUATION [{self.name}]  "
                    f"{len(self.all_videos)} videos  "
                    f"[20% held-out eval split — no training overlap]")
        logger.info(f"{'='*60}")
        for d in [self.vid_out, self.met_out, self.csv_out]:
            d.mkdir(parents=True, exist_ok=True)
        rows=[]
        for idx, vp in enumerate(self.all_videos, 1):
            save_vid = (idx <= VISUAL_VIDEOS)
            vid_out  = self.vid_out / f"{vp.name}.mp4" if save_vid else None
            frames   = sorted(vp.glob("*.jpg"))
            if not frames: continue
            first    = cv2.imread(str(frames[0]))
            if first is None: continue
            h, w     = first.shape[:2]
            self.tracker.reset(); preds=[]; writer=None
            if save_vid and vid_out:
                writer = cv2.VideoWriter(
                    str(vid_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS, (w, h))
            for fi, fp in enumerate(frames):
                img=cv2.imread(str(fp))
                if img is None: continue
                dets=self.detector.detect(img)
                trks=self.tracker.update(dets,img)
                preds.append({"frame_id":fi,
                              "tracks":[(t.id,t.bbox.tolist(),
                                         t.class_id,t.confidence)
                                        for t in trks]})
                if writer:
                    writer.write(_render_tracks(img, trks))
            if writer:
                writer.release()
                logger.info(f"  [{self.name}] 🎬 Video saved → {vid_out}")
            m=_evaluate_video(preds,self.annotations,vp.name,w,h)
            if m:
                rows.append({"video":vp.name,**m})
                # Silent progress tick — no per-video metrics printed
                if idx % 50 == 0 or idx == len(self.all_videos):
                    logger.info(f"  [{self.name}] Progress: {idx}/{len(self.all_videos)} videos processed…")
        pd.DataFrame(rows).to_csv(self.csv_out/"per_video_metrics.csv",index=False)
        agg = self._aggregate(rows, self.name)

        # ── Print aggregated per-method summary (overall metrics only) ────────
        _print_method_summary(self.name, agg)

        with open(self.met_out/"aggregated_metrics.json","w") as f:
            json.dump(agg, f, indent=2)
        return agg

    @staticmethod
    def _aggregate(rows: List[dict], name: str) -> dict:
        if not rows: return {"reid_name": name}
        tp=int(np.sum([r["TP"] for r in rows])); fp=int(np.sum([r["FP"] for r in rows]))
        fn=int(np.sum([r["FN"] for r in rows])); tn=int(np.sum([r["TN"] for r in rows]))
        prec=tp/(tp+fp) if (tp+fp)>0 else 0.
        rec =tp/(tp+fn) if (tp+fn)>0 else 0.
        f1  =2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.
        spec=tn/(tn+fp) if (tn+fp)>0 else 0.
        return {
            "reid_name":   name,
            "n_videos":    len(rows),
            "mota":        float(np.mean([r["mota"]        for r in rows])),
            "idf1":        float(np.mean([r["idf1"]        for r in rows])),
            "ade":         float(np.mean([r["ade"]         for r in rows])),
            "raw_ade":     float(np.mean([r["raw_ade"]     for r in rows])),
            "id_switches": int(np.sum(   [r["id_switches"] for r in rows])),
            "mt":          int(np.sum(   [r["mt"]          for r in rows])),
            "ml":          int(np.sum(   [r["ml"]          for r in rows])),
            "TP":tp, "FP":fp, "FN":fn, "TN":tn,
            "precision":round(prec,6),"recall":round(rec,6),
            "f1":round(f1,6),"specificity":round(spec,6),
        }


# ============================================================================
# SECTION 15: VISUALIZATION
# ============================================================================

_PALETTE = [
    (255,56,56),(255,157,51),(51,255,255),(56,255,255),(255,56,132),
    (131,255,56),(56,131,255),(255,210,51),(51,255,131),(131,56,255),
    (255,56,210),(100,200,100),(200,100,200),(100,150,255),
]

def _render_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    """
    Draws every track — including fully occluded ones.

    Detected  (is_kalman_pred=False): solid thick box, white label bg.
    Occluded  (is_kalman_pred=True) : solid box same thickness + semi-transparent
                                      filled overlay + 'OCC' tag so the predicted
                                      position is clearly visible even when the
                                      object is behind another object.
    """
    img = frame.copy()
    H, W = img.shape[:2]

    for t in tracks:
        # ── clamp Kalman-predicted coords to frame boundaries ─────────────────
        x1 = int(max(0,   min(t.bbox[0], W-2)))
        y1 = int(max(0,   min(t.bbox[1], H-2)))
        x2 = int(max(x1+1,min(t.bbox[2], W-1)))
        y2 = int(max(y1+1,min(t.bbox[3], H-1)))

        # Skip boxes that collapsed to nothing after clamping
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        col = _PALETTE[t.id % len(_PALETTE)]

        if t.is_kalman_pred:
            # ── OCCLUDED: semi-transparent filled rect + solid border ─────────
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)   # filled
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)       # blend 25%
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)         # solid border

            lbl = f"ID{t.id}[OCC]"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw, ty), col, -1)
            cv2.putText(img, lbl, (x1, ty - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            # ── DETECTED: standard solid box ─────────────────────────────────
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

            lbl = f"ID{t.id} {t.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw, ty), col, -1)
            cv2.putText(img, lbl, (x1, ty - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── trajectory tail (last 20 centres, always drawn) ───────────────────
        for i in range(1, min(len(t.trajectory), 20)):
            p1 = (int((t.trajectory[-i  ][0] + t.trajectory[-i  ][2]) / 2),
                  int((t.trajectory[-i  ][1] + t.trajectory[-i  ][3]) / 2))
            p2 = (int((t.trajectory[-i-1][0] + t.trajectory[-i-1][2]) / 2),
                  int((t.trajectory[-i-1][1] + t.trajectory[-i-1][3]) / 2))
            cv2.line(img, p1, p2, col, 1)

    return img


# ============================================================================
# SECTION 16: COMPARISON TABLE  — TP / FP / FN / TN prominently displayed
# ============================================================================

def _print_method_summary(name: str, agg: dict):
    """Prints a compact per-method result box immediately after evaluation."""
    tp=agg.get("TP",0); fp=agg.get("FP",0)
    fn=agg.get("FN",0); tn=agg.get("TN",0)
    total = tp+fp+fn+tn
    acc   = (tp+tn)/total if total>0 else 0.
    bar   = "─"*58
    print(f"\n  ┌{bar}┐")
    print(f"  │  RESULT SUMMARY: {name:<38s}│")
    print(f"  ├{bar}┤")
    print(f"  │  Tracking                                              │")
    print(f"  │    MOTA         : {agg.get('mota',0):>8.4f}                          │")
    print(f"  │    IDF1         : {agg.get('idf1',0):>8.4f}                          │")
    print(f"  │    ADE (norm.)  : {agg.get('ade',0):>8.4f}                          │")
    print(f"  │    ID Switches  : {agg.get('id_switches',0):>8d}                          │")
    print(f"  │    MT / ML      : {agg.get('mt',0):>4d} / {agg.get('ml',0):<4d}                        │")
    print(f"  ├{bar}┤")
    print(f"  │  Detection Confusion Matrix                            │")
    print(f"  │                                                        │")
    print(f"  │         Predicted +    Predicted -                     │")
    print(f"  │  GT +   TP={tp:>9,}   FN={fn:>9,}                  │")
    print(f"  │  GT -   FP={fp:>9,}   TN={tn:>9,}                  │")
    print(f"  │                                                        │")
    print(f"  │  Precision  : {agg.get('precision',0):>8.4f}                          │")
    print(f"  │  Recall     : {agg.get('recall',0):>8.4f}                          │")
    print(f"  │  F1         : {agg.get('f1',0):>8.4f}                          │")
    print(f"  │  Specificity: {agg.get('specificity',0):>8.4f}                          │")
    print(f"  │  Accuracy   : {acc:>8.4f}  (TP+TN)/{total}               │")
    print(f"  └{bar}┘")


def _save_and_print_comparison(all_results: List[dict]):
    """Write reid_comparison.csv and print a full formatted comparison table."""
    if not all_results:
        logger.warning("No results to compare."); return

    COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["reid_name",
            "mota","idf1","ade","raw_ade","id_switches","mt","ml",
            "TP","FP","FN","TN",
            "precision","recall","f1","specificity"]
    df = pd.DataFrame(all_results)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(COMPARE_CSV, index=False)
    logger.info(f"Comparison CSV → {COMPARE_CSV}")

    w = 130
    print(); print("="*w)
    print("  FINAL COMPARISON — ALL 6 Re-ID METHODS  (trained on OVIS data)")
    print("="*w)

    # ── Block 1: Tracking metrics ─────────────────────────────────────────────
    print(f"\n  ── TRACKING METRICS ──")
    print(f"  {'Method':<16} {'MOTA':>8} {'IDF1':>8} {'ADE':>8} "
          f"{'RawADE':>9} {'IDsw':>7} {'MT':>6} {'ML':>6}")
    print("  " + "─"*74)
    for _, r in df.iterrows():
        print(f"  {r['reid_name']:<16} "
              f"{r['mota']:>8.4f} {r['idf1']:>8.4f} {r['ade']:>8.4f} "
              f"{r['raw_ade']:>9.2f} {int(r['id_switches']):>7d} "
              f"{int(r['mt']):>6d} {int(r['ml']):>6d}")

    # ── Block 2: Confusion matrix counts ─────────────────────────────────────
    print(f"\n  ── DETECTION CONFUSION MATRIX ──")
    print(f"  {'Method':<16} {'TP':>12} {'FP':>12} {'FN':>12} {'TN':>12}")
    print("  " + "─"*56)
    for _, r in df.iterrows():
        print(f"  {r['reid_name']:<16} "
              f"{int(r['TP']):>12,} {int(r['FP']):>12,} "
              f"{int(r['FN']):>12,} {int(r['TN']):>12,}")

    # ── Block 3: Derived detection metrics ───────────────────────────────────
    print(f"\n  ── DETECTION DERIVED METRICS ──")
    print(f"  {'Method':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'Specificity':>12} {'Accuracy':>10}")
    print("  " + "─"*68)
    for _, r in df.iterrows():
        tp=int(r['TP']); fp=int(r['FP']); fn=int(r['FN']); tn=int(r['TN'])
        total=tp+fp+fn+tn
        acc=(tp+tn)/total if total>0 else 0.
        print(f"  {r['reid_name']:<16} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} "
              f"{r['specificity']:>12.4f} {acc:>10.4f}")

    # ── Block 4: Best per metric ──────────────────────────────────────────────
    print(f"\n  ── BEST METHOD PER METRIC ──")
    print("  " + "─"*50)
    for met in ["mota","idf1","f1","precision","recall","specificity"]:
        if met in df.columns:
            idx  = df[met].idxmax()
            best = df.loc[idx]
            print(f"    {met:<14}  →  {best['reid_name']:<16}  ({best[met]:.4f})")
    for met_raw, met_label in [("TP","TP (highest)"),("TN","TN (highest)"),
                                ("FP","FP (lowest)"),("FN","FN (lowest)")]:
        if met_raw in df.columns:
            if "lowest" in met_label:
                idx = df[met_raw].idxmin(); val = df.loc[idx,met_raw]
            else:
                idx = df[met_raw].idxmax(); val = df.loc[idx,met_raw]
            print(f"    {met_label:<14}  →  {df.loc[idx,'reid_name']:<16}  ({int(val):,})")
    print("="*w); print()


# ============================================================================
# SECTION 16b: COMPARISON GRAPHS
#
#  Pipeline being compared across 6 combinations:
#    Trained YOLO11  →  Kalman Filter (position under full & long-term occlusion)
#                    →  Re-ID method  (identity recovery after occlusion)
#
#  Each combination = one bar per method in every chart below.
#  Three chart files are saved to EXPERIMENT_OUT/results/graphs/:
#    1. tracking_metrics.png   — MOTA, IDF1, ADE, RawADE, IDsw, MT, ML
#    2. confusion_matrix.png   — TP, FP, FN, TN (absolute counts)
#    3. derived_metrics.png    — Precision, Recall, F1, Specificity
# ============================================================================

# Consistent colour per method across all charts
_METHOD_COLORS = {
    "OccludedReID": "#E74C3C",
    "PGFA":         "#3498DB",
    "HOReID":       "#2ECC71",
    "PAT":          "#F39C12",
    "TransReID":    "#9B59B6",
    "OAMN":         "#1ABC9C",
}
_DEFAULT_COLORS = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6","#1ABC9C"]


def _bar_group(ax, methods, values_dict, title, ylabel, pct=False,
               highlight_max=True):
    """
    Draw a grouped horizontal bar chart on `ax`.
    values_dict = {metric_name: [val_per_method, ...]}
    """
    import numpy as np
    metrics   = list(values_dict.keys())
    n_metrics = len(metrics)
    n_methods = len(methods)
    bar_h     = 0.8 / n_methods
    y_base    = np.arange(n_metrics)

    for mi, method in enumerate(methods):
        vals  = [values_dict[m][mi] for m in metrics]
        color = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        bars  = ax.barh(y_base - mi * bar_h + (n_methods-1)*bar_h/2,
                        vals, bar_h * 0.85,
                        label=method, color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            txt = f"{val:.1f}%" if pct else (f"{val:.4f}" if val < 10 else f"{int(val):,}")
            ax.text(bar.get_width() + max(abs(bar.get_width())*0.01, 0.002),
                    bar.get_y() + bar.get_height()/2,
                    txt, va="center", ha="left", fontsize=7.5, color="#333333")

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
    Produces three publication-quality comparison charts and saves them to
    EXPERIMENT_OUT/results/graphs/.

    Chart 1 — Tracking metrics  : MOTA, IDF1, ADE, RawADE, IDsw, MT, ML
    Chart 2 — Confusion matrix  : TP, FP, FN, TN  (absolute counts)
    Chart 3 — Derived metrics   : Precision, Recall, F1, Specificity
    """
    if not all_results:
        logger.warning("[Graphs] No results — skipping graph generation.")
        return

    graph_dir = EXPERIMENT_OUT / "results" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    methods = [r["reid_name"] for r in all_results]
    df      = pd.DataFrame(all_results)

    # ── helper: safe column list ──────────────────────────────────────────────
    def col(name, default=0.0):
        return [float(r.get(name, default)) for r in all_results]

    pipeline_tag = "Trained YOLO11 + Kalman Filter + Re-ID method"

    # =========================================================================
    # CHART 1 — Tracking metrics
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Tracking Metrics — {pipeline_tag}",
                 fontsize=13, fontweight="bold", y=1.01)

    # Left: rate/score metrics (0-1 range)
    _bar_group(axes[0], methods,
               {"MOTA":  col("mota"),
                "IDF1":  col("idf1"),
                "ADE\n(norm.)": col("ade")},
               title="Score Metrics (higher = better)",
               ylabel="Score  [0 – 1]")

    # Right: count / integer metrics
    _bar_group(axes[1], methods,
               {"RawADE\n(pixels)": col("raw_ade"),
                "ID Switches":      col("id_switches"),
                "Mostly\nTracked":  col("mt"),
                "Mostly\nLost":     col("ml")},
               title="Count Metrics",
               ylabel="Count / Pixels")

    # Annotate: lower = better for IDsw / ML / RawADE
    axes[1].set_title("Count Metrics  (↓ better for IDsw, ML, RawADE)",
                       fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    out1 = graph_dir / "1_tracking_metrics.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Graphs] Saved → {out1}")

    # =========================================================================
    # CHART 2 — Confusion matrix counts
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Detection Confusion Matrix (absolute counts) — {pipeline_tag}",
                 fontsize=13, fontweight="bold", y=1.01)

    n_methods = len(methods)
    x         = np.arange(4)   # TP FP FN TN
    bar_w     = 0.8 / n_methods
    cm_labels  = ["TP", "FP", "FN", "TN"]
    cm_vals    = {
        m: [r.get("TP",0), r.get("FP",0), r.get("FN",0), r.get("TN",0)]
        for m, r in zip(methods, all_results)
    }

    for mi, method in enumerate(methods):
        color  = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        offset = (mi - n_methods/2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, cm_vals[method], bar_w * 0.85,
                        label=method, color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, cm_vals[method]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        f"{int(val):,}", ha="center", va="bottom",
                        fontsize=7, rotation=45, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(cm_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("TP: correct detections   FP: false alarms   "
                 "FN: missed objects   TN: true empty frames",
                 fontsize=9, color="#555555", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, framealpha=0.7)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.tight_layout()
    out2 = graph_dir / "2_confusion_matrix.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Graphs] Saved → {out2}")

    # =========================================================================
    # CHART 3 — Derived metrics (Precision / Recall / F1 / Specificity)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Derived Detection Metrics — {pipeline_tag}",
                 fontsize=13, fontweight="bold", y=1.01)

    der_labels = ["Precision", "Recall", "F1", "Specificity"]
    der_vals   = {
        m: [r.get("precision",0), r.get("recall",0),
            r.get("f1",0),        r.get("specificity",0)]
        for m, r in zip(methods, all_results)
    }

    for mi, method in enumerate(methods):
        color  = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        offset = (mi - n_methods/2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, der_vals[method], bar_w * 0.85,
                        label=method, color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, der_vals[method]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=45, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(der_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score  [0 – 1]", fontsize=10)
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

    # =========================================================================
    # CHART 4 — Summary radar / spider chart  (all score metrics in one view)
    # =========================================================================
    try:
        radar_metrics = ["MOTA", "IDF1", "Precision", "Recall", "F1", "Specificity"]
        radar_keys    = ["mota","idf1","precision","recall","f1","specificity"]
        N             = len(radar_metrics)
        angles        = [n / float(N) * 2 * np.pi for n in range(N)]
        angles        += angles[:1]   # close the polygon

        fig, ax = plt.subplots(figsize=(8, 8),
                               subplot_kw=dict(polar=True))
        fig.suptitle(f"Overall Score Radar — {pipeline_tag}",
                     fontsize=12, fontweight="bold", y=1.02)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7,
                            color="#888888")
        ax.grid(color="#CCCCCC", linestyle="--", linewidth=0.6)

        for mi, (method, r) in enumerate(zip(methods, all_results)):
            vals   = [float(r.get(k, 0)) for k in radar_keys]
            vals  += vals[:1]
            color  = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
            ax.plot(angles, vals, linewidth=1.8, linestyle="solid",
                    label=method, color=color)
            ax.fill(angles, vals, alpha=0.08, color=color)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                  fontsize=8.5, framealpha=0.8)

        plt.tight_layout()
        out4 = graph_dir / "4_radar_summary.png"
        fig.savefig(out4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Graphs] Saved → {out4}")
    except Exception as e:
        logger.warning(f"[Graphs] Radar chart skipped: {e}")

    logger.info(f"[Graphs] All charts saved to → {graph_dir}")
    print(f"\n  Graphs saved to → {graph_dir}")
    print(f"    1_tracking_metrics.png  — MOTA, IDF1, ADE, IDsw, MT, ML")
    print(f"    2_confusion_matrix.png  — TP, FP, FN, TN")
    print(f"    3_derived_metrics.png   — Precision, Recall, F1, Specificity")
    print(f"    4_radar_summary.png     — All score metrics in one spider chart")


# ============================================================================
# SECTION 17: MAIN
# ============================================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def main():
    set_seed(RANDOM_SEED)

    print(); print("="*70)
    print("  MOT Re-ID COMPARISON — ALL 6 METHODS")
    _dev_label = (f"GPU · {torch.cuda.get_device_name(0)}" if DEVICE == "cuda"
                  else "Apple MPS (integrated GPU)" if DEVICE == "mps"
                  else f"CPU ({os.cpu_count()} threads)")
    print(f"  Device       : {_dev_label}")
    print(f"  Methods      : {', '.join(ALL_METHODS)}")
    print(f"  Train/       : {TRAIN_DIR}")
    print(f"  Outputs      : {EXPERIMENT_OUT}")
    print(f"  Epochs       : {EPOCHS}  (early-stop patience={EARLY_STOP_PAT})")
    print(f"  Output videos: {VISUAL_VIDEOS} .mp4 per method")
    print(f"  Split        : {int(TRAIN_SPLIT*100)}% train / {100-int(TRAIN_SPLIT*100)}% eval  (no overlap)")
    # Resolve which YOLO model will actually be loaded
    _yolo_display = YOLO_TRAINED_PATH
    for _p in [YOLO_TRAINED_PATH] + YOLO_FALLBACK_PATHS:
        if Path(_p).exists():
            _yolo_display = _p; break
    else:
        _yolo_display = YOLO_PRETRAINED_FALLBACK + "  ⚠ (pretrained fallback)"
    print(f"  YOLO         : {_yolo_display}")
    print("="*70)

    # ── Validate paths ────────────────────────────────────────────────────────
    if not ANNOTATIONS_FILE.exists():
        sys.exit(f"[ERROR] Annotations not found: {ANNOTATIONS_FILE}")
    if not TRAIN_DIR.exists():
        sys.exit(f"[ERROR] Train directory not found: {TRAIN_DIR}")

    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    logger.info(f"Annotations loaded — "
                f"{len(annotations.get('videos',[]))} videos.")

    all_vids = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    if not all_vids:
        sys.exit("[ERROR] No video folders found inside Train/")

    # ── 80 / 20 split — no overlap between train and eval ────────────────────
    split_idx  = int(len(all_vids) * TRAIN_SPLIT)
    train_vids = all_vids[:split_idx]
    eval_vids  = all_vids[split_idx:]
    logger.info(f"Total videos : {len(all_vids)}")
    logger.info(f"Train videos : {len(train_vids)}  (80% — used for crop "
                f"extraction + val MOTA, {min(5,len(train_vids))} random/epoch)")
    logger.info(f"Eval  videos : {len(eval_vids)}  (20% — held-out, "
                f"never seen during training)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Build / load crop dataset ONCE  (shared by all 6 methods)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 1 / 3 — Building crop dataset from OVIS annotations")
    print(f"{'─'*70}")
    crop_dataset = OVISCropDataset(
        video_root       = TRAIN_DIR,
        annotations_file = ANNOTATIONS_FILE,
        cache_dir        = CROP_CACHE,
        allowed_folders  = {v.name for v in train_vids},   # 80% train only
    )
    if crop_dataset.num_classes < 2:
        sys.exit("[ERROR] Fewer than 2 valid identities found in crop dataset. "
                 "Check TRAIN_DIR and ANNOTATIONS_FILE paths.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Load YOLO ONCE  (fixed, shared by all methods)
    # ─────────────────────────────────────────────────────────────────────────
    detector = YOLODetector()   # loads trained model via _load_trained_yolo()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Train → Evaluate → collect results  (one loop per method)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEP 2 / 3 — Training & evaluating {len(ALL_METHODS)} methods")
    print(f"{'─'*70}")

    all_results: List[dict] = []

    for method_name in ALL_METHODS:
        print()
        print(f"  ╔{'═'*58}╗")
        print(f"  ║  {'METHOD : ' + method_name:<56s}║")
        print(f"  ╚{'═'*58}╝")

        set_seed(RANDOM_SEED)

        reid = build_reid(method_name)
        reid.to(reid._device)

        # ── Train ─────────────────────────────────────────────────────────
        trainer = ReIDTrainer(
            method_name  = method_name,
            reid         = reid,
            detector     = detector,
            all_videos   = train_vids,       # 80% — train split only
            annotations  = annotations,
            crop_dataset = crop_dataset,
        )
        trainer.run()
        trainer.load_best()

        # ── Evaluate ──────────────────────────────────────────────────────
        tracker   = UnifiedTracker(reid)
        evaluator = ReIDEvaluator(
            method_name = method_name,
            tracker     = tracker,
            detector    = detector,
            all_videos  = eval_vids,         # 20% — held-out eval split only
            annotations = annotations,
        )
        agg = evaluator.run()

        if agg:
            all_results.append(agg)
            logger.info(
                f"  [{method_name}] DONE  "
                f"MOTA={agg.get('mota',0):.4f}  "
                f"IDF1={agg.get('idf1',0):.4f}  "
                f"F1={agg.get('f1',0):.4f}  "
                f"TP={agg.get('TP',0):,}  "
                f"FP={agg.get('FP',0):,}  "
                f"FN={agg.get('FN',0):,}  "
                f"TN={agg.get('TN',0):,}")
        else:
            logger.warning(f"  [{method_name}] No metrics returned.")

        # Release memory before next method
        del reid, trainer, tracker, evaluator

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Print final comparison table
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 3 / 3 — Final comparison")
    print(f"{'─'*70}")
    _save_and_print_comparison(all_results)
    _save_comparison_graphs(all_results)

    print(f"\n  Output videos → {EXPERIMENT_OUT / 'outputs' / '<method>' / 'videos'}")
    print(f"  Metrics CSV   → {COMPARE_CSV}")
    print(f"  Per-video CSV → {EXPERIMENT_OUT / 'outputs' / '<method>' / 'csv'}")
    print(f"  Charts        → {EXPERIMENT_OUT / 'results' / 'graphs'}")
    logger.info(f"All outputs saved to → {EXPERIMENT_OUT}")
    logger.info("Run complete.")


if __name__ == "__main__":
    main()
