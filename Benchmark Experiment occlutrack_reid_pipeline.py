"""
================================================================================
  MULTI-OBJECT TRACKING — OccluTrack on File 2 Pipeline
================================================================================
  PIPELINE ARCHITECTURE (identical to the 6 Re-ID experiments):
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
    │  OccluTrack         │  ← inter-track IoU occlusion scoring
    │  Association        │    per-track cost relaxation for occluded tracks
    │                     │    horizontal-stripe visibility weighting
    │                     │    appearance gallery (same as Re-ID experiments)
    └─────────────────────┘

  WHAT IS IDENTICAL TO THE 6 Re-ID EXPERIMENTS:
    • YOLO11  (same fallback chain)
    • Kalman Filter — same 7-state formulation + dynamic Q inflation
    • 4-state track machine: TENTATIVE → CONFIRMED → OCCLUDED → LOST
    • occlusion_duration tracking + 150-frame max lifetime
    • Long-term occlusion bypass at REID_LT_OCC_FRAMES (30 frames)
    • OVISCropDataset (crop extraction, caching, 80/20 split)
    • Training loop (TripletLoss + CrossEntropy, same epochs / LR / patience)
    • Evaluation loop (same metrics: MOTA, IDF1, ADE, TP/FP/FN/TN …)
    • Video rendering (_render_tracks — detected vs Kalman-predicted display)
    • Comparison graphs (tracking / confusion matrix / derived / radar)
    • Device selection (CUDA → MPS → CPU)

  WHAT IS DIFFERENT (OccluTrack-specific):
    • Association uses inter-track IoU occlusion scoring on top of the
      4-state machine — any track whose predicted bbox overlaps another
      track's predicted bbox above OCCLU_IOU_THR is additionally flagged
      "geometry-occluded" and gets a relaxed matching cost threshold.
    • Combined cost = APP_WEIGHT * cosine_dist + (1-APP_WEIGHT) * iou_dist
      (OccluTrack balanced weighting, same formula as original OccluTrack)
    • Long-term occlusion (>REID_LT_OCC_FRAMES) shifts weight to 0.1/0.9
      IoU/appearance — same as File 2's UnifiedTracker for fair comparison.
    • Appearance gallery: raw list capped at NN_BUDGET (max-similarity
      retrieval), same strategy as original OccluTrack.
    • OCCLU_COST_RELAX additive relaxation on gated cost threshold for
      geometry-occluded tracks.

  OUTPUTS (same structure as the 6 Re-ID experiments):
    • outputs/OccluTrack/videos/      — 20 annotated .mp4 files
    • outputs/OccluTrack/metrics/     — aggregated_metrics.json
    • outputs/OccluTrack/csv/         — per_video_metrics.csv
    • results/reid_comparison.csv     — appended with OccluTrack row
    • results/graphs/                 — all 4 chart types
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS  +  DEVICE SELECTION
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

import torch

def _select_device() -> str:
    torch.set_num_threads(os.cpu_count() or 4)
    if torch.cuda.is_available():
        dev = "cuda"
        logger.info(f"[Device] GPU detected — using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        logger.info("[Device] Apple MPS (integrated GPU) detected — using MPS")
    else:
        dev = "cpu"
        print("[Device] No GPU detected — using CPU")
    return dev

# Logger must exist before _select_device is called
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = _select_device()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import motmetrics as mm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")


# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

METHOD_NAME = "OccluTrack"

# ── Paths (match File 2 layout exactly) ──────────────────────────────────────
BASE_DIR         = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate")
EXPERIMENT_OUT   = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 6th term\Graduation Project 2\Experiments")
TRAIN_DIR        = BASE_DIR / "train"
ANNOTATIONS_FILE = BASE_DIR / "annotations_train.json"
CROP_CACHE       = EXPERIMENT_OUT / "crop_cache"
COMPARE_CSV      = EXPERIMENT_OUT / "results" / "reid_comparison.csv"

# ── YOLO (identical fallback chain to File 2) ─────────────────────────────────
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
YOLO_PRETRAINED_FALLBACK = "yolo11n.pt"
YOLO_CONF     = 0.50
YOLO_IOU      = 0.45
YOLO_IMG_SIZE = 640
YOLO_VERBOSE  = False

# ── Kalman (identical to File 2) ──────────────────────────────────────────────
VIDEO_FPS         = 15
LONG_TERM_OCC_SEC = 10
LONG_TERM_MAX_AGE = VIDEO_FPS * LONG_TERM_OCC_SEC    # 150 frames
KF_MIN_HITS       = 3

# ── Association (shared with File 2) ─────────────────────────────────────────
REID_EMBED_DIM    = 512
REID_IOU_GATE     = 0.20
REID_MATCH_THRESH = 0.55
W_IOU             = 0.40
W_APP             = 0.60
DET_IOU_THRESH    = 0.50

# ── Occlusion thresholds (identical semantics to File 2) ─────────────────────
REID_LT_OCC_FRAMES = 30   # below: IoU gate active; above: appearance-only

# ── OccluTrack-specific knobs ─────────────────────────────────────────────────
# OCCLU_IOU_THR   — IoU between two *track* predicted bboxes above which both
#                   are flagged as geometry-occluded (separate from the 4-state
#                   machine which triggers on missed detections).
# OCCLU_COST_RELAX— additive relaxation on the matching threshold for any track
#                   flagged geometry-occluded, giving it a wider acceptance band.
# APP_WEIGHT      — weight of the appearance term in the combined cost matrix.
#                   (1 - APP_WEIGHT) goes to IoU distance.
# N_STRIPES       — number of horizontal body stripes for visibility weighting
#                   in the appearance extractor (same as OccludedReID in File 2).
# VISIBILITY_THR  — gradient-energy threshold below which a stripe is treated
#                   as occluded and down-weighted.
# NN_BUDGET       — max gallery embeddings kept per track (max-similarity
#                   retrieval, same strategy as original OccluTrack paper).
OCCLU_IOU_THR    = 0.30
OCCLU_COST_RELAX = 0.15
APP_WEIGHT       = 0.50
N_STRIPES        = 6
VISIBILITY_THR   = 0.40
NN_BUDGET        = 100

# ── Training (identical to File 2) ───────────────────────────────────────────
EPOCHS           = 10
EARLY_STOP_PAT   = 4
LEARNING_RATE    = 3e-4
TRIPLET_MARGIN   = 0.3
RANDOM_SEED      = 42
MIN_CROPS_PER_ID = 4
MAX_CROPS_PER_ID = 50
VISUAL_VIDEOS    = 20
TRAIN_SPLIT      = 0.80


# ============================================================================
# SECTION 3: DATA STRUCTURES  (identical to File 2)
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
# SECTION 4: OVIS LAZY DATASET
# ============================================================================
#
#  Stores only (frame_path, x, y, w, h, label) tuples — zero pixel data in
#  RAM.  Each crop is read from disk only when __getitem__ is called during
#  training.  No cache file is written or read.  No MemoryError possible.
#
# ============================================================================

class OVISCropDataset(Dataset):
    """
    Lazy on-the-fly crop reader.
    Scans annotations_train.json once at startup to build an index of
    (frame_path, bbox, label) tuples.  Pixel data is never held in memory —
    each frame is opened, cropped, resized, and discarded inside __getitem__.

    This replaces the cache-based approach entirely: no pickle, no RAM spike,
    works on any machine regardless of dataset size.
    """

    def __init__(self, video_root: Path, annotations_file: Path,
                 cache_dir: Optional[Path] = None,          # kept for API compat, unused
                 allowed_folders: Optional[set] = None):
        import torchvision.transforms as T
        self.allowed_folders = allowed_folders
        # Each entry: (frame_path_str, x1, y1, x2, y2, label_int)
        self.samples:    List[Tuple] = []
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
        self._build_index(video_root, annotations_file)

    def _build_index(self, video_root: Path, annotations_file: Path):
        """
        One-pass scan of the JSON.  Stores file paths + bbox coords only.
        No images are read here — total memory cost is a few MB of strings.
        """
        logger.info("[Dataset] Building lazy index from annotations…")
        with open(annotations_file) as f:
            ann_data = json.load(f)

        # Map video_id → folder name
        vid_map: Dict[int, str] = {}
        for v in ann_data.get("videos", []):
            names = v.get("file_names", [])
            if names:
                folder = names[0].replace("\\", "/").split("/")[0]
                vid_map[v["id"]] = folder

        # Map video_id → sorted list of frame file paths
        frame_map: Dict[int, List[str]] = {}

        # First pass: collect (global_id → list of (frame_path, x1,y1,x2,y2))
        id_entries: Dict[int, List[Tuple]] = defaultdict(list)

        for ann in ann_data.get("annotations", []):
            vid_id = ann["video_id"]
            folder = vid_map.get(vid_id)
            if folder is None:
                continue
            if self.allowed_folders is not None and folder not in self.allowed_folders:
                continue

            vid_path = video_root / folder
            if not vid_path.is_dir():
                continue

            # Build frame list once per video
            if vid_id not in frame_map:
                frame_map[vid_id] = sorted(
                    str(p) for p in vid_path.glob("*.jpg"))
            frame_files = frame_map[vid_id]
            if not frame_files:
                continue

            track_id  = (ann.get("instance_id") or
                         ann.get("track_id")    or
                         ann["id"])
            global_id = vid_id * 100_000 + track_id

            count = 0
            for fi, bbox in enumerate(ann.get("bboxes", [])):
                if count >= MAX_CROPS_PER_ID:
                    break
                if bbox is None or len(bbox) != 4:
                    continue
                x, y, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    continue
                if fi >= len(frame_files):
                    continue
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = int(x + bw)   # clamping done in __getitem__ after imread
                y2 = int(y + bh)
                id_entries[global_id].append((frame_files[fi], x1, y1, x2, y2))
                count += 1

        # Keep only identities with enough crops
        valid_ids = sorted([gid for gid, entries in id_entries.items()
                            if len(entries) >= MIN_CROPS_PER_ID])
        label_map = {gid: idx for idx, gid in enumerate(valid_ids)}

        for gid in valid_ids:
            lbl = label_map[gid]
            for entry in id_entries[gid]:
                self.samples.append((*entry, lbl))

        self.num_classes = len(valid_ids)
        logger.info(f"[Dataset] Index built: {len(self.samples)} entries  |  "
                    f"{self.num_classes} identities  (no pixel data in RAM)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, x1, y1, x2, y2, label = self.samples[idx]
        img = cv2.imread(frame_path)
        if img is None:
            # Fallback: return a black crop if the frame can't be read
            import torchvision.transforms as T
            dummy = np.zeros((256, 128, 3), dtype=np.uint8)
            return self.transform(dummy), label
        # Clamp to actual image dimensions
        h, w = img.shape[:2]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            dummy = np.zeros((256, 128, 3), dtype=np.uint8)
            return self.transform(dummy), label
        crop = cv2.resize(img[y1:y2, x1:x2], (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(crop), label


# ============================================================================
# SECTION 5: PK BATCH SAMPLER  (identical to File 2)
# ============================================================================

class TripletBatchSampler:
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
# SECTION 6: LOSS FUNCTIONS  (identical to File 2)
# ============================================================================

class TripletLoss(nn.Module):
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
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


# ============================================================================
# SECTION 7: YOLO DETECTOR  (identical to File 2)
# ============================================================================

def _load_trained_yolo() -> YOLO:
    if Path(YOLO_TRAINED_PATH).exists():
        logger.info(f"[YOLO] ✓ Trained model found: {YOLO_TRAINED_PATH}")
        model = YOLO(YOLO_TRAINED_PATH)
        model.to(DEVICE)
        logger.info(f"[YOLO] Classes ({len(model.names)}): "
                    + ", ".join(f"{k}:{v}" for k, v in model.names.items()))
        return model

    logger.warning(f"[YOLO] ⚠ Primary path not found: {YOLO_TRAINED_PATH}")
    for alt in YOLO_FALLBACK_PATHS:
        if alt != YOLO_TRAINED_PATH and Path(alt).exists():
            logger.info(f"[YOLO] ✓ Trained model found at fallback: {alt}")
            model = YOLO(alt)
            model.to(DEVICE)
            return model

    logger.warning(f"[YOLO] ❌ No trained model found — falling back to {YOLO_PRETRAINED_FALLBACK}")
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
# SECTION 8: KALMAN PREDICTOR  (identical to File 2 — dynamic Q inflation)
# ============================================================================

class KalmanPredictor:
    """
    Standard 7-state Kalman filter [cx, cy, s, r, vx, vy, vs].
    During occlusion, process noise Q is inflated proportionally to how long
    the track has been occluded — same as File 2's KalmanPredictor.
    """
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
        z = self._to_z(b)
        self.kf.x[:4] = z
        self.kf.update(z)

    def update(self, b: np.ndarray):
        self.kf.update(self._to_z(b))
        self._occ_frames = 0
        self.kf.Q = self._base_Q.copy()

    def predict(self, is_occluded: bool = False) -> np.ndarray:
        """
        Inflates Q during occlusion so the filter correctly widens its
        uncertainty as Kalman drift accumulates — identical to File 2.
        """
        if is_occluded:
            self._occ_frames += 1
            factor = (1.0 + 2.0 * min(self._occ_frames, LONG_TERM_MAX_AGE)
                      / LONG_TERM_MAX_AGE)
            self.kf.Q = self._base_Q * factor
        self.kf.predict()
        return self._to_bbox(self.kf.x)

    @staticmethod
    def _to_z(b):
        w = b[2] - b[0]; h = b[3] - b[1]
        return np.array([[b[0]+w/2], [b[1]+h/2],
                          [w*h], [w/max(float(h), 1e-6)]], dtype=float)

    @staticmethod
    def _to_bbox(x):
        s, r = float(x[2]), float(x[3])
        if s > 0 and r > 0:
            w = np.sqrt(s * r); h = s / w
        else:
            w = h = 0.0
        cx, cy = float(x[0]), float(x[1])
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


# ============================================================================
# SECTION 9: OCCLUTRACK APPEARANCE EXTRACTOR
# ============================================================================
#
#  Backbone   = MobileNetV2 pretrained on ImageNet (last 5 blocks unfrozen).
#  Head       = OccluTrack horizontal-stripe visibility weighting → REID_EMBED_DIM.
#  Training   = TripletLoss (batch-hard) + CrossEntropy (ID classification),
#               identical training recipe to the 6 Re-ID methods in File 2.
#
#  The stripe-visibility logic is OccluTrack's own contribution:
#    • Feature map is split into N_STRIPES horizontal bands.
#    • Each stripe's contribution is weighted by its gradient energy
#      (a proxy for how much of that body region is visible).
#    • Stripes below VISIBILITY_THR are masked out.
#    • Global average pool is the fallback when all stripes are masked.
#
# ============================================================================

class OccluTrackExtractor(nn.Module):
    BACKBONE_DIM = 1280   # MobileNetV2 final feature channels

    def __init__(self):
        super().__init__()
        self._device = DEVICE
        self.n_stripes      = N_STRIPES
        self.visibility_thr = VISIBILITY_THR
        self._transform_eval = None
        self.backbone = self._load_backbone().to(self._device)
        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, REID_EMBED_DIM),
        ).to(self._device)
        logger.info(f"[OccluTrack] {self.n_stripes}-stripe trainable extractor ready.")

    # ── backbone loading (mirrors BaseReIDExtractor._load_backbone) ───────────

    def _load_backbone(self) -> nn.Module:
        import torchvision.models as tvm
        import torchvision.transforms as T
        mn = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        bb = nn.Sequential(*list(mn.features.children()))
        # Freeze early blocks, fine-tune last 5
        for child in list(bb.children())[:-5]:
            for p in child.parameters(): p.requires_grad = False
        for child in list(bb.children())[-5:]:
            for p in child.parameters(): p.requires_grad = True
        self._transform_eval = T.Compose([
            T.ToPILImage(), T.Resize((256, 128)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return bb

    # ── stripe visibility helper ──────────────────────────────────────────────

    def _gradient_visibility(self, gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        return min(1.0, float(np.std(cv2.Laplacian(gray, cv2.CV_64F))) / 255.0 * 8.0)

    def _gray_batch(self, x: torch.Tensor) -> List[np.ndarray]:
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        imgs = np.clip(
            (x.detach().cpu().permute(0,2,3,1).numpy() * std + mean) * 255,
            0, 255).astype(np.uint8)
        return [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        B, C, H, W = fmap.shape
        grays = self._gray_batch(x)
        sh    = max(1, H // self.n_stripes)
        c_sh  = max(1, x.shape[2] // self.n_stripes)
        agg   = torch.zeros(B, C, device=self._device)
        w_sum = torch.zeros(B,    device=self._device)

        for i in range(self.n_stripes):
            sf = fmap[:, :, i*sh:(i+1)*sh, :]
            if sf.shape[2] == 0:
                continue
            sf_pool = F.adaptive_avg_pool2d(sf, 1).squeeze(-1).squeeze(-1)
            vis = torch.tensor(
                [self._gradient_visibility(g[i*c_sh:(i+1)*c_sh, :])
                 for g in grays],
                device=self._device)
            mask  = (vis >= self.visibility_thr).float()
            agg   += sf_pool * (vis * mask).unsqueeze(1)
            w_sum += vis * mask

        # Fall back to global average pool when all stripes are masked
        gpool = F.adaptive_avg_pool2d(fmap, 1).squeeze(-1).squeeze(-1)
        agg   = torch.where(
            (w_sum == 0).unsqueeze(1),
            gpool,
            agg / (w_sum.unsqueeze(1) + 1e-9))

        return F.normalize(self.head(agg), dim=1)

    # ── inference helper (mirrors BaseReIDExtractor.extract) ─────────────────

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(REID_EMBED_DIM, dtype=np.float32)
        try:
            crop    = cv2.resize(frame[y1:y2, x1:x2], (128, 256))
            rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp     = self._transform_eval(rgb).unsqueeze(0).to(self._device)
            self.eval()
            with torch.no_grad():
                emb = self.forward(inp).squeeze(0)
            return emb.cpu().numpy()
        except Exception:
            return np.zeros(REID_EMBED_DIM, dtype=np.float32)


# ============================================================================
# SECTION 10: OCCLUTRACK TRACKER
# ============================================================================
#
#  Inherits the 4-state machine and long-term occlusion logic from File 2's
#  UnifiedTracker, and adds OccluTrack's geometry-based occlusion scoring.
#
#  How the two occlusion mechanisms interact
#  -----------------------------------------
#  1. STATE-MACHINE occlusion (from File 2):
#     Triggered when a CONFIRMED track receives NO matched detection in a frame.
#     → track.state = OCCLUDED; occlusion_duration starts counting.
#     → At REID_LT_OCC_FRAMES (30 frames): IoU gate bypassed, 0.10/0.90 mix.
#     → At LONG_TERM_MAX_AGE  (150 frames): track → LOST.
#
#  2. GEOMETRY occlusion (OccluTrack addition):
#     Before building the cost matrix each frame, any track whose predicted
#     bbox overlaps another track's predicted bbox ≥ OCCLU_IOU_THR is flagged
#     geometry-occluded (_geom_occ dict).  This can fire even when the track
#     DID receive a detection (partial occlusion, not a missed frame).
#     Effect: the acceptance threshold for that track is relaxed by
#     OCCLU_COST_RELAX so the slightly-worse appearance match from a
#     partially-occluded crop is still accepted.
#
#  Combined cost matrix
#  --------------------
#  For a normal (non-long-term) match:
#      cost = APP_WEIGHT * cosine_dist + (1 - APP_WEIGHT) * iou_dist
#
#  For a long-term occluded track (same as File 2's UnifiedTracker):
#      cost = 0.10 * iou_dist + 0.90 * cosine_dist
#
#  Greedy matching threshold:
#      base  = 1.0 - REID_MATCH_THRESH
#      if geometry-occluded: threshold += OCCLU_COST_RELAX
#
# ============================================================================

def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u if u > 0 else 0.0


def _greedy_match(cost: np.ndarray, thresholds: np.ndarray):
    """
    Greedy matching with per-column thresholds.
    thresholds[ti] = acceptance threshold for track column ti.
    A (di, ti) pair is accepted only if cost[di, ti] <= thresholds[ti].
    """
    matched, used_r, used_c = [], set(), set()
    # Build flat list with per-pair threshold gate
    candidates = [
        (cost[r, c], r, c)
        for r in range(cost.shape[0])
        for c in range(cost.shape[1])
        if cost[r, c] <= thresholds[c]
    ]
    candidates.sort()
    for val, r, c in candidates:
        if r not in used_r and c not in used_c:
            matched.append((r, c))
            used_r.add(r)
            used_c.add(c)
    return (matched,
            [r for r in range(cost.shape[0]) if r not in used_r],
            [c for c in range(cost.shape[1]) if c not in used_c])


class OccluTracker:
    """
    OccluTrack association wrapped inside File 2's full pipeline.
    Drop-in replacement for UnifiedTracker — same public interface.
    """

    def __init__(self, extractor: OccluTrackExtractor):
        self.extractor = extractor
        self._tracks:  List[Track]                 = []
        self._kf:      Dict[int, KalmanPredictor]  = {}
        self._gallery: Dict[int, List[np.ndarray]] = {}
        self._geom_occ: Dict[int, bool]            = {}   # geometry-occlusion flag
        self._next_id  = 1

    def reset(self):
        self._tracks.clear()
        self._kf.clear()
        self._gallery.clear()
        self._geom_occ.clear()
        self._next_id = 1

    # ── OccluTrack: inter-track geometry occlusion scoring ───────────────────

    def _compute_geometry_occlusion(self):
        """
        For every active track, check if its predicted bbox overlaps any other
        active track's predicted bbox above OCCLU_IOU_THR.
        Tracks flagged here get a relaxed cost threshold in the association step.
        This runs on predicted positions AFTER Kalman predict, BEFORE matching.
        """
        active = [t for t in self._tracks if t.state != OcclusionState.LOST]
        for i, ti in enumerate(active):
            geom_occluded = False
            for j, tj in enumerate(active):
                if i == j:
                    continue
                if _iou(ti.bbox, tj.bbox) >= OCCLU_IOU_THR:
                    geom_occluded = True
                    break
            self._geom_occ[ti.id] = geom_occluded

    # ── appearance: max-similarity gallery retrieval (OccluTrack style) ──────

    def _cosine_dist(self, query: np.ndarray, track_id: int) -> float:
        gallery = self._gallery.get(track_id, [])
        if not gallery:
            return 1.0
        g_arr = np.stack(gallery[-NN_BUDGET:])
        nq = np.linalg.norm(query)
        ng = np.linalg.norm(g_arr, axis=1)
        if nq == 0 or not np.any(ng > 0):
            return 1.0
        sims = g_arr @ query / (ng * nq + 1e-9)
        return float(1.0 - sims.max())

    # ── main update ───────────────────────────────────────────────────────────

    def update(self, detections: List[Detection],
               frame: np.ndarray) -> List[Track]:

        # 1. Kalman predict — identical to File 2's UnifiedTracker
        for t in self._tracks:
            is_occ = (t.state == OcclusionState.OCCLUDED)
            t.bbox = self._kf[t.id].predict(is_occluded=is_occ)
            t.time_since_update += 1
            t.age += 1
            if is_occ:
                t.occlusion_duration += 1

        # 2. OccluTrack geometry occlusion scoring (after predict, before match)
        self._compute_geometry_occlusion()

        # 3. Extract appearance for detections
        det_feats = [self.extractor.extract(frame, d.bbox) for d in detections]

        # 4. Build OccluTrack combined cost matrix
        active = [t for t in self._tracks if t.state != OcclusionState.LOST]

        if active and detections:
            cost       = np.ones((len(detections), len(active)))
            thresholds = np.zeros(len(active))

            for ti, trk in enumerate(active):
                # Base threshold — same as File 2's greedy_match call
                base_thr = 1.0 - REID_MATCH_THRESH

                # OccluTrack relaxation for geometry-occluded tracks
                if self._geom_occ.get(trk.id, False):
                    base_thr += OCCLU_COST_RELAX

                thresholds[ti] = base_thr

                # Long-term occlusion flag — same threshold as File 2
                long_term = (trk.state == OcclusionState.OCCLUDED and
                             trk.occlusion_duration > REID_LT_OCC_FRAMES)

                for di, det in enumerate(detections):
                    iou_v = _iou(det.bbox, trk.bbox)
                    iou_c = 1.0 - iou_v

                    use_appearance = (trk.id in self._gallery and
                                      (iou_v >= REID_IOU_GATE or long_term))

                    if use_appearance:
                        app_c = self._cosine_dist(det_feats[di], trk.id)
                        if long_term:
                            # Identical to File 2's long-term weight shift
                            cost[di, ti] = 0.10 * iou_c + 0.90 * app_c
                        else:
                            # OccluTrack balanced weighting
                            cost[di, ti] = APP_WEIGHT * app_c + (1.0 - APP_WEIGHT) * iou_c
                    else:
                        # No appearance → pure IoU distance
                        cost[di, ti] = iou_c

            matched, unm_d, unm_t = _greedy_match(cost, thresholds)

        else:
            matched = []
            unm_d   = list(range(len(detections)))
            unm_t   = list(range(len(active)))

        # 5. Update matched tracks
        for di, ti in matched:
            t = active[ti]
            t.bbox              = detections[di].bbox
            t.confidence        = detections[di].confidence
            t.hits             += 1
            t.time_since_update = 0
            t.occlusion_duration = 0
            t.is_kalman_pred    = False
            t.trajectory.append(detections[di].bbox.copy())
            self._kf[t.id].update(detections[di].bbox)
            self._gallery.setdefault(t.id, []).append(det_feats[di])
            self._geom_occ[t.id] = False   # successful match → reset geometry flag

            # State transitions — identical to File 2
            if t.state == OcclusionState.OCCLUDED:
                t.state = OcclusionState.CONFIRMED
            elif t.state == OcclusionState.TENTATIVE and t.hits >= KF_MIN_HITS:
                t.state = OcclusionState.CONFIRMED

        # 6. Handle unmatched tracks — identical state machine to File 2
        for ti in unm_t:
            t = active[ti]
            if   t.state == OcclusionState.TENTATIVE:
                t.state = OcclusionState.LOST
            elif t.state == OcclusionState.CONFIRMED:
                t.state          = OcclusionState.OCCLUDED
                t.is_kalman_pred = True
            elif t.state == OcclusionState.OCCLUDED:
                if t.occlusion_duration >= LONG_TERM_MAX_AGE:
                    t.state = OcclusionState.LOST
                else:
                    t.is_kalman_pred = True

        # 7. Spawn new tracks for unmatched detections
        for di in unm_d:
            det = detections[di]
            tid = self._next_id; self._next_id += 1
            kf  = KalmanPredictor()
            kf.initialize(det.bbox)
            self._tracks.append(Track(
                id=tid, bbox=det.bbox, class_id=det.class_id,
                confidence=det.confidence, state=OcclusionState.TENTATIVE,
                trajectory=[det.bbox.copy()], hits=1))
            self._kf[tid]      = kf
            self._gallery[tid] = [det_feats[di]]
            self._geom_occ[tid] = False

        # 8. Remove LOST tracks
        self._tracks = [t for t in self._tracks if t.state != OcclusionState.LOST]

        return [t for t in self._tracks
                if t.state in (OcclusionState.CONFIRMED, OcclusionState.OCCLUDED)]


# ============================================================================
# SECTION 11: GROUND-TRUTH HELPERS  (identical to File 2)
# ============================================================================

def _build_gt(annotations: dict, video_name: str):
    video_info = None
    for v in annotations.get("videos", []):
        folder = v.get("file_names", [""])[0].replace("\\", "/").split("/")[0]
        if folder == video_name:
            video_info = v; break
    if video_info is None:
        return None, {}
    vid_id  = video_info["id"]
    vid_len = video_info.get("length", len(video_info.get("file_names", [])))
    gt: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations.get("annotations", []):
        if ann["video_id"] != vid_id: continue
        tid = ann.get("instance_id") or ann.get("track_id") or ann["id"]
        for fi, bbox in enumerate(ann.get("bboxes", [])):
            if fi >= vid_len or not bbox or len(bbox) != 4: continue
            x, y, bw, bh = bbox
            if bw > 0 and bh > 0:
                gt[fi].append({"id": tid, "bbox": [x, y, x+bw, y+bh]})
    return video_info, gt


# ============================================================================
# SECTION 12: METRICS  (identical to File 2)
# ============================================================================

def _compute_detection_metrics(predictions, gt_by_frame, vid_len,
                                iou_thr=DET_IOU_THRESH):
    TP = FP = FN = TN = 0
    pred_lut = {p["frame_id"]: p for p in predictions}
    for fid in range(vid_len):
        gt_objs  = gt_by_frame.get(fid, [])
        pi       = pred_lut.get(fid)
        pred_bbs = [np.array(t[1]) for t in pi["tracks"]] if pi else []
        gt_bbs   = [np.array(g["bbox"]) for g in gt_objs]
        ng, np_  = len(gt_bbs), len(pred_bbs)
        if ng == 0 and np_ == 0: TN += 1; continue
        if ng == 0: FP += np_; continue
        if np_ == 0: FN += ng; continue
        mg = set(); mp = set()
        for pi2, gi, iv in sorted(
                [(pi2, gi, _iou(pred_bbs[pi2], gt_bbs[gi]))
                 for pi2 in range(np_) for gi in range(ng)],
                key=lambda x: -x[2]):
            if pi2 in mp or gi in mg: continue
            if iv >= iou_thr: TP += 1; mp.add(pi2); mg.add(gi)
        FP += np_ - len(mp); FN += ng - len(mg)
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


def _evaluate_video(predictions, annotations, video_name, img_w, img_h):
    video_info, gt_by_frame = _build_gt(annotations, video_name)
    if video_info is None: return None
    vid_len  = video_info.get("length", 0)
    img_diag = np.sqrt(img_w**2 + img_h**2) if (img_w and img_h) else 1.0
    acc = mm.MOTAccumulator(auto_id=True)
    pred_trajs: Dict[int, list] = defaultdict(list)
    gt_trajs:   Dict[int, list] = defaultdict(list)
    for pf in predictions:
        fid = pf["frame_id"]
        if fid >= vid_len: continue
        gt_objs = gt_by_frame.get(fid, [])
        gt_ids  = [o["id"]   for o in gt_objs]
        gt_bbs  = [o["bbox"] for o in gt_objs]
        p_ids   = [t[0] for t in pf["tracks"]]
        p_bbs   = [t[1] for t in pf["tracks"]]
        for pid, pb in zip(p_ids, p_bbs): pred_trajs[pid].append(pb)
        for gid, gb in zip(gt_ids, gt_bbs): gt_trajs[gid].append(gb)
        dist = (np.array([[1. - _iou(np.array(gb), np.array(pb))
                            for pb in p_bbs] for gb in gt_bbs])
                if gt_bbs and p_bbs else np.empty((len(gt_bbs), len(p_bbs))))
        acc.update(gt_ids, p_ids, dist)
    mh  = mm.metrics.create()
    smr = mh.compute(acc,
                     metrics=["mota", "idf1", "num_switches",
                               "mostly_tracked", "mostly_lost"],
                     name="acc")
    def _g(col, d=0.0):
        v = smr[col].values[0] if col in smr.columns else d
        return float(v) if not np.isnan(v) else d
    errs = []
    for pt in pred_trajs.values():
        for gt in gt_trajs.values():
            n = min(len(pt), len(gt))
            if n == 0: continue
            pc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in pt[:n]])
            gc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in gt[:n]])
            errs.extend(np.linalg.norm(pc - gc, axis=1))
    raw_ade = float(np.mean(errs)) if errs else 0.0
    ade     = raw_ade / img_diag if img_diag > 0 else raw_ade
    cfm  = _compute_detection_metrics(predictions, gt_by_frame, vid_len)
    tp, fp, fn, tn = cfm["TP"], cfm["FP"], cfm["FN"], cfm["TN"]
    prec = tp / (tp+fp) if (tp+fp) > 0 else 0.
    rec  = tp / (tp+fn) if (tp+fn) > 0 else 0.
    f1   = 2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.
    spec = tn / (tn+fp) if (tn+fp) > 0 else 0.
    return {
        "mota": max(-1., _g("mota")), "idf1": max(0., _g("idf1")),
        "ade": ade, "raw_ade": raw_ade,
        "id_switches": int(_g("num_switches")),
        "mt": int(_g("mostly_tracked")), "ml": int(_g("mostly_lost")),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": prec, "recall": rec, "f1": f1, "specificity": spec,
    }


# ============================================================================
# SECTION 13: TRAINING LOOP  (identical to File 2's ReIDTrainer)
# ============================================================================

class OccluTrackTrainer:
    LAMBDA_TRIP = 1.0
    LAMBDA_CE   = 0.5

    def __init__(self, extractor: OccluTrackExtractor,
                 detector: YOLODetector,
                 train_videos: List[Path],
                 annotations: dict,
                 crop_dataset: OVISCropDataset):
        self.extractor   = extractor
        self.detector    = detector
        self.train_vids  = train_videos
        self.annotations = annotations
        self.dataset     = crop_dataset
        self.device      = extractor._device
        self.weights_dir = EXPERIMENT_OUT / "weights" / METHOD_NAME
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = crop_dataset.num_classes
        self.classifier  = IDClassifier(REID_EMBED_DIM,
                                        self.num_classes).to(self.device)
        self.optimizer   = optim.Adam(
            list(extractor.parameters()) + list(self.classifier.parameters()),
            lr=LEARNING_RATE, weight_decay=5e-4)
        self.scheduler   = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=1e-6)
        self.triplet_loss = TripletLoss(TRIPLET_MARGIN)
        self.ce_loss      = nn.CrossEntropyLoss()

    def run(self) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  TRAINING [OccluTrack]  "
                    f"{len(self.dataset)} crops / {self.num_classes} IDs")
        logger.info(f"{'='*60}")
        labels  = [self.dataset.samples[i][-1] for i in range(len(self.dataset))]
        sampler = TripletBatchSampler(labels, P=4, K=4)
        loader  = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0)
        best_mota = float("-inf"); no_improve = 0; history = []
        for epoch in range(1, EPOCHS + 1):
            loss  = self._train_epoch(loader)
            val_m = self._val_mota(epoch)
            self.scheduler.step()
            history.append({"epoch": epoch, "loss": loss, "val_mota": val_m})
            logger.info(f"  [OccluTrack] Ep {epoch:02d}/{EPOCHS}  "
                        f"Loss={loss:.4f}  ValMOTA={val_m:.4f}  "
                        f"LR={self.scheduler.get_last_lr()[0]:.2e}")
            if val_m > best_mota:
                best_mota = val_m; no_improve = 0; self._save("best.pth")
                logger.info(f"    ✔ [OccluTrack] Best ValMOTA={best_mota:.4f}")
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PAT:
                    logger.info(f"  [OccluTrack] Early stop at epoch {epoch}.")
                    break
        self._save("last.pth")
        logger.info(f"  [OccluTrack] Training done. Best={best_mota:.4f}")
        return {"history": history, "best_mota": best_mota}

    def load_best(self):
        ckpt = self.weights_dir / "best.pth"
        if ckpt.exists():
            st = torch.load(ckpt, map_location=self.device)
            self.extractor.load_state_dict(st["reid_state"])
            logger.info(f"  [OccluTrack] Loaded best weights ← {ckpt}")
        else:
            logger.warning("  [OccluTrack] No best.pth — using last weights.")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.extractor.train(); self.classifier.train()
        total = 0.; n = 0
        for imgs, labels in loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)
            emb    = self.extractor(imgs)
            logits = self.classifier(emb)
            loss   = (self.LAMBDA_TRIP * self.triplet_loss(emb, labels)
                      + self.LAMBDA_CE * self.ce_loss(logits, labels))
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.extractor.parameters(), 5.0)
            self.optimizer.step()
            total += loss.item(); n += 1
        return total / max(1, n)

    def _val_mota(self, epoch: int) -> float:
        random.seed(RANDOM_SEED + epoch)
        sample  = random.sample(self.train_vids, min(5, len(self.train_vids)))
        tracker = OccluTracker(self.extractor)
        motas   = []
        for vp in sample:
            frames = sorted(vp.glob("*.jpg"))
            if not frames: continue
            first  = cv2.imread(str(frames[0]))
            if first is None: continue
            h, w   = first.shape[:2]
            tracker.reset(); preds = []
            for fi, fp in enumerate(frames):
                img = cv2.imread(str(fp))
                if img is None: continue
                dets = self.detector.detect(img)
                trks = tracker.update(dets, img)
                preds.append({"frame_id": fi,
                              "tracks": [(t.id, t.bbox.tolist(),
                                          t.class_id, t.confidence)
                                         for t in trks]})
            m = _evaluate_video(preds, self.annotations, vp.name, w, h)
            if m: motas.append(m["mota"])
        return float(np.mean(motas)) if motas else 0.0

    def _save(self, filename: str):
        torch.save({
            "reid_name":  METHOD_NAME,
            "params": {
                "n_stripes":        N_STRIPES,
                "visibility_thr":   VISIBILITY_THR,
                "occlu_iou_thr":    OCCLU_IOU_THR,
                "occlu_cost_relax": OCCLU_COST_RELAX,
                "app_weight":       APP_WEIGHT,
                "nn_budget":        NN_BUDGET,
            },
            "timestamp":  datetime.now().isoformat(),
            "reid_state": self.extractor.state_dict(),
        }, self.weights_dir / filename)


# ============================================================================
# SECTION 14: EVALUATION LOOP  (identical to File 2's ReIDEvaluator)
# ============================================================================

class OccluTrackEvaluator:
    def __init__(self, tracker: OccluTracker,
                 detector: YOLODetector,
                 eval_videos: List[Path],
                 annotations: dict):
        self.tracker     = tracker
        self.detector    = detector
        self.eval_vids   = eval_videos
        self.annotations = annotations
        self.vid_out = EXPERIMENT_OUT / "outputs" / METHOD_NAME / "videos"
        self.met_out = EXPERIMENT_OUT / "outputs" / METHOD_NAME / "metrics"
        self.csv_out = EXPERIMENT_OUT / "outputs" / METHOD_NAME / "csv"

    def run(self) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  EVALUATION [OccluTrack]  {len(self.eval_vids)} videos  "
                    f"[20% held-out eval split]")
        logger.info(f"{'='*60}")
        for d in [self.vid_out, self.met_out, self.csv_out]:
            d.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, vp in enumerate(self.eval_vids, 1):
            save_vid = (idx <= VISUAL_VIDEOS)
            vid_out  = self.vid_out / f"{vp.name}.mp4" if save_vid else None
            frames   = sorted(vp.glob("*.jpg"))
            if not frames: continue
            first    = cv2.imread(str(frames[0]))
            if first is None: continue
            h, w     = first.shape[:2]
            self.tracker.reset(); preds = []; writer = None
            if save_vid and vid_out:
                writer = cv2.VideoWriter(
                    str(vid_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS, (w, h))
            for fi, fp in enumerate(frames):
                img = cv2.imread(str(fp))
                if img is None: continue
                dets = self.detector.detect(img)
                trks = self.tracker.update(dets, img)
                preds.append({"frame_id": fi,
                              "tracks": [(t.id, t.bbox.tolist(),
                                          t.class_id, t.confidence)
                                         for t in trks]})
                if writer:
                    writer.write(_render_tracks(img, trks))
            if writer:
                writer.release()
                logger.info(f"  [OccluTrack] 🎬 Video saved → {vid_out}")
            m = _evaluate_video(preds, self.annotations, vp.name, w, h)
            if m:
                rows.append({"video": vp.name, **m})
                if idx % 50 == 0 or idx == len(self.eval_vids):
                    logger.info(f"  [OccluTrack] Progress: "
                                f"{idx}/{len(self.eval_vids)} videos…")
        pd.DataFrame(rows).to_csv(self.csv_out / "per_video_metrics.csv", index=False)
        agg = self._aggregate(rows)
        _print_method_summary(METHOD_NAME, agg)
        with open(self.met_out / "aggregated_metrics.json", "w") as f:
            json.dump(agg, f, indent=2)
        return agg

    @staticmethod
    def _aggregate(rows: List[dict]) -> dict:
        if not rows: return {"reid_name": METHOD_NAME}
        tp = int(np.sum([r["TP"] for r in rows]))
        fp = int(np.sum([r["FP"] for r in rows]))
        fn = int(np.sum([r["FN"] for r in rows]))
        tn = int(np.sum([r["TN"] for r in rows]))
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0.
        rec  = tp / (tp+fn) if (tp+fn) > 0 else 0.
        f1   = 2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.
        spec = tn / (tn+fp) if (tn+fp) > 0 else 0.
        return {
            "reid_name":   METHOD_NAME,
            "n_videos":    len(rows),
            "mota":        float(np.mean([r["mota"]        for r in rows])),
            "idf1":        float(np.mean([r["idf1"]        for r in rows])),
            "ade":         float(np.mean([r["ade"]         for r in rows])),
            "raw_ade":     float(np.mean([r["raw_ade"]     for r in rows])),
            "id_switches": int(np.sum(   [r["id_switches"] for r in rows])),
            "mt":          int(np.sum(   [r["mt"]          for r in rows])),
            "ml":          int(np.sum(   [r["ml"]          for r in rows])),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision":   round(prec, 6), "recall": round(rec, 6),
            "f1":          round(f1, 6),   "specificity": round(spec, 6),
        }


# ============================================================================
# SECTION 15: VIDEO RENDERING  (identical to File 2)
# ============================================================================

_PALETTE = [
    (255,56,56),(255,157,51),(51,255,255),(56,255,255),(255,56,132),
    (131,255,56),(56,131,255),(255,210,51),(51,255,131),(131,56,255),
    (255,56,210),(100,200,100),(200,100,200),(100,150,255),
]


def _render_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    img  = frame.copy()
    H, W = img.shape[:2]
    for t in tracks:
        x1 = int(max(0,    min(t.bbox[0], W-2)))
        y1 = int(max(0,    min(t.bbox[1], H-2)))
        x2 = int(max(x1+1, min(t.bbox[2], W-1)))
        y2 = int(max(y1+1, min(t.bbox[3], H-1)))
        if x2 - x1 < 2 or y2 - y1 < 2: continue
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
        cv2.putText(img, lbl, (x1, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for i in range(1, min(len(t.trajectory), 20)):
            p1 = (int((t.trajectory[-i  ][0] + t.trajectory[-i  ][2]) / 2),
                  int((t.trajectory[-i  ][1] + t.trajectory[-i  ][3]) / 2))
            p2 = (int((t.trajectory[-i-1][0] + t.trajectory[-i-1][2]) / 2),
                  int((t.trajectory[-i-1][1] + t.trajectory[-i-1][3]) / 2))
            cv2.line(img, p1, p2, col, 1)
    return img


# ============================================================================
# SECTION 16: RESULTS TABLE + GRAPHS  (identical to File 2)
# ============================================================================

def _print_method_summary(name: str, agg: dict):
    tp = agg.get("TP", 0); fp = agg.get("FP", 0)
    fn = agg.get("FN", 0); tn = agg.get("TN", 0)
    total = tp + fp + fn + tn
    acc   = (tp + tn) / total if total > 0 else 0.
    bar   = "─" * 58
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


def _save_comparison_graphs(all_results: List[dict]):
    if not all_results:
        return
    graph_dir = EXPERIMENT_OUT / "results" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    methods = [r["reid_name"] for r in all_results]

    _METHOD_COLORS = {
        "OccludedReID": "#E74C3C", "PGFA":      "#3498DB",
        "HOReID":       "#2ECC71", "PAT":        "#F39C12",
        "TransReID":    "#9B59B6", "OAMN":       "#1ABC9C",
        "OccluTrack":   "#FF6B35",   # distinct orange for OccluTrack
    }
    _DEFAULT_COLORS = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6","#1ABC9C","#FF6B35"]

    def col(name, default=0.0):
        return [float(r.get(name, default)) for r in all_results]

    pipeline_tag = "Trained YOLO11 + Kalman Filter + OccluTrack"

    # Chart 1 — Tracking metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Tracking Metrics — {pipeline_tag}",
                 fontsize=13, fontweight="bold", y=1.01)
    _bar_group(axes[0], methods,
               {"MOTA": col("mota"), "IDF1": col("idf1"), "ADE\n(norm.)": col("ade")},
               "Score Metrics (higher = better)", "Score  [0 – 1]",
               _METHOD_COLORS, _DEFAULT_COLORS)
    _bar_group(axes[1], methods,
               {"RawADE\n(pixels)": col("raw_ade"), "ID Switches": col("id_switches"),
                "Mostly\nTracked": col("mt"), "Mostly\nLost": col("ml")},
               "Count Metrics  (↓ better for IDsw, ML, RawADE)", "Count / Pixels",
               _METHOD_COLORS, _DEFAULT_COLORS)
    plt.tight_layout()
    fig.savefig(graph_dir / "1_tracking_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 2 — Confusion matrix
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Detection Confusion Matrix — {pipeline_tag}",
                 fontsize=13, fontweight="bold", y=1.01)
    n_methods = len(methods)
    x = np.arange(4); bar_w = 0.8 / n_methods
    for mi, (method, r) in enumerate(zip(methods, all_results)):
        color  = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        offset = (mi - n_methods/2 + 0.5) * bar_w
        vals   = [r.get("TP",0), r.get("FP",0), r.get("FN",0), r.get("TN",0)]
        bars   = ax.bar(x + offset, vals, bar_w * 0.85,
                        label=method, color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01, f"{int(val):,}",
                        ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(["TP", "FP", "FN", "TN"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, framealpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    fig.savefig(graph_dir / "2_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 3 — Derived metrics
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Derived Detection Metrics — {pipeline_tag}",
                 fontsize=13, fontweight="bold", y=1.01)
    for mi, (method, r) in enumerate(zip(methods, all_results)):
        color  = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
        offset = (mi - n_methods/2 + 0.5) * bar_w
        vals   = [r.get("precision",0), r.get("recall",0),
                  r.get("f1",0), r.get("specificity",0)]
        bars   = ax.bar(x + offset, vals, bar_w * 0.85,
                        label=method, color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision","Recall","F1","Specificity"],
                       fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score  [0 – 1]", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, framealpha=0.7)
    plt.tight_layout()
    fig.savefig(graph_dir / "3_derived_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 4 — Radar
    try:
        radar_metrics = ["MOTA","IDF1","Precision","Recall","F1","Specificity"]
        radar_keys    = ["mota","idf1","precision","recall","f1","specificity"]
        N      = len(radar_metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.suptitle(f"Overall Score Radar — {pipeline_tag}",
                     fontsize=12, fontweight="bold", y=1.02)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=9)
        ax.set_ylim(0, 1)
        for mi, (method, r) in enumerate(zip(methods, all_results)):
            vals  = [float(r.get(k, 0)) for k in radar_keys] + [float(r.get(radar_keys[0], 0))]
            color = _METHOD_COLORS.get(method, _DEFAULT_COLORS[mi % len(_DEFAULT_COLORS)])
            ax.plot(angles, vals, linewidth=1.8, label=method, color=color)
            ax.fill(angles, vals, alpha=0.08, color=color)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8.5)
        plt.tight_layout()
        fig.savefig(graph_dir / "4_radar_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"[Graphs] Radar chart skipped: {e}")

    logger.info(f"[Graphs] All charts saved to → {graph_dir}")


def _bar_group(ax, methods, values_dict, title, ylabel,
               method_colors, default_colors):
    metrics   = list(values_dict.keys())
    n_metrics = len(metrics)
    n_methods = len(methods)
    bar_h     = 0.8 / n_methods
    y_base    = np.arange(n_metrics)
    for mi, method in enumerate(methods):
        vals  = [values_dict[m][mi] for m in metrics]
        color = method_colors.get(method, default_colors[mi % len(default_colors)])
        bars  = ax.barh(y_base - mi*bar_h + (n_methods-1)*bar_h/2,
                        vals, bar_h*0.85, label=method, color=color,
                        alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            txt = (f"{val:.4f}" if val < 10 else f"{int(val):,}")
            ax.text(bar.get_width() + max(abs(bar.get_width())*0.01, 0.002),
                    bar.get_y() + bar.get_height()/2,
                    txt, va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.7)


def _append_to_comparison_csv(agg: dict):
    """
    Appends the OccluTrack result to reid_comparison.csv so it sits
    alongside the 6 Re-ID method results from File 2.
    """
    COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["reid_name","mota","idf1","ade","raw_ade","id_switches","mt","ml",
            "TP","FP","FN","TN","precision","recall","f1","specificity"]
    new_row = pd.DataFrame([{c: agg.get(c, "") for c in cols}])
    if COMPARE_CSV.exists():
        existing = pd.read_csv(COMPARE_CSV)
        # Remove any previous OccluTrack row to avoid duplicates
        existing = existing[existing["reid_name"] != METHOD_NAME]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(COMPARE_CSV, index=False)
    logger.info(f"[CSV] OccluTrack result appended → {COMPARE_CSV}")


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
    print("  MOT — OccluTrack Tracker (File 2 Pipeline)")
    _dev_label = (f"GPU · {torch.cuda.get_device_name(0)}" if DEVICE == "cuda"
                  else "Apple MPS" if DEVICE == "mps"
                  else f"CPU ({os.cpu_count()} threads)")
    print(f"  Device       : {_dev_label}")
    print(f"  Tracker      : OccluTrack (inter-track geometry occlusion scoring)")
    print(f"  Train dir    : {TRAIN_DIR}")
    print(f"  Outputs      : {EXPERIMENT_OUT}")
    print(f"  Epochs       : {EPOCHS}  (early-stop patience={EARLY_STOP_PAT})")
    print(f"  Split        : {int(TRAIN_SPLIT*100)}% train / "
          f"{100-int(TRAIN_SPLIT*100)}% eval  (no overlap)")
    print(f"  OccluTrack params:")
    print(f"    OCCLU_IOU_THR    = {OCCLU_IOU_THR}   (geometry occlusion gate)")
    print(f"    OCCLU_COST_RELAX = {OCCLU_COST_RELAX}  (threshold relaxation for occluded tracks)")
    print(f"    APP_WEIGHT       = {APP_WEIGHT}   (appearance vs IoU balance)")
    print(f"    N_STRIPES        = {N_STRIPES}       (stripe visibility bands)")
    print(f"    NN_BUDGET        = {NN_BUDGET}     (gallery depth per track)")
    print(f"    REID_LT_OCC_FRAMES = {REID_LT_OCC_FRAMES}  (long-term occlusion bypass)")
    print(f"    LONG_TERM_MAX_AGE  = {LONG_TERM_MAX_AGE} (max track lifetime, frames)")
    _yolo_display = YOLO_PRETRAINED_FALLBACK + "  ⚠ (pretrained fallback)"
    for _p in [YOLO_TRAINED_PATH] + YOLO_FALLBACK_PATHS:
        if Path(_p).exists():
            _yolo_display = _p; break
    print(f"  YOLO         : {_yolo_display}")
    print("="*70)

    # ── Validate paths ────────────────────────────────────────────────────────
    if not ANNOTATIONS_FILE.exists():
        sys.exit(f"[ERROR] Annotations not found: {ANNOTATIONS_FILE}")
    if not TRAIN_DIR.exists():
        sys.exit(f"[ERROR] Train directory not found: {TRAIN_DIR}")

    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    logger.info(f"Annotations loaded — {len(annotations.get('videos',[]))} videos.")

    all_vids = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    if not all_vids:
        sys.exit("[ERROR] No video folders found inside Train/")

    split_idx  = int(len(all_vids) * TRAIN_SPLIT)
    train_vids = all_vids[:split_idx]
    eval_vids  = all_vids[split_idx:]
    logger.info(f"Total: {len(all_vids)}  Train: {len(train_vids)}  "
                f"Eval: {len(eval_vids)}")

    # ── Step 1: Crop dataset ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 1 / 3 — Building crop dataset from OVIS annotations")
    print(f"{'─'*70}")
    crop_dataset = OVISCropDataset(
        video_root       = TRAIN_DIR,
        annotations_file = ANNOTATIONS_FILE,
        cache_dir        = CROP_CACHE,
        allowed_folders  = {v.name for v in train_vids},
    )
    if crop_dataset.num_classes < 2:
        sys.exit("[ERROR] Fewer than 2 valid identities found. "
                 "Check TRAIN_DIR and ANNOTATIONS_FILE.")

    # ── Step 2: YOLO ──────────────────────────────────────────────────────────
    detector = YOLODetector()

    # ── Step 3: Build extractor + tracker ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 2 / 3 — Training OccluTrack appearance extractor")
    print(f"{'─'*70}")
    set_seed(RANDOM_SEED)
    extractor = OccluTrackExtractor()
    extractor.to(extractor._device)

    trainer = OccluTrackTrainer(
        extractor    = extractor,
        detector     = detector,
        train_videos = train_vids,
        annotations  = annotations,
        crop_dataset = crop_dataset,
    )
    trainer.run()
    trainer.load_best()

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 3 / 3 — Evaluating OccluTrack on held-out eval split")
    print(f"{'─'*70}")
    tracker   = OccluTracker(extractor)
    evaluator = OccluTrackEvaluator(
        tracker     = tracker,
        detector    = detector,
        eval_videos = eval_vids,
        annotations = annotations,
    )
    agg = evaluator.run()

    # ── Results ───────────────────────────────────────────────────────────────
    if agg:
        _append_to_comparison_csv(agg)

        # Load all results from CSV to regenerate combined graphs
        if COMPARE_CSV.exists():
            all_results = pd.read_csv(COMPARE_CSV).to_dict("records")
        else:
            all_results = [agg]
        _save_comparison_graphs(all_results)

        logger.info(
            f"  [OccluTrack] DONE  "
            f"MOTA={agg.get('mota',0):.4f}  "
            f"IDF1={agg.get('idf1',0):.4f}  "
            f"F1={agg.get('f1',0):.4f}  "
            f"TP={agg.get('TP',0):,}  FP={agg.get('FP',0):,}  "
            f"FN={agg.get('FN',0):,}  TN={agg.get('TN',0):,}")
    else:
        logger.warning("[OccluTrack] No metrics returned — check paths.")

    print(f"\n  Output videos → {EXPERIMENT_OUT / 'outputs' / METHOD_NAME / 'videos'}")
    print(f"  Metrics JSON  → {EXPERIMENT_OUT / 'outputs' / METHOD_NAME / 'metrics'}")
    print(f"  Per-video CSV → {EXPERIMENT_OUT / 'outputs' / METHOD_NAME / 'csv'}")
    print(f"  Comparison CSV→ {COMPARE_CSV}")
    print(f"  Charts        → {EXPERIMENT_OUT / 'results' / 'graphs'}")
    logger.info("Run complete.")


if __name__ == "__main__":
    main()
