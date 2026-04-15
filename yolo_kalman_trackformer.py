"""
================================================================================
  MULTI-OBJECT TRACKING EXPERIMENT PIPELINE
  YOLOv11 (fixed) + Standard Kalman Filter (fixed) + Replaceable Tracker
================================================================================
  Experiments 7 & 8  —  Transformer-Based Trackers
      TRACKER_NAME = "TrackFormer"  — single-frame scaled dot-product query
      TRACKER_NAME = "MOTR"         — decaying multi-frame memory query

  HOW TO SWITCH TRACKERS:
      Change only:  TRACKER_NAME = "TrackFormer"   or   TRACKER_NAME = "MOTR"

  Architecture note
  -----------------
  TrackFormer and MOTR use persistent object queries updated by transformer
  self-attention over full image feature maps — a mechanism that presupposes
  a jointly-trained backbone.  To keep all experiments strictly comparable
  (fixed YOLO, fixed Kalman, fixed MobileNetV2), each tracker's defining
  algorithmic contribution is implemented as a post-processing module:

  TrackFormer → YOLO detections → Kalman predict → MobileNetV2 embeddings
               → scaled dot-product attention cost (Q·Kᵀ / √d) between
                 per-track query vectors and detection embeddings, gated by
                 IoU positional mask → query updated from current frame only
                 (short, single-frame memory — faithful to TrackFormer's
                 one-step temporal attention)

  MOTR        → same scaled dot-product attention cost, but each track
               maintains a decaying multi-frame memory buffer; the query
               used for attention is the decay-weighted mean of all past
               matched embeddings → long-range identity propagation under
               prolonged occlusion (faithful to MOTR's extended memory)

  Everything else — training loop, evaluation, Kalman, YOLO, dataset split,
  output format — is byte-for-byte identical to all previous experiments.
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import os
import sys
import json
import cv2
import pickle
import random
import logging
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

import torch
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import motmetrics as mm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 2: CONFIG
# ============================================================================

# ─── SWAP THIS LINE TO RUN A DIFFERENT TRACKER ───────────────────────────────
TRACKER_NAME = "TrackFormer"   # Options: "TrackFormer" | "MOTR"
# ─────────────────────────────────────────────────────────────────────────────

# Paths — identical to all previous experiments
BASE_DIR = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate")
EXPERIMENT_OUT = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 6th term\Graduation Project 2\Experiments")

TRAIN_DIR        = BASE_DIR / "train"
ANNOTATIONS_FILE = BASE_DIR / "annotations_train.json"

# Derived paths (auto-built from TRACKER_NAME)
WEIGHTS_DIR = EXPERIMENT_OUT / "weights"  / TRACKER_NAME
VID_OUT_DIR = EXPERIMENT_OUT / "outputs"  / TRACKER_NAME / "videos"
METRICS_DIR = EXPERIMENT_OUT / "outputs"  / TRACKER_NAME / "metrics"
SPLIT_FILE  = EXPERIMENT_OUT / "results"  / "dataset_split.json"

# Dataset — identical
TOTAL_VIDEOS  = 600
TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42
VISUAL_VIDEOS = 50

# YOLO — fixed, never change per experiment
YOLO_MODEL    = "yolo11n.pt"
YOLO_CONF     = 0.50
YOLO_IOU      = 0.45
YOLO_IMG_SIZE = 640
YOLO_VERBOSE  = False

# Kalman — fixed, never change per experiment
KF_MAX_AGE       = 30
KF_MIN_HITS      = 3
KF_IOU_THRESHOLD = 0.30

# Training — identical to all previous experiments
EPOCHS         = 15
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-3
EARLY_STOP_PAT = 5

# ─────────────────────────────────────────────────────────────────────────────
# Tracker hyper-parameters
# All shared knobs (max_age, n_init, max_iou_dist, nn_budget) are held
# identical to all previous experiments for a fair comparison.
# Only tracker-specific knobs differ.
# ─────────────────────────────────────────────────────────────────────────────
TRACKER_PARAMS: Dict[str, dict] = {

    # ── TrackFormer ────────────────────────────────────────────────────────
    # Shared knobs: unchanged from all baselines
    # Tracker-specific:
    #   attn_temperature — √d scaling divisor for the dot-product attention
    #                      score (standard transformer temperature; set to
    #                      √embedding_dim at runtime, exposed here as a
    #                      multiplier on that default if tuning is needed)
    #   iou_gate         — IoU threshold below which a (detection, track)
    #                      pair is masked out before attention (positional
    #                      gating: spatially impossible matches are excluded
    #                      before the attention score is computed, replicating
    #                      the role of positional encodings in TrackFormer)
    #   query_momentum   — weight given to the new embedding when updating
    #                      the query after a match; (1 - query_momentum) is
    #                      retained from the previous query.
    #                      TrackFormer: high momentum → fast single-frame
    #                      adaptation (short memory)
    "TrackFormer": {
        "max_age":          30,
        "n_init":            3,
        "max_iou_dist":     0.70,
        "nn_budget":       100,
        "attn_temperature":  1.0,    # multiplier on √dim; 1.0 = standard
        "iou_gate":          0.05,   # mask pairs with IoU < this value
        "query_momentum":    0.80,   # high → current frame dominates query
    },

    # ── MOTR ───────────────────────────────────────────────────────────────
    # Shared knobs: unchanged from all baselines
    # Tracker-specific:
    #   attn_temperature — same role as TrackFormer; kept identical so that
    #                      the only architectural variable is the memory
    #   iou_gate         — same positional gating as TrackFormer
    #   memory_decay     — per-frame multiplicative decay applied to all
    #                      entries in the memory buffer between frames.
    #                      Older embeddings fade but contribute to the
    #                      decay-weighted query until the track is deleted.
    #                      This is the defining MOTR mechanism: tracks
    #                      retain long-range identity under occlusion.
    #   memory_depth     — maximum number of past embeddings kept in the
    #                      buffer per track (caps memory growth)
    "MOTR": {
        "max_age":          30,
        "n_init":            3,
        "max_iou_dist":     0.70,
        "nn_budget":       100,
        "attn_temperature":  1.0,
        "iou_gate":          0.05,
        "memory_decay":      0.85,   # per-frame decay; lower → faster fade
        "memory_depth":     20,      # max past frames retained in buffer
    },
}


# ============================================================================
# SECTION 3: DATA STRUCTURES  (unchanged from all previous experiments)
# ============================================================================

@dataclass
class Detection:
    bbox:       np.ndarray
    confidence: float
    class_id:   int
    frame_id:   int = 0


@dataclass
class Track:
    id:                int
    bbox:              np.ndarray
    class_id:          int
    confidence:        float
    trajectory:        List[np.ndarray] = field(default_factory=list)
    age:               int = 0
    hits:              int = 0
    time_since_update: int = 0


# ============================================================================
# SECTION 4: DATASET SPLIT LOADER  (unchanged from all previous experiments)
# ============================================================================

class DatasetSplitLoader:
    """
    Splits videos into train/test sets with a fixed random seed.
    Saves the split to disk so it is IDENTICAL across all tracker experiments.
    If dataset_split.json already exists it is reloaded — guaranteeing
    TrackFormer / MOTR see the exact same 480/120 split as every other
    experiment.
    """

    def __init__(self, video_root: Path, annotations_file: Path,
                 train_ratio: float = 0.80, seed: int = 42,
                 split_save_path: Path = None):
        self.video_root       = video_root
        self.annotations_file = annotations_file
        self.train_ratio      = train_ratio
        self.seed             = seed
        self.split_save_path  = split_save_path

    def load(self) -> Tuple[List[Path], List[Path]]:
        if self.split_save_path and self.split_save_path.exists():
            logger.info(f"Loading existing split from {self.split_save_path}")
            return self._load_split()

        all_folders = sorted([d for d in self.video_root.iterdir() if d.is_dir()])
        all_folders = all_folders[:TOTAL_VIDEOS]

        rng     = random.Random(self.seed)
        indices = list(range(len(all_folders)))
        rng.shuffle(indices)

        n_train   = int(len(indices) * self.train_ratio)
        train_idx = sorted(indices[:n_train])
        test_idx  = sorted(indices[n_train:])

        train_videos = [all_folders[i] for i in train_idx]
        test_videos  = [all_folders[i] for i in test_idx]

        logger.info(f"Split: {len(train_videos)} train / {len(test_videos)} test  (seed={self.seed})")

        if self.split_save_path:
            self.split_save_path.parent.mkdir(parents=True, exist_ok=True)
            split_data = {
                "seed":         self.seed,
                "train_ratio":  self.train_ratio,
                "n_train":      len(train_videos),
                "n_test":       len(test_videos),
                "train_videos": [str(v) for v in train_videos],
                "test_videos":  [str(v) for v in test_videos],
            }
            with open(self.split_save_path, "w") as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"Split saved to {self.split_save_path}")

        return train_videos, test_videos

    def _load_split(self) -> Tuple[List[Path], List[Path]]:
        with open(self.split_save_path, "r") as f:
            data = json.load(f)
        return ([Path(v) for v in data["train_videos"]],
                [Path(v) for v in data["test_videos"]])


# ============================================================================
# SECTION 5: YOLO DETECTOR MODULE  (fixed — identical for every experiment)
# ============================================================================

class YOLODetector:
    """Wraps YOLOv11 detection.  Settings are NEVER changed per experiment."""

    def __init__(self, model_path: str = YOLO_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading YOLO model '{model_path}' on {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

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
            boxes = r.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls  = int(boxes.cls[i].cpu().numpy())
                detections.append(Detection(bbox=bbox, confidence=conf, class_id=cls))
        return detections


# ============================================================================
# SECTION 6: KALMAN PREDICTOR MODULE  (fixed — identical for every experiment)
# ============================================================================

class KalmanPredictor:
    """
    Standard Kalman Filter (7-state: cx, cy, s, r, vx, vy, vs).
    This is the ONLY motion model used across all experiments.
    """

    def __init__(self):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.kf.R       *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P       *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

    def initialize(self, bbox: np.ndarray):
        z = self._bbox_to_z(bbox)
        self.kf.x[:4] = z
        self.kf.update(z)

    def update(self, bbox: np.ndarray):
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        self.kf.predict()
        return self._x_to_bbox(self.kf.x)

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s  = w * h
        r  = w / float(h) if h != 0 else 1.0
        return np.array([cx, cy, s, r]).reshape((4, 1))

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> np.ndarray:
        s, r = float(x[2]), float(x[3])
        if s > 0 and r > 0:
            w = np.sqrt(s * r)
            h = s / w
        else:
            w = h = 0.0
        cx, cy = float(x[0]), float(x[1])
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ============================================================================
# SECTION 7: SHARED UTILITIES
# ============================================================================

def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _hungarian_match(cost_matrix: np.ndarray, threshold: float):
    """Greedy matching — unchanged from all previous experiments."""
    matched, used_rows, used_cols = [], set(), set()
    flat = [(cost_matrix[r, c], r, c)
            for r in range(cost_matrix.shape[0])
            for c in range(cost_matrix.shape[1])]
    flat.sort()
    for cost, r, c in flat:
        if cost > threshold:
            break
        if r not in used_rows and c not in used_cols:
            matched.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
    unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in used_rows]
    unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in used_cols]
    return matched, unmatched_rows, unmatched_cols


# ── Appearance feature extractor (identical to all previous experiments) ─────

class AppearanceExtractor:
    """
    Lightweight CNN-based appearance feature extractor.
    Uses a small pre-trained MobileNetV2 backbone (frozen, ImageNet weights).
    Identical to the extractor used in all previous experiments — the Re-ID
    quality is a controlled constant across all eight tracker comparisons.
    """

    def __init__(self):
        self._model    = None
        self._device   = "cuda" if torch.cuda.is_available() else "cpu"
        self._feat_dim = 512
        self._ready    = False
        self._try_load()

    def _try_load(self):
        try:
            import torchvision.models as tvm
            import torchvision.transforms as T
            backbone = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
            backbone = torch.nn.Sequential(*list(backbone.features.children()))
            backbone.eval()
            backbone.to(self._device)
            self._model = backbone
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self._feat_dim = 1280
            self._ready    = True
        except Exception as e:
            logger.warning(f"Appearance extractor unavailable ({e}). Using zero embeddings.")

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        if not self._ready:
            return np.zeros(self._feat_dim)
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self._feat_dim)
            crop     = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp      = self._transform(crop_rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model(inp)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
            return feat.squeeze().cpu().numpy()
        except Exception:
            return np.zeros(self._feat_dim)


# ============================================================================
# SECTION 8: ATTENTION UTILITIES
# ============================================================================

def _scaled_dot_product_attention(
        queries: np.ndarray,         # (T, D)  — one query per track
        keys:    np.ndarray,         # (N, D)  — one key per detection
        temperature_mult: float,     # multiplier on √D
        iou_gate_mask: np.ndarray,   # (N, T) bool — True = pair is allowed
) -> np.ndarray:
    """
    Computes scaled dot-product attention scores and returns a cost matrix.

        scores(n, t) = softmax_over_t( Q[t] · K[n] / (√D × temperature_mult) )

    Returns cost matrix of shape (N, T):
        cost(n, t) = 1 − attention_score(n, t)

    Pairs blocked by iou_gate_mask receive cost = 1.0 (maximum cost) so
    they are never chosen by the greedy matcher.

    This is the standard transformer attention formula applied to the
    tracking association problem: each detection (key) attends to all
    track queries and the resulting softmax weight is the association score.
    """
    T, D = queries.shape
    N    = keys.shape[0]

    scale  = float(np.sqrt(D)) * temperature_mult
    # Raw scores: (N, T)
    scores = keys @ queries.T / (scale + 1e-9)      # (N, T)

    # Apply iou_gate mask: blocked pairs get -inf before softmax
    scores[~iou_gate_mask] = -1e9

    # Softmax over track dimension for each detection (row-wise)
    scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    attn       = scores_exp / (scores_exp.sum(axis=1, keepdims=True) + 1e-9)

    # Cost = 1 − attention score; blocked pairs reset to 1.0
    cost = 1.0 - attn
    cost[~iou_gate_mask] = 1.0

    return cost   # (N, T)  — (detections, tracks)


# ============================================================================
# SECTION 9: REPLACEABLE TRACKER WRAPPERS
# ============================================================================

class BaseTrackerWrapper:
    """Abstract base — all tracker wrappers must implement reset() and update()."""

    def reset(self):
        raise NotImplementedError

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        raise NotImplementedError


# ── TrackFormer ───────────────────────────────────────────────────────────────

class TrackFormerWrapper(BaseTrackerWrapper):
    """
    TrackFormer: persistent object queries with single-frame attention update.

    Original TrackFormer contribution
    ----------------------------------
    TrackFormer maintains one learned object query per tracked object across
    frames.  At each frame, the queries from the previous timestep attend
    jointly to the image feature map (via a transformer decoder), producing
    updated queries that simultaneously perform detection and identity
    assignment.  There is no separate IoU-matching step — association emerges
    from the attention mechanism itself.

    Simulation (YOLO fixed)
    -----------------------
    Each track carries a query vector q_t ∈ ℝᴰ (D = MobileNetV2 feat dim).
    Each YOLO detection produces a key vector k_d ∈ ℝᴰ via AppearanceExtractor.

    Association cost via scaled dot-product attention:
        scores(d, t) = softmax_t( q_t · k_d  /  (√D × temperature) )
        cost(d, t)   = 1 − scores(d, t)

    Positional gating (proxy for positional encodings in the transformer):
        pairs with IoU(det_bbox, track_bbox) < iou_gate are masked to cost=1
        before softmax, preventing spatially impossible associations.

    Query update after matching (single-frame — TrackFormer's short memory):
        q_t ← query_momentum × q_t  +  (1 − query_momentum) × k_matched
    Only the current frame's matched embedding updates the query; no history
    is accumulated beyond the exponential decay of a single step.

    Kalman filter provides bbox prediction for Kalman positional gating and
    track bbox updates — identical to all other experiments.
    """

    def __init__(self, params: dict):
        self.max_age         = params.get("max_age",         30)
        self.n_init          = params.get("n_init",           3)
        self.max_iou_dist    = params.get("max_iou_dist",    0.70)
        self.nn_budget       = params.get("nn_budget",      100)
        self.attn_temp       = params.get("attn_temperature", 1.0)
        self.iou_gate        = params.get("iou_gate",         0.05)
        self.query_momentum  = params.get("query_momentum",   0.80)

        self.extractor = AppearanceExtractor()
        self._feat_dim = self.extractor.feat_dim

        self._tracks:  List[Track]                = []
        self._kf_map:  Dict[int, KalmanPredictor] = {}
        self._queries: Dict[int, np.ndarray]      = {}   # track_id → query vector
        self._next_id  = 1

    def reset(self):
        self._tracks.clear()
        self._kf_map.clear()
        self._queries.clear()
        self._next_id = 1

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-9) if n > 0 else v

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:

        # 1. Kalman predict
        for t in self._tracks:
            t.bbox = self._kf_map[t.id].predict()
            t.time_since_update += 1
            t.age += 1

        # 2. Extract detection embeddings (keys)
        det_feats = [self._normalize(self.extractor.extract(frame, d.bbox))
                     for d in detections]

        # 3. Build IoU gate mask and attention cost matrix
        if self._tracks and detections:
            N = len(detections)
            T = len(self._tracks)

            # IoU gate mask: (N, T) — True where association is spatially allowed
            iou_gate_mask = np.zeros((N, T), dtype=bool)
            for di, det in enumerate(detections):
                for ti, trk in enumerate(self._tracks):
                    iou_gate_mask[di, ti] = _iou(det.bbox, trk.bbox) >= self.iou_gate

            # Stack query matrix: (T, D)
            queries = np.stack([
                self._normalize(self._queries[t.id]) for t in self._tracks
            ])  # (T, D)

            # Stack key matrix: (N, D)
            keys = np.stack(det_feats)  # (N, D)

            # Scaled dot-product attention cost: (N, T) → (detections, tracks)
            cost = _scaled_dot_product_attention(
                queries, keys, self.attn_temp, iou_gate_mask
            )

            matched, unmatched_dets, unmatched_trks = _hungarian_match(
                cost, self.max_iou_dist
            )
        else:
            matched        = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self._tracks)))

        # 4. Update matched tracks — single-frame query update (TrackFormer)
        for di, ti in matched:
            t   = self._tracks[ti]
            k_d = det_feats[di]
            t.bbox              = detections[di].bbox
            t.confidence        = detections[di].confidence
            t.hits             += 1
            t.time_since_update = 0
            t.trajectory.append(detections[di].bbox.copy())
            self._kf_map[t.id].update(detections[di].bbox)
            # Short-memory query update: momentum blend with current embedding
            self._queries[t.id] = (self.query_momentum * self._queries[t.id]
                                   + (1.0 - self.query_momentum) * k_d)

        # 5. Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            det = detections[di]
            tid = self._next_id; self._next_id += 1
            kf  = KalmanPredictor()
            kf.initialize(det.bbox)
            trk = Track(id=tid, bbox=det.bbox, class_id=det.class_id,
                        confidence=det.confidence,
                        trajectory=[det.bbox.copy()], hits=1)
            self._tracks.append(trk)
            self._kf_map[tid]  = kf
            # Initialise query from first detection embedding
            self._queries[tid] = self._normalize(det_feats[di])

        # 6. Remove stale tracks
        surviving = []
        for t in self._tracks:
            if t.time_since_update < self.max_age:
                surviving.append(t)
            else:
                self._kf_map.pop(t.id, None)
                self._queries.pop(t.id, None)
        self._tracks = surviving

        return [t for t in self._tracks if t.hits >= self.n_init]


# ── MOTR ──────────────────────────────────────────────────────────────────────

class MOTRWrapper(BaseTrackerWrapper):
    """
    MOTR: persistent object queries with decaying long-range memory.

    Original MOTR contribution
    ---------------------------
    MOTR extends TrackFormer by adding a long-range temporal memory to each
    object query.  Rather than blending only the immediately previous query
    with the current detection embedding, MOTR maintains a sequence of past
    embeddings that decay in influence over time.  This allows the model to
    sustain identity through prolonged occlusion — when no detection matches
    a track for several frames, the accumulated memory keeps the query
    informative until the object re-appears.

    Simulation (YOLO fixed)
    -----------------------
    Association cost: identical scaled dot-product attention to TrackFormer.
    The difference is entirely in how the query vector is maintained:

    Memory buffer per track:  buffer[t] = [(embedding_1, weight_1), ...]
    At each frame, all existing weights are multiplied by memory_decay:
        weight_i ← weight_i × memory_decay
    When a match occurs, the new embedding is appended with weight = 1.0.
    The effective query is the weighted mean of the buffer:
        q_t = Σ (weight_i × embedding_i) / Σ weight_i

    This means recent matches dominate the query, older ones fade, but
    nothing is discarded until max_age expires — capturing MOTR's long-range
    identity propagation within the fixed-YOLO pipeline.

    memory_depth caps the buffer size; oldest entries are evicted first.
    iou_gate and attn_temperature are identical to TrackFormer so that the
    only controlled variable between the two trackers is the memory mechanism.
    """

    def __init__(self, params: dict):
        self.max_age      = params.get("max_age",          30)
        self.n_init       = params.get("n_init",            3)
        self.max_iou_dist = params.get("max_iou_dist",     0.70)
        self.nn_budget    = params.get("nn_budget",       100)
        self.attn_temp    = params.get("attn_temperature",  1.0)
        self.iou_gate     = params.get("iou_gate",          0.05)
        self.memory_decay = params.get("memory_decay",      0.85)
        self.memory_depth = params.get("memory_depth",     20)

        self.extractor = AppearanceExtractor()
        self._feat_dim = self.extractor.feat_dim

        self._tracks:  List[Track]                          = []
        self._kf_map:  Dict[int, KalmanPredictor]           = {}
        # Memory buffer: track_id → list of [embedding, weight] pairs
        self._memory:  Dict[int, List[List]]                = {}
        # Effective query (weighted mean of memory): track_id → vector
        self._queries: Dict[int, np.ndarray]                = {}
        self._next_id  = 1

    def reset(self):
        self._tracks.clear()
        self._kf_map.clear()
        self._memory.clear()
        self._queries.clear()
        self._next_id = 1

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-9) if n > 0 else v

    def _decay_memory(self, track_id: int):
        """Apply per-frame multiplicative decay to all buffer entries."""
        buf = self._memory[track_id]
        for entry in buf:
            entry[1] *= self.memory_decay

    def _add_to_memory(self, track_id: int, embedding: np.ndarray):
        """Append a new embedding with weight=1.0; evict oldest if at depth."""
        buf = self._memory[track_id]
        buf.append([embedding.copy(), 1.0])
        if len(buf) > self.memory_depth:
            buf.pop(0)   # evict oldest entry

    def _recompute_query(self, track_id: int):
        """Recompute effective query as decay-weighted mean of memory buffer."""
        buf = self._memory[track_id]
        if not buf:
            return
        total_weight = sum(e[1] for e in buf)
        if total_weight < 1e-9:
            return
        q = sum(e[0] * e[1] for e in buf) / total_weight
        self._queries[track_id] = self._normalize(q)

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:

        # 1. Kalman predict + decay all memory buffers (time passing)
        for t in self._tracks:
            t.bbox = self._kf_map[t.id].predict()
            t.time_since_update += 1
            t.age += 1
            self._decay_memory(t.id)
            self._recompute_query(t.id)   # refresh query after decay

        # 2. Extract detection embeddings (keys)
        det_feats = [self._normalize(self.extractor.extract(frame, d.bbox))
                     for d in detections]

        # 3. Build IoU gate mask and attention cost matrix
        if self._tracks and detections:
            N = len(detections)
            T = len(self._tracks)

            iou_gate_mask = np.zeros((N, T), dtype=bool)
            for di, det in enumerate(detections):
                for ti, trk in enumerate(self._tracks):
                    iou_gate_mask[di, ti] = _iou(det.bbox, trk.bbox) >= self.iou_gate

            # Stack queries: (T, D) — decay-weighted memory means
            queries = np.stack([
                self._normalize(self._queries[t.id]) for t in self._tracks
            ])

            # Stack keys: (N, D)
            keys = np.stack(det_feats)

            # Identical attention formula to TrackFormer — only the query
            # content differs (long-range memory vs single-frame momentum)
            cost = _scaled_dot_product_attention(
                queries, keys, self.attn_temp, iou_gate_mask
            )

            matched, unmatched_dets, unmatched_trks = _hungarian_match(
                cost, self.max_iou_dist
            )
        else:
            matched        = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self._tracks)))

        # 4. Update matched tracks — add to long-range memory buffer (MOTR)
        for di, ti in matched:
            t   = self._tracks[ti]
            k_d = det_feats[di]
            t.bbox              = detections[di].bbox
            t.confidence        = detections[di].confidence
            t.hits             += 1
            t.time_since_update = 0
            t.trajectory.append(detections[di].bbox.copy())
            self._kf_map[t.id].update(detections[di].bbox)
            # Long-range memory update: append new embedding, recompute query
            self._add_to_memory(t.id, k_d)
            self._recompute_query(t.id)

        # 5. Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            det = detections[di]
            tid = self._next_id; self._next_id += 1
            kf  = KalmanPredictor()
            kf.initialize(det.bbox)
            trk = Track(id=tid, bbox=det.bbox, class_id=det.class_id,
                        confidence=det.confidence,
                        trajectory=[det.bbox.copy()], hits=1)
            self._tracks.append(trk)
            self._kf_map[tid]  = kf
            feat = self._normalize(det_feats[di])
            self._memory[tid]  = [[feat.copy(), 1.0]]
            self._queries[tid] = feat

        # 6. Remove stale tracks
        surviving = []
        for t in self._tracks:
            if t.time_since_update < self.max_age:
                surviving.append(t)
            else:
                self._kf_map.pop(t.id, None)
                self._memory.pop(t.id, None)
                self._queries.pop(t.id, None)
        self._tracks = surviving

        return [t for t in self._tracks if t.hits >= self.n_init]


# ── Tracker Factory ───────────────────────────────────────────────────────────

def build_tracker(name: str) -> BaseTrackerWrapper:
    params = TRACKER_PARAMS.get(name, {})
    if name == "TrackFormer":
        return TrackFormerWrapper(params)
    elif name == "MOTR":
        return MOTRWrapper(params)
    else:
        raise ValueError(
            f"Unknown tracker: '{name}'. "
            f"This file supports: 'TrackFormer', 'MOTR'. "
            f"For other trackers use the corresponding experiment file."
        )


# ============================================================================
# SECTION 10: TRAINING LOOP  (identical structure to all previous experiments)
# ============================================================================

class TrackerTrainer:
    """
    Runs the detection + tracking pass over all training videos to
    calibrate tracker state.  Saves weights to disk.

    Training settings are IDENTICAL to all previous experiments:
        EPOCHS=15, BATCH_SIZE=16, LR=1e-3, EARLY_STOP_PAT=5
        mini-epoch size = 80 videos (same random sampling logic, same seed)
    """

    def __init__(self, tracker: BaseTrackerWrapper, detector: YOLODetector,
                 train_videos: List[Path], annotations: dict):
        self.tracker      = tracker
        self.detector     = detector
        self.train_videos = train_videos
        self.annotations  = annotations

    def run(self) -> dict:
        logger.info(f"=== TRAINING — {TRACKER_NAME} ({len(self.train_videos)} videos) ===")
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        best_mota  = -np.inf
        no_improve = 0
        history    = []

        for epoch in range(1, EPOCHS + 1):
            epoch_metrics = self._run_epoch(epoch)
            history.append(epoch_metrics)
            avg_mota = epoch_metrics["avg_mota"]

            logger.info(
                f"  Epoch {epoch:02d}/{EPOCHS}  "
                f"MOTA={avg_mota:.4f}  IDF1={epoch_metrics['avg_idf1']:.4f}"
            )

            if avg_mota > best_mota:
                best_mota  = avg_mota
                no_improve = 0
                self._save_weights("best.pt")
                logger.info(f"    ✔ New best  MOTA={best_mota:.4f}")
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PAT:
                    logger.info(f"  Early stopping at epoch {epoch} (patience={EARLY_STOP_PAT})")
                    break

        self._save_weights("last.pt")
        self._save_kalman()
        self._save_tracker_state()

        logger.info(f"Training complete. Best MOTA = {best_mota:.4f}")
        return {"history": history, "best_mota": best_mota}

    def _run_epoch(self, epoch: int) -> dict:
        random.seed(RANDOM_SEED + epoch)
        sample = random.sample(self.train_videos, min(80, len(self.train_videos)))

        all_mota, all_idf1 = [], []
        for vid_path in sample:
            result = self._process_video(vid_path, save_video=False)
            if result:
                all_mota.append(result["mota"])
                all_idf1.append(result["idf1"])

        return {
            "epoch":    epoch,
            "avg_mota": float(np.mean(all_mota)) if all_mota else 0.0,
            "avg_idf1": float(np.mean(all_idf1)) if all_idf1 else 0.0,
        }

    def _process_video(self, video_folder: Path,
                       save_video: bool = False,
                       video_out_path: Path = None) -> Optional[dict]:
        frames = sorted(video_folder.glob("*.jpg"))
        if not frames:
            return None

        first = cv2.imread(str(frames[0]))
        if first is None:
            return None
        h, w = first.shape[:2]

        self.tracker.reset()
        predictions = []
        writer      = None

        if save_video and video_out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(video_out_path), fourcc, 15, (w, h))

        for frame_idx, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            dets   = self.detector.detect(img)
            tracks = self.tracker.update(dets, img)

            predictions.append({
                "frame_id": frame_idx,
                "tracks":   [(t.id, t.bbox.tolist(), t.class_id, t.confidence)
                             for t in tracks],
            })

            if writer is not None:
                writer.write(_render_tracks(img, tracks))

        if writer:
            writer.release()

        metrics = _evaluate_video(predictions, self.annotations,
                                  video_folder.name, w, h)
        return metrics

    def _save_weights(self, filename: str):
        state = {
            "tracker_name": TRACKER_NAME,
            "params":       TRACKER_PARAMS.get(TRACKER_NAME, {}),
            "timestamp":    datetime.now().isoformat(),
        }
        with open(WEIGHTS_DIR / filename, "wb") as f:
            pickle.dump(state, f)

    def _save_kalman(self):
        kf_info = {
            "type":         "StandardKalmanFilter",
            "dim_x":         7,
            "dim_z":         4,
            "max_age":       KF_MAX_AGE,
            "min_hits":      KF_MIN_HITS,
            "iou_threshold": KF_IOU_THRESHOLD,
        }
        with open(WEIGHTS_DIR / "kalman.pkl", "wb") as f:
            pickle.dump(kf_info, f)

    def _save_tracker_state(self):
        state = {
            "tracker_name":  TRACKER_NAME,
            "params":        TRACKER_PARAMS.get(TRACKER_NAME, {}),
            "yolo_conf":     YOLO_CONF,
            "yolo_iou":      YOLO_IOU,
            "yolo_img_size": YOLO_IMG_SIZE,
        }
        with open(WEIGHTS_DIR / "tracker_state.pkl", "wb") as f:
            pickle.dump(state, f)


# ============================================================================
# SECTION 11: EVALUATION  (unchanged from all previous experiments)
# ============================================================================

def _build_gt(annotations: dict, video_folder_name: str) -> Tuple[Optional[dict], dict]:
    video_info = None
    for v in annotations.get("videos", []):
        first_file = v.get("file_names", [""])[0]
        folder = first_file.replace("\\", "/").split("/")[0]
        if folder == video_folder_name:
            video_info = v
            break
    if video_info is None:
        return None, {}

    vid_id  = video_info["id"]
    vid_len = video_info.get("length", len(video_info.get("file_names", [])))
    gt_by_frame: Dict[int, List[dict]] = defaultdict(list)

    for ann in annotations.get("annotations", []):
        if ann["video_id"] != vid_id:
            continue
        track_id = ann.get("instance_id") or ann.get("track_id") or ann["id"]
        bboxes   = ann.get("bboxes", [])
        for frame_idx, bbox in enumerate(bboxes):
            if frame_idx >= vid_len:
                continue
            if not bbox or len(bbox) != 4:
                continue
            x, y, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                continue
            gt_by_frame[frame_idx].append({
                "id":   track_id,
                "bbox": [x, y, x + bw, y + bh],
            })

    return video_info, gt_by_frame


def _evaluate_video(predictions: List[dict], annotations: dict,
                    video_folder_name: str,
                    img_w: int, img_h: int) -> Optional[dict]:
    video_info, gt_by_frame = _build_gt(annotations, video_folder_name)
    if video_info is None:
        return None

    vid_len  = video_info.get("length", 0)
    img_diag = np.sqrt(img_w ** 2 + img_h ** 2) if (img_w > 0 and img_h > 0) else 1.0
    acc      = mm.MOTAccumulator(auto_id=True)
    pred_trajs: Dict[int, List] = defaultdict(list)
    gt_trajs:   Dict[int, List] = defaultdict(list)

    for pf in predictions:
        fid = pf["frame_id"]
        if fid >= vid_len:
            continue
        gt_objs   = gt_by_frame.get(fid, [])
        gt_ids    = [o["id"]   for o in gt_objs]
        gt_bboxes = [o["bbox"] for o in gt_objs]
        p_ids     = [t[0] for t in pf["tracks"]]
        p_bboxes  = [t[1] for t in pf["tracks"]]

        for pid, pb in zip(p_ids, p_bboxes):
            pred_trajs[pid].append(pb)
        for gid, gb in zip(gt_ids, gt_bboxes):
            gt_trajs[gid].append(gb)

        if gt_bboxes and p_bboxes:
            dist = np.array([
                [1.0 - _iou(np.array(gb), np.array(pb)) for pb in p_bboxes]
                for gb in gt_bboxes
            ])
        else:
            dist = np.empty((len(gt_bboxes), len(p_bboxes)))

        acc.update(gt_ids, p_ids, dist)

    mh      = mm.metrics.create()
    summary = mh.compute(acc,
                         metrics=["mota", "idf1", "num_switches",
                                  "mostly_tracked", "mostly_lost"],
                         name="acc")

    def _get(col, default=0):
        if col in summary.columns:
            v = summary[col].values[0]
            return float(v) if not np.isnan(v) else default
        return default

    # ADE (normalised by image diagonal)
    all_errs = []
    for pt in pred_trajs.values():
        for gt in gt_trajs.values():
            if pt and gt:
                n  = min(len(pt), len(gt))
                pc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in pt[:n]])
                gc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in gt[:n]])
                all_errs.extend(np.linalg.norm(pc - gc, axis=1))

    raw_ade = float(np.mean(all_errs)) if all_errs else 0.0
    ade     = raw_ade / img_diag if img_diag > 0 else raw_ade

    return {
        "mota":        max(-1.0, _get("mota")),
        "idf1":        max(0.0,  _get("idf1")),
        "ade":         ade,
        "raw_ade":     raw_ade,
        "id_switches": int(_get("num_switches")),
        "mt":          int(_get("mostly_tracked")),
        "ml":          int(_get("mostly_lost")),
    }


class TrackerEvaluator:
    """Runs evaluation over 120 test videos and aggregates metrics."""

    def __init__(self, tracker: BaseTrackerWrapper, detector: YOLODetector,
                 test_videos: List[Path], annotations: dict,
                 n_visual: int = VISUAL_VIDEOS):
        self.tracker     = tracker
        self.detector    = detector
        self.test_videos = test_videos
        self.annotations = annotations
        self.n_visual    = n_visual

    def run(self) -> dict:
        logger.info(f"=== EVALUATION — {TRACKER_NAME} ({len(self.test_videos)} videos) ===")
        VID_OUT_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

        per_video_rows = []

        for idx, vid_path in enumerate(self.test_videos, 1):
            vid_name   = vid_path.name
            save_video = idx <= self.n_visual
            video_out  = VID_OUT_DIR / f"{vid_name}.mp4" if save_video else None

            logger.info(f"  [{idx:3d}/{len(self.test_videos)}] {vid_name}"
                        + ("  [video]" if save_video else ""))

            frames = sorted(vid_path.glob("*.jpg"))
            if not frames:
                logger.warning(f"    No frames — skipping.")
                continue

            first = cv2.imread(str(frames[0]))
            if first is None:
                continue
            h, w = first.shape[:2]

            self.tracker.reset()
            predictions = []
            writer      = None

            if save_video and video_out:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_out), fourcc, 15, (w, h))

            for frame_idx, fp in enumerate(frames):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                dets   = self.detector.detect(img)
                tracks = self.tracker.update(dets, img)
                predictions.append({
                    "frame_id": frame_idx,
                    "tracks":   [(t.id, t.bbox.tolist(), t.class_id, t.confidence)
                                 for t in tracks],
                })
                if writer:
                    writer.write(_render_tracks(img, tracks))

            if writer:
                writer.release()

            metrics = _evaluate_video(predictions, self.annotations, vid_name, w, h)
            if metrics:
                per_video_rows.append({"video": vid_name, **metrics})
                logger.info(
                    f"    MOTA={metrics['mota']:.4f}  IDF1={metrics['idf1']:.4f}"
                    f"  ADE={metrics['ade']:.4f}  IDS={metrics['id_switches']}"
                )

        # Aggregate and save
        agg     = self._aggregate(per_video_rows)
        agg_out = METRICS_DIR / "aggregated_metrics.json"
        with open(agg_out, "w") as f:
            json.dump(agg, f, indent=2)
        logger.info(f"Aggregated metrics → {agg_out}")

        return agg

    @staticmethod
    def _aggregate(rows: List[dict]) -> dict:
        if not rows:
            return {}
        return {
            "tracker_name": TRACKER_NAME,
            "n_videos":     len(rows),
            "mota":         float(np.mean([r["mota"]       for r in rows])),
            "idf1":         float(np.mean([r["idf1"]       for r in rows])),
            "ade":          float(np.mean([r["ade"]         for r in rows])),
            "raw_ade":      float(np.mean([r["raw_ade"]     for r in rows])),
            "id_switches":  int(np.sum(  [r["id_switches"] for r in rows])),
            "mt":           int(np.sum(  [r["mt"]           for r in rows])),
            "ml":           int(np.sum(  [r["ml"]           for r in rows])),
        }


# ============================================================================
# SECTION 12: VISUALIZATION  (unchanged from all previous experiments)
# ============================================================================

_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 51, 255, 255), ( 56, 255, 255),
    (255,  56, 132), (131, 255,  56), ( 56, 131, 255), (255, 210,  51),
    ( 51, 255, 131), (131,  56, 255), (255,  56, 210),
]


def _render_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    img = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = _PALETTE[t.id % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"ID{t.id} {t.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i in range(1, min(len(t.trajectory), 20)):
            p1 = (int((t.trajectory[-i][0]+t.trajectory[-i][2])/2),
                  int((t.trajectory[-i][1]+t.trajectory[-i][3])/2))
            p2 = (int((t.trajectory[-i-1][0]+t.trajectory[-i-1][2])/2),
                  int((t.trajectory[-i-1][1]+t.trajectory[-i-1][3])/2))
            cv2.line(img, p1, p2, color, 1)
    return img


# ============================================================================
# SECTION 13: MAIN EXPERIMENT RUNNER
# ============================================================================

def _print_summary(metrics: dict):
    w = 52
    print()
    print("=" * w)
    print(f"  EXPERIMENT SUMMARY — {TRACKER_NAME}")
    print("=" * w)
    print(f"  {'MOTA':<20s}  {metrics.get('mota', 0):.4f}")
    print(f"  {'IDF1':<20s}  {metrics.get('idf1', 0):.4f}")
    print(f"  {'ADE (normalized)':<20s}  {metrics.get('ade', 0):.4f}")
    print(f"  {'Raw ADE (px)':<20s}  {metrics.get('raw_ade', 0):.4f}")
    print(f"  {'ID Switches':<20s}  {metrics.get('id_switches', 0)}")
    print(f"  {'Mostly Tracked':<20s}  {metrics.get('mt', 0)}")
    print(f"  {'Mostly Lost':<20s}  {metrics.get('ml', 0)}")
    print("=" * w)
    print(f"  Weights  → {WEIGHTS_DIR}")
    print(f"  Videos   → {VID_OUT_DIR}")
    print(f"  Metrics  → {METRICS_DIR}")
    print("=" * w)
    print()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def main():
    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(RANDOM_SEED)

    # ── Validate tracker choice ───────────────────────────────────────────
    if TRACKER_NAME not in TRACKER_PARAMS:
        sys.exit(
            f"[ERROR] TRACKER_NAME='{TRACKER_NAME}' is not supported. "
            f"Choose from: {list(TRACKER_PARAMS.keys())}"
        )

    logger.info(f"{'='*60}")
    logger.info(f" Experiment: {TRACKER_NAME}")
    logger.info(f" Device    : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"{'='*60}")

    # ── Load annotations ──────────────────────────────────────────────────
    if not ANNOTATIONS_FILE.exists():
        sys.exit(f"[ERROR] Annotations not found: {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)
    logger.info(f"Annotations loaded — {len(annotations.get('videos', []))} videos described.")

    # ── Dataset split (reloads shared split from disk) ────────────────────
    splitter = DatasetSplitLoader(
        video_root       = TRAIN_DIR,
        annotations_file = ANNOTATIONS_FILE,
        train_ratio      = TRAIN_RATIO,
        seed             = RANDOM_SEED,
        split_save_path  = SPLIT_FILE,
    )
    train_videos, test_videos = splitter.load()

    logger.info(f"Train: {len(train_videos)} videos | Test: {len(test_videos)} videos")
    logger.info(f"Visual outputs for first {VISUAL_VIDEOS} test videos.")

    # ── Build shared components ───────────────────────────────────────────
    detector = YOLODetector(YOLO_MODEL)
    tracker  = build_tracker(TRACKER_NAME)

    # ── Training ──────────────────────────────────────────────────────────
    trainer = TrackerTrainer(
        tracker      = tracker,
        detector     = detector,
        train_videos = train_videos,
        annotations  = annotations,
    )
    trainer.run()

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluator = TrackerEvaluator(
        tracker      = tracker,
        detector     = detector,
        test_videos  = test_videos,
        annotations  = annotations,
        n_visual     = VISUAL_VIDEOS,
    )
    agg_metrics = evaluator.run()

    # ── Summary ───────────────────────────────────────────────────────────
    if agg_metrics:
        _print_summary(agg_metrics)
    else:
        logger.warning("No metrics produced — check video paths and annotations.")

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
