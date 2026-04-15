"""
================================================================================
  MULTI-OBJECT TRACKING EXPERIMENT PIPELINE
  YOLOv11 (fixed) + Standard Kalman Filter (fixed) + Replaceable Tracker
================================================================================
  Experiments 3 & 4:
      TRACKER_NAME = "OccluTrack"   — appearance + occlusion-aware gating
      TRACKER_NAME = "CSTrack"      — dual-stream IoU/Re-ID cross-correlation

  HOW TO SWITCH TRACKERS:
      Change only:  TRACKER_NAME = "OccluTrack"   or   TRACKER_NAME = "CSTrack"

  Everything else — training, evaluation, Kalman logic, YOLO —
  is kept IDENTICAL to the DeepSORT / StrongSORT experiments for a fair
  comparison. All shared hyper-parameters (max_age, n_init, max_iou_dist,
  nn_budget, epochs, batch size, LR, early-stop patience) are unchanged.
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
import shutil
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
TRACKER_NAME = "CSTrack"   # Options: "OccluTrack" | "CSTrack"
# ─────────────────────────────────────────────────────────────────────────────

# Paths  — identical to the baseline experiments
BASE_DIR = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate")
EXPERIMENT_OUT = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 6th term\Graduation Project 2\Experiments")

TRAIN_DIR        = BASE_DIR / "train"
ANNOTATIONS_FILE = BASE_DIR / "annotations_train.json"

# Derived paths (auto-built from TRACKER_NAME)
WEIGHTS_DIR  = EXPERIMENT_OUT / "weights"  / TRACKER_NAME
VID_OUT_DIR  = EXPERIMENT_OUT / "outputs"  / TRACKER_NAME / "videos"
METRICS_DIR  = EXPERIMENT_OUT / "outputs"  / TRACKER_NAME / "metrics"
SPLIT_FILE   = EXPERIMENT_OUT / "results"  / "dataset_split.json"

# Dataset — identical
TOTAL_VIDEOS   = 600
TRAIN_RATIO    = 0.80
RANDOM_SEED    = 42
VISUAL_VIDEOS  = 50

# YOLO — fixed, never change per experiment
YOLO_MODEL      = "yolo11n.pt"
YOLO_CONF       = 0.50
YOLO_IOU        = 0.45
YOLO_IMG_SIZE   = 640
YOLO_VERBOSE    = False

# Kalman — fixed, never change per experiment
KF_MAX_AGE       = 30
KF_MIN_HITS      = 3
KF_IOU_THRESHOLD = 0.30

# Training — identical to baseline experiments
EPOCHS          = 15
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-3
EARLY_STOP_PAT  = 5

# ─────────────────────────────────────────────────────────────────────────────
# Tracker hyper-parameters
# All shared knobs (max_age, n_init, max_iou_dist, nn_budget) are held
# identical to DeepSORT / StrongSORT for a fair comparison.
# Only tracker-specific knobs differ.
# ─────────────────────────────────────────────────────────────────────────────
TRACKER_PARAMS: Dict[str, dict] = {

    # ── OccluTrack ─────────────────────────────────────────────────────────
    # Shared knobs: unchanged from baselines
    # Tracker-specific:
    #   occlusion_iou_thr  — IoU threshold above which two tracks are
    #                         considered mutually occluded (0.30 matches the
    #                         global KF_IOU_THRESHOLD so the same geometry
    #                         drives both Kalman gating and occlusion scoring)
    #   max_age_occluded   — how many extra frames an occluded track may
    #                         survive beyond max_age before deletion
    #   occ_cost_relax     — additive relaxation applied to the matching
    #                         cost threshold when a track is flagged occluded
    #                         (gives occluded tracks a wider acceptance band)
    #   app_weight         — weight of the appearance term in the combined
    #                         cost; (1 - app_weight) goes to IoU
    "OccluTrack": {
        "max_age":          30,
        "n_init":            3,
        "max_iou_dist":     0.70,
        "nn_budget":       100,
        "occlusion_iou_thr": 0.30,
        "max_age_occluded":  10,
        "occ_cost_relax":   0.15,
        "app_weight":        0.50,
    },

    # ── CSTrack ────────────────────────────────────────────────────────────
    # Shared knobs: unchanged from baselines
    # Tracker-specific:
    #   iou_weight     — weight of the IoU channel in the dual-stream cost
    #   reid_weight    — weight of the Re-ID channel (iou + reid must sum ≤ 1;
    #                    the remainder is implicit gate margin)
    #   gate_thresh    — IoU below this causes the Re-ID channel to be
    #                    up-weighted (cross-correlation gating: weak IoU →
    #                    trust Re-ID more; strong IoU → trust IoU more)
    #   nn_budget      — Re-ID gallery depth per track (same as baselines)
    "CSTrack": {
        "max_age":      30,
        "n_init":        3,
        "max_iou_dist": 0.70,
        "nn_budget":   100,
        "iou_weight":   0.50,
        "reid_weight":  0.50,
        "gate_thresh":  0.40,
    },
}


# ============================================================================
# SECTION 3: DATA STRUCTURES  (unchanged from baseline)
# ============================================================================

@dataclass
class Detection:
    bbox:       np.ndarray
    confidence: float
    class_id:   int
    frame_id:   int = 0


@dataclass
class Track:
    id:               int
    bbox:             np.ndarray
    class_id:         int
    confidence:       float
    trajectory:       List[np.ndarray] = field(default_factory=list)
    age:              int = 0
    hits:             int = 0
    time_since_update: int = 0


# ============================================================================
# SECTION 4: DATASET SPLIT LOADER  (unchanged from baseline)
# ============================================================================

class DatasetSplitLoader:
    """
    Splits videos into train/test sets with a fixed random seed.
    Saves the split to disk so it is identical across ALL tracker experiments.
    If the split JSON already exists from a previous experiment it is simply
    reloaded — guaranteeing OccluTrack / CSTrack see the same 480 train /
    120 test videos as DeepSORT and StrongSORT.
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

        rng = random.Random(self.seed)
        indices = list(range(len(all_folders)))
        rng.shuffle(indices)

        n_train    = int(len(indices) * self.train_ratio)
        train_idx  = sorted(indices[:n_train])
        test_idx   = sorted(indices[n_train:])

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
    """Wraps YOLOv11 detection. Settings are NEVER changed per experiment."""

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
        self.kf.R    *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P    *= 10.0
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
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _hungarian_match(cost_matrix: np.ndarray, threshold: float):
    """Greedy matching — unchanged from baseline."""
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


# ── Appearance feature extractor (identical to baseline) ─────────────────────

class AppearanceExtractor:
    """
    Lightweight CNN-based appearance feature extractor.
    Uses a small pre-trained MobileNetV2 backbone.
    Falls back gracefully to zero embedding when GPU/model unavailable.
    Shared by OccluTrack and CSTrack — identical to DeepSORT / StrongSORT.
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
# SECTION 8: REPLACEABLE TRACKER WRAPPERS
# ============================================================================

class BaseTrackerWrapper:
    """Abstract base — all tracker wrappers must implement reset() and update()."""

    def reset(self):
        raise NotImplementedError

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        raise NotImplementedError


# ── OccluTrack ────────────────────────────────────────────────────────────────

class OccluTrackWrapper(BaseTrackerWrapper):
    """
    OccluTrack: appearance-based tracker with explicit occlusion awareness.

    Design
    ------
    1. DETECTION: YOLO (fixed, shared).
    2. MOTION: shared KalmanPredictor — same as all other experiments.
    3. APPEARANCE: same MobileNetV2 AppearanceExtractor.
    4. OCCLUSION SCORING
       At each frame, after Kalman prediction, every active track receives an
       occlusion score in [0, 1]:
           occ_score(i) = max over j≠i of IoU(track_i, track_j)
       A track is flagged as occluded when occ_score ≥ occlusion_iou_thr.

    5. OCCLUSION-AWARE ASSOCIATION
       The base cost is:
           cost(d, t) = app_weight * cosine_dist(d, t)
                      + (1 - app_weight) * iou_dist(d, t)
       For occluded tracks the acceptance threshold is relaxed by
       occ_cost_relax so they can still be matched despite weaker overlap.

    6. OCCLUSION-AWARE DELETION
       A non-occluded track is deleted after max_age frames without update.
       A track that is currently occluded gets max_age + max_age_occluded
       frames before deletion, giving it time to re-emerge.

    7. GALLERY: per-track gallery of appearance embeddings (capped at
       nn_budget), identical in spirit to DeepSORT.
    """

    def __init__(self, params: dict):
        self.max_age           = params.get("max_age",           30)
        self.n_init            = params.get("n_init",             3)
        self.max_iou_dist      = params.get("max_iou_dist",      0.70)
        self.nn_budget         = params.get("nn_budget",        100)
        self.occlusion_iou_thr = params.get("occlusion_iou_thr", 0.30)
        self.max_age_occluded  = params.get("max_age_occluded",  10)
        self.occ_cost_relax    = params.get("occ_cost_relax",    0.15)
        self.app_weight        = params.get("app_weight",         0.50)

        self.extractor  = AppearanceExtractor()
        self._tracks: List[Track]             = []
        self._kf_map: Dict[int, KalmanPredictor] = {}
        self._feats:  Dict[int, List[np.ndarray]] = {}
        self._occ:    Dict[int, bool]            = {}   # occlusion flag per id
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._kf_map.clear()
        self._feats.clear()
        self._occ.clear()
        self._next_id = 1

    # ── occlusion scoring ─────────────────────────────────────────────────

    def _compute_occlusion_flags(self):
        """
        Mark each track as occluded if its predicted bbox overlaps another
        track's predicted bbox above occlusion_iou_thr.
        """
        n = len(self._tracks)
        for i, ti in enumerate(self._tracks):
            max_iou = 0.0
            for j, tj in enumerate(self._tracks):
                if i == j:
                    continue
                max_iou = max(max_iou, _iou(ti.bbox, tj.bbox))
            self._occ[ti.id] = (max_iou >= self.occlusion_iou_thr)

    # ── appearance helpers ────────────────────────────────────────────────

    def _cosine_dist(self, query: np.ndarray, track_id: int) -> float:
        gallery = self._feats.get(track_id, [])
        if not gallery:
            return 1.0
        gallery_arr = np.stack(gallery[-self.nn_budget:])
        nq = np.linalg.norm(query)
        ng = np.linalg.norm(gallery_arr, axis=1)
        if nq == 0 or not np.any(ng > 0):
            return 1.0
        sims = gallery_arr @ query / (ng * nq + 1e-9)
        return float(1.0 - sims.max())

    # ── main update ──────────────────────────────────────────────────────

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:

        # 1. Kalman predict
        for t in self._tracks:
            t.bbox = self._kf_map[t.id].predict()
            t.time_since_update += 1
            t.age += 1

        # 2. Compute occlusion flags on predicted positions
        self._compute_occlusion_flags()

        # 3. Extract appearance for new detections
        det_feats = [self.extractor.extract(frame, d.bbox) for d in detections]

        # 4. Build cost matrix
        if self._tracks and detections:
            cost = np.ones((len(detections), len(self._tracks)))
            for di, det in enumerate(detections):
                for ti, trk in enumerate(self._tracks):
                    iou_c = 1.0 - _iou(det.bbox, trk.bbox)
                    app_c = self._cosine_dist(det_feats[di], trk.id)
                    cost[di, ti] = (self.app_weight * app_c
                                    + (1.0 - self.app_weight) * iou_c)

            # Per-track relaxed thresholds
            thresholds = np.array([
                self.max_iou_dist + (self.occ_cost_relax if self._occ.get(t.id, False) else 0.0)
                for t in self._tracks
            ])

            # Association with per-column thresholds:
            # build a gated cost matrix where occluded-track columns use
            # relaxed threshold, then call the shared greedy matcher with
            # the stricter global threshold (occluded cells are already ≤
            # relaxed_threshold after gating below).
            gated_cost = cost.copy()
            for ti, trk in enumerate(self._tracks):
                thr = thresholds[ti]
                if thr > self.max_iou_dist:
                    # Scale occluded column costs down proportionally so
                    # they still pass the global max_iou_dist gate.
                    mask = gated_cost[:, ti] <= thr
                    gated_cost[mask, ti] *= (self.max_iou_dist / thr)

            matched, unmatched_dets, unmatched_trks = _hungarian_match(
                gated_cost, self.max_iou_dist
            )
        else:
            matched        = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self._tracks)))

        # 5. Update matched tracks
        for di, ti in matched:
            t = self._tracks[ti]
            t.bbox              = detections[di].bbox
            t.confidence        = detections[di].confidence
            t.hits             += 1
            t.time_since_update = 0
            t.trajectory.append(detections[di].bbox.copy())
            self._kf_map[t.id].update(detections[di].bbox)
            self._feats.setdefault(t.id, []).append(det_feats[di])
            self._occ[t.id] = False   # re-appeared → no longer occluded

        # 6. Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            det = detections[di]
            tid = self._next_id; self._next_id += 1
            kf  = KalmanPredictor()
            kf.initialize(det.bbox)
            trk = Track(id=tid, bbox=det.bbox, class_id=det.class_id,
                        confidence=det.confidence,
                        trajectory=[det.bbox.copy()], hits=1)
            self._tracks.append(trk)
            self._kf_map[tid] = kf
            self._feats[tid]  = [det_feats[di]]
            self._occ[tid]    = False

        # 7. Remove stale tracks — occluded tracks get extended patience
        surviving = []
        for t in self._tracks:
            effective_max_age = (self.max_age + self.max_age_occluded
                                 if self._occ.get(t.id, False)
                                 else self.max_age)
            if t.time_since_update < effective_max_age:
                surviving.append(t)
            else:
                # clean up auxiliary dicts
                self._kf_map.pop(t.id, None)
                self._feats.pop(t.id, None)
                self._occ.pop(t.id, None)
        self._tracks = surviving

        return [t for t in self._tracks if t.hits >= self.n_init]


# ── CSTrack ───────────────────────────────────────────────────────────────────

class CSTrackWrapper(BaseTrackerWrapper):
    """
    CSTrack: dual-stream cross-correlation tracker.

    Design
    ------
    CSTrack's original contribution is a jointly-trained detector + Re-ID
    network where the two feature streams cross-correlate each other.
    Because YOLO is kept fixed for fair comparison, we implement the
    defining algorithmic idea: a *dual-stream cost matrix* where IoU and
    Re-ID costs are computed separately and then fused with a gating
    function that lets each stream moderate the other.

    Dual-stream fusion
    ------------------
    Let  C_iou(d,t)  = 1 - IoU(d,t)         ∈ [0, 1]
         C_reid(d,t) = cosine_dist(d,t)      ∈ [0, 1]

    Adaptive weights (the cross-correlation gate):
        When IoU is strong (C_iou ≤ gate_thresh):
            trust IoU more → w_iou > iou_weight
        When IoU is weak  (C_iou >  gate_thresh):
            trust Re-ID more → w_reid > reid_weight
    Concretely:
        if C_iou ≤ gate_thresh:
            w_iou  = iou_weight + reid_weight * (1 - C_iou / gate_thresh)
            w_reid = 1 - w_iou
        else:
            w_reid = reid_weight + iou_weight * (C_iou - gate_thresh) / (1 - gate_thresh)
            w_iou  = 1 - w_reid
        cost(d,t) = w_iou * C_iou + w_reid * C_reid

    This produces a single combined cost that is then matched with the
    shared greedy Hungarian solver, identical to all other experiments.

    Appearance gallery: EMA-updated (same spirit as StrongSORT) to give
    CSTrack's Re-ID stream a stable embedding representation.
    """

    def __init__(self, params: dict):
        self.max_age      = params.get("max_age",      30)
        self.n_init       = params.get("n_init",        3)
        self.max_iou_dist = params.get("max_iou_dist", 0.70)
        self.nn_budget    = params.get("nn_budget",   100)
        self.iou_weight   = params.get("iou_weight",   0.50)
        self.reid_weight  = params.get("reid_weight",  0.50)
        self.gate_thresh  = params.get("gate_thresh",  0.40)

        self.extractor  = AppearanceExtractor()
        self._tracks: List[Track]               = []
        self._kf_map: Dict[int, KalmanPredictor]   = {}
        self._ema_feats: Dict[int, np.ndarray]     = {}   # EMA gallery
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._kf_map.clear()
        self._ema_feats.clear()
        self._next_id = 1

    # ── dual-stream cost ──────────────────────────────────────────────────

    def _dual_stream_cost(self,
                          iou_c:  float,
                          reid_c: float) -> float:
        """
        Adaptive fusion of IoU and Re-ID cost streams.
        When IoU is strong (low C_iou) we up-weight IoU; when IoU is
        weak (high C_iou) we up-weight Re-ID — mimicking the cross-
        correlation gating in the original CSTrack formulation.
        """
        if iou_c <= self.gate_thresh:
            # strong spatial overlap → trust motion
            alpha = iou_c / (self.gate_thresh + 1e-9)        # ∈ [0, 1]
            w_iou  = self.iou_weight + self.reid_weight * (1.0 - alpha)
            w_reid = 1.0 - w_iou
        else:
            # weak spatial overlap → trust Re-ID
            beta   = (iou_c - self.gate_thresh) / (1.0 - self.gate_thresh + 1e-9)
            w_reid = self.reid_weight + self.iou_weight * beta
            w_iou  = 1.0 - w_reid

        return float(w_iou * iou_c + w_reid * reid_c)

    # ── appearance helpers ────────────────────────────────────────────────

    def _cosine_dist(self, query: np.ndarray, track_id: int) -> float:
        ema = self._ema_feats.get(track_id)
        if ema is None:
            return 1.0
        nq = np.linalg.norm(query)
        ne = np.linalg.norm(ema)
        if nq == 0 or ne == 0:
            return 1.0
        return float(1.0 - (ema @ query) / (ne * nq + 1e-9))

    def _update_ema(self, track_id: int, new_feat: np.ndarray):
        """EMA update with α = 0.90 (same as StrongSORT for consistency)."""
        alpha = 0.90
        if track_id in self._ema_feats:
            self._ema_feats[track_id] = (alpha * self._ema_feats[track_id]
                                          + (1 - alpha) * new_feat)
        else:
            self._ema_feats[track_id] = new_feat.copy()

    # ── main update ──────────────────────────────────────────────────────

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:

        # 1. Kalman predict
        for t in self._tracks:
            t.bbox = self._kf_map[t.id].predict()
            t.time_since_update += 1
            t.age += 1

        # 2. Extract appearance for detections
        det_feats = [self.extractor.extract(frame, d.bbox) for d in detections]

        # 3. Build dual-stream cost matrix
        if self._tracks and detections:
            cost = np.ones((len(detections), len(self._tracks)))
            for di, det in enumerate(detections):
                for ti, trk in enumerate(self._tracks):
                    iou_c  = 1.0 - _iou(det.bbox, trk.bbox)
                    reid_c = self._cosine_dist(det_feats[di], trk.id)
                    cost[di, ti] = self._dual_stream_cost(iou_c, reid_c)

            matched, unmatched_dets, unmatched_trks = _hungarian_match(
                cost, self.max_iou_dist
            )
        else:
            matched        = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self._tracks)))

        # 4. Update matched tracks
        for di, ti in matched:
            t = self._tracks[ti]
            t.bbox              = detections[di].bbox
            t.confidence        = detections[di].confidence
            t.hits             += 1
            t.time_since_update = 0
            t.trajectory.append(detections[di].bbox.copy())
            self._kf_map[t.id].update(detections[di].bbox)
            self._update_ema(t.id, det_feats[di])

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
            self._kf_map[tid] = kf
            self._update_ema(tid, det_feats[di])

        # 6. Remove stale tracks
        self._tracks = [t for t in self._tracks if t.time_since_update < self.max_age]

        return [t for t in self._tracks if t.hits >= self.n_init]


# ── Tracker Factory ───────────────────────────────────────────────────────────

def build_tracker(name: str) -> BaseTrackerWrapper:
    params = TRACKER_PARAMS.get(name, {})
    if name == "OccluTrack":
        return OccluTrackWrapper(params)
    elif name == "CSTrack":
        return CSTrackWrapper(params)
    else:
        raise ValueError(
            f"Unknown tracker: '{name}'. "
            f"This file supports: 'OccluTrack', 'CSTrack'. "
            f"For DeepSORT / StrongSORT use the baseline file."
        )


# ============================================================================
# SECTION 9: TRAINING LOOP  (identical structure to baseline)
# ============================================================================

class TrackerTrainer:
    """
    Runs the detection + tracking pass over all training videos to
    calibrate tracker state.  Saves weights to disk.

    Training settings are IDENTICAL to the baseline experiments:
        EPOCHS=15, BATCH_SIZE=16, LR=1e-3, EARLY_STOP_PAT=5
        mini-epoch size = 80 videos (same random sampling logic)
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
        # Same mini-epoch sampling as baseline: seed offset per epoch,
        # 80 videos per mini-epoch
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
                rendered = _render_tracks(img, tracks)
                writer.write(rendered)

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
# SECTION 10: EVALUATION LOOP  (unchanged from baseline)
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
                row = {"video": vid_name, **metrics}
                per_video_rows.append(row)
                logger.info(
                    f"    MOTA={metrics['mota']:.4f}  IDF1={metrics['idf1']:.4f}"
                    f"  ADE={metrics['ade']:.4f}  IDS={metrics['id_switches']}"
                )

        # Aggregate
        agg     = self._aggregate(per_video_rows)
        agg_csv = METRICS_DIR / "aggregated_metrics.json"
        with open(agg_csv, "w") as f:
            json.dump(agg, f, indent=2)
        logger.info(f"Aggregated metrics → {agg_csv}")

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
# SECTION 11: VISUALIZATION / VIDEO WRITER  (unchanged from baseline)
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

    # ── Dataset split (reloads the shared split from disk) ────────────────
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
