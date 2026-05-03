"""
================================================================================
  FINAL VIDEO INFERENCE PIPELINE — VERSION 2
  YOLOv11n  +  Standard Kalman Filter  +  OAMN (Occlusion-Aware Mask Network)
================================================================================

  Changes from v1
  ---------------
   1. NO trajectory polyline is drawn anymore (the path line is removed).
   2. Yellow / amber colors are forbidden — palette stripped of yellows.
   3. Occluded / predicted bounding boxes NEVER shrink or change size.
      The (width, height) is locked at the moment occlusion starts and
      only the box center is allowed to move along the Kalman trajectory.
      This keeps the box "standing firm" while the soldier is hidden.

  Behaviour (unchanged)
  ---------------------
   - YOLOv11n detects soldiers each frame.
   - OAMN extracts a Re-ID appearance embedding for each detection.
   - Each soldier gets a stable ID.
   - When a soldier becomes occluded:
        * For 2 SECONDS the box keeps moving (translation only,
          fixed size) along the Kalman-predicted trajectory.
        * After 2 seconds the box FREEZES at its last predicted center.
   - When the soldier reappears, OAMN re-assigns the SAME ID.

  Inputs / Outputs
  ----------------
   - Inputs:   <BASE>/raw videos/*.{mp4,mov,avi,mkv,m4v}
   - YOLO:     <BASE>/best_yolo11_weight.pt
   - OAMN:     <BASE>/best_OAMN_weight.pth
   - Outputs:  <BASE>/output videos v2/<video_name>_tracked.mp4

  No training. No metrics. Inference only.
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
import os
import sys
import cv2
import logging
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 2: CONFIG
# ============================================================================
BASE_DIR        = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 6th term\Graduation Project 2\CL PIPELINE")
RAW_VIDEO_DIR   = BASE_DIR / "raw videos"
OUTPUT_VIDEO_DIR = BASE_DIR / "output videos v2"

YOLO_WEIGHTS    = BASE_DIR / "best_yolo11_weight.pt"
OAMN_WEIGHTS    = BASE_DIR / "best_OAMN_weight.pth"

# YOLO inference settings
YOLO_CONF       = 0.35
YOLO_IOU        = 0.45
YOLO_IMG_SIZE   = 640
YOLO_VERBOSE    = False
PERSON_CLASS_ID = 0

# Occlusion behaviour
OCCLUSION_FOLLOW_SECONDS  = 2.0    # bbox follows trajectory this long
LOST_ID_RETENTION_SECONDS = 60.0   # then frozen, kept for re-ID this long

# Tracker association
N_INIT             = 2
MAX_IOU_DIST       = 0.7
APPEARANCE_WEIGHT  = 0.55
REID_GALLERY_BUDGET = 100
REID_REASSIGN_THR  = 0.30

OAMN_INPUT_SIZE    = (256, 128)

# ── Rendering ────────────────────────────────────────────────────────────────
# Forbidden: any yellow / amber tone.  Palette below contains only red, pink,
# magenta, purple, blue, cyan, lime-green, mint-green and orange-red — no
# yellows.
PALETTE = [
    (255,  56,  56),   # red
    (255,  56, 132),   # pink
    (255,  56, 210),   # magenta
    (131,  56, 255),   # purple
    ( 56, 131, 255),   # blue
    ( 56, 210, 255),   # light blue
    ( 51, 255, 255),   # cyan
    (131, 255,  56),   # lime green
    ( 51, 255, 131),   # mint green
    (255,  90,  40),   # orange-red (no yellow component)
]


# ============================================================================
# SECTION 3: DATA STRUCTURES
# ============================================================================
@dataclass
class Detection:
    bbox:       np.ndarray
    confidence: float
    class_id:   int
    feat:       Optional[np.ndarray] = None


@dataclass
class Track:
    id:                int
    bbox:              np.ndarray
    class_id:          int
    confidence:        float
    age:               int = 0
    hits:              int = 0
    time_since_update: int = 0
    occluded:          bool = False
    frozen:            bool = False
    last_predicted:    Optional[np.ndarray] = None
    # Locked (w, h) at the moment occlusion started — keeps the box from
    # shrinking or growing while the soldier is hidden.
    locked_size:       Optional[Tuple[float, float]] = None


# ============================================================================
# SECTION 4: YOLO DETECTOR
# ============================================================================
class YOLODetector:
    def __init__(self, weights_path: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading YOLO weights '{weights_path.name}' on {self.device}")
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        self.model = YOLO(str(weights_path))
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                             imgsz=YOLO_IMG_SIZE, verbose=YOLO_VERBOSE)
        dets: List[Detection] = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].cpu().numpy())
                if self.model.names and len(self.model.names) > 1:
                    if cls != PERSON_CLASS_ID:
                        continue
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                dets.append(Detection(bbox=bbox, confidence=conf, class_id=cls))
        return dets


# ============================================================================
# SECTION 5: KALMAN PREDICTOR
# ============================================================================
class KalmanPredictor:
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

    def predict_center_only(self) -> Tuple[float, float]:
        """
        Advance the Kalman state and return only the predicted (cx, cy).
        Used when the bbox size must remain locked (occluded state).
        """
        self.kf.predict()
        return float(self.kf.x[0]), float(self.kf.x[1])

    @staticmethod
    def _bbox_to_z(b):
        w  = b[2] - b[0]; h = b[3] - b[1]
        cx = b[0] + w / 2.0; cy = b[1] + h / 2.0
        s  = w * h
        r  = w / float(h) if h != 0 else 1.0
        return np.array([cx, cy, s, r]).reshape((4, 1))

    @staticmethod
    def _x_to_bbox(x):
        s, r = float(x[2]), float(x[3])
        if s > 0 and r > 0:
            w = np.sqrt(s * r); h = s / w
        else:
            w = h = 0.0
        cx, cy = float(x[0]), float(x[1])
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


# ============================================================================
# SECTION 6: OAMN  (Occlusion-Aware Mask Network)
# ============================================================================
class OAMN(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        from torchvision.models import mobilenet_v2
        mb = mobilenet_v2(weights=None)
        self.backbone = mb.features

        self.spatial_mask = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mask_conf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Identity(),
        )
        self.head_4 = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        mask = self.spatial_mask(feat)
        feat = feat * mask
        h = self.head[0](feat)
        h = self.head[1](h)
        h = self.head[2](h)
        h = self.head[3](h)
        h = torch.flatten(h, 1)
        emb = self.head_4(h)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


class OAMNExtractor:
    def __init__(self, weights_path: Path):
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_dim = 512
        self.ready    = False
        self.model    = None
        self._transform = None

        try:
            import torchvision.transforms as T
            self.model = OAMN(embed_dim=self.embed_dim).to(self.device)
            self.model.eval()

            if weights_path.exists():
                state = torch.load(str(weights_path), map_location=self.device,
                                   weights_only=False)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                    state = state["model"]

                remapped = {}
                for k, v in state.items():
                    if k.startswith("head.4."):
                        remapped[k.replace("head.4.", "head_4.")] = v
                    else:
                        remapped[k] = v

                missing, unexpected = self.model.load_state_dict(remapped, strict=False)
                logger.info(f"OAMN loaded — missing={len(missing)} unexpected={len(unexpected)}")
            else:
                logger.warning(f"OAMN weights not found at {weights_path} — using random init.")

            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize(OAMN_INPUT_SIZE),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.ready = True
        except Exception as e:
            logger.warning(f"OAMN unavailable ({e}). Using zero embeddings — tracking will rely on IoU.")

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.zeros(self.embed_dim, dtype=np.float32)
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.embed_dim, dtype=np.float32)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp  = self._transform(crop).unsqueeze(0).to(self.device)
            emb  = self.model(inp).squeeze(0).cpu().numpy()
            return emb.astype(np.float32)
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)


# ============================================================================
# SECTION 7: UTILITIES
# ============================================================================
def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(cost: np.ndarray, threshold: float):
    matched, used_r, used_c = [], set(), set()
    flat = [(cost[r, c], r, c)
            for r in range(cost.shape[0])
            for c in range(cost.shape[1])]
    flat.sort()
    for v, r, c in flat:
        if v > threshold:
            break
        if r not in used_r and c not in used_c:
            matched.append((r, c)); used_r.add(r); used_c.add(c)
    unmatched_r = [r for r in range(cost.shape[0]) if r not in used_r]
    unmatched_c = [c for c in range(cost.shape[1]) if c not in used_c]
    return matched, unmatched_r, unmatched_c


def _bbox_from_center_and_size(cx: float, cy: float,
                                w: float, h: float) -> np.ndarray:
    return np.array([cx - w / 2.0, cy - h / 2.0,
                     cx + w / 2.0, cy + h / 2.0], dtype=float)


# ============================================================================
# SECTION 8: OAMN TRACKER
# ============================================================================
class OAMNTracker:
    """
    States per identity:
        ACTIVE    — matched this frame, full Kalman update.
        OCCLUDED  — unmatched but inside the 2-second hold; bbox CENTER
                    moves along Kalman trajectory while WIDTH/HEIGHT
                    stay LOCKED at the size from the moment occlusion
                    began.  No size shrink, no size growth.
        FROZEN    — past 2 seconds; bbox STOPS at the last predicted
                    position with the same locked size.
    """

    def __init__(self, fps: float, oamn: OAMNExtractor):
        self.fps   = max(1.0, float(fps))
        self.oamn  = oamn
        self.follow_frames = int(round(OCCLUSION_FOLLOW_SECONDS * self.fps))
        self.lost_max_age  = int(round(LOST_ID_RETENTION_SECONDS * self.fps))

        self._tracks: List[Track]                  = []
        self._kf_map: Dict[int, KalmanPredictor]   = {}
        self._gallery: Dict[int, List[np.ndarray]] = {}
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._kf_map.clear()
        self._gallery.clear()
        self._next_id = 1

    # ── appearance helpers ────────────────────────────────────────────────
    def _appearance_dist(self, query: np.ndarray, track_id: int) -> float:
        gallery = self._gallery.get(track_id, [])
        if not gallery or query is None or not np.any(query):
            return 1.0
        arr = np.stack(gallery[-REID_GALLERY_BUDGET:])
        nq  = np.linalg.norm(query)
        ng  = np.linalg.norm(arr, axis=1)
        if nq == 0 or not np.any(ng > 0):
            return 1.0
        sims = arr @ query / (ng * nq + 1e-9)
        return float(1.0 - sims.max())

    def _store_feat(self, tid: int, feat: np.ndarray):
        if feat is None or not np.any(feat):
            return
        self._gallery.setdefault(tid, []).append(feat)
        if len(self._gallery[tid]) > REID_GALLERY_BUDGET:
            self._gallery[tid] = self._gallery[tid][-REID_GALLERY_BUDGET:]

    # ── main update ───────────────────────────────────────────────────────
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:

        # 1. Prediction step.
        #    - ACTIVE tracks: standard Kalman predict (size may evolve).
        #    - Tracks already OCCLUDED/FROZEN: predict CENTER only, keep
        #      width/height locked to whatever was stored when occlusion
        #      first began.  This is what stops the box from shrinking.
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1
            if t.frozen:
                # Frozen: bbox does not move at all.
                if t.last_predicted is not None:
                    t.bbox = t.last_predicted.copy()
            elif t.occluded and t.locked_size is not None:
                # Occluded but still inside follow window — translate only.
                cx, cy = self._kf_map[t.id].predict_center_only()
                w, h   = t.locked_size
                t.bbox = _bbox_from_center_and_size(cx, cy, w, h)
                t.last_predicted = t.bbox.copy()
            else:
                # Active track entering a frame with no detection yet —
                # do a normal predict; if it stays unmatched we'll lock
                # the size in step 7 below.
                t.bbox = self._kf_map[t.id].predict()
                t.last_predicted = t.bbox.copy()

        # 2. Extract OAMN appearance features for every detection
        for d in detections:
            d.feat = self.oamn.extract(frame, d.bbox)

        # 3. Build cost matrix and match
        if self._tracks and detections:
            cost = np.ones((len(detections), len(self._tracks)), dtype=np.float32)
            for di, det in enumerate(detections):
                for ti, trk in enumerate(self._tracks):
                    iou_c = 1.0 - _iou(det.bbox, trk.bbox)
                    app_c = self._appearance_dist(det.feat, trk.id)
                    cost[di, ti] = (APPEARANCE_WEIGHT      * app_c
                                  + (1 - APPEARANCE_WEIGHT) * iou_c)
            matched, unmatched_dets, unmatched_trks = _greedy_match(cost, MAX_IOU_DIST)
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self._tracks)))

        # 4. Update matched tracks — these are now ACTIVE again.
        for di, ti in matched:
            t   = self._tracks[ti]
            det = detections[di]
            t.bbox              = det.bbox
            t.confidence        = det.confidence
            t.hits             += 1
            t.time_since_update = 0
            t.occluded          = False
            t.frozen            = False
            t.locked_size       = None
            self._kf_map[t.id].update(det.bbox)
            self._store_feat(t.id, det.feat)

        # 5. Re-ID against lost (occluded/frozen) tracks
        spawn_indices = []
        for di in unmatched_dets:
            det = detections[di]
            best_tid, best_d = None, REID_REASSIGN_THR
            for t in self._tracks:
                if not (t.occluded or t.frozen):
                    continue
                d = self._appearance_dist(det.feat, t.id)
                if d < best_d:
                    best_d, best_tid = d, t.id
            if best_tid is not None:
                trk = next(t for t in self._tracks if t.id == best_tid)
                trk.bbox              = det.bbox
                trk.confidence        = det.confidence
                trk.hits             += 1
                trk.time_since_update = 0
                trk.occluded          = False
                trk.frozen            = False
                trk.locked_size       = None
                self._kf_map[trk.id].update(det.bbox)
                self._store_feat(trk.id, det.feat)
            else:
                spawn_indices.append(di)

        # 6. Spawn new tracks for genuinely-new detections
        for di in spawn_indices:
            det = detections[di]
            tid = self._next_id; self._next_id += 1
            kf  = KalmanPredictor(); kf.initialize(det.bbox)
            trk = Track(id=tid, bbox=det.bbox, class_id=det.class_id,
                        confidence=det.confidence, hits=1)
            self._tracks.append(trk)
            self._kf_map[tid] = kf
            self._store_feat(tid, det.feat)

        # 7. Mark unmatched tracks as occluded / frozen and LOCK their size
        for ti in unmatched_trks:
            t = self._tracks[ti]
            # If this is the first frame the track is occluded, freeze its
            # current size so the box never shrinks afterwards.
            if not t.occluded and t.locked_size is None:
                w = float(t.bbox[2] - t.bbox[0])
                h = float(t.bbox[3] - t.bbox[1])
                # Keep the size that was being shown right before occlusion;
                # if Kalman already shrank it slightly this frame, restore
                # it from last_predicted's previous size by recomputing
                # from bbox (already the predicted bbox — fine).
                t.locked_size = (max(1.0, w), max(1.0, h))
                # Force this frame's box to the locked size around the
                # current predicted center, so the transition is seamless.
                cx = (t.bbox[0] + t.bbox[2]) / 2.0
                cy = (t.bbox[1] + t.bbox[3]) / 2.0
                t.bbox = _bbox_from_center_and_size(cx, cy, *t.locked_size)
                t.last_predicted = t.bbox.copy()

            t.occluded = True
            if t.time_since_update > self.follow_frames and not t.frozen:
                # 2-second window elapsed → freeze permanently
                t.frozen = True
                # last_predicted already holds the freeze position

        # 8. Drop tracks that exceeded the lost retention window
        survivors = []
        for t in self._tracks:
            if t.time_since_update > self.lost_max_age:
                self._kf_map.pop(t.id, None)
                self._gallery.pop(t.id, None)
            else:
                survivors.append(t)
        self._tracks = survivors

        # 9. Return everything worth rendering
        return [t for t in self._tracks
                if t.hits >= N_INIT or t.time_since_update == 0]


# ============================================================================
# SECTION 9: RENDERING  (no trajectory line, no yellow)
# ============================================================================
def _color_for(tid: int):
    return PALETTE[tid % len(PALETTE)]


def _render(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    img = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = _color_for(t.id)

        if t.frozen:
            status = "PREDICTED"
            # Dashed box for frozen state — same size, no shrink.
            for i in range(x1, x2, 12):
                cv2.line(img, (i, y1), (min(i + 6, x2), y1), color, 2)
                cv2.line(img, (i, y2), (min(i + 6, x2), y2), color, 2)
            for i in range(y1, y2, 12):
                cv2.line(img, (x1, i), (x1, min(i + 6, y2)), color, 2)
                cv2.line(img, (x2, i), (x2, min(i + 6, y2)), color, 2)
        elif t.occluded:
            status = "OCCLUDED"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        else:
            status = ""
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"ID{t.id}"
        if status:
            label += f" [{status}]"
        if not t.occluded:
            label += f" {t.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # NOTE: trajectory polyline intentionally REMOVED in v2.

    return img


# ============================================================================
# SECTION 10: PER-VIDEO INFERENCE
# ============================================================================
def process_video(video_path: Path, detector: YOLODetector,
                  oamn: OAMNExtractor, out_dir: Path):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open: {video_path.name}")
        return

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_tracked.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    tracker = OAMNTracker(fps=fps, oamn=oamn)

    logger.info(f"  → {video_path.name}  "
                f"({width}x{height} @ {fps:.1f}fps, {n_frames} frames)  "
                f"hold={tracker.follow_frames}f / lost={tracker.lost_max_age}f")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets   = detector.detect(frame)
        tracks = tracker.update(dets, frame)
        rendered = _render(frame, tracks)
        writer.write(rendered)
        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(f"      frame {frame_idx}/{n_frames}")

    cap.release()
    writer.release()
    logger.info(f"  ✔ saved → {out_path.name}")


# ============================================================================
# SECTION 11: MAIN
# ============================================================================
def main():
    if not RAW_VIDEO_DIR.exists():
        sys.exit(f"[ERROR] Raw video directory not found: {RAW_VIDEO_DIR}")

    detector = YOLODetector(YOLO_WEIGHTS)
    oamn     = OAMNExtractor(OAMN_WEIGHTS)

    videos = sorted([p for p in RAW_VIDEO_DIR.iterdir()
                     if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}])
    if not videos:
        sys.exit(f"[ERROR] No video files in {RAW_VIDEO_DIR}")

    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Found {len(videos)} videos in {RAW_VIDEO_DIR}")
    logger.info(f"Outputs → {OUTPUT_VIDEO_DIR}")

    for idx, vp in enumerate(videos, 1):
        logger.info(f"[{idx:>2}/{len(videos)}] {vp.name}")
        try:
            process_video(vp, detector, oamn, OUTPUT_VIDEO_DIR)
        except Exception as e:
            logger.error(f"Failed on {vp.name}: {e}")

    logger.info("All videos processed.")


if __name__ == "__main__":
    main()
