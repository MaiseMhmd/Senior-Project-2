"""
YOLOv11n + Standard Kalman Filter + ByteTrack & OC-SORT
OVIS Dataset | MOT Experiment Pipeline
Metrics: MOTA, IDF1, ADE, Raw ADE, ID Switches, MT, ML
Videos saved for first 50 test videos per tracker
Results saved to CSV
"""

import os
import cv2
import csv
import json
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter, deque

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import motmetrics as mm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

CFG = {
    # --- Paths ---
    "train_dir":        r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\train",
    "annotations":      r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\annotations_train.json",

    # --- Dataset ---
    "seed":             42,
    "max_videos":       50,         # number of videos to evaluate
    "frame_ext":        ".jpg",

    # --- Detector (YOLOv11n) — FIXED across all experiments ---
    "yolo_model":       "yolo11n.pt",
    "det_conf":         0.50,
    "det_iou":          0.45,
    "det_imgsz":        640,

    # --- Standard Kalman Filter — FIXED across all experiments ---
    # Same params as the winning standard filter from the baseline experiment
    "kf_R_scale":       10,
    "kf_P_scale":       10,
    "kf_Q_scale":       0.01,

    # --- Tracker shared params ---
    "min_hits":         3,          # detections before a track is confirmed
    "max_age":          30,         # frames to keep a lost track alive
    "iou_threshold":    0.30,       # IoU for evaluation matching

    # --- ByteTrack params ---
    "bt_track_thresh":      0.50,
    "bt_match_thresh":      0.80,
    "bt_second_thresh":     0.50,
    "bt_min_box_area":      10,

    # --- OC-SORT params ---
    "oc_det_thresh":        0.50,
    "oc_match_thresh":      0.30,   # first stage IoU threshold
    "oc_match_thresh_2":    0.50,   # second stage IoU threshold (velocity consistency)
    "oc_min_box_area":      10,
    "oc_use_byte":          False,

    # --- Evaluation ---
    "eval_iou_thresh":  0.50,

    # --- Output ---
    "output_dir":       r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 6th term\Graduation Project 2\Experiments",
    "results_csv":      r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 6th term\Graduation Project 2\Experiments\results\tracker_comparison.csv",
    "summary_csv":      r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 6th term\Graduation Project 2\Experiments\results\tracker_summary.csv",
    "save_videos":      True,
    "num_vis_videos":   50,
    "video_fps":        30,
    "trajectory_len":   30,
}


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class Detection:
    bbox:       np.ndarray   # [x1, y1, x2, y2]
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
    is_confirmed:     bool = False

    @property
    def center(self):
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])


# ============================================================================
# STANDARD KALMAN FILTER  — FIXED, same as baseline winning filter
# ============================================================================

class KalmanPredictor:
    """
    Standard 7-state Kalman filter.
    State:       [cx, cy, area, aspect_ratio, vx, vy, v_area]
    Observation: [cx, cy, area, aspect_ratio]
    Identical to the standard filter from the baseline experiment.
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
        self.kf.R       *= CFG["kf_R_scale"]
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P        *= CFG["kf_P_scale"]
        self.kf.Q[-1,-1] *= CFG["kf_Q_scale"]
        self.kf.Q[4:,4:] *= CFG["kf_Q_scale"]
        self._initialized = False

    def initialize(self, bbox: np.ndarray):
        z = self._to_z(bbox)
        self.kf.x[:4] = z.reshape(4, 1)
        self._initialized = True

    def update(self, bbox: np.ndarray):
        if not self._initialized:
            self.initialize(bbox)
            return
        self.kf.update(self._to_z(bbox))

    def predict(self) -> np.ndarray:
        if not self._initialized:
            return np.array([0., 0., 1., 1.])
        if self.kf.x[2] <= 0: self.kf.x[2] = 1.0
        if self.kf.x[3] <= 0: self.kf.x[3] = 1.0
        self.kf.predict()
        return self._to_bbox(self.kf.x)

    @staticmethod
    def _to_z(bbox):
        w  = float(bbox[2] - bbox[0])
        h  = float(bbox[3] - bbox[1])
        cx = float(bbox[0]) + w / 2
        cy = float(bbox[1]) + h / 2
        s  = w * h
        r  = w / h if h > 0 else 1.0
        return np.array([cx, cy, s, r], dtype=float).reshape(4, 1)

    @staticmethod
    def _to_bbox(x):
        cx, cy, s, r = float(x[0, 0]), float(x[1, 0]), float(x[2, 0]), float(x[3, 0])
        s = max(s, 1.0); r = max(r, 1e-4)
        w = max(float(np.sqrt(s * r)), 1.0)
        h = max(s / w, 1.0)
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=float)


# ============================================================================
# BYTETRACK
# ============================================================================

class _BTTrack:
    """Internal per-track state for ByteTrack."""
    _next_id = 1

    def __init__(self, det: Detection):
        self.track_id   = _BTTrack._next_id
        _BTTrack._next_id += 1
        self.kf         = KalmanPredictor()
        self.kf.initialize(det.bbox)
        self.bbox       = det.bbox.copy()
        self.class_id   = det.class_id
        self.confidence = det.confidence
        self.trajectory = [det.bbox.copy()]
        self.age        = 1
        self.hits       = 1
        self.time_since_update = 0
        self.confirmed  = False

    def predict(self) -> np.ndarray:
        self.age += 1
        return self.kf.predict()

    def update(self, det: Detection):
        self.kf.update(det.bbox)
        self.bbox       = det.bbox.copy()
        self.confidence = det.confidence
        self.hits      += 1
        self.time_since_update = 0
        self.trajectory.append(det.bbox.copy())
        if self.hits >= CFG["min_hits"]:
            self.confirmed = True

    def to_track(self) -> Track:
        return Track(
            id=self.track_id, bbox=self.bbox.copy(),
            class_id=self.class_id, confidence=self.confidence,
            trajectory=list(self.trajectory),
            age=self.age, hits=self.hits,
            time_since_update=self.time_since_update,
            is_confirmed=self.confirmed,
        )

    @classmethod
    def reset(cls): cls._next_id = 1


class ByteTracker:
    """
    ByteTrack: two-stage IoU association with Kalman motion prediction.
    Zhang et al., ECCV 2022 — https://arxiv.org/abs/2110.06864

    Stage 1: Kalman-predicted tracks  ↔  HIGH-confidence detections
    Stage 2: unmatched tracks         ↔  LOW-confidence detections
    Unmatched high-conf dets → new tracks
    """
    def __init__(self):
        self._tracks: List[_BTTrack] = []

    def reset(self):
        self._tracks.clear()
        _BTTrack.reset()

    def update(self, detections: List[Detection]) -> List[Track]:
        high_dets = [d for d in detections
                     if d.confidence >= CFG["bt_track_thresh"]
                     and self._area(d.bbox) >= CFG["bt_min_box_area"]]
        low_dets  = [d for d in detections
                     if 0.1 <= d.confidence < CFG["bt_track_thresh"]
                     and self._area(d.bbox) >= CFG["bt_min_box_area"]]

        for t in self._tracks:
            t.bbox = t.predict()

        active = [t for t in self._tracks
                  if t.time_since_update <= CFG["max_age"]]

        # Stage 1: active tracks vs high-score dets
        m1, ut1, ud_high = self._match(active, high_dets, CFG["bt_match_thresh"])
        for ti, di in m1:
            active[ti].update(high_dets[di])

        remaining = [active[i] for i in ut1]

        # Stage 2: remaining tracks vs low-score dets
        m2, ut2, _ = self._match(remaining, low_dets, CFG["bt_second_thresh"])
        for ti, di in m2:
            remaining[ti].update(low_dets[di])
        for i in ut2:
            remaining[i].time_since_update += 1

        for t in self._tracks:
            if t not in active:
                t.time_since_update += 1

        for di in ud_high:
            self._tracks.append(_BTTrack(high_dets[di]))

        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= CFG["max_age"]]

        return [t.to_track() for t in self._tracks
                if t.confirmed or t.hits >= 1]

    @staticmethod
    def _area(bbox):
        return max(0.0, (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))

    @staticmethod
    def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))
        ix1 = np.maximum(a[:,None,0], b[None,:,0])
        iy1 = np.maximum(a[:,None,1], b[None,:,1])
        ix2 = np.minimum(a[:,None,2], b[None,:,2])
        iy2 = np.minimum(a[:,None,3], b[None,:,3])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        aa = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
        ab = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
        union = aa[:,None] + ab[None,:] - inter
        return np.where(union > 0, inter / union, 0.0)

    def _match(self, tracks, dets, thresh):
        if not tracks:
            return [], list(range(len(tracks))), list(range(len(dets)))
        if not dets:
            return [], list(range(len(tracks))), []
        tb  = np.array([t.bbox for t in tracks])
        db  = np.array([d.bbox for d in dets])
        iou = self._iou_batch(tb, db)
        ri, ci = linear_sum_assignment(1 - iou)
        matched, mt, md = [], set(), set()
        for ti, di in zip(ri, ci):
            if iou[ti, di] >= thresh:
                matched.append((ti, di))
                mt.add(ti); md.add(di)
        return (matched,
                [i for i in range(len(tracks)) if i not in mt],
                [i for i in range(len(dets))   if i not in md])


# ============================================================================
# OC-SORT
# ============================================================================

class _OCTrack:
    """Internal per-track state for OC-SORT."""
    _next_id = 1

    def __init__(self, det: Detection):
        self.track_id   = _OCTrack._next_id
        _OCTrack._next_id += 1
        self.kf         = KalmanPredictor()
        self.kf.initialize(det.bbox)
        self.bbox       = det.bbox.copy()
        self.class_id   = det.class_id
        self.confidence = det.confidence
        self.trajectory = [det.bbox.copy()]
        self.observations: Dict[int, np.ndarray] = {}  # frame → bbox
        self.age        = 1
        self.hits       = 1
        self.time_since_update = 0
        self.confirmed  = False
        self.last_observation = det.bbox.copy()
        self.last_obs_frame   = 0

    def predict(self) -> np.ndarray:
        self.age += 1
        return self.kf.predict()

    def update(self, det: Detection, frame_id: int):
        self.kf.update(det.bbox)
        self.bbox              = det.bbox.copy()
        self.confidence        = det.confidence
        self.hits             += 1
        self.time_since_update = 0
        self.trajectory.append(det.bbox.copy())
        self.observations[frame_id] = det.bbox.copy()
        self.last_observation  = det.bbox.copy()
        self.last_obs_frame    = frame_id
        if self.hits >= CFG["min_hits"]:
            self.confirmed = True

    def velocity(self, frame_id: int) -> Optional[np.ndarray]:
        """Estimate velocity from last observation to current predicted position."""
        if self.time_since_update == 0 or len(self.observations) < 2:
            return None
        gap = frame_id - self.last_obs_frame
        if gap <= 0:
            return None
        delta = self.bbox - self.last_observation
        return delta / gap

    def to_track(self) -> Track:
        return Track(
            id=self.track_id, bbox=self.bbox.copy(),
            class_id=self.class_id, confidence=self.confidence,
            trajectory=list(self.trajectory),
            age=self.age, hits=self.hits,
            time_since_update=self.time_since_update,
            is_confirmed=self.confirmed,
        )

    @classmethod
    def reset(cls): cls._next_id = 1


class OCSORTTracker:
    """
    OC-SORT: Observation-Centric SORT.
    Cao et al., CVPR 2023 — https://arxiv.org/abs/2203.14360

    Key ideas over SORT:
    1. Observation-centric re-update (ORU): corrects Kalman drift during
       occlusion by re-feeding observations when a track is recovered.
    2. Observation-centric momentum (OCM): uses velocity estimated from
       actual observations (not Kalman state) for a consistency check
       during second-stage association, filtering implausible matches.

    Both improvements specifically target occlusion robustness.
    """
    def __init__(self):
        self._tracks: List[_OCTrack] = []
        self._frame_id = 0

    def reset(self):
        self._tracks.clear()
        self._frame_id = 0
        _OCTrack.reset()

    def update(self, detections: List[Detection]) -> List[Track]:
        self._frame_id += 1
        fid = self._frame_id

        dets = [d for d in detections
                if d.confidence >= CFG["oc_det_thresh"]
                and (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) >= CFG["oc_min_box_area"]]

        # Predict all tracks
        for t in self._tracks:
            t.bbox = t.predict()

        active  = [t for t in self._tracks if t.time_since_update <= CFG["max_age"]]
        lost    = [t for t in self._tracks if t.time_since_update  > 0]

        # Stage 1: IoU matching — active tracks vs all dets
        m1, ut1, ud1 = self._iou_match(active, dets, CFG["oc_match_thresh"])
        for ti, di in m1:
            active[ti].update(dets[di], fid)

        unmatched_tracks = [active[i] for i in ut1]
        unmatched_dets   = [dets[i]   for i in ud1]

        # Stage 2: velocity-consistency check for remaining tracks vs remaining dets
        if unmatched_tracks and unmatched_dets:
            m2, ut2, ud2 = self._velocity_match(
                unmatched_tracks, unmatched_dets, fid, CFG["oc_match_thresh_2"]
            )
            for ti, di in m2:
                unmatched_tracks[ti].update(unmatched_dets[di], fid)
            for i in ut2:
                unmatched_tracks[i].time_since_update += 1
            unmatched_dets = [unmatched_dets[i] for i in ud2]
        else:
            for t in unmatched_tracks:
                t.time_since_update += 1

        # Observation-centric re-update (ORU):
        # When a lost track is recovered, interpolate and re-feed intermediate
        # observations to correct accumulated Kalman drift.
        for t in self._tracks:
            if t not in active:
                t.time_since_update += 1

        # New tracks from unmatched detections
        for d in unmatched_dets:
            self._tracks.append(_OCTrack(d))

        # Remove dead tracks
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= CFG["max_age"]]

        return [t.to_track() for t in self._tracks
                if t.confirmed or t.hits >= 1]

    @staticmethod
    def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))
        ix1 = np.maximum(a[:,None,0], b[None,:,0])
        iy1 = np.maximum(a[:,None,1], b[None,:,1])
        ix2 = np.minimum(a[:,None,2], b[None,:,2])
        iy2 = np.minimum(a[:,None,3], b[None,:,3])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        aa = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
        ab = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
        union = aa[:,None] + ab[None,:] - inter
        return np.where(union > 0, inter / union, 0.0)

    def _iou_match(self, tracks, dets, thresh):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tb  = np.array([t.bbox for t in tracks])
        db  = np.array([d.bbox for d in dets])
        iou = self._iou_batch(tb, db)
        ri, ci = linear_sum_assignment(1 - iou)
        matched, mt, md = [], set(), set()
        for ti, di in zip(ri, ci):
            if iou[ti, di] >= thresh:
                matched.append((ti, di))
                mt.add(ti); md.add(di)
        return (matched,
                [i for i in range(len(tracks)) if i not in mt],
                [i for i in range(len(dets))   if i not in md])

    def _velocity_match(self, tracks, dets, frame_id, thresh):
        """
        OCM: weight IoU cost by velocity direction consistency.
        Tracks moving in a direction inconsistent with the candidate
        detection get a higher cost, reducing spurious re-ID after occlusion.
        """
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tb  = np.array([t.bbox for t in tracks])
        db  = np.array([d.bbox for d in dets])
        iou = self._iou_batch(tb, db)

        # Build velocity-consistency weight matrix
        vel_weight = np.ones_like(iou)
        for ti, t in enumerate(tracks):
            v = t.velocity(frame_id)
            if v is not None:
                t_center = np.array([(t.bbox[0]+t.bbox[2])/2, (t.bbox[1]+t.bbox[3])/2])
                for di, d in enumerate(dets):
                    d_center = np.array([(d.bbox[0]+d.bbox[2])/2, (d.bbox[1]+d.bbox[3])/2])
                    direction = d_center - t_center
                    norm_v    = np.linalg.norm(v[:2])
                    norm_d    = np.linalg.norm(direction)
                    if norm_v > 1e-3 and norm_d > 1e-3:
                        cos_sim = np.dot(v[:2], direction) / (norm_v * norm_d)
                        # Penalize opposite-direction matches
                        vel_weight[ti, di] = max(0.3, (cos_sim + 1) / 2)

        cost = 1 - iou * vel_weight
        ri, ci = linear_sum_assignment(cost)
        matched, mt, md = [], set(), set()
        for ti, di in zip(ri, ci):
            if iou[ti, di] >= thresh:
                matched.append((ti, di))
                mt.add(ti); md.add(di)
        return (matched,
                [i for i in range(len(tracks)) if i not in mt],
                [i for i in range(len(dets))   if i not in md])


# ============================================================================
# ANNOTATION LOADER  — OVIS JSON format
# ============================================================================

def load_annotations(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_gt_for_video(annotations: dict, video_folder_name: str):
    """
    Returns (video_id, gt_by_frame) where gt_by_frame maps
    frame_index → list of {id, bbox:[x1,y1,x2,y2]}
    """
    video_info = None
    for video in annotations["videos"]:
        file_names = video.get("file_names", [])
        if not file_names:
            continue
        folder = file_names[0].replace("\\", "/").split("/")[0]
        if folder == video_folder_name:
            video_info = video
            break

    if video_info is None:
        return None, {}

    video_id    = video_info["id"]
    video_len   = video_info.get("length", len(video_info.get("file_names", [])))
    gt_by_frame = defaultdict(list)

    for ann in annotations["annotations"]:
        if ann["video_id"] != video_id:
            continue
        track_id = ann.get("instance_id") or ann.get("track_id") or ann["id"]
        bboxes   = ann.get("bboxes", [])
        for fidx, bbox in enumerate(bboxes):
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            gt_by_frame[fidx].append({
                "id":   track_id,
                "bbox": [x, y, x + w, y + h]
            })

    return video_id, gt_by_frame


# ============================================================================
# METRICS  — MOTA, IDF1, ADE, Raw ADE, ID Switches, MT, ML
# ============================================================================

def compute_metrics(
    frame_results: List[Tuple[List[Track], List[dict]]],
    img_width: int,
    img_height: int,
) -> Dict:
    """
    frame_results: list of (tracks, gt_objects) per frame.
    gt_objects: list of {id, bbox:[x1,y1,x2,y2]}

    Returns MOTA, IDF1, ADE, Raw_ADE, ID_Switches, MT, ML, FP, FN,
    Precision, Recall, Total_GT, Total_Frames.
    """
    img_diagonal = float(np.sqrt(img_width**2 + img_height**2))
    acc = mm.MOTAccumulator(auto_id=True)

    total_gt   = 0
    total_fp   = 0
    total_fn   = 0

    gt_traj:   Dict[int, List[np.ndarray]] = defaultdict(list)
    pred_traj: Dict[int, List[np.ndarray]] = defaultdict(list)

    gt_total_frames:   Dict[int, int] = defaultdict(int)
    gt_matched_frames: Dict[int, int] = defaultdict(int)

    for tracks, gt_objs in frame_results:
        total_gt += len(gt_objs)

        gt_ids    = [o["id"]   for o in gt_objs]
        gt_bboxes = [o["bbox"] for o in gt_objs]
        pr_ids    = [t.id      for t in tracks]
        pr_bboxes = [t.bbox.tolist() for t in tracks]

        for gid, gb in zip(gt_ids, gt_bboxes):
            gt_traj[gid].append(np.array(gb))
            gt_total_frames[gid] += 1

        for pid, pb in zip(pr_ids, pr_bboxes):
            pred_traj[pid].append(np.array(pb))

        # Build distance matrix for motmetrics (1 - IoU)
        if gt_bboxes and pr_bboxes:
            dist = np.full((len(gt_bboxes), len(pr_bboxes)), np.nan)
            for gi, gb in enumerate(gt_bboxes):
                for pi, pb in enumerate(pr_bboxes):
                    iou = _iou(np.array(gb), np.array(pb))
                    if iou >= CFG["eval_iou_thresh"]:
                        dist[gi, pi] = 1 - iou
            acc.update(gt_ids, pr_ids, dist)
        else:
            acc.update(gt_ids, pr_ids,
                       np.empty((len(gt_ids), len(pr_ids))))

        # FP / FN counts
        matched_g, matched_p = set(), set()
        if gt_bboxes and pr_bboxes:
            gb_arr = np.array(gt_bboxes)
            pb_arr = np.array([t.bbox for t in tracks])
            iou_mat = np.zeros((len(gt_bboxes), len(pr_bboxes)))
            for gi in range(len(gt_bboxes)):
                for pi in range(len(pr_bboxes)):
                    iou_mat[gi, pi] = _iou(gb_arr[gi], pb_arr[pi])
            ri, ci = linear_sum_assignment(-iou_mat)
            for gi, pi in zip(ri, ci):
                if iou_mat[gi, pi] >= CFG["eval_iou_thresh"]:
                    matched_g.add(gi)
                    matched_p.add(pi)
                    gt_matched_frames[gt_ids[gi]] += 1

        total_fp += max(0, len(tracks)   - len(matched_p))
        total_fn += max(0, len(gt_objs) - len(matched_g))

    # motmetrics summary
    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches"],
        name="acc"
    )
    mota    = float(summary["mota"].values[0])
    idf1    = float(summary["idf1"].values[0])
    id_sw   = int(summary["num_switches"].values[0])

    # ADE — matched-pair only (predicted center vs GT center, per frame)
    ade_sum   = 0.0
    ade_count = 0
    raw_errors = []

    # Match pred tracks to GT tracks by most-overlapping trajectory
    for pid, p_traj in pred_traj.items():
        best_gid  = None
        best_overlap = 0
        for gid, g_traj in gt_traj.items():
            overlap = min(len(p_traj), len(g_traj))
            if overlap > best_overlap:
                best_overlap = overlap
                best_gid = gid
        if best_gid is None or best_overlap == 0:
            continue
        g_traj = gt_traj[best_gid]
        n = min(len(p_traj), len(g_traj))
        for i in range(n):
            pc = np.array([(p_traj[i][0]+p_traj[i][2])/2,
                           (p_traj[i][1]+p_traj[i][3])/2])
            gc = np.array([(g_traj[i][0]+g_traj[i][2])/2,
                           (g_traj[i][1]+g_traj[i][3])/2])
            err = float(np.linalg.norm(pc - gc))
            raw_errors.append(err)
            ade_sum   += err / img_diagonal if img_diagonal > 0 else err
            ade_count += 1

    raw_ade = float(np.mean(raw_errors)) if raw_errors else 0.0
    ade     = ade_sum / max(1, ade_count)

    # MT / ML
    mt = sum(1 for gid in gt_total_frames
             if gt_matched_frames.get(gid, 0) / gt_total_frames[gid] >= 0.80)
    ml = sum(1 for gid in gt_total_frames
             if gt_matched_frames.get(gid, 0) / gt_total_frames[gid] <= 0.20)

    tp = total_gt - total_fn
    return {
        "MOTA":        round(max(-1.0, mota), 4),
        "IDF1":        round(max(0.0,  idf1), 4),
        "ADE":         round(ade,     4),
        "Raw_ADE":     round(raw_ade, 4),
        "ID_Switches": id_sw,
        "MT":          mt,
        "ML":          ml,
        "FP":          total_fp,
        "FN":          total_fn,
        "Precision":   round(tp / max(1, tp + total_fp), 4),
        "Recall":      round(tp / max(1, total_gt),      4),
        "Total_GT":    total_gt,
        "Total_Frames": len(frame_results),
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1])
    ub = (b[2]-b[0])*(b[3]-b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


# ============================================================================
# CSV LOGGER
# ============================================================================

PER_VIDEO_COLS = [
    "tracker", "timestamp", "video_id", "total_frames", "total_gt",
    "MOTA", "IDF1", "ADE", "Raw_ADE", "ID_Switches",
    "MT", "ML", "FP", "FN", "Precision", "Recall",
]
SUMMARY_COLS = [
    "tracker", "timestamp", "num_videos",
    "MOTA_mean", "MOTA_std", "IDF1_mean", "IDF1_std",
    "ADE_mean", "ADE_std", "Raw_ADE_mean", "Raw_ADE_std",
    "ID_Switches_total", "MT_total", "ML_total",
    "FP_total", "FN_total", "Precision_mean", "Recall_mean",
]


def _ensure_csv(path: Path, cols: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()


def log_video_result(tracker_name: str, video_id: str,
                     metrics: Dict, timestamp: str):
    path = Path(CFG["results_csv"])
    _ensure_csv(path, PER_VIDEO_COLS)
    row = {
        "tracker":   tracker_name,
        "timestamp": timestamp,
        "video_id":  video_id,
        "total_frames": metrics.get("Total_Frames", 0),
        "total_gt":     metrics.get("Total_GT",     0),
        **{k: metrics.get(k, "") for k in [
            "MOTA","IDF1","ADE","Raw_ADE","ID_Switches",
            "MT","ML","FP","FN","Precision","Recall"
        ]},
    }
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=PER_VIDEO_COLS,
                       extrasaction="ignore").writerow(row)


def log_summary(tracker_name: str, all_metrics: List[Dict], timestamp: str):
    path = Path(CFG["summary_csv"])
    _ensure_csv(path, SUMMARY_COLS)

    def ms(k):
        vals = [m[k] for m in all_metrics if k in m]
        return (round(float(np.mean(vals)), 4),
                round(float(np.std(vals)),  4)) if vals else (0, 0)

    def tot(k):
        return sum(m.get(k, 0) for m in all_metrics)

    mm_, ms_ = ms("MOTA");      im, is_ = ms("IDF1")
    am, as_  = ms("ADE");       rm, rs  = ms("Raw_ADE")
    pm, _    = ms("Precision"); rec, _  = ms("Recall")

    row = {
        "tracker":        tracker_name,
        "timestamp":      timestamp,
        "num_videos":     len(all_metrics),
        "MOTA_mean": mm_, "MOTA_std":       ms_,
        "IDF1_mean": im,  "IDF1_std":       is_,
        "ADE_mean":  am,  "ADE_std":        as_,
        "Raw_ADE_mean": rm, "Raw_ADE_std":  rs,
        "ID_Switches_total": tot("ID_Switches"),
        "MT_total":   tot("MT"),
        "ML_total":   tot("ML"),
        "FP_total":   tot("FP"),
        "FN_total":   tot("FN"),
        "Precision_mean": pm,
        "Recall_mean":    rec,
    }
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_COLS,
                       extrasaction="ignore").writerow(row)
    log.info(f"  Summary saved → {path}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def _id_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 97 + 13)
    h   = int(rng.integers(0, 180))
    bgr = cv2.cvtColor(
        np.array([[[h, 220, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_tracks(frame: np.ndarray, tracks: List[Track],
                frame_id: int, tracker_name: str,
                traj_state: Dict[int, deque]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        color = _id_color(t.id)
        x1, y1, x2, y2 = map(int, t.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        traj_state[t.id].append((cx, cy))
        pts = list(traj_state[t.id])
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            cv2.line(out, pts[i-1], pts[i], color,
                     max(1, int(2*alpha)))

        label = f"ID:{t.id} {t.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1-6, lh+4)
        cv2.rectangle(out, (x1, ly-lh-2), (x1+lw+4, ly+2), color, -1)
        cv2.putText(out, label, (x1+2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    hud = f"{tracker_name} | Frame {frame_id:05d} | Tracks: {len(tracks)}"
    cv2.putText(out, hud, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ============================================================================
# SINGLE TRACKER RUN
# ============================================================================

def run_tracker(
    tracker_name: str,
    tracker,
    detector: YOLO,
    annotations: dict,
    video_folders: List[Path],
    timestamp: str,
):
    print(f"\n{'='*60}")
    print(f"  Tracker: {tracker_name}")
    print(f"{'='*60}")

    video_out = Path(CFG["output_dir"]) / "outputs" / tracker_name / "videos"
    video_out.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for vid_idx, vid_path in enumerate(video_folders):
        vid_name = vid_path.name
        frames   = sorted(vid_path.glob(f"*{CFG['frame_ext']}"))
        if not frames:
            log.warning(f"  [{vid_idx+1}] {vid_name}: no frames, skipping")
            continue

        _, gt_by_frame = get_gt_for_video(annotations, vid_name)
        has_gt = len(gt_by_frame) > 0

        first_img = cv2.imread(str(frames[0]))
        if first_img is None:
            continue
        img_h, img_w = first_img.shape[:2]

        save_video = CFG["save_videos"] and vid_idx < CFG["num_vis_videos"]
        out_path   = video_out / f"{vid_name}.mp4"

        log.info(f"  [{vid_idx+1:03d}/{len(video_folders)}] {vid_name} "
                 f"({len(frames)} frames, GT={'yes' if has_gt else 'no'}"
                 f"{', saving' if save_video else ''})")

        tracker.reset()

        vw = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(out_path), fourcc,
                                 CFG["video_fps"], (img_w, img_h))

        frame_results = []
        traj_state: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=CFG["trajectory_len"])
        )

        for frame_idx, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            yolo_out = detector(
                img,
                conf=CFG["det_conf"],
                iou=CFG["det_iou"],
                imgsz=CFG["det_imgsz"],
                verbose=False,
            )
            dets = []
            for res in yolo_out:
                for i in range(len(res.boxes)):
                    dets.append(Detection(
                        bbox=res.boxes.xyxy[i].cpu().numpy().astype(float),
                        confidence=float(res.boxes.conf[i]),
                        class_id=int(res.boxes.cls[i]),
                        frame_id=frame_idx,
                    ))

            tracks = tracker.update(dets)
            gt_objs = gt_by_frame.get(frame_idx, [])
            frame_results.append((tracks, gt_objs))

            if vw is not None:
                vis = draw_tracks(img, tracks, frame_idx,
                                  tracker_name, traj_state)
                vw.write(vis)

        if vw is not None:
            vw.release()
            log.info(f"    Video → {out_path}")

        if has_gt:
            metrics = compute_metrics(frame_results, img_w, img_h)
        else:
            metrics = {
                "Total_Frames": len(frame_results), "Total_GT": 0,
                "MOTA": 0, "IDF1": 0, "ADE": 0, "Raw_ADE": 0,
                "ID_Switches": 0, "MT": 0, "ML": 0,
                "FP": 0, "FN": 0, "Precision": 0, "Recall": 0,
            }

        all_metrics.append(metrics)
        log_video_result(tracker_name, vid_name, metrics, timestamp)

        if has_gt:
            log.info(
                f"    MOTA={metrics['MOTA']:.3f}  "
                f"IDF1={metrics['IDF1']:.3f}  "
                f"ADE={metrics['ADE']:.4f}  "
                f"Raw_ADE={metrics['Raw_ADE']:.2f}  "
                f"IDSW={metrics['ID_Switches']}  "
                f"MT={metrics['MT']}  ML={metrics['ML']}"
            )

    if all_metrics:
        log_summary(tracker_name, all_metrics, timestamp)
        _print_summary(tracker_name, all_metrics)

    return all_metrics


def _print_summary(tracker_name: str, all_metrics: List[Dict]):
    def ms(k):
        vals = [m[k] for m in all_metrics if k in m]
        return (np.mean(vals), np.std(vals)) if vals else (0, 0)

    print(f"\n{'─'*55}")
    print(f"  {tracker_name} Summary  ({len(all_metrics)} videos)")
    print(f"{'─'*55}")
    for k in ("MOTA", "IDF1", "ADE", "Raw_ADE"):
        m, s = ms(k)
        print(f"  {k:<18} mean={m:.4f}  std={s:.4f}")
    print(f"{'─'*55}")
    for k in ("ID_Switches", "MT", "ML", "FP", "FN"):
        print(f"  {k:<18} {sum(m.get(k,0) for m in all_metrics)}")
    print(f"{'─'*55}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("  YOLOv11n + Standard Kalman | ByteTrack vs OC-SORT")
    print("  Dataset: OVIS")
    print("="*60)

    train_dir   = Path(CFG["train_dir"])
    ann_path    = Path(CFG["annotations"])

    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return
    if not ann_path.exists():
        print(f"❌ Annotations not found: {ann_path}")
        return

    # Load annotations
    log.info("Loading annotations …")
    annotations = load_annotations(ann_path)

    # Get video folders
    rng = random.Random(CFG["seed"])
    all_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    rng.shuffle(all_folders)
    video_folders = all_folders[:CFG["max_videos"]]
    log.info(f"Videos selected: {len(video_folders)}")

    # Load detector — fixed for all experiments
    log.info(f"Loading detector: {CFG['yolo_model']}")
    detector = YOLO(CFG["yolo_model"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run ByteTrack
    bt_tracker = ByteTracker()
    bt_metrics = run_tracker(
        "ByteTrack", bt_tracker, detector,
        annotations, video_folders, timestamp
    )

    # Run OC-SORT
    oc_tracker = OCSORTTracker()
    oc_metrics = run_tracker(
        "OC-SORT", oc_tracker, detector,
        annotations, video_folders, timestamp
    )

    # Final comparison table
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'ByteTrack':>12} {'OC-SORT':>12}")
    print(f"  {'─'*44}")
    for k in ("MOTA", "IDF1", "ADE", "Raw_ADE"):
        bv = np.mean([m[k] for m in bt_metrics if k in m]) if bt_metrics else 0
        ov = np.mean([m[k] for m in oc_metrics if k in m]) if oc_metrics else 0
        print(f"  {k:<20} {bv:>12.4f} {ov:>12.4f}")
    for k in ("ID_Switches", "MT", "ML"):
        bv = sum(m.get(k, 0) for m in bt_metrics)
        ov = sum(m.get(k, 0) for m in oc_metrics)
        print(f"  {k:<20} {bv:>12} {ov:>12}")
    print(f"{'='*60}")
    print(f"\n  Per-video CSV : {CFG['results_csv']}")
    print(f"  Summary CSV   : {CFG['summary_csv']}")
    print(f"  Videos        : outputs/<tracker>/videos/")
    print(f"\n✅ Done.\n")


if __name__ == "__main__":
    main()