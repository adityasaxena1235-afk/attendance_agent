"""Lightweight multi-face tracker (IoU + centre distance, greedy matching).

It is deliberately identity-free: a track is "somebody who has been continuously
in view", not a recognised person. That keeps the system privacy-preserving.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import BBox, Detection, TrackObs


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(b: BBox):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    last_seen: float
    first_seen: float


class CentroidIoUTracker:
    def __init__(self, max_missing_seconds: float = 3.0, iou_threshold: float = 0.2,
                 max_center_dist: float = 1.5):
        self.max_missing = max_missing_seconds
        self.iou_threshold = iou_threshold
        self.max_center_dist = max_center_dist
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def update(self, t: float, detections: List[Detection]) -> List[TrackObs]:
        # drop stale tracks
        for tid in [k for k, tr in self._tracks.items() if t - tr.last_seen > self.max_missing]:
            del self._tracks[tid]

        # score every (track, detection) pair
        pairs = []
        for tid, tr in self._tracks.items():
            tcx, tcy = _center(tr.bbox)
            for di, det in enumerate(detections):
                dcx, dcy = _center(det.bbox)
                dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5 / max(tr.bbox[2], 1e-6)
                ov = iou(tr.bbox, det.bbox)
                # the allowed jump grows with the time a track has been unseen, so a
                # person who is missed for a few frames while moving is still re-matched
                gate = self.max_center_dist * 0.5 + 1.0 * (t - tr.last_seen)
                if ov >= self.iou_threshold or dist <= gate:
                    # higher is better: overlap dominates, distance breaks ties
                    pairs.append((ov - 0.1 * dist, tid, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        out: List[TrackObs] = []
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            det = detections[di]
            tr = self._tracks[tid]
            tr.bbox, tr.last_seen = det.bbox, t
            out.append(TrackObs(tid, det.bbox, det.yaw_deg))

        for di, det in enumerate(detections):
            if di in used_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(tid, det.bbox, t, t)
            out.append(TrackObs(tid, det.bbox, det.yaw_deg))
        return out

    @property
    def active_ids(self) -> List[int]:
        return list(self._tracks)
