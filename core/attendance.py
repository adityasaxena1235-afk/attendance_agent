"""Sustained-presence attendance.

A person is only marked present after being continuously in view for
`presence_seconds` (short detection drop-outs up to `gap_tolerance_seconds` do not
break the run). One-frame false positives and people just walking past never reach
that bar, so they never appear in the attendance log.

Tracks that disappear and re-appear under a new track id (occlusion, detector
miss) are merged back into the same participant by spatial re-association.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import Config
from .models import Event, TrackObs
from .tracker import iou


@dataclass
class Record:
    rid: int
    first_seen: float
    run_start: float
    last_seen: float
    last_bbox: tuple
    present_time: float = 0.0
    confirmed_at: Optional[float] = None
    check_in: Optional[float] = None   # when the person actually arrived (start of confirming run)
    label: Optional[str] = None
    status: str = "pending"            # pending | present | left
    visits: int = 1
    left_at: Optional[float] = None


class AttendanceManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.records: Dict[int, Record] = {}
        self._track_to_rid: Dict[int, int] = {}
        self._next_rid = 1
        self._next_label = 1
        self.events: List[Event] = []

    # ------------------------------------------------------------------
    def _new_record(self, t: float, bbox: tuple) -> Record:
        rec = Record(self._next_rid, t, t, t, bbox)
        self.records[rec.rid] = rec
        self._next_rid += 1
        return rec

    def _find_lost_match(self, t: float, bbox: tuple, taken: set) -> Optional[Record]:
        """Re-associate a brand-new track with a recently lost, confirmed participant."""
        best, best_score = None, 0.3
        for rec in self.records.values():
            if rec.rid in taken or rec.confirmed_at is None:
                continue
            if t - rec.last_seen > self.cfg.absence_seconds:
                continue
            score = iou(rec.last_bbox, bbox)
            if score > best_score:
                best, best_score = rec, score
        return best

    # ------------------------------------------------------------------
    def update(self, t: float, observations: List[TrackObs]) -> Dict[int, int]:
        """Process one analysed frame. Returns {track_id: record_id} for this frame."""
        cfg = self.cfg
        seen_now: Dict[int, int] = {}
        taken = set()

        for ob in observations:
            rid = self._track_to_rid.get(ob.track_id)
            if rid is None:
                match = self._find_lost_match(t, ob.bbox, taken)
                rec = match if match is not None else self._new_record(t, ob.bbox)
                self._track_to_rid[ob.track_id] = rec.rid
                rid = rec.rid
            rec = self.records[rid]
            taken.add(rid)
            seen_now[ob.track_id] = rid

            gap = t - rec.last_seen
            if gap > 0:
                if gap <= cfg.gap_tolerance_seconds:
                    rec.present_time += gap
                else:
                    # a real gap: a new continuous run starts
                    if rec.confirmed_at is None:
                        rec.run_start = t
                    else:
                        rec.visits += 1
                        if rec.status == "left":
                            rec.status = "present"
                            rec.left_at = None
                            self.events.append(Event(t, "RETURNED", rec.label or f"#{rec.rid}",
                                                     f"absent {gap:.0f}s"))
            rec.last_seen = t
            rec.last_bbox = ob.bbox

            if rec.confirmed_at is None and t - rec.run_start >= cfg.presence_seconds:
                rec.confirmed_at = t
                rec.check_in = rec.run_start
                rec.status = "present"
                rec.label = f"Participant {self._next_label}"
                self._next_label += 1
                self.events.append(Event(rec.check_in, "CHECK_IN", rec.label,
                                         f"confirmed after {cfg.presence_seconds:.0f}s of presence"))

        # departures
        for rec in self.records.values():
            if rec.status == "present" and rec.rid not in taken \
                    and t - rec.last_seen > cfg.absence_seconds:
                rec.status = "left"
                rec.left_at = rec.last_seen
                self.events.append(Event(rec.last_seen, "LEFT", rec.label or f"#{rec.rid}",
                                         f"not seen for {cfg.absence_seconds:.0f}s"))
        return seen_now

    # ------------------------------------------------------------------
    def rename(self, rid: int, label: str) -> bool:
        rec = self.records.get(rid)
        if rec is None or rec.confirmed_at is None:
            return False
        rec.label = label
        return True

    def confirmed(self) -> List[Record]:
        return sorted((r for r in self.records.values() if r.confirmed_at is not None),
                      key=lambda r: r.check_in or 0)

    def transient_count(self) -> int:
        """Detections that never reached sustained presence (filtered false positives)."""
        return sum(1 for r in self.records.values() if r.confirmed_at is None)

    def present_count(self) -> int:
        return sum(1 for r in self.records.values() if r.status == "present")

    def snapshot(self, session_start: datetime, now: float) -> List[dict]:
        out = []
        for r in self.confirmed():
            out.append({
                "id": r.rid,
                "label": r.label,
                "status": r.status,
                "check_in": (session_start + timedelta(seconds=r.check_in)).isoformat(timespec="seconds"),
                "check_in_offset_s": round(r.check_in, 1),
                "last_seen": (session_start + timedelta(seconds=r.last_seen)).isoformat(timespec="seconds"),
                "left_at": (session_start + timedelta(seconds=r.left_at)).isoformat(timespec="seconds")
                if r.left_at is not None else None,
                "present_seconds": round(r.present_time, 1),
                "presence_pct": round(100.0 * r.present_time / now, 1) if now > 0 else 0.0,
                "visits": r.visits,
            })
        return out
