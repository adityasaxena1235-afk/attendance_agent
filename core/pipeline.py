"""SessionPipeline: wires every module together for one class session.

Video frames and audio chunks can arrive in any interleaving; everything is keyed by
session-relative time `t` (seconds), so recorded files and live feeds work the same way.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from config import Config
from .agent import TeachingAgent
from .attendance import AttendanceManager
from .audio import SpeechDetector
from .engagement import EngagementAnalyzer
from .fusion import EngagementFusion
from .models import Detection, TrackObs
from .tracker import CentroidIoUTracker


class SessionPipeline:
    def __init__(self, cfg: Config, name: str = "Class session", source: str = "unknown",
                 detector=None):
        self.cfg = cfg
        self.id = uuid.uuid4().hex[:10]
        self.name = name
        self.source = source
        self.started_at = datetime.now()
        self.detector = detector

        self.tracker = CentroidIoUTracker(cfg.track_max_missing_seconds, cfg.track_iou_threshold,
                                          cfg.track_max_center_dist)
        self.attendance = AttendanceManager(cfg)
        self.engagement = EngagementAnalyzer(cfg)
        self.fusion = EngagementFusion(cfg)
        self.speech = SpeechDetector(cfg)
        self.agent = TeachingAgent(cfg)

        self.t = 0.0
        self.suggestion_log: List[dict] = []
        self.audio_seen = False
        self.audio_note: Optional[str] = None
        self._last_video_t: Optional[float] = None
        self._lock = threading.RLock()

    # ---- ingestion ------------------------------------------------------
    def process_detections(self, t: float, detections: List[Detection]) -> None:
        with self._lock:
            dt = t - self._last_video_t if self._last_video_t is not None else 1.0 / self.cfg.target_fps
            dt = min(max(dt, 1e-3), self.cfg.gap_tolerance_seconds)
            self._last_video_t = t
            self.t = max(self.t, t)

            obs: List[TrackObs] = self.tracker.update(t, detections)
            mapping = self.attendance.update(t, obs)
            for ob in obs:
                rid = mapping[ob.track_id]
                sample = self.engagement.update(rid, t, ob)
                # only confirmed participants contribute to engagement
                if self.attendance.records[rid].confirmed_at is not None:
                    self.fusion.add_visual(t, rid, sample, dt)

    def process_frame(self, t: float, frame_bgr: np.ndarray) -> None:
        if self.detector is None:
            raise RuntimeError("No face detector configured for process_frame().")
        self.process_detections(t, self.detector.detect(frame_bgr))

    def process_audio(self, t: float, samples: np.ndarray, sr: int) -> None:
        with self._lock:
            self.audio_seen = True
            self.t = max(self.t, t)
            self.fusion.add_audio(self.speech.process(t, samples, sr))

    # ---- views ----------------------------------------------------------
    def _participants(self) -> List[dict]:
        att = {a["id"]: a for a in self.attendance.snapshot(self.started_at, max(self.t, 1e-6))}
        out = []
        for rid, a in att.items():
            stats = self.fusion.participant_stats(rid)
            out.append({**a, **stats})
        return out

    def snapshot(self, advise: bool = True) -> dict:
        with self._lock:
            windows = self.fusion.windows()
            trends = self.fusion.trends(windows)
            participants = self._participants()
            attendance = [{k: p[k] for k in ("id", "label", "status", "check_in", "check_in_offset_s",
                                             "last_seen", "left_at", "present_seconds",
                                             "presence_pct", "visits")} for p in participants]
            if advise:
                # advise only on completed windows: a half-filled window is too noisy to act on
                done = [w for w in windows if w["end_s"] <= self.t + 1e-6]
                for s in self.agent.advise(self.t, done, self.fusion.trends(done), participants,
                                           attendance, live=True):
                    self.suggestion_log.append(s.__dict__)
            return {
                "id": self.id, "name": self.name, "elapsed_s": round(self.t, 1),
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "present_now": self.attendance.present_count(),
                "attendees": len(attendance),
                "filtered_transients": self.attendance.transient_count(),
                "attendance": attendance,
                "participants": participants,
                "windows": windows,
                "trends": trends,
                "events": [self._event(e) for e in self.attendance.events],
                "suggestions": self.suggestion_log[-12:][::-1],
                "has_audio": self.audio_seen,
                "audio_note": self.audio_note,
            }

    def _event(self, e) -> dict:
        return {"time": (self.started_at + timedelta(seconds=e.t)).isoformat(timespec="seconds"),
                "offset_s": round(e.t, 1), "kind": e.kind, "participant": e.participant,
                "detail": e.detail}

    def finalize(self) -> dict:
        """Close the session and return the complete report (also used for storage)."""
        with self._lock:
            windows = self.fusion.windows()
            trends = self.fusion.trends(windows)
            participants = self._participants()
            attendance = [{k: p[k] for k in ("id", "label", "status", "check_in", "check_in_offset_s",
                                             "last_seen", "left_at", "present_seconds",
                                             "presence_pct", "visits")} for p in participants]
            final_sugg = self.agent.advise(self.t, windows, trends, participants, attendance, live=False)
            llm_note = self.agent.llm_summary(final_sugg, trends)
            return {
                "id": self.id, "name": self.name, "source": self.source,
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "duration_s": round(self.t, 1),
                "attendance": attendance,
                "participants": participants,
                "windows": windows,
                "trends": trends,
                "events": [self._event(e) for e in self.attendance.events],
                "suggestions": [s.__dict__ for s in final_sugg],
                "coach_note": llm_note,
                "filtered_transients": self.attendance.transient_count(),
                "settings": {"presence_seconds": self.cfg.presence_seconds,
                             "window_seconds": self.cfg.window_seconds},
                "privacy": {"offline": True, "frames_stored": False, "audio_stored": False,
                            "identity_recognition": False},
            }
