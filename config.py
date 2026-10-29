"""Central configuration for the Multimodal Attendance & Engagement Agent.

All processing is local. Nothing in this project makes network calls unless you
explicitly point `LOCAL_LLM_URL` at a model server running on your own machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "attendance.db"
REPORTS_DIR = DATA_DIR / "reports"
UPLOADS_DIR = DATA_DIR / "uploads"


@dataclass
class Config:
    # ---- Attendance (sustained presence) --------------------------------
    presence_seconds: float = 8.0      # continuous presence needed to confirm attendance
    gap_tolerance_seconds: float = 3.0  # short detection dropouts that do not break a run
    absence_seconds: float = 20.0      # unseen this long -> participant considered "left"

    # ---- Tracking -------------------------------------------------------
    track_max_missing_seconds: float = 3.0
    track_iou_threshold: float = 0.2
    track_max_center_dist: float = 1.5  # in units of box width

    # ---- Engagement signals ---------------------------------------------
    yaw_attentive_deg: float = 25.0     # |yaw| below this = looking at the front
    yaw_away_deg: float = 60.0          # |yaw| above this = fully looking away
    still_motion_thresh: float = 0.03   # normalised displacement/sec below this = still
    restless_motion_thresh: float = 0.60
    inactivity_seconds: float = 45.0    # still for this long -> "inactive" flag

    # ---- Audio ----------------------------------------------------------
    speech_margin_db: float = 9.0       # dB above adaptive noise floor to count as speech
    min_speech_dB: float = -45.0        # absolute floor so silence never counts as speech
    turn_min_seconds: float = 1.0       # speech burst length that counts as a "turn"

    # ---- Fusion & windows -----------------------------------------------
    window_seconds: float = 30.0
    visual_weight: float = 0.75
    audio_weight: float = 0.25
    motion_weight: float = 0.4          # how much movement/stillness can scale visual engagement
    low_engagement: float = 50.0        # window score below this is a "low" window
    drop_points: float = 15.0           # fall vs previous window flagged as a drop
    peak_points: float = 12.0           # rise vs previous window flagged as a peak

    # ---- Processing -----------------------------------------------------
    target_fps: float = 5.0             # frames/sec actually analysed
    face_backend: str = "auto"          # auto | yunet | haar | mediapipe
    min_face_size: int = 40             # pixels

    # ---- Privacy --------------------------------------------------------
    store_frames: bool = False          # raw frames are never written to disk by default
    retention_days: int = 90            # old sessions are purged on startup

    # ---- Optional local LLM (e.g. Ollama on localhost). Empty = rule-based only.
    local_llm_url: str = field(default_factory=lambda: os.environ.get("LOCAL_LLM_URL", ""))
    local_llm_model: str = field(default_factory=lambda: os.environ.get("LOCAL_LLM_MODEL", "llama3.2"))


def ensure_dirs() -> None:
    for p in (DATA_DIR, MODELS_DIR, REPORTS_DIR, UPLOADS_DIR):
        p.mkdir(parents=True, exist_ok=True)
