"""Plain data containers shared across modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

BBox = Tuple[float, float, float, float]  # x, y, w, h in pixels (or any consistent unit)


@dataclass
class Detection:
    """One face found in one frame."""
    bbox: BBox
    score: float = 1.0
    yaw_deg: Optional[float] = None  # head turn estimate, 0 = facing camera


@dataclass
class TrackObs:
    """A detection that has been assigned to a persistent track."""
    track_id: int
    bbox: BBox
    yaw_deg: Optional[float] = None


@dataclass
class AudioFeatures:
    """Result of analysing one audio chunk."""
    t: float
    duration: float
    level_db: float
    noise_floor_db: float
    speech: bool


@dataclass
class Event:
    t: float
    kind: str          # CHECK_IN | LEFT | RETURNED
    participant: str
    detail: str = ""


@dataclass
class Suggestion:
    t: float
    priority: str      # high | medium | low
    title: str
    rationale: str
    evidence: dict = field(default_factory=dict)
    source: str = "rules"
