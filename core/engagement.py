"""Per-participant behavioural signals: head pose, movement, inactivity."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from config import Config
from .models import TrackObs

# Movement is measured as displacement over ~1 second (not frame to frame) so that
# detector jitter, which cancels out over time, is not mistaken for body movement.
_SPEED_WINDOW_S = 1.0


@dataclass
class Sample:
    attention: float      # 0..1 from head pose
    motion: float         # 0..1 activity level (still -> low, natural movement -> high)
    inactive: bool
    restless: bool
    yaw: Optional[float]
    speed: float          # face-widths per second, smoothed


def attention_from_yaw(yaw: Optional[float], cfg: Config) -> float:
    """1.0 when looking forward, falling linearly to 0.0 when fully turned away."""
    if yaw is None:
        return 0.5  # unknown pose: neutral
    a = abs(yaw)
    if a <= cfg.yaw_attentive_deg:
        return 1.0
    if a >= cfg.yaw_away_deg:
        return 0.0
    return 1.0 - (a - cfg.yaw_attentive_deg) / (cfg.yaw_away_deg - cfg.yaw_attentive_deg)


class _State:
    def __init__(self, t: float):
        self.hist: Deque[Tuple[float, float, float]] = deque()  # (t, cx, cy)
        self.last_motion_t = t
        self.speed_hist: Deque[float] = deque(maxlen=8)


class EngagementAnalyzer:
    """Keeps a small state per participant and turns observations into samples."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: Dict[int, _State] = {}

    def update(self, rid: int, t: float, ob: TrackObs) -> Sample:
        cfg = self.cfg
        cx, cy = ob.bbox[0] + ob.bbox[2] / 2, ob.bbox[1] + ob.bbox[3] / 2
        st = self._state.get(rid)
        if st is None:
            st = self._state[rid] = _State(t)

        # drop history older than the speed window and compare with the oldest kept point
        while st.hist and t - st.hist[0][0] > _SPEED_WINDOW_S:
            st.hist.popleft()
        if st.hist:
            t0, x0, y0 = st.hist[0]
            dist = ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5
            speed = dist / max(ob.bbox[2], 1e-6) / max(t - t0, 1e-3)
        else:
            speed = 0.0
        st.hist.append((t, cx, cy))
        st.speed_hist.append(speed)
        smooth = sum(st.speed_hist) / len(st.speed_hist)

        if smooth > cfg.still_motion_thresh:
            st.last_motion_t = t
        still_for = t - st.last_motion_t
        inactive = still_for >= cfg.inactivity_seconds
        restless = smooth > cfg.restless_motion_thresh

        # Natural small movement is healthy. Long total stillness and wild movement
        # both lower the score.
        if restless:
            motion = 0.5
        elif inactive:
            motion = 0.15
        else:
            decay = min(1.0, still_for / cfg.inactivity_seconds)
            motion = 1.0 - 0.7 * decay

        return Sample(
            attention=attention_from_yaw(ob.yaw_deg, cfg),
            motion=motion,
            inactive=inactive,
            restless=restless,
            yaw=ob.yaw_deg,
            speed=smooth,
        )
