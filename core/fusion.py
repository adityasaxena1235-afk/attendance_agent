"""Multimodal fusion: visual behaviour + audio activity -> engagement scores & trends."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from config import Config
from .audio import AudioWindow, classify_interaction, interaction_score
from .engagement import Sample
from .models import AudioFeatures


class _PAcc:
    """Time-weighted accumulator for one participant inside one window."""
    __slots__ = ("w", "att", "mot", "vis", "inactive", "restless", "yaw_sum", "yaw_n")

    def __init__(self):
        self.w = self.att = self.mot = self.vis = self.inactive = self.restless = 0.0
        self.yaw_sum = 0.0
        self.yaw_n = 0

    def add(self, s: Sample, dt: float, motion_weight: float):
        self.w += dt
        self.att += s.attention * dt
        self.mot += s.motion * dt
        # Visual engagement: looking away zeroes it whatever the body is doing, while
        # healthy movement vs long stillness scales it between (1-mw) and 1.
        self.vis += s.attention * ((1.0 - motion_weight) + motion_weight * s.motion) * dt
        if s.inactive:
            self.inactive += dt
        if s.restless:
            self.restless += dt
        if s.yaw is not None:
            self.yaw_sum += abs(s.yaw)
            self.yaw_n += 1


class _WindowAcc:
    def __init__(self, cfg: Config):
        self.people: Dict[int, _PAcc] = defaultdict(_PAcc)
        self.audio = AudioWindow(cfg)


class EngagementFusion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._wins: Dict[int, _WindowAcc] = {}
        self._totals: Dict[int, _PAcc] = defaultdict(_PAcc)
        self._per_person_windows: Dict[int, Dict[int, float]] = defaultdict(dict)

    # ---- ingestion -----------------------------------------------------
    def _win(self, t: float) -> _WindowAcc:
        idx = int(t // self.cfg.window_seconds)
        if idx not in self._wins:
            self._wins[idx] = _WindowAcc(self.cfg)
        return self._wins[idx]

    def add_visual(self, t: float, rid: int, sample: Sample, dt: float) -> None:
        mw = self.cfg.motion_weight
        self._win(t).people[rid].add(sample, dt, mw)
        self._totals[rid].add(sample, dt, mw)

    def add_audio(self, feat: AudioFeatures) -> None:
        self._win(feat.t).audio.add(feat)

    # ---- scoring -------------------------------------------------------
    def _visual_score(self, p: _PAcc) -> float:
        return p.vis / p.w if p.w > 0 else 0.0

    def _fuse(self, visual: Optional[float], audio: float, has_audio: bool) -> Optional[float]:
        cfg = self.cfg
        if visual is None:
            return None
        if not has_audio:
            return 100.0 * visual
        total_w = cfg.visual_weight + cfg.audio_weight
        return 100.0 * (cfg.visual_weight * visual + cfg.audio_weight * audio) / total_w

    def window_result(self, idx: int) -> Optional[dict]:
        w = self._wins.get(idx)
        if w is None:
            return None
        cfg = self.cfg
        a_sum = w.audio.summary()
        a_score = interaction_score(a_sum)
        has_audio = a_sum["has_audio"]

        people = {rid: p for rid, p in w.people.items() if p.w > 0}
        visual = float(np.mean([self._visual_score(p) for p in people.values()])) if people else None
        class_score = self._fuse(visual, a_score, has_audio)

        per_person = {}
        for rid, p in people.items():
            sc = self._fuse(self._visual_score(p), a_score, has_audio)
            per_person[rid] = round(sc, 1)
            self._per_person_windows[rid][idx] = sc

        attentive = [p.att / p.w >= 0.7 for p in people.values()]
        return {
            "index": idx,
            "start_s": idx * cfg.window_seconds,
            "end_s": (idx + 1) * cfg.window_seconds,
            "score": round(class_score, 1) if class_score is not None else None,
            "visual_score": round(100 * visual, 1) if visual is not None else None,
            "audio_score": round(100 * a_score, 1) if has_audio else None,
            "participants": len(people),
            "attentive_pct": round(100.0 * sum(attentive) / len(attentive), 1) if attentive else None,
            "inactive_count": sum(1 for p in people.values() if p.inactive / max(p.w, 1e-6) > 0.5),
            "restless_count": sum(1 for p in people.values() if p.restless / max(p.w, 1e-6) > 0.5),
            "interaction": classify_interaction(a_sum),
            "speech_ratio": a_sum["speech_ratio"] if has_audio else None,
            "turns": a_sum["turns"] if has_audio else None,
            "longest_silence_s": a_sum["longest_silence_s"] if has_audio else None,
            "per_person": per_person,
        }

    def windows(self) -> List[dict]:
        out = []
        for idx in sorted(self._wins):
            r = self.window_result(idx)
            if r is not None and r["score"] is not None:
                out.append(r)
        return out

    # ---- trends --------------------------------------------------------
    def trends(self, windows: Optional[List[dict]] = None) -> dict:
        cfg = self.cfg
        wins = windows if windows is not None else self.windows()
        if not wins:
            return {"direction": "n/a", "slope_per_min": 0.0, "drops": [], "peaks": [],
                    "low_windows": [], "best": None, "worst": None, "mean": None}
        scores = np.array([w["score"] for w in wins], dtype=float)
        mins = np.array([w["start_s"] / 60.0 for w in wins], dtype=float)
        slope = float(np.polyfit(mins, scores, 1)[0]) if len(wins) >= 3 else 0.0
        direction = "declining" if slope < -1.0 else "improving" if slope > 1.0 else "steady"

        drops, peaks = [], []
        for prev, cur in zip(wins, wins[1:]):
            delta = cur["score"] - prev["score"]
            if delta <= -cfg.drop_points:
                drops.append({"at_s": cur["start_s"], "from": prev["score"], "to": cur["score"],
                              "delta": round(delta, 1)})
            elif delta >= cfg.peak_points:
                peaks.append({"at_s": cur["start_s"], "from": prev["score"], "to": cur["score"],
                              "delta": round(delta, 1)})
        low = [w["start_s"] for w in wins if w["score"] < cfg.low_engagement]
        best = max(wins, key=lambda w: w["score"])
        worst = min(wins, key=lambda w: w["score"])
        return {
            "direction": direction,
            "slope_per_min": round(slope, 2),
            "drops": drops,
            "peaks": peaks,
            "low_windows": low,
            "best": {"at_s": best["start_s"], "score": best["score"]},
            "worst": {"at_s": worst["start_s"], "score": worst["score"]},
            "mean": round(float(scores.mean()), 1),
        }

    # ---- per-participant analytics ------------------------------------
    def participant_stats(self, rid: int) -> dict:
        p = self._totals.get(rid)
        if p is None or p.w <= 0:
            return {"engagement": None, "attention_pct": None, "inactive_s": 0.0,
                    "restless_s": 0.0, "timeline": []}
        self.windows()  # refresh per-person window scores
        tl = [{"start_s": idx * self.cfg.window_seconds, "score": round(sc, 1)}
              for idx, sc in sorted(self._per_person_windows.get(rid, {}).items())]
        scores = [x["score"] for x in tl]
        return {
            "engagement": round(float(np.mean(scores)), 1) if scores else None,
            "attention_pct": round(100 * p.att / p.w, 1),
            "inactive_s": round(p.inactive, 1),
            "restless_s": round(p.restless, 1),
            "avg_head_turn_deg": round(p.yaw_sum / p.yaw_n, 1) if p.yaw_n else None,
            "timeline": tl,
        }
