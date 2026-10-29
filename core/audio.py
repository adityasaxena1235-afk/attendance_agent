"""Energy-based speech / interaction detection (no cloud, no speech-to-text).

The audio stream is analysed only for *whether* people are talking and how
conversation is distributed over time. Audio itself is never stored and the
words are never transcribed.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from config import Config
from .models import AudioFeatures


class SpeechDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.noise_floor_db: Optional[float] = None
        self._speaking = False

    @staticmethod
    def _level_db(x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        return 20.0 * math.log10(max(rms, 1e-9))

    @staticmethod
    def _zcr(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

    def process(self, t: float, samples: np.ndarray, sr: int) -> AudioFeatures:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        level = self._level_db(x)
        zcr = self._zcr(x)

        # adaptive noise floor: drops quickly, rises very slowly
        if self.noise_floor_db is None:
            self.noise_floor_db = level
        elif level < self.noise_floor_db:
            self.noise_floor_db += 0.3 * (level - self.noise_floor_db)
        else:
            self.noise_floor_db += 0.002 * (level - self.noise_floor_db)

        on_thresh = max(self.noise_floor_db + self.cfg.speech_margin_db, self.cfg.min_speech_dB)
        off_thresh = on_thresh - 3.0  # hysteresis avoids flicker
        thresh = off_thresh if self._speaking else on_thresh
        self._speaking = level > thresh and zcr < 0.4
        return AudioFeatures(t=t, duration=len(x) / float(sr), level_db=level,
                             noise_floor_db=self.noise_floor_db, speech=self._speaking)


class AudioWindow:
    """Accumulates AudioFeatures and reports interaction statistics."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.speech = 0.0
        self.turns = 0
        self.longest_speech = 0.0
        self.longest_silence = 0.0
        self._run_speech = 0.0
        self._run_silence = 0.0
        self._in_speech = False
        self._levels: List[float] = []

    def add(self, f: AudioFeatures) -> None:
        self.total += f.duration
        self._levels.append(f.level_db)
        if f.speech:
            self.speech += f.duration
            self._run_speech += f.duration
            if self._run_silence > 0:
                self.longest_silence = max(self.longest_silence, self._run_silence)
            self._run_silence = 0.0
            if not self._in_speech and self._run_speech >= 0:
                self._in_speech = True
        else:
            if self._in_speech:
                if self._run_speech >= self.cfg.turn_min_seconds:
                    self.turns += 1
                self.longest_speech = max(self.longest_speech, self._run_speech)
                self._in_speech = False
            self._run_speech = 0.0
            self._run_silence += f.duration

    def summary(self) -> dict:
        ratio = self.speech / self.total if self.total > 0 else 0.0
        silence = max(self.longest_silence, self._run_silence)
        speech_run = max(self.longest_speech, self._run_speech)
        return {
            "speech_ratio": round(ratio, 3),
            "turns": self.turns + (1 if self._in_speech and self._run_speech >= self.cfg.turn_min_seconds else 0),
            "longest_speech_s": round(speech_run, 1),
            "longest_silence_s": round(silence, 1),
            "avg_level_db": round(float(np.mean(self._levels)), 1) if self._levels else None,
            "has_audio": self.total > 0,
        }


def interaction_score(summary: dict) -> float:
    """0..1 score of how 'alive' the audio environment is.

    * Complete silence is weak engagement (nothing happening).
    * A moderate speech ratio with several turn changes (discussion / Q&A) is best.
    * Non-stop single-voice talk is neutral-to-slightly-low (pure lecture).
    """
    if not summary.get("has_audio"):
        return 0.5
    r = summary["speech_ratio"]
    turns = summary["turns"]
    if r < 0.05:
        base = 0.25
    elif r < 0.15:
        base = 0.45
    elif r < 0.8:
        base = 0.85
    elif r < 0.9:
        base = 0.65
    else:
        base = 0.5
    # turn-taking only signals interaction when it is not just one voice talking
    turn_bonus = min(turns, 6) * 0.025 if r < 0.8 else 0.0
    return float(min(1.0, base + turn_bonus))


def classify_interaction(summary: dict) -> str:
    if not summary.get("has_audio"):
        return "unknown"
    r, turns = summary["speech_ratio"], summary["turns"]
    if r < 0.05:
        return "silent"
    # Without speaker identification, near-constant speech means one voice is talking
    # (brief pauses for breath or slides do not change that).
    if r >= 0.8:
        return "continuous lecture"
    if turns >= 4 and 0.15 <= r < 0.8:
        return "discussion"
    return "mixed"
