"""Synthetic classroom generator so the whole system runs without a camera or microphone.

It emits the same kind of data the real detectors produce (face boxes with yaw,
audio chunks), including the awkward cases the system must handle:

* late arrivals and an early leaver
* people just walking past (must NOT be marked present)
* a participant occluded for several seconds (tracker loses them, attendance must merge them back)
* random detector drop-outs
* a class timeline with lecture, discussion, long one-voice lecture (fatigue), silent work and an activity
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

from core.models import Detection

SR = 16000
CHUNK_S = 0.5


@dataclass
class Profile:
    name: str
    seat: Tuple[float, float]
    arrive: float = 0.0
    leave: float = 1e9
    trait: float = 1.0          # >1 = easily distracted
    stillness: float = 0.0      # extra tendency to sit still
    occlusion: Optional[Tuple[float, float]] = None


# (start, end, label, distract_prob, restless_prob, still_prob)
PHASES = [
    (0, 240, "lecture", 0.10, 0.02, 0.05),
    (240, 360, "discussion", 0.05, 0.05, 0.02),
    (360, 540, "long lecture", None, 0.03, 0.10),     # distraction ramps up (fatigue)
    (540, 600, "silent work", 0.30, 0.01, 0.45),
    (600, 720, "activity", 0.05, 0.30, 0.01),
    (720, 900, "lecture", 0.12, 0.03, 0.06),
]


def _phase(t: float):
    for p in PHASES:
        if p[0] <= t < p[1]:
            return p
    return PHASES[-1]


def _distract_prob(t: float, ph) -> float:
    if ph[3] is not None:
        return ph[3]
    frac = (t - ph[0]) / (ph[1] - ph[0])
    return 0.10 + 0.45 * frac


def default_profiles() -> List[Profile]:
    seats = [(120, 120), (320, 120), (520, 120), (720, 120),
             (120, 300), (320, 300), (520, 300), (720, 300)]
    return [
        Profile("p1", seats[0], arrive=0, trait=0.7),
        Profile("p2", seats[1], arrive=0, trait=1.0),
        Profile("p3", seats[2], arrive=5, trait=1.1, occlusion=(300, 306)),  # occluded 6 s
        Profile("p4", seats[3], arrive=10, trait=1.0),
        Profile("p5", seats[4], arrive=0, trait=2.2, stillness=0.35),         # often disengaged
        Profile("p6", seats[5], arrive=150, trait=1.0),                        # late
        Profile("p7", seats[6], arrive=400, trait=1.2),                        # very late
        Profile("p8", seats[7], arrive=0, trait=1.0, leave=700),               # leaves early
    ]


def _speech_pattern(rng: np.random.Generator, duration: float) -> List[bool]:
    n = int(duration / CHUNK_S)
    out: List[bool] = []
    i = 0
    while i < n:
        t = i * CHUNK_S
        lab = _phase(t)[2]
        if lab == "discussion" or lab == "activity":
            burst = int(rng.integers(3, 8))       # 1.5-4 s of speech
            gap = int(rng.integers(2, 4))         # 1-2 s of silence between speakers
            out += [True] * burst + [False] * gap
            i += burst + gap
        elif lab == "silent work":
            out.append(False)
            i += 1
        else:  # lecture: mostly speech with short pauses
            run = int(rng.integers(10, 30))
            out += [True] * run + [False] * 1
            i += run + 1
    return out[:n]


def _audio_chunk(rng: np.random.Generator, speaking: bool) -> np.ndarray:
    n = int(SR * CHUNK_S)
    t = np.arange(n) / SR
    if speaking:
        f0 = float(rng.uniform(110, 240))
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28))
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
        return (0.05 * env * sig + rng.normal(0, 0.003, n)).astype(np.float32)
    return rng.normal(0, 0.003, n).astype(np.float32)


class _Person:
    def __init__(self, prof: Profile, rng: np.random.Generator):
        self.p = prof
        self.rng = rng
        self.state = "attentive"
        self.state_until = prof.arrive
        self.yaw_target = 0.0
        self.phase_off = rng.uniform(0, 6.28)
        self.period = rng.uniform(5.0, 8.0)

    def _resample(self, t: float) -> None:
        ph = _phase(t)
        dp = min(0.9, _distract_prob(t, ph) * self.p.trait)
        rp = ph[4]
        sp = min(0.9, ph[5] + self.p.stillness)
        r = self.rng.random()
        if r < dp:
            self.state, self.yaw_target = "distracted", float(self.rng.choice([-1, 1]) * self.rng.uniform(45, 85))
        elif r < dp + rp:
            self.state = "restless"
        elif r < dp + rp + sp:
            self.state = "still"
        else:
            self.state = "attentive"
        if self.state != "distracted":
            self.yaw_target = 0.0
        self.state_until = t + float(self.rng.uniform(6, 18))

    def detection(self, t: float) -> Optional[Detection]:
        p = self.p
        if not (p.arrive <= t < p.leave):
            return None
        if p.occlusion and p.occlusion[0] <= t < p.occlusion[1]:
            return None
        if self.rng.random() < 0.03:       # random detector miss
            return None
        if t >= self.state_until:
            self._resample(t)

        w, h = 90.0, 110.0
        x, y = p.seat
        if self.state == "still":
            dx = dy = 0.0
        elif self.state == "restless":
            dx = 0.5 * w * math.sin(2 * math.pi * t / 2.0 + self.phase_off)
            dy = 0.3 * w * math.cos(2 * math.pi * t / 1.7 + self.phase_off)
        else:   # slow natural sway
            dx = 0.08 * w * math.sin(2 * math.pi * t / self.period + self.phase_off)
            dy = 0.05 * w * math.cos(2 * math.pi * t / self.period + self.phase_off)
        jx, jy = self.rng.normal(0, 0.008 * w, 2)          # detector jitter
        yaw = self.yaw_target + float(self.rng.normal(0, 6.0))
        return Detection((x + dx + jx, y + dy + jy, w, h), 0.95, yaw)


def simulate(seed: int = 7, duration_s: float = 900.0, fps: float = 5.0,
             profiles: Optional[List[Profile]] = None, with_audio: bool = True,
             passersby: Tuple[Tuple[float, float], ...] = ((100.0, 2.5), (500.0, 3.0))
             ) -> Iterator[Tuple[str, float, object]]:
    """Yield ('video', t, [Detection]) and ('audio', t, (samples, sr)) in time order."""
    rng = np.random.default_rng(seed)
    people = [_Person(pr, rng) for pr in (profiles or default_profiles())]
    speech = _speech_pattern(rng, duration_s) if with_audio else []

    n_frames = int(duration_s * fps)
    next_audio = 0.0
    a_idx = 0
    for i in range(n_frames):
        t = i / fps
        if with_audio:
            while next_audio <= t and a_idx < len(speech):
                yield "audio", next_audio, (_audio_chunk(rng, speech[a_idx]), SR)
                a_idx += 1
                next_audio += CHUNK_S
        dets = [d for d in (pp.detection(t) for pp in people) if d is not None]
        for k, (start, length) in enumerate(passersby):      # briefly visible walkers
            if start <= t < start + length:
                dets.append(Detection((900.0 + 50 * k + 60 * (t - start), 200.0, 80.0, 100.0), 0.9, 10.0))
        yield "video", t, dets


def replay(pipeline, events: Iterator[Tuple[str, float, object]]) -> None:
    """Feed a simulated stream into a SessionPipeline."""
    for kind, t, payload in events:
        if kind == "video":
            pipeline.process_detections(t, payload)
        else:
            samples, sr = payload
            pipeline.process_audio(t, samples, sr)
