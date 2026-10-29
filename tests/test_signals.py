import math

import numpy as np

from config import Config
from core.audio import AudioWindow, SpeechDetector, classify_interaction, interaction_score
from core.engagement import EngagementAnalyzer, attention_from_yaw
from core.models import TrackObs
from core.vision import yaw_from_landmarks

SR = 16000


def tone(seconds=0.5, amp=0.05, f=180.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def noise(seconds=0.5, amp=0.003, seed=0):
    return np.random.default_rng(seed).normal(0, amp, int(SR * seconds)).astype(np.float32)


# ---------------- head pose / attention ----------------
def test_attention_mapping():
    cfg = Config()
    assert attention_from_yaw(0, cfg) == 1.0
    assert attention_from_yaw(-20, cfg) == 1.0
    assert attention_from_yaw(90, cfg) == 0.0
    assert 0.0 < attention_from_yaw(45, cfg) < 1.0
    assert attention_from_yaw(None, cfg) == 0.5


def test_yaw_from_landmarks_is_signed_and_zero_when_frontal():
    # right eye, left eye, nose, right mouth, left mouth (image coordinates)
    frontal = yaw_from_landmarks((40, 50), (80, 50), (60, 70), (45, 90), (75, 90))
    right = yaw_from_landmarks((40, 50), (80, 50), (74, 70), (50, 90), (80, 90))
    left = yaw_from_landmarks((40, 50), (80, 50), (46, 70), (40, 90), (70, 90))
    assert abs(frontal) < 1
    assert right > 15 and left < -15


# ---------------- movement / inactivity ----------------
def feed(an, seconds, fn, fps=5):
    s = None
    for i in range(int(seconds * fps)):
        t = i / fps
        s = an.update(1, t, TrackObs(1, fn(t), 0.0))
    return s


def test_still_person_becomes_inactive_and_swaying_person_does_not():
    cfg = Config(inactivity_seconds=45)
    still = feed(EngagementAnalyzer(cfg), 60, lambda t: (100.0, 100.0, 90.0, 110.0))
    assert still.inactive and still.motion < 0.3
    sway = feed(EngagementAnalyzer(cfg), 60,
                lambda t: (100 + 8 * math.sin(t), 100.0, 90.0, 110.0))
    assert not sway.inactive and sway.motion > 0.9


def test_detector_jitter_is_not_mistaken_for_movement():
    cfg = Config(inactivity_seconds=45)
    rng = np.random.default_rng(1)
    s = feed(EngagementAnalyzer(cfg), 60,
             lambda t: (100 + rng.normal(0, 0.7), 100 + rng.normal(0, 0.7), 90.0, 110.0))
    assert s.inactive


def test_restless_person_is_flagged():
    cfg = Config()
    s = feed(EngagementAnalyzer(cfg), 20, lambda t: (100 + 45 * math.sin(2 * math.pi * t / 2), 100.0, 90.0, 110.0))
    assert s.restless


# ---------------- audio ----------------
def test_silence_is_not_speech_and_speech_is():
    det, cfg = SpeechDetector(Config()), Config()
    for i in range(6):
        assert not det.process(i * 0.5, noise(seed=i), SR).speech
    assert det.process(3.0, tone(), SR).speech
    assert not det.process(3.5, noise(seed=9), SR).speech


def test_noise_floor_adapts_to_a_noisy_room():
    det = SpeechDetector(Config())
    for i in range(40):
        det.process(i * 0.5, noise(amp=0.02, seed=i), SR)    # steady -34 dB hum
    assert not det.process(20.0, noise(amp=0.02, seed=99), SR).speech
    assert det.process(20.5, tone(amp=0.2), SR).speech


def test_one_voice_vs_discussion_classification():
    cfg = Config()
    det = SpeechDetector(cfg)
    lecture = AudioWindow(cfg)
    for i in range(60):
        lecture.add(det.process(i * 0.5, tone() if i % 20 else noise(), SR))
    assert classify_interaction(lecture.summary()) == "continuous lecture"

    det2, disc = SpeechDetector(cfg), AudioWindow(cfg)
    pattern = ([True] * 4 + [False] * 3) * 6
    for i, sp in enumerate(pattern):
        disc.add(det2.process(i * 0.5, tone() if sp else noise(seed=i), SR))
    s = disc.summary()
    assert classify_interaction(s) == "discussion" and s["turns"] >= 4
    assert interaction_score(s) > interaction_score(lecture.summary())


def test_silent_room_scores_low():
    cfg = Config()
    det, w = SpeechDetector(cfg), AudioWindow(cfg)
    for i in range(40):
        w.add(det.process(i * 0.5, noise(seed=i), SR))
    assert classify_interaction(w.summary()) == "silent"
    assert interaction_score(w.summary()) < 0.4
