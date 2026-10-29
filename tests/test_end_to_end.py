import csv
import io
import wave
from pathlib import Path

import numpy as np
import pytest

from config import Config
from core.pipeline import SessionPipeline
from core.simulator import default_profiles, replay, simulate


@pytest.fixture(scope="module")
def report():
    pl = SessionPipeline(Config(), "e2e", "simulator")
    replay(pl, simulate(seed=7))
    return pl.finalize()


def test_attendance_matches_ground_truth(report):
    truth = default_profiles()
    assert len(report["attendance"]) == len(truth) == 8          # no ghosts, nobody missed
    assert report["filtered_transients"] == 2                    # the two walkers were rejected


def test_checkin_times_match_real_arrivals(report):
    arrivals = sorted(p.arrive for p in default_profiles())
    got = sorted(a["check_in_offset_s"] for a in report["attendance"])
    assert max(abs(g - a) for g, a in zip(got, arrivals)) <= 1.0


def test_early_leaver_and_occlusion_are_handled(report):
    left_events = [e for e in report["events"] if e["kind"] == "LEFT"]
    assert len(left_events) == 1 and 695 <= left_events[0]["offset_s"] <= 705   # p8 leaves at 700 s
    assert sum(1 for a in report["attendance"] if a["status"] == "left") == 1
    assert sum(1 for a in report["attendance"] if a["visits"] == 2) == 1   # occluded 6 s, merged


def test_engagement_dips_in_fatigue_and_recovers_in_discussion(report):
    by = {w["start_s"]: w["score"] for w in report["windows"]}
    fatigue = np.mean([by[s] for s in (450, 480, 510)])
    discussion = np.mean([by[s] for s in (270, 300, 330)])
    assert discussion - fatigue > 25


def test_agent_localises_the_problem(report):
    titles = " | ".join(s["title"] for s in report["suggestions"])
    assert "06:00-09:00" in titles                              # the long lecture
    assert "What worked" in titles


def test_report_declares_privacy_posture(report):
    assert report["privacy"] == {"offline": True, "frames_stored": False, "audio_stored": False,
                                 "identity_recognition": False}


def test_simulation_is_deterministic():
    a, b = SessionPipeline(Config(), "a", "s"), SessionPipeline(Config(), "b", "s")
    replay(a, simulate(seed=3, duration_s=300))
    replay(b, simulate(seed=3, duration_s=300))
    assert [w["score"] for w in a.finalize()["windows"]] == [w["score"] for w in b.finalize()["windows"]]


def test_works_without_audio():
    pl = SessionPipeline(Config(), "noaudio", "s")
    replay(pl, simulate(seed=2, duration_s=300, with_audio=False))
    r = pl.finalize()
    assert r["windows"] and all(w["audio_score"] is None for w in r["windows"])
    assert len(r["attendance"]) >= 4


# ---------------- file ingestion (synthetic video + wav, fake detector) ----------------
class FakeDetector:
    backend = "fake"

    def detect(self, frame):
        from core.models import Detection
        return [Detection((60.0, 40.0, 50.0, 60.0), 1.0, 0.0)]


def test_video_and_wav_files_flow_through_the_pipeline(tmp_path):
    import cv2
    from core.sources import VideoSource, wav_chunks

    vpath, apath = tmp_path / "c.avi", tmp_path / "c.wav"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    for _ in range(300):                                          # 30 s at 10 fps
        vw.write(np.zeros((120, 160, 3), np.uint8))
    vw.release()
    t = np.arange(16000 * 30) / 16000
    pcm = (0.1 * np.sin(2 * np.pi * 180 * t) * 32767).astype(np.int16)
    with wave.open(str(apath), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())

    pl = SessionPipeline(Config(), "file", "video", detector=FakeDetector())
    audio = wav_chunks(str(apath))
    pending = next(audio, None)
    vs = VideoSource(str(vpath), 5.0)
    for tt, frame in vs.frames():
        while pending is not None and pending[0] <= tt:
            pl.process_audio(*pending)
            pending = next(audio, None)
        pl.process_frame(tt, frame)
    vs.release()
    r = pl.finalize()
    assert len(r["attendance"]) == 1 and r["attendance"][0]["check_in_offset_s"] == 0.0
    assert r["windows"][0]["audio_score"] is not None and r["duration_s"] > 25
