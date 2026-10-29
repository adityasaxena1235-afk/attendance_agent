"""Runs one session at a time in a background thread (simulated, uploaded or live)."""
from __future__ import annotations

import threading
import time
import traceback
from typing import Optional

from config import Config
from .pipeline import SessionPipeline
from .simulator import simulate
from .sources import MicSource, VideoSource, wav_chunks
from .storage import Storage


class SessionRunner:
    def __init__(self, cfg: Config, storage: Storage):
        self.cfg = cfg
        self.storage = storage
        self.pipeline: Optional[SessionPipeline] = None
        self.status = "idle"       # idle | running | saving | done | error
        self.error: Optional[str] = None
        self.progress: Optional[float] = None   # 0..1 when total length is known
        self.saved_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---- lifecycle ------------------------------------------------------
    def busy(self) -> bool:
        return self.status in ("running", "saving")

    def _start(self, pipeline: SessionPipeline, target) -> None:
        with self._lock:
            if self.busy():
                raise RuntimeError("A session is already running. Stop it first.")
            self.pipeline = pipeline
            self.status, self.error, self.progress, self.saved_id = "running", None, 0.0, None
            self._stop.clear()
            self._thread = threading.Thread(target=self._guard, args=(target,), daemon=True)
            self._thread.start()

    def _guard(self, target) -> None:
        try:
            target()
            self._finish()
        except Exception as exc:  # surface problems to the UI instead of dying silently
            traceback.print_exc()
            self.error, self.status = str(exc), "error"

    def _finish(self) -> None:
        self.status = "saving"
        p = self.pipeline
        report = p.finalize()
        self.storage.save_session(p.id, p.name, p.started_at, report["duration_s"], p.source, report)
        self.saved_id, self.progress, self.status = p.id, 1.0, "done"

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---- simulated -----------------------------------------------------
    def start_simulation(self, name: str, duration_s: float, seed: int, speed: float) -> None:
        pl = SessionPipeline(self.cfg, name, "simulator")

        def run():
            wall0 = time.time()
            for kind, t, payload in simulate(seed=seed, duration_s=duration_s, fps=self.cfg.target_fps):
                if self._stop.is_set():
                    break
                if speed > 0:
                    delay = t / speed - (time.time() - wall0)
                    if delay > 0:
                        time.sleep(min(delay, 0.5))
                if kind == "video":
                    pl.process_detections(t, payload)
                    self.progress = t / duration_s
                else:
                    pl.process_audio(t, *payload)
        self._start(pl, run)

    # ---- uploaded recording -------------------------------------------
    def start_file(self, name: str, video_path: str, audio_path: Optional[str], detector) -> None:
        pl = SessionPipeline(self.cfg, name, "video file", detector=detector)

        def run():
            vs = VideoSource(video_path, self.cfg.target_fps)
            total_frames = vs.cap.get(7) or 0   # CAP_PROP_FRAME_COUNT
            fps = vs._fps
            audio_iter = wav_chunks(audio_path) if audio_path else iter(())
            pending = next(audio_iter, None)
            try:
                for t, frame in vs.frames():
                    if self._stop.is_set():
                        break
                    while pending is not None and pending[0] <= t:
                        pl.process_audio(pending[0], pending[1], pending[2])
                        pending = next(audio_iter, None)
                    pl.process_frame(t, frame)
                    if total_frames:
                        self.progress = min(0.99, t * fps / total_frames)
                while pending is not None:          # audio longer than the video
                    pl.process_audio(pending[0], pending[1], pending[2])
                    pending = next(audio_iter, None)
            finally:
                vs.release()
        self._start(pl, run)

    # ---- live camera + mic --------------------------------------------
    def start_live(self, name: str, camera: int, use_mic: bool, detector) -> None:
        pl = SessionPipeline(self.cfg, name, f"camera {camera}", detector=detector)

        def run():
            vs = VideoSource(camera, self.cfg.target_fps)
            mic = None
            mic_thread = None
            t0 = time.time()
            if use_mic:
                try:
                    mic = MicSource()
                    mic.start()
                    mic_thread = threading.Thread(
                        target=lambda: [pl.process_audio(*c) for c in mic.chunks(t0, self._stop)],
                        daemon=True)
                    mic_thread.start()
                except RuntimeError as exc:
                    pl.audio_note = str(exc)   # continue with video only
            try:
                for t, frame in vs.frames():
                    if self._stop.is_set():
                        break
                    pl.process_frame(t, frame)
            finally:
                self._stop.set()
                vs.release()
                if mic:
                    mic.stop()
                if mic_thread:
                    mic_thread.join(2)
            self.progress = None
        self._start(pl, run)
