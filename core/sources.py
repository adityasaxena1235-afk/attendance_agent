"""Video and audio sources: recorded files, webcam/microphone."""
from __future__ import annotations

import queue
import wave
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np


class VideoSource:
    """Yields (t_seconds, frame) at roughly `target_fps` from a file or camera index."""

    def __init__(self, source: Union[str, int], target_fps: float = 5.0, max_width: int = 960):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        self.is_live = isinstance(source, int)
        self.target_fps = target_fps
        self.max_width = max_width
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def frames(self) -> Iterator[Tuple[float, np.ndarray]]:
        import time
        step = max(1, int(round(self._fps / self.target_fps))) if not self.is_live else 1
        t0 = time.time()
        i = 0
        last_emit = -1e9
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.is_live:
                t = time.time() - t0
                if t - last_emit < 1.0 / self.target_fps:
                    continue
            else:
                if i % step:
                    i += 1
                    continue
                pos = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                t = pos / 1000.0 if pos and pos > 0 else i / self._fps
            i += 1
            last_emit = t
            h, w = frame.shape[:2]
            if w > self.max_width:
                frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)))
            yield t, frame

    def release(self) -> None:
        self.cap.release()


def wav_chunks(path: str, chunk_seconds: float = 0.5) -> Iterator[Tuple[float, np.ndarray, int]]:
    """Yield (t, mono float32 samples, sample_rate) from a PCM .wav file."""
    with wave.open(path, "rb") as wf:
        sr, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        n = int(sr * chunk_seconds)
        t = 0.0
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        while True:
            raw = wf.readframes(n)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if width == 1:
                x = (x - 128.0) / 128.0
            else:
                x = x / float(np.iinfo(dtype).max)
            if ch > 1:
                x = x.reshape(-1, ch).mean(axis=1)
            yield t, x, sr
            t += len(x) / sr


class MicSource:
    """Live microphone via the optional `sounddevice` package."""

    def __init__(self, sr: int = 16000, chunk_seconds: float = 0.5, device: Optional[int] = None):
        try:
            import sounddevice as sd
        except Exception as exc:  # PortAudio missing, package missing...
            raise RuntimeError(f"Microphone unavailable ({exc}). Install `sounddevice` "
                               "and PortAudio, or analyse a .wav file instead.") from exc
        self._sd = sd
        self.sr = sr
        self.chunk = int(sr * chunk_seconds)
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = sd.InputStream(samplerate=sr, channels=1, blocksize=self.chunk,
                                     device=device, callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        self.q.put(indata[:, 0].copy())

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def chunks(self, t0: float, stop_event) -> Iterator[Tuple[float, np.ndarray, int]]:
        import time
        while not stop_event.is_set():
            try:
                x = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            yield time.time() - t0, x, self.sr
