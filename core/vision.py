"""Face detection + head-pose estimation from a video frame (fully local).

Backends, tried in this order when `face_backend == "auto"`:

1. YuNet (OpenCV FaceDetectorYN). Fast and accurate, and its 5 facial landmarks give a
   good yaw estimate. Needs `models/face_detection_yunet.onnx` (run
   `python scripts/download_models.py` once).
2. Haar cascades bundled with opencv-python (frontal + profile). Works out of the box;
   yaw is coarse (frontal ~ 0 deg, profile ~ 60 deg).

Frames are analysed in memory and discarded; nothing is written to disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import MODELS_DIR, Config
from .models import Detection

YUNET_PATH = MODELS_DIR / "face_detection_yunet.onnx"


def yaw_from_landmarks(re, le, nose, rm, lm) -> float:
    """Estimate head yaw (deg) from YuNet's 5 landmarks.

    The nose tip sits midway between the eyes when the face looks at the camera and
    slides toward one eye as the head turns. The offset relative to the inter-eye
    distance is almost linear in yaw for +-60 deg, so we scale it.
    """
    eye_mid_x = (re[0] + le[0]) / 2.0
    eye_dist = max(abs(le[0] - re[0]), 1e-6)
    mouth_mid_x = (rm[0] + lm[0]) / 2.0
    offset = ((nose[0] - eye_mid_x) + (nose[0] - mouth_mid_x)) / 2.0 / eye_dist
    # image-left landmark is the person's right eye in YuNet's ordering
    return float(np.clip(offset * 90.0, -90.0, 90.0))


class FaceDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = None
        self._yunet = None
        self._frontal = None
        self._profile = None
        self._init_backend(cfg.face_backend)

    # ------------------------------------------------------------------
    def _init_backend(self, wanted: str) -> None:
        order = ["yunet", "haar"] if wanted == "auto" else [wanted]
        errors = []
        for name in order:
            try:
                if name == "yunet":
                    self._init_yunet()
                elif name == "haar":
                    self._init_haar()
                else:
                    raise ValueError(f"unknown backend {name}")
                self.backend = name
                return
            except Exception as exc:  # try the next backend
                errors.append(f"{name}: {exc}")
        raise RuntimeError("No face detector available. " + " | ".join(errors) +
                           ". Run `python scripts/download_models.py`.")

    def _init_yunet(self) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV build has no FaceDetectorYN")
        if not Path(YUNET_PATH).exists():
            raise FileNotFoundError(f"{YUNET_PATH} missing")
        self._yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)

    def _init_haar(self) -> None:
        base = Path(getattr(cv2.data, "haarcascades", "")) if hasattr(cv2, "data") else Path("")
        f, p = base / "haarcascade_frontalface_default.xml", base / "haarcascade_profileface.xml"
        if not f.exists():
            raise FileNotFoundError("Haar cascade files not bundled with this OpenCV build")
        self._frontal = cv2.CascadeClassifier(str(f))
        self._profile = cv2.CascadeClassifier(str(p)) if p.exists() else None

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.backend == "yunet":
            return self._detect_yunet(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_yunet(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(frame)
        out: List[Detection] = []
        if faces is None:
            return out
        for f in faces:
            x, y, fw, fh = [float(v) for v in f[:4]]
            if min(fw, fh) < self.cfg.min_face_size:
                continue
            re, le, nose, rm, lm = [(f[4 + 2 * i], f[5 + 2 * i]) for i in range(5)]
            out.append(Detection((x, y, fw, fh), float(f[14]), yaw_from_landmarks(re, le, nose, rm, lm)))
        return out

    def _detect_haar(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        ms = (self.cfg.min_face_size, self.cfg.min_face_size)
        dets: List[Detection] = []
        for (x, y, w, h) in self._frontal.detectMultiScale(gray, 1.1, 5, minSize=ms):
            dets.append(Detection((float(x), float(y), float(w), float(h)), 1.0, 0.0))
        if self._profile is not None:
            width = gray.shape[1]
            profiles = [(x, y, w, h) for (x, y, w, h) in self._profile.detectMultiScale(gray, 1.1, 5, minSize=ms)]
            flipped = cv2.flip(gray, 1)
            profiles += [(width - x - w, y, w, h)
                         for (x, y, w, h) in self._profile.detectMultiScale(flipped, 1.1, 5, minSize=ms)]
            for (x, y, w, h) in profiles:
                box = (float(x), float(y), float(w), float(h))
                if all(_iou(box, d.bbox) < 0.3 for d in dets):  # not already a frontal hit
                    dets.append(Detection(box, 0.8, 60.0))
        return dets


def _iou(a, b) -> float:
    from .tracker import iou
    return iou(a, b)
