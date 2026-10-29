"""One-time download of the YuNet face-detection model (about 230 KB).

This is the ONLY step in the project that needs internet. After it has run once,
the system works completely offline. You can also copy the file in by hand:

    models/face_detection_yunet.onnx
"""
import sys
import urllib.request
from pathlib import Path

URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
       "face_detection_yunet/face_detection_yunet_2023mar.onnx")
DEST = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet.onnx"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 100_000:
        print(f"Already present: {DEST}")
        return 0
    print(f"Downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:
        print(f"Download failed: {exc}\nDownload the file manually and save it as {DEST}")
        return 1
    if DEST.stat().st_size < 100_000:   # an error page or git-lfs pointer, not a model
        DEST.unlink()
        print("Downloaded file is too small to be the model. Download it manually from the OpenCV Zoo.")
        return 1
    print(f"Saved {DEST} ({DEST.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
