"""Command-line runner (no browser needed).

    python run_demo.py --simulate                 # synthetic 15-minute class
    python run_demo.py --video class.mp4 [--audio class.wav]
"""
import argparse
import json
import sys
from pathlib import Path

from config import DB_PATH, REPORTS_DIR, Config, ensure_dirs
from core import reports
from core.pipeline import SessionPipeline
from core.storage import Storage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="run the built-in classroom simulator")
    ap.add_argument("--video", help="path to a recorded video")
    ap.add_argument("--audio", help="optional PCM .wav with the classroom audio")
    ap.add_argument("--name", default="CLI session")
    ap.add_argument("--duration", type=float, default=900.0, help="simulated seconds")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not args.simulate and not args.video:
        ap.error("choose --simulate or --video")

    ensure_dirs()
    cfg = Config()

    if args.simulate:
        from core.simulator import replay, simulate
        pl = SessionPipeline(cfg, args.name, "simulator")
        replay(pl, simulate(seed=args.seed, duration_s=args.duration, fps=cfg.target_fps))
    else:
        from core.sources import VideoSource, wav_chunks
        from core.vision import FaceDetector
        pl = SessionPipeline(cfg, args.name, "video file", detector=FaceDetector(cfg))
        print(f"face detector: {pl.detector.backend}")
        audio = wav_chunks(args.audio) if args.audio else iter(())
        pending = next(audio, None)
        vs = VideoSource(args.video, cfg.target_fps)
        for t, frame in vs.frames():
            while pending is not None and pending[0] <= t:
                pl.process_audio(*pending)
                pending = next(audio, None)
            pl.process_frame(t, frame)
        vs.release()

    report = pl.finalize()
    Storage(DB_PATH).save_session(pl.id, pl.name, pl.started_at, report["duration_s"], pl.source, report)
    out = REPORTS_DIR / pl.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "attendance.csv").write_text(reports.attendance_csv(report))
    (out / "engagement.csv").write_text(reports.engagement_csv(report))
    (out / "events.csv").write_text(reports.events_csv(report))

    print(f"\nSession {pl.id}: {report['duration_s']:.0f}s, {len(report['attendance'])} attendees, "
          f"{report['filtered_transients']} brief detections filtered, mean engagement "
          f"{report['trends']['mean']} ({report['trends']['direction']})")
    for a in report["attendance"]:
        print(f"  {a['label']:<14} {a['status']:<8} check-in {a['check_in'][11:19]}  presence {a['presence_pct']:>5}%")
    print("\nSuggestions:")
    for s in report["suggestions"]:
        print(f"  [{s['priority']}] {s['title']}")
    print(f"\nReports written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
