"""Flask API + dashboard for the Multimodal Attendance & Engagement Agent.

Everything runs on this machine. Bind address defaults to 127.0.0.1 so the dashboard is only
reachable locally.

    python app.py            -> http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request

from config import DB_PATH, UPLOADS_DIR, Config, ensure_dirs
from core import reports
from core.runner import SessionRunner
from core.storage import Storage

_detector = None
_detector_error: Optional[str] = None


def get_detector(cfg: Config):
    """Create the face detector lazily so the app (and simulator) start without a camera stack."""
    global _detector, _detector_error
    if _detector is None and _detector_error is None:
        try:
            from core.vision import FaceDetector
            _detector = FaceDetector(cfg)
        except Exception as exc:
            _detector_error = str(exc)
    if _detector is None:
        raise RuntimeError(_detector_error)
    return _detector


def create_app(cfg: Optional[Config] = None, db_path=DB_PATH) -> Flask:
    ensure_dirs()
    cfg = cfg or Config()
    storage = Storage(db_path)
    storage.purge_older_than(cfg.retention_days)
    runner = SessionRunner(cfg, storage)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3  # 2 GB recordings
    app.extensions["runner"] = runner
    app.extensions["storage"] = storage

    def err(msg: str, code: int = 400):
        return jsonify({"error": msg}), code

    # ---- pages ----------------------------------------------------------
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        backend, detector_ok, note = None, False, None
        try:
            backend, detector_ok = get_detector(cfg).backend, True
        except Exception as exc:
            note = str(exc)
        return jsonify({"offline": True, "detector": backend, "detector_ok": detector_ok,
                        "detector_note": note, "local_llm": bool(cfg.local_llm_url),
                        "presence_seconds": cfg.presence_seconds, "window_seconds": cfg.window_seconds})

    # ---- start sessions -------------------------------------------------
    @app.post("/api/sessions/simulate")
    def start_sim():
        body = request.get_json(silent=True) or {}
        try:
            runner.start_simulation(body.get("name", "Simulated class"),
                                    float(body.get("duration_s", 900)), int(body.get("seed", 7)),
                                    float(body.get("speed", 60)))
        except RuntimeError as exc:
            return err(str(exc), 409)
        return jsonify({"status": "started", "id": runner.pipeline.id})

    @app.post("/api/sessions/upload")
    def start_upload():
        video = request.files.get("video")
        if video is None or not video.filename:
            return err("Attach a video file in the 'video' field.")
        try:
            detector = get_detector(cfg)
        except Exception as exc:
            return err(f"Face detector unavailable: {exc}", 500)
        ext = os.path.splitext(video.filename)[1].lower()
        vpath = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext or '.mp4'}"
        video.save(vpath)
        apath = None
        audio = request.files.get("audio")
        if audio is not None and audio.filename:
            apath = UPLOADS_DIR / f"{uuid.uuid4().hex}.wav"
            audio.save(apath)
        try:
            runner.start_file(request.form.get("name", "Uploaded recording"), str(vpath),
                              str(apath) if apath else None, detector)
        except RuntimeError as exc:
            return err(str(exc), 409)
        return jsonify({"status": "started", "id": runner.pipeline.id})

    @app.post("/api/sessions/live")
    def start_live():
        body = request.get_json(silent=True) or {}
        try:
            detector = get_detector(cfg)
        except Exception as exc:
            return err(f"Face detector unavailable: {exc}", 500)
        try:
            runner.start_live(body.get("name", "Live class"), int(body.get("camera", 0)),
                              bool(body.get("mic", True)), detector)
        except RuntimeError as exc:
            return err(str(exc), 409)
        return jsonify({"status": "started", "id": runner.pipeline.id})

    @app.post("/api/sessions/stop")
    def stop():
        if not runner.busy():
            return err("No session is running.", 409)
        runner.stop()
        return jsonify({"status": "stopping"})

    # ---- live view ------------------------------------------------------
    @app.get("/api/live")
    def live():
        if runner.pipeline is None:
            return jsonify({"status": "idle"})
        snap = runner.pipeline.snapshot(advise=runner.status == "running")
        snap.update({"status": runner.status, "error": runner.error, "progress": runner.progress,
                     "saved_id": runner.saved_id})
        return jsonify(snap)

    # ---- saved sessions -------------------------------------------------
    @app.get("/api/sessions")
    def sessions():
        return jsonify(storage.list_sessions())

    @app.get("/api/sessions/<sid>")
    def session(sid):
        rep = storage.get_session(sid)
        return jsonify(rep) if rep else err("Session not found.", 404)

    @app.delete("/api/sessions/<sid>")
    def delete(sid):
        if runner.busy() and runner.pipeline is not None and runner.pipeline.id == sid:
            return err("Stop the session before deleting it.", 409)
        return jsonify({"deleted": True}) if storage.delete_session(sid) else err("Session not found.", 404)

    @app.patch("/api/sessions/<sid>/labels")
    def rename(sid):
        rep = storage.get_session(sid)
        if rep is None:
            return err("Session not found.", 404)
        mapping = {k: v.strip() for k, v in (request.get_json(silent=True) or {}).items()
                   if isinstance(v, str) and v.strip()}
        if not mapping:
            return err('Send a JSON object like {"Participant 1": "Asha"}.')
        existing = {a["label"] for a in rep["attendance"]}
        unknown = [k for k in mapping if k not in existing]
        if unknown:
            return err(f"Unknown participant label(s): {', '.join(unknown)}", 404)
        meta = {k: rep.pop(k) for k in ("id", "name", "started_at", "duration_s", "source")}
        rep = reports.apply_labels(rep, mapping)
        storage.save_session(meta["id"], meta["name"], datetime.fromisoformat(meta["started_at"]),
                             meta["duration_s"], meta["source"], rep)
        return jsonify(storage.get_session(sid))

    # ---- exports --------------------------------------------------------
    def _export(sid, fn, name, mime):
        rep = storage.get_session(sid)
        if rep is None:
            return err("Session not found.", 404)
        return Response(fn(rep), mimetype=mime,
                        headers={"Content-Disposition": f"attachment; filename={name}-{sid}.csv"})

    @app.get("/api/sessions/<sid>/export/attendance.csv")
    def export_attendance(sid):
        return _export(sid, reports.attendance_csv, "attendance", "text/csv")

    @app.get("/api/sessions/<sid>/export/engagement.csv")
    def export_engagement(sid):
        return _export(sid, reports.engagement_csv, "engagement", "text/csv")

    @app.get("/api/sessions/<sid>/export/events.csv")
    def export_events(sid):
        return _export(sid, reports.events_csv, "events", "text/csv")

    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    create_app().run(host=host, port=port, threaded=True)
