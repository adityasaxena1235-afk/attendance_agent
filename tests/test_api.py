import pytest

from app import create_app
from config import Config


@pytest.fixture()
def client(tmp_path):
    app = create_app(Config(), db_path=tmp_path / "t.db")
    return app.test_client(), app.extensions["runner"]


def run_sim(client, runner, duration=240):
    r = client.post("/api/sessions/simulate", json={"name": "T", "duration_s": duration, "speed": 0})
    assert r.status_code == 200
    runner.join(60)
    assert runner.status == "done", runner.error
    return runner.saved_id


def test_dashboard_page_and_assets_are_served(client):
    c, _ = client
    page = c.get("/")
    assert page.status_code == 200 and b"Attendance" in page.data
    assert c.get("/static/app.js").status_code == 200 and c.get("/static/style.css").status_code == 200
    assert b"http://" not in page.data and b"https://" not in page.data        # nothing fetched from the web


def test_health_reports_detector_state_without_crashing(client):
    c, _ = client
    j = c.get("/api/health").get_json()
    assert j["offline"] is True and "detector_ok" in j


def test_full_session_lifecycle_and_exports(client):
    c, runner = client
    sid = run_sim(c, runner)
    live = c.get("/api/live").get_json()
    assert live["status"] == "done" and live["attendees"] >= 4
    assert any(s["id"] == sid for s in c.get("/api/sessions").get_json())
    rep = c.get(f"/api/sessions/{sid}").get_json()
    assert rep["windows"] and rep["attendance"] and rep["privacy"]["offline"]
    for name in ("attendance", "engagement", "events"):
        r = c.get(f"/api/sessions/{sid}/export/{name}.csv")
        assert r.status_code == 200 and r.mimetype == "text/csv" and r.data.count(b"\n") >= 2


def test_rename_participant_updates_everywhere(client):
    c, runner = client
    sid = run_sim(c, runner)
    r = c.patch(f"/api/sessions/{sid}/labels", json={"Participant 1": "Asha"})
    assert r.status_code == 200
    rep = r.get_json()
    assert any(a["label"] == "Asha" for a in rep["attendance"])
    assert any(e["participant"] == "Asha" for e in rep["events"])
    assert b"Asha" in c.get(f"/api/sessions/{sid}/export/attendance.csv").data
    assert c.patch(f"/api/sessions/{sid}/labels", json={"Nobody": "x"}).status_code == 404
    assert c.patch(f"/api/sessions/{sid}/labels", json={}).status_code == 400


def test_delete_removes_session(client):
    c, runner = client
    sid = run_sim(c, runner)
    assert c.delete(f"/api/sessions/{sid}").status_code == 200
    assert c.get(f"/api/sessions/{sid}").status_code == 404
    assert c.delete(f"/api/sessions/{sid}").status_code == 404


def test_cannot_start_two_sessions_and_stop_works(client):
    c, runner = client
    assert c.post("/api/sessions/simulate", json={"duration_s": 900, "speed": 30}).status_code == 200
    assert c.post("/api/sessions/simulate", json={"duration_s": 900, "speed": 30}).status_code == 409
    assert c.post("/api/sessions/stop").status_code == 200
    runner.join(30)
    assert runner.status == "done"                 # stopping early still saves what was captured
    assert c.post("/api/sessions/stop").status_code == 409


def test_upload_without_video_is_rejected(client):
    c, _ = client
    assert c.post("/api/sessions/upload", data={}).status_code in (400, 500)


def test_unknown_session_is_404(client):
    c, _ = client
    assert c.get("/api/sessions/nope").status_code == 404
    assert c.get("/api/sessions/nope/export/attendance.csv").status_code == 404
