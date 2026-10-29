# Multimodal Attendance & Engagement Agent

An offline, privacy-preserving classroom intelligence system. It watches a live or recorded
video feed and listens to the room audio to:

1. **Mark attendance automatically** from *sustained presence* (not single-frame detection) and write timestamped logs.
2. **Measure engagement** from head pose, movement, inactivity and audio activity, fused into a score per participant and per 30-second window.
3. **Advise the instructor** with explainable, context-aware teaching suggestions (pacing, interaction, classroom management).
4. **Show everything on a local dashboard** with CSV exports and per-participant analytics.

Everything runs on your machine. No frames, audio or face images are stored; there is no face *recognition*, only anonymous tracking.

---

## Quick start

```bash
cd attendance_agent
pip install -r requirements.txt

python app.py                    # open http://127.0.0.1:5000 and click "Run demo"
```

The **demo** runs a simulated 15-minute class (8 learners, late arrivals, someone leaving early,
people walking past, an occlusion, a long lecture that causes fatigue, a discussion, silent work
and an activity), so you can see every feature without a camera.

Command line, no browser:

```bash
python run_demo.py --simulate
python run_demo.py --video class.mp4 --audio class.wav     # real recording
```

### Using a real camera or recording

```bash
python scripts/download_models.py      # one-time, ~230 KB. The only step that needs internet.
```

This fetches the YuNet face model into `models/`. Without it, the system falls back to OpenCV's
bundled Haar cascades (coarser head pose). Then, in the dashboard, choose **Live camera** or
**Analyse recording** (video + optional `.wav` of the classroom audio).

Live microphone input needs the optional `sounddevice` package (and the PortAudio system library).
Without it the system still runs on video alone, and says so in the status bar.

---

## How it maps to the project brief

| Brief step | Where it lives |
|---|---|
| Define objectives (attendance, engagement, AI suggestions) | this README, `config.py` |
| Design architecture & workflow | [Architecture](#architecture) below, `core/pipeline.py` |
| Video input (live / recorded) | `core/sources.py` (`VideoSource`), `core/runner.py` |
| Audio input (microphone / file) | `core/sources.py` (`MicSource`, `wav_chunks`) |
| Face detection & tracking | `core/vision.py`, `core/tracker.py` |
| Sustained-presence logic | `core/attendance.py` |
| Timestamped attendance logs | `core/attendance.py`, `core/reports.py` (CSV), events log |
| Head pose, movement, inactivity | `core/vision.py` (yaw), `core/engagement.py` |
| Speech / interaction from audio | `core/audio.py` |
| Multimodal fusion | `core/fusion.py` |
| Engagement scores & session trends | `core/fusion.py` (`windows`, `trends`) |
| AI agent for teaching suggestions | `core/agent.py` |
| Dashboard for attendance, analytics, reports | `templates/`, `static/`, `app.py` |
| Offline processing & local storage | SQLite (`core/storage.py`), no network code paths |
| Test accuracy & performance | `tests/` (48 tests), `scripts/evaluate.py` |

---

## Architecture

The layered structure follows the diagram you shared (Frontend / Service / Utility / Storage), with the
cloud AI services replaced by local components so that everything works offline.

```
┌──────────────────────────── Frontend Layer ─────────────────────────────┐
│  HTML5 template · CSS · JavaScript dashboard (no CDN, SVG charts)        │
│             │ polls /api/live every 1.5 s                                │
│  Flask API  ▼  app.py: sessions, live view, exports, rename, delete      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────── Service Layer (core/, all local) ──────────────────────┐
│  SessionRunner (thread) ── VideoSource / MicSource / wav_chunks          │
│          │                                                               │
│  SessionPipeline                                                         │
│   video frame ─► FaceDetector (YuNet | Haar) ─► Tracker                  │
│                    │ boxes + yaw                  │ track ids            │
│                    ▼                              ▼                      │
│            AttendanceManager  ◄── sustained presence, re-association     │
│            EngagementAnalyzer ◄── head pose, movement, inactivity        │
│   audio chunk ─► SpeechDetector ─► AudioWindow (speech ratio, turns)     │
│                    └──────────────┬───────────────┘                      │
│                              EngagementFusion ─► window scores, trends   │
│                                   │                                      │
│                             TeachingAgent ─► suggestions + rationale     │
│                                   (optional LOCAL LLM coach note)        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌──────────────── Storage / Utility Layer ─────────────────────────────────┐
│  SQLite  data/attendance.db   (derived metrics only)                     │
│  CSV / JSON reports  data/reports/   · retention purge · delete session  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## How it works

### Attendance: sustained presence
A face must be **continuously in view for `presence_seconds` (default 8 s)** before it is logged.
Short detector drop-outs (up to `gap_tolerance_seconds`, 3 s) do not break the run. The log records
the time the person *actually arrived* (start of the confirming run), not when it was confirmed.

* People walking past, reflections and one-frame false positives never reach the threshold, so they never appear (counted as "false detections filtered").
* If the tracker loses someone (occlusion, turned away) and re-creates them under a new id, `AttendanceManager` merges the new track back into the same participant by position, so nobody is counted twice.
* A participant unseen for `absence_seconds` (20 s) is marked **left**; coming back logs **returned**.

### Engagement signals (per participant)
* **Head pose / attention**: yaw from YuNet's 5 landmarks (nose position relative to the eyes and mouth). 1.0 within 25° of forward, falling linearly to 0 at 60°.
* **Movement**: displacement over about 1 s, measured in face-widths per second. Using a 1 s baseline instead of frame-to-frame cancels detector jitter.
* **Inactivity**: no meaningful movement for 45 s. **Restless**: very high movement.

### Audio
Energy-based speech detection with an adaptive noise floor and hysteresis; no transcription, nothing stored.
Per window it reports speech ratio, turn count and longest silence, then labels the window *continuous lecture*, *discussion*, *mixed* or *silent*.
There is no speaker identification, so "continuous lecture" means "one voice is talking almost the whole time".

### Fusion and scoring
```
visual_i  = attention_i × (0.6 + 0.4 × motion_i)          # looking away zeroes it, whatever the body does
audio     = interaction_score(speech ratio, turns)         # silence is low, discussion is high
score_i   = 100 × (0.75 × visual_i + 0.25 × audio) / 1.0   # without audio: visual only
class     = mean over participants
```
Per-window scores give the timeline; the trend is a linear fit, plus detected **drops** (≥15 points), **peaks** and low stretches.
All weights and thresholds are in `config.py`.

### Teaching agent
An observe → reason → suggest loop with explainable rules (each suggestion carries a rationale and its evidence):

* **Live mode** acts on the latest *completed* window, with cool-downs so it doesn't nag: re-engage now, attention drop, break up long one-voice talk, restlessness, silent-and-still room.
* **Review mode** analyses the whole session: low stretches with times, biggest drops, long one-voice stretches that coincided with falling engagement, what worked (peaks), late arrivals, early leavers, learners who may need a private check-in.
* It never enforces anything. Optionally, set `LOCAL_LLM_URL` (e.g. an Ollama server on `localhost`) to have a local model condense findings into a short coach note; nothing leaves the machine.

---

## Configuration

Edit `config.py` (a `Config` dataclass). The ones you will most likely tune:

| Setting | Default | Meaning |
|---|---|---|
| `presence_seconds` | 8 | continuous presence needed to count someone as present |
| `gap_tolerance_seconds` | 3 | dropout that does not break a run |
| `absence_seconds` | 20 | unseen this long means "left" |
| `window_seconds` | 30 | engagement window length |
| `yaw_attentive_deg` / `yaw_away_deg` | 25 / 60 | head turn thresholds |
| `inactivity_seconds` | 45 | stillness that counts as inactive |
| `visual_weight` / `audio_weight` | 0.75 / 0.25 | fusion weights |
| `low_engagement` | 50 | below this a window is "low" |
| `retention_days` | 90 | older sessions are purged on startup |
| `face_backend` | auto | `auto`, `yunet` or `haar` |

---

## Privacy and ethics

* **Offline**: no network calls anywhere in the runtime. `scripts/download_models.py` is the only thing that touches the internet, and only once. The dashboard loads no external fonts, scripts or images (a test checks this).
* **No raw media stored**: frames are analysed in memory and dropped; audio is analysed in chunks and dropped. Only derived metrics go to SQLite.
* **No identification**: participants are anonymous ("Participant 3") unless the instructor renames them in the dashboard. There is no face recognition or embedding database.
* **Local only**: the server binds to `127.0.0.1` by default.
* **Your responsibility**: tell participants what is measured and get consent where required (school policy, GDPR, FERPA, etc.). Engagement is a *proxy* from posture and sound, so use it to improve teaching, not to grade or discipline individuals.

---

## API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | detector status and settings |
| `POST /api/sessions/simulate` | `{name, duration_s, seed, speed}` run the simulator (`speed: 0` = as fast as possible) |
| `POST /api/sessions/upload` | multipart `video` (+ optional `audio` .wav, `name`) |
| `POST /api/sessions/live` | `{name, camera, mic}` |
| `POST /api/sessions/stop` | stop and save the running session |
| `GET /api/live` | live snapshot: attendance, windows, trends, suggestions, events |
| `GET /api/sessions` · `GET /api/sessions/<id>` · `DELETE /api/sessions/<id>` | saved sessions |
| `PATCH /api/sessions/<id>/labels` | `{"Participant 1": "Asha"}` rename |
| `GET /api/sessions/<id>/export/{attendance,engagement,events}.csv` | exports |

---

## Testing and accuracy

```bash
python -m pytest tests -q          # 48 tests, about 6 s
python scripts/evaluate.py         # attendance accuracy over 10 seeds, about 1 min
```

Results from `scripts/evaluate.py` on the simulator (8 true participants + 2 walkers per run, 10 seeds):

| `presence_seconds` | precision | recall | ghost attendees / run |
|---|---|---|---|
| 0.2 (single-frame baseline) | 80% | 100% | 2.0 |
| 3 to 30 | 100% | 100% | 0.0 |

Check-in time error is 0 s (arrival is logged from the start of the confirming run), and the early
leaver was detected in 100% of runs. End-to-end tests also verify that the engagement score dips in the
fatigue stretch and recovers in discussion, and that the agent points to the right minutes.

**Read these numbers honestly**: they measure the *logic* (tracking, merging, thresholds, scoring,
agent) against synthetic ground truth. They do not measure how accurately a real detector finds faces
and estimates head pose in your classroom. For that, run a short recording where you know who was
present, compare with the attendance CSV, and tune `presence_seconds`, `yaw_*` and `still_motion_thresh`.

## Known limitations

* Head pose from five landmarks is approximate (about ±15° in good light); the Haar fallback only distinguishes frontal from profile. Low light, masks, strong backlight and faces smaller than `min_face_size` reduce accuracy.
* Looking down at notes or a laptop looks like "looking away". Silent reading looks like "inactive". The agent words these cases as questions to check, not conclusions.
* Audio has no speaker separation, so it cannot tell instructor from student speech.
* One camera view only; people outside it are not counted.
* One session runs at a time.

## Project structure

```
attendance_agent/
├── app.py                  Flask API + dashboard server
├── config.py               all settings
├── run_demo.py             CLI runner (simulate or video file)
├── core/
│   ├── vision.py           face detection + head-pose (YuNet / Haar)
│   ├── tracker.py          multi-face tracker
│   ├── attendance.py       sustained presence, check-in/out, re-association
│   ├── engagement.py       attention, movement, inactivity
│   ├── audio.py            speech detection, interaction stats
│   ├── fusion.py           multimodal scores, windows, trends
│   ├── agent.py            teaching suggestions (live + review)
│   ├── pipeline.py         ties it together for one session
│   ├── runner.py           background session runner
│   ├── sources.py          video / microphone / wav sources
│   ├── simulator.py        synthetic classroom (ground truth for tests)
│   ├── storage.py          SQLite
│   └── reports.py          CSV exports
├── templates/ static/      dashboard (no external dependencies)
├── scripts/                download_models.py, evaluate.py
├── tests/                  48 tests
├── models/                 face model goes here
└── data/                   SQLite DB, reports (created on first run)
```
