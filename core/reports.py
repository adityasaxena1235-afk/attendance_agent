"""CSV / JSON exports built from a finished session report."""
from __future__ import annotations

import csv
import io
from typing import List


def _csv(rows: List[list], header: List[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def attendance_csv(report: dict) -> str:
    eng = {p["id"]: p for p in report.get("participants", [])}
    rows = []
    for a in report.get("attendance", []):
        e = eng.get(a["id"], {})
        rows.append([a["label"], a["status"], a["check_in"], a.get("left_at") or "",
                     a["present_seconds"], a["presence_pct"], a["visits"],
                     e.get("engagement", ""), e.get("attention_pct", ""), e.get("inactive_s", "")])
    return _csv(rows, ["participant", "status", "check_in", "left_at", "present_seconds",
                       "presence_pct", "visits", "avg_engagement", "attention_pct", "inactive_seconds"])


def engagement_csv(report: dict) -> str:
    rows = []
    for w in report.get("windows", []):
        rows.append([w["start_s"], w["end_s"], w["score"], w["visual_score"], w["audio_score"],
                     w["participants"], w["attentive_pct"], w["interaction"], w["speech_ratio"],
                     w["turns"], w["inactive_count"], w["restless_count"]])
    return _csv(rows, ["start_s", "end_s", "engagement", "visual", "audio", "participants",
                       "attentive_pct", "interaction", "speech_ratio", "turns",
                       "inactive_count", "restless_count"])


def events_csv(report: dict) -> str:
    rows = [[e["time"], e["kind"], e["participant"], e["detail"]] for e in report.get("events", [])]
    return _csv(rows, ["time", "event", "participant", "detail"])


def apply_labels(report: dict, mapping: dict) -> dict:
    """Rename participants everywhere in a report. mapping: {old_label: new_label}."""
    for section in ("attendance", "participants"):
        for row in report.get(section, []):
            if row["label"] in mapping:
                row["label"] = mapping[row["label"]]
    for e in report.get("events", []):
        if e["participant"] in mapping:
            e["participant"] = mapping[e["participant"]]
    for s in report.get("suggestions", []):
        for old, new in mapping.items():
            s["rationale"] = s["rationale"].replace(old, new)
    return report
