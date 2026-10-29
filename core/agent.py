"""Teaching-suggestion agent.

The agent follows an observe -> reason -> suggest loop over the session analytics.
Reasoning is rule-based, so it is deterministic, explainable and fully offline.
Every suggestion carries its rationale and the evidence behind it.

Two modes:
* live   - looks at the most recent window and fires suggestions the instructor can act on now
            (each rule has a cool-down so it does not nag).
* review - looks over the whole finished session and reports what happened, where, and what to try.

Optionally, if LOCAL_LLM_URL points at a model server running on this machine (for example
Ollama), the agent can ask it to condense the findings into a short coaching note. That call
never leaves localhost and is skipped on any error.

The agent only advises. It never enforces anything.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Callable, Dict, List, Optional

from config import Config
from .models import Suggestion

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def _runs(windows: List[dict], pred: Callable[[dict], bool], step: float) -> List[List[dict]]:
    """Maximal runs of consecutive (time-contiguous) windows satisfying `pred`."""
    runs, cur = [], []
    for w in windows:
        if pred(w) and (not cur or abs(w["start_s"] - cur[-1]["start_s"] - step) < 1e-6):
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = [w] if pred(w) else []
    if cur:
        runs.append(cur)
    return runs


class TeachingAgent:
    def __init__(self, cfg: Config, cooldown_s: float = 120.0):
        self.cfg = cfg
        self.cooldown = cooldown_s
        self._fired: Dict[str, float] = {}

    def _ok(self, key: str, now: float, live: bool) -> bool:
        if not live:
            return True
        last = self._fired.get(key)
        if last is not None and now - last < self.cooldown:
            return False
        self._fired[key] = now
        return True

    # ------------------------------------------------------------------
    def advise(self, now: float, windows: List[dict], trends: dict,
               participants: List[dict], attendance: List[dict],
               live: bool = False) -> List[Suggestion]:
        if not windows:
            return []
        out: List[Suggestion] = []

        def add(key, priority, title, why, evidence, t=None):
            if self._ok(key, now, live):
                out.append(Suggestion(t=now if t is None else t, priority=priority, title=title,
                                      rationale=why, evidence=evidence))

        if live:
            self._live_rules(add, windows, trends)
        else:
            self._review_rules(add, windows, trends)
        self._cohort_rules(add, now, windows, trends, participants, attendance, live)

        if not out and not live and windows[-1]["score"] >= 65:
            add("steady", "low", "Engagement is healthy",
                f"Engagement is {windows[-1]['score']:.0f}/100 with a {trends['direction']} trend. "
                "No action needed; keep the current pacing.", {"score": windows[-1]["score"]})
        out.sort(key=lambda s: (PRIORITY_ORDER[s.priority], s.t))
        return out

    # ---- live: what to do right now -----------------------------------
    def _live_rules(self, add, windows, trends) -> None:
        cfg = self.cfg
        last = windows[-1]
        span = f"{mmss(last['start_s'])}-{mmss(last['end_s'])}"

        if last["score"] < cfg.low_engagement:
            add("low_now", "high", "Re-engage the class now",
                f"Engagement in the latest window ({span}) is {last['score']:.0f}/100, below the "
                f"{cfg.low_engagement:.0f} threshold. A short interactive task (quick poll, "
                "think-pair-share, or a 2-minute stretch) can reset attention.",
                {"score": last["score"], "attentive_pct": last["attentive_pct"]}, last["start_s"])

        recent = [d for d in trends.get("drops", [])
                  if d["at_s"] >= last["start_s"] - cfg.window_seconds and d["to"] < 75.0]
        if recent:
            d = recent[-1]
            add(f"drop_{int(d['at_s'])}", "high", f"Attention dropped at {mmss(d['at_s'])}",
                f"Engagement fell {abs(d['delta']):.0f} points ({d['from']:.0f} to {d['to']:.0f}). "
                "Check what changed: a dense slide, a topic switch, or a long explanation. "
                "Splitting it into smaller chunks or adding an example may help.", d, d["at_s"])

        lecture = _runs(windows, lambda w: w["interaction"] == "continuous lecture", cfg.window_seconds)
        if lecture and lecture[-1][-1] is last and len(lecture[-1]) >= 3 and \
                (last["score"] < 65 or lecture[-1][0]["score"] - last["score"] >= 10):
            n = len(lecture[-1])
            add("lecture_fatigue", "medium", "Break up the continuous talk",
                f"One voice has been talking for about {n * cfg.window_seconds / 60:.1f} minutes while "
                "engagement slips. Ask a question, invite a student to explain, or switch to a short activity.",
                {"lecture_windows": n, "score_now": last["score"]})

        self._state_rules(add, last, suffix="")

        if len(windows) >= 6 and trends["slope_per_min"] < -1.5:
            add("declining", "medium", "Engagement is trending down",
                f"Average engagement is falling about {abs(trends['slope_per_min']):.1f} points per minute. "
                "Plan a pacing change: shorter segments, an interactive block, or a break.",
                {"slope_per_min": trends["slope_per_min"], "mean": trends["mean"]})

    def _state_rules(self, add, w: dict, suffix: str) -> None:
        n = w["participants"]
        if not n:
            return
        when = f" around {mmss(w['start_s'])}" if suffix == "review" else ""
        if w["interaction"] == "silent" and w["inactive_count"] / n >= 0.4:
            add(f"silent_still{suffix}{int(w['start_s']) if suffix else ''}", "medium",
                f"Check understanding{when}",
                "The room was silent and many participants had been still for a long time. That can be "
                "quiet work, but also disengagement. A comprehension check or quick show of hands "
                "will tell you which.", {"inactive": w["inactive_count"], "participants": n}, w["start_s"])
        if w["restless_count"] / n >= 0.4:
            add(f"restless{suffix}{int(w['start_s']) if suffix else ''}", "medium",
                f"The class was restless{when}",
                f"{w['restless_count']} of {n} participants showed high movement. Consider a brief "
                "movement break, clearer transitions, or restating the goal of the current activity.",
                {"restless": w["restless_count"], "participants": n}, w["start_s"])

    # ---- review: what happened over the whole session ------------------
    def _review_rules(self, add, windows, trends) -> None:
        cfg = self.cfg
        step = cfg.window_seconds
        mean = trends["mean"]

        # sustained low segments
        low_runs = _runs(windows, lambda w: w["score"] < cfg.low_engagement, step)
        for run in low_runs:
            a, b = run[0]["start_s"], run[-1]["end_s"]
            kinds = {w["interaction"] for w in run}
            add(f"low_{int(a)}", "high", f"Low engagement {mmss(a)}-{mmss(b)}",
                f"Engagement stayed below {cfg.low_engagement:.0f} (lowest {min(w['score'] for w in run):.0f}). "
                f"The audio pattern in that stretch was: {', '.join(sorted(kinds))}. "
                "Consider an interactive task, a shorter explanation, or a break around this point next time.",
                {"from_s": a, "to_s": b, "min_score": min(w["score"] for w in run)}, a)

        # biggest drops
        for d in sorted(trends["drops"], key=lambda d: d["delta"])[:2]:
            add(f"drop_{int(d['at_s'])}", "medium", f"Sharp attention drop at {mmss(d['at_s'])}",
                f"Engagement fell {abs(d['delta']):.0f} points in one step ({d['from']:.0f} to {d['to']:.0f}). "
                "Look at what you were presenting at that moment: topic switch, dense slide, or long explanation.",
                d, d["at_s"])

        # lecture fatigue segments
        for run in _runs(windows, lambda w: w["interaction"] == "continuous lecture", step):
            if len(run) >= 3 and run[0]["score"] - run[-1]["score"] >= 12:
                a, b = run[0]["start_s"], run[-1]["end_s"]
                add(f"fatigue_{int(a)}", "medium", f"Long one-voice stretch {mmss(a)}-{mmss(b)}",
                    f"{(b - a) / 60:.1f} minutes of near-constant talking saw engagement fall from "
                    f"{run[0]['score']:.0f} to {run[-1]['score']:.0f}. Add a question, a poll, or a short "
                    "activity every few minutes.", {"from": run[0]["score"], "to": run[-1]["score"]}, a)

        # silent / restless windows (first occurrence only to avoid noise)
        seen = set()
        for w in windows:
            if w["participants"] and w["interaction"] == "silent" and \
                    w["inactive_count"] / w["participants"] >= 0.4 and "s" not in seen:
                seen.add("s")
                self._state_rules(add, w, "review")
            if w["participants"] and w["restless_count"] / w["participants"] >= 0.4 and "r" not in seen:
                seen.add("r")
                self._state_rules(add, w, "review")

        if len(windows) >= 4 and trends["direction"] == "declining":
            add("declining", "medium", "Engagement trended down across the session",
                f"Average engagement fell about {abs(trends['slope_per_min']):.1f} points per minute. "
                "Plan a pacing change: shorter segments, an interactive block, or a scheduled break.",
                {"slope_per_min": trends["slope_per_min"], "mean": mean})

    # ---- shared between modes -----------------------------------------
    def _cohort_rules(self, add, now, windows, trends, participants, attendance, live) -> None:
        cfg = self.cfg

        weak = [p for p in participants if p.get("engagement") is not None
                and p["engagement"] < cfg.low_engagement and len(p.get("timeline", [])) >= 2]
        if weak and not live:
            names = ", ".join(p["label"] for p in weak[:4])
            add("individual", "low", "Gentle check-in for a few learners",
                f"{len(weak)} participant(s) were below the engagement threshold for most of the session "
                f"({names}). A friendly, private check-in after class is usually more useful than "
                "singling anyone out during the session.", {"participants": [p["label"] for p in weak]})

        if attendance and not live:
            cutoff = max(120.0, 0.1 * now)
            late = [a for a in attendance if a["check_in_offset_s"] > cutoff]
            if len(late) >= 3:
                add("late", "low", "Several learners arrived late",
                    f"{len(late)} participants checked in after the first {mmss(cutoff)}. Consider a quick "
                    "recap at the start or a short 'what you missed' slide.", {"late": len(late)})
        if attendance:
            left = [a for a in attendance if a["status"] == "left"]
            if len(left) >= max(2, len(attendance) // 4):
                add("left_early", "medium", "Participants are leaving",
                    f"{len(left)} of {len(attendance)} participants have left. If this is unplanned, "
                    "check whether session length or schedule is a factor.",
                    {"left": len(left), "total": len(attendance)})

        peaks = trends.get("peaks", [])
        if live and windows:
            peaks = [p for p in peaks if p["at_s"] >= windows[-1]["start_s"] - cfg.window_seconds]
        if peaks:
            best = max(peaks, key=lambda p: p["to"])
            win = next((w for w in windows if w["start_s"] == best["at_s"]), None)
            fmt = f" ({win['interaction']})" if win and win["interaction"] != "unknown" else ""
            add("peak", "low", f"What worked around {mmss(best['at_s'])}",
                f"Engagement jumped {best['delta']:.0f} points{fmt}. Whatever you did here worked, so "
                "consider repeating the approach in later segments.", best, best["at_s"])

    # ------------------------------------------------------------------
    def llm_summary(self, suggestions: List[Suggestion], trends: dict) -> Optional[str]:
        """Optional: ask a LOCAL model to condense findings. Returns None if unavailable."""
        if not self.cfg.local_llm_url or not suggestions:
            return None
        bullet = "\n".join(f"- {s.title}: {s.rationale}" for s in suggestions[:6])
        prompt = ("You are a supportive teaching coach. In 3-4 sentences, summarise these classroom "
                  "observations for the instructor and suggest the most useful next step. "
                  f"Trend: {trends.get('direction')}.\n{bullet}")
        body = json.dumps({"model": self.cfg.local_llm_model, "prompt": prompt, "stream": False}).encode()
        try:
            req = urllib.request.Request(self.cfg.local_llm_url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode()).get("response", "").strip() or None
        except Exception:
            return None
