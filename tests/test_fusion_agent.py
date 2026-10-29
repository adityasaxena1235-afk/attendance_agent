from config import Config
from core.agent import TeachingAgent
from core.engagement import Sample
from core.fusion import EngagementFusion
from core.models import AudioFeatures


def sample(att=1.0, motion=1.0, inactive=False, restless=False, yaw=0.0):
    return Sample(att, motion, inactive, restless, yaw, 0.1)


def fill(f, t0, t1, people, att, speech=True, motion=1.0, inactive=False, fps=5):
    t = t0
    while t < t1:
        for rid in people:
            f.add_visual(t, rid, sample(att=att, motion=motion, inactive=inactive), 1 / fps)
        t += 1 / fps
    t = t0
    while t < t1:
        f.add_audio(AudioFeatures(t, 0.5, -20 if speech else -55, -55, speech))
        t += 0.5


def test_attentive_class_scores_high_and_away_class_scores_low():
    cfg = Config()
    f = EngagementFusion(cfg)
    fill(f, 0, 30, [1, 2, 3], att=1.0)
    fill(f, 30, 60, [1, 2, 3], att=0.0)
    w = f.windows()
    assert w[0]["score"] > 80 and w[1]["score"] < 40
    assert w[0]["attentive_pct"] == 100.0 and w[1]["attentive_pct"] == 0.0


def test_looking_away_is_not_rescued_by_movement():
    f = EngagementFusion(Config())
    fill(f, 0, 30, [1], att=0.0, motion=1.0)
    assert f.windows()[0]["visual_score"] == 0.0


def test_inactivity_lowers_score_but_less_than_looking_away():
    f = EngagementFusion(Config())
    fill(f, 0, 30, [1], att=1.0, motion=0.15, inactive=True)
    fill(f, 30, 60, [1], att=1.0, motion=1.0)
    w = f.windows()
    assert w[0]["score"] < w[1]["score"]
    assert w[0]["visual_score"] > 50
    assert w[0]["inactive_count"] == 1


def test_fusion_without_audio_uses_visual_only():
    f = EngagementFusion(Config())
    t = 0.0
    while t < 30:
        f.add_visual(t, 1, sample(), 0.2)
        t += 0.2
    w = f.windows()[0]
    assert w["audio_score"] is None and w["score"] == w["visual_score"]


def test_trends_detect_drop_and_decline():
    cfg = Config()
    f = EngagementFusion(cfg)
    for i, a in enumerate([1.0, 1.0, 0.9, 0.5, 0.4, 0.3]):
        fill(f, i * 30, i * 30 + 30, [1, 2], att=a)
    tr = f.trends()
    assert tr["direction"] == "declining"
    assert any(d["at_s"] == 90 for d in tr["drops"])
    assert tr["worst"]["at_s"] == 150


def test_participant_stats_separate_good_and_bad_learner():
    f = EngagementFusion(Config())
    for i in range(4):
        fill(f, i * 30, i * 30 + 30, [1], att=1.0)
        fill(f, i * 30, i * 30 + 30, [2], att=0.1)
    f.windows()
    assert f.participant_stats(1)["engagement"] > f.participant_stats(2)["engagement"] + 30


# ---------------- agent ----------------
def windows_from(scores, interactions=None):
    out = []
    for i, sc in enumerate(scores):
        out.append({"index": i, "start_s": i * 30.0, "end_s": i * 30.0 + 30, "score": sc,
                    "visual_score": sc, "audio_score": 70, "participants": 5, "attentive_pct": sc,
                    "inactive_count": 0, "restless_count": 0,
                    "interaction": (interactions or ["mixed"] * len(scores))[i],
                    "speech_ratio": 0.9, "turns": 1, "longest_silence_s": 1})
    return out


def trends_for(cfg, wins):
    f = EngagementFusion(cfg)
    return f.trends(wins)


def test_agent_live_flags_low_engagement_with_cooldown():
    cfg = Config()
    ag = TeachingAgent(cfg)
    wins = windows_from([85, 80, 70, 40])
    out = ag.advise(120, wins, trends_for(cfg, wins), [], [], live=True)
    assert any(s.priority == "high" and "Re-engage" in s.title for s in out)
    again = ag.advise(130, wins, trends_for(cfg, wins), [], [], live=True)
    assert not any("Re-engage" in s.title for s in again)       # cooldown


def test_agent_review_finds_long_lecture_fatigue_and_low_segment():
    cfg = Config()
    ag = TeachingAgent(cfg)
    wins = windows_from([85, 80, 78, 66, 55, 45, 44, 80],
                        ["mixed", "continuous lecture", "continuous lecture", "continuous lecture",
                         "continuous lecture", "continuous lecture", "continuous lecture", "discussion"])
    titles = [s.title for s in ag.advise(240, wins, trends_for(cfg, wins), [], [], live=False)]
    assert any("Long one-voice stretch" in t for t in titles)
    assert any(t.startswith("Low engagement") for t in titles)
    assert any(t.startswith("What worked") for t in titles)


def test_agent_never_fires_on_healthy_live_class():
    cfg = Config()
    ag = TeachingAgent(cfg)
    wins = windows_from([88, 90, 87, 89, 91, 90, 88])
    assert ag.advise(210, wins, trends_for(cfg, wins), [], [], live=True) == []


def test_agent_suggestions_always_explain_themselves():
    cfg = Config()
    ag = TeachingAgent(cfg)
    wins = windows_from([85, 80, 70, 40])
    for s in ag.advise(120, wins, trends_for(cfg, wins), [], [], live=True):
        assert s.rationale and s.evidence is not None


def test_llm_summary_is_skipped_when_no_local_model_configured():
    cfg = Config(local_llm_url="")
    ag = TeachingAgent(cfg)
    wins = windows_from([85, 80, 70, 40])
    sug = ag.advise(120, wins, trends_for(cfg, wins), [], [], live=True)
    assert ag.llm_summary(sug, {"direction": "steady"}) is None
