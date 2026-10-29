from config import Config
from core.attendance import AttendanceManager
from core.models import Detection, TrackObs
from core.tracker import CentroidIoUTracker


def box(x, y=100, w=90, h=110):
    return (float(x), float(y), float(w), float(h))


def run(att, track_id, bbox, t0, t1, fps=5.0):
    t = t0
    while t < t1 - 1e-9:
        att.update(t, [TrackObs(track_id, bbox)])
        t += 1 / fps


# ---------------- tracker ----------------
def test_tracker_keeps_id_for_moving_face():
    tr = CentroidIoUTracker()
    ids = set()
    for i in range(30):
        obs = tr.update(i * 0.2, [Detection(box(100 + i * 4))])
        ids.add(obs[0].track_id)
    assert len(ids) == 1


def test_tracker_separates_two_people():
    tr = CentroidIoUTracker()
    obs = tr.update(0, [Detection(box(100)), Detection(box(500))])
    assert len({o.track_id for o in obs}) == 2
    obs2 = tr.update(0.2, [Detection(box(104)), Detection(box(498))])
    assert {o.track_id for o in obs2} == {o.track_id for o in obs}


def test_tracker_rematches_after_short_miss_with_large_jump():
    tr = CentroidIoUTracker()
    tid = tr.update(0.0, [Detection(box(100))])[0].track_id
    # person is missed for 0.6 s and has moved most of a face width
    assert tr.update(0.6, [Detection(box(170))])[0].track_id == tid


def test_tracker_new_id_after_long_absence():
    tr = CentroidIoUTracker(max_missing_seconds=3)
    tid = tr.update(0.0, [Detection(box(100))])[0].track_id
    assert tr.update(10.0, [Detection(box(100))])[0].track_id != tid


# ---------------- attendance ----------------
def test_brief_visitor_is_not_marked_present():
    cfg = Config(presence_seconds=8)
    att = AttendanceManager(cfg)
    run(att, 1, box(900), 0, 3)                  # walks past for 3 s
    assert att.confirmed() == []
    assert att.transient_count() == 1
    assert att.events == []


def test_sustained_presence_is_confirmed_with_true_arrival_time():
    cfg = Config(presence_seconds=8)
    att = AttendanceManager(cfg)
    run(att, 1, box(100), 12.0, 30.0)
    rec = att.confirmed()[0]
    assert rec.check_in == 12.0                  # logged at arrival, not when confirmed
    assert rec.confirmed_at >= 12.0 + 8
    assert [e.kind for e in att.events] == ["CHECK_IN"]


def test_short_dropout_does_not_break_the_run():
    cfg = Config(presence_seconds=8, gap_tolerance_seconds=3)
    att = AttendanceManager(cfg)
    run(att, 1, box(100), 0, 5)
    run(att, 1, box(100), 7.0, 12)               # 2 s gap, inside tolerance
    assert att.confirmed()[0].check_in == 0.0


def test_long_dropout_before_confirmation_restarts_the_run():
    cfg = Config(presence_seconds=8, gap_tolerance_seconds=3)
    att = AttendanceManager(cfg)
    run(att, 1, box(100), 0, 5)
    run(att, 1, box(100), 10.0, 15)              # 5 s gap, then only 5 s more
    assert att.confirmed() == []


def test_leaves_and_returns():
    cfg = Config(presence_seconds=5, absence_seconds=20)
    att = AttendanceManager(cfg)
    run(att, 1, box(100), 0, 10)
    att.update(40.0, [])                         # unseen for 30 s
    assert att.confirmed()[0].status == "left"
    run(att, 1, box(100), 50.0, 52.0)
    assert att.confirmed()[0].status == "present"
    assert [e.kind for e in att.events] == ["CHECK_IN", "LEFT", "RETURNED"]


def test_new_track_id_at_same_spot_is_merged_into_same_participant():
    cfg = Config(presence_seconds=5)
    att = AttendanceManager(cfg)
    run(att, 1, box(100), 0, 10)
    run(att, 2, box(104), 14.0, 20.0)            # tracker lost them, new id, same seat
    assert len(att.confirmed()) == 1
    assert att.confirmed()[0].visits == 2


def test_two_neighbours_are_not_merged():
    cfg = Config(presence_seconds=5)
    att = AttendanceManager(cfg)
    run(att, 1, box(100), 0, 10)
    run(att, 2, box(500), 14.0, 25.0)
    assert len(att.confirmed()) == 2
