"""Accuracy evaluation against the simulator's ground truth.

Ground truth: 8 real participants (late arrivals, an early leaver, an occluded person) plus 2
people who just walk past. We run many random seeds and report attendance precision/recall,
check-in time error, and how the sustained-presence threshold trades off the two.

    python scripts/evaluate.py
"""
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from core.pipeline import SessionPipeline  # noqa: E402
from core.simulator import default_profiles, replay, simulate  # noqa: E402


def evaluate(presence_seconds: float, seed: int):
    cfg = Config(presence_seconds=presence_seconds)
    pl = SessionPipeline(cfg, "eval", "simulator")
    replay(pl, simulate(seed=seed, with_audio=False))
    truth = default_profiles()
    recs = pl.attendance.confirmed()

    matched, false_pos, errs = set(), 0, []
    for r in recs:
        x, y = r.last_bbox[0] + r.last_bbox[2] / 2 - 45, r.last_bbox[1] + r.last_bbox[3] / 2 - 55
        best = min(truth, key=lambda p: (p.seat[0] - x) ** 2 + (p.seat[1] - y) ** 2)
        d = ((best.seat[0] - x) ** 2 + (best.seat[1] - y) ** 2) ** 0.5
        if d < 110 and best.name not in matched:
            matched.add(best.name)
            errs.append(abs(r.check_in - best.arrive))
        else:
            false_pos += 1
    leave_ok = any(e.kind == "LEFT" for e in pl.attendance.events)
    return len(recs), len(matched), false_pos, errs, leave_ok


def main():
    seeds = range(10)
    print(f"{'presence_s':>10} {'precision':>10} {'recall':>8} {'ghosts/run':>11} {'check-in MAE (s)':>17} {'leaver found':>13}")
    for ps in (0.2, 3, 5, 8, 15, 30):
        tp = fp = n_true = 0
        errs, leavers = [], 0
        for sd in seeds:
            n_conf, matched, false_pos, e, lv = evaluate(ps, sd)
            tp += matched
            fp += false_pos
            n_true += len(default_profiles())
            errs += e
            leavers += lv
        prec = tp / (tp + fp) if tp + fp else 0.0
        print(f"{ps:>10} {prec:>10.2%} {tp / n_true:>8.2%} {fp / len(seeds):>11.1f} "
              f"{mean(errs) if errs else float('nan'):>17.2f} {leavers / len(seeds):>13.0%}")
    print("\npresence_s = 0.2 is effectively 'one frame is enough', i.e. naive single-frame detection.")


if __name__ == "__main__":
    main()
