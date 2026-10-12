"""
Offline checks for Phase 49 (no camera, no PLC, no models):  python test_phase49_tracking.py

  1. position interpolation / extrapolation
  2. REPLAY of the real Phase 45 recording: every RFID1 read finds exactly one box
     in the read zone, and each tag lands on the same lane/x again one lap later
     (NOTE: the default zone was derived from this same recording, so this checks
     the logic and the zone numbers, it is NOT an independent test)
  3. SIMULATION of 6 pallets on the loop with the real read timing, tracker-id
     changes after occlusions and a missed detection: no wrong number may ever
     be shown ("?" is allowed), and most pallets must be identified
  4. learn mode, label rules, bad-input safety
"""
import csv
import importlib.util
import random
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

spec = importlib.util.spec_from_file_location("p49", HERE / "49_conference_rfid1_physical_tracking.py")
P49 = importlib.util.module_from_spec(spec)
sys.modules["p49"] = P49
spec.loader.exec_module(P49)
P49.OUT_DIR = Path(tempfile.mkdtemp())

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (" " + str(extra) if extra else ""))


def box(cx, cy, w=160, h=120):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


# ---------------------------------------------------------------- 1. history
h = P49.TrackHistory()
for i in range(6):
    h.update(100 + i * 0.1, [{"id": 7, "box": box(1000 - 20 * i, 500)}])
check("interpolates between frames", abs(h.position_at(7, 100.25)[0] - 950) < 1e-6)
check("extrapolates forward (<=0.8 s)", abs(h.position_at(7, 100.7)[0] - (900 - 40)) < 1.0, h.position_at(7, 100.7))
check("no position far outside the track's life", h.position_at(7, 105.0) is None and h.position_at(7, 90.0) is None)
check("unknown id -> None", h.position_at(99, 100.0) is None)

# ---------------------------------------------------------------- 2. replay of REAL live runs
# 3 live runs of Phase 49 (Oct 9): every read lists the boxes that were seen
# (track/lane/cx).  Rules checked: exactly ONE top-lane box in the read zone per read
# (duplicates merged), and every tag lands at about the same x lap after lap.
# NOTE: the zone was derived from these same runs - this checks logic and numbers,
# it is not an independent test.
RUNS = HERE.parent / "RESULTS" / "49_RFID1_Physical_Tracking"
zone = P49.ReadZone()
per_tag = defaultdict(list)
checked = unique = 0
bad = []
for name in ("144029", "144432", "144455"):
    for row in csv.DictReader(open(RUNS / f"phase49_20261009_{name}.csv", encoding="utf-8")):
        boxes = []
        for item in row["boxes_seen(track/lane/cx/in_zone)"].split(";"):
            if item:
                track, lane, cx, _ = item.split("/")
                boxes.append((lane, float(cx)))
        if len(boxes) < 4:
            continue                      # start-up reads, no boxes tracked yet
        top = sorted(cx for lane, cx in boxes if zone.contains(lane, cx))
        merged = []
        for cx in top:
            if not merged or cx - merged[-1] > 75:   # a box is ~160 px wide: closer = same pallet
                merged.append(cx)
        checked += 1
        if len(merged) == 1:
            unique += 1
            per_tag[int(row["tag"])].append(merged[0])
        else:
            bad.append((name, row["event"], row["tag"], merged))
check("replay of 3 live runs: every read has exactly ONE box in the top-lane zone", checked >= 25 and unique == checked, (checked, bad[:3]))
spread = {tag: round(max(v) - min(v)) for tag, v in per_tag.items()}
check("replay: each tag is read at about the same x lap after lap (spread <= 130 px)",
      all(s <= 130 for s in spread.values()) and len(per_tag) == 6, spread)
check("the bottom lane is not part of the default zone (only passing pallets there)", not zone.contains("bottom", 1250.0))

# ---------------------------------------------------------------- 3. simulation
P49.EVENT_SETTLE_SEC = 0.6
SPEED = 170.0                                   # px/s along the loop
TOP_Y, BOT_Y, X_LEFT, X_RIGHT = 545.0, 880.0, 330.0, 1750.0
TOP_LEN = X_RIGHT - X_LEFT
TURN = BOT_Y - TOP_Y
LOOP = 2 * TOP_LEN + 2 * TURN


def point_at(s):
    s %= LOOP
    if s < TOP_LEN:
        return X_RIGHT - s, TOP_Y, "top"
    s -= TOP_LEN
    if s < TURN:
        return X_LEFT, TOP_Y + s, "turn"
    s -= TURN
    if s < TOP_LEN:
        return X_LEFT + s, BOT_Y, "bottom"
    s -= TOP_LEN
    return X_RIGHT, BOT_Y - s, "turn"


def s_of_top(x):
    return X_RIGHT - x


def s_of_bottom(x):
    return TOP_LEN + TURN + (x - X_LEFT)


# Pallet spacing taken from the newest live run (read intervals x speed), in travel
# order 6,1,2,3,4,5 - irregular, every tag is read on the TOP lane at its own x.
ORDER = [6, 1, 2, 3, 4, 5]
GAP_BEHIND = {6: 417.0, 1: 663.0, 2: 737.0, 3: 386.0, 4: 450.0, 5: None}
GAP_BEHIND[5] = LOOP - sum(v for v in GAP_BEHIND.values() if v)
TAG_X = {3: 1145.0, 4: 1170.0, 5: 1390.0, 6: 1370.0, 1: 1330.0, 2: 1120.0}     # measured per tag
START_S = {}
position = 1000.0
for tag in ORDER:
    START_S[tag] = position % LOOP
    position -= GAP_BEHIND[tag]
TAG_TIMES = {tag: 0.0 for tag in ORDER}


def run_simulation(seconds=150.0, miss=None, seed=3, hide=(0.3, 1.2), every=(12.0, 25.0), noise=0.0, wobble=0.0):
    """miss: (tag, t0, t1) -> that pallet is NOT detected in [t0,t1] (e.g. at its read)."""
    rng = random.Random(seed)
    app = P49.App(P49.queue.Queue(), P49.ReadZone(), learn=False, debug=False)
    app.split_y = 700.0
    next_id = [1]
    state = {}
    for tag in TAG_TIMES:
        state[tag] = {"id": next_id[0], "hidden_until": 0.0, "new_after": 0.0, "pending_id": None, "next_hide": rng.uniform(6, every[0] + 6)}
        next_id[0] += 1
    truth = {}
    app.truth = truth
    previous_s = {tag: None for tag in TAG_TIMES}
    present = {tag: False for tag in TAG_TIMES}
    wrong, labelled, total = [], 0, 0
    dt = 0.11
    t = 1000.0
    steps = int(seconds / dt)
    for step in range(steps):
        t += dt
        clock = t - 1000.0
        tracks = []
        for tag in TAG_TIMES:
            st = state[tag]
            s = START_S[tag] + SPEED * clock + wobble * 60.0 * __import__("math").sin(0.7 * clock + tag)
            x, y, lane = point_at(s)
            if noise:
                x += rng.uniform(-noise, noise)
                y += rng.uniform(-noise, noise)
            w, hgt = (160, 120) if lane == "top" else (200, 150) if lane == "bottom" else (180, 135)

            # RFID read model: every tag is read on the top lane, from its own x until 240 px later
            inside = lane == "top" and TAG_X[tag] - 240.0 <= x <= TAG_X[tag]
            if inside and not present[tag]:
                app.events.put({"type": "appear", "number": tag, "time": t + 0.04})
                present[tag] = True
            elif not inside and present[tag]:
                app.events.put({"type": "leave", "number": tag, "time": t})
                present[tag] = False

            # occlusion: the pallet vanishes for a while and comes back under a NEW id
            if clock >= st["next_hide"] and st["hidden_until"] <= clock:
                st["hidden_until"] = clock + rng.uniform(*hide)
                st["next_hide"] = clock + rng.uniform(*every)
                st["pending_id"] = next_id[0]
                next_id[0] += 1
            hidden = clock < st["hidden_until"]
            if st["pending_id"] is not None and not hidden:
                if clock >= st["hidden_until"] + 0.55:      # 5 frames to confirm the new id
                    st["id"] = st["pending_id"]
                    st["pending_id"] = None
                else:
                    hidden = True
            if miss and miss[0] == tag and miss[1] <= clock <= miss[2]:
                hidden = True
            if hidden:
                continue
            truth[st["id"]] = tag
            tracks.append({"id": st["id"], "box": box(x, y, w, hgt)})

        app.ingest(t, tracks, (1.0, 1.0))

        if clock > 50.0:
            for tr in tracks:
                total += 1
                number = app.identities.number_of(tr["id"])
                if number is not None:
                    labelled += 1
                    if number != truth[tr["id"]]:
                        wrong.append((round(clock, 1), tr["id"], number, truth[tr["id"]]))
    return app, wrong, labelled, total


app, wrong, labelled, total = run_simulation()
check("simulation: NEVER a wrong physical number on a box", not wrong, wrong[:6])
check("simulation: confirmed numbers appear on most box-frames after 50 s", total and labelled / total >= 0.5, f"{labelled}/{total}")
check("simulation: later laps verify earlier reads", app.identities.verified >= 6, f"verified={app.identities.verified} conflicts={app.identities.conflicts}")
results = [r[3] for r in app.resolver.log_rows]
check("simulation: reads mostly resolve to MAP", results.count("MAP") >= 0.7 * len(results), {k: results.count(k) for k in set(results)})

app2, wrong2, lab2, tot2 = run_simulation(seed=11, miss=(4, 31.5, 36.0))
check("simulation 2 (a pallet NOT detected at its read): still no wrong number", not wrong2, wrong2[:6])
check("simulation 2: the missed read is reported, not guessed",
      any(r[3] in ("NO_CANDIDATE", "AMBIGUOUS") for r in app2.resolver.log_rows), [r[3] for r in app2.resolver.log_rows][-10:])


# ---------------------------------------------------------------- 3b. safety sweep
sweep_wrong = 0
sweep_runs = 0
for seed in range(1, 9):
    for cfg in ({}, {"noise": 6.0}, {"hide": (0.1, 0.6), "every": (6.0, 12.0)}, {"hide": (0.5, 2.5), "every": (8.0, 16.0)}):
        _, bad, _, _ = run_simulation(seconds=110.0, seed=seed, **cfg)
        sweep_wrong += len(bad)
        sweep_runs += 1
check("sweep: 32 simulated runs (noise, short and LONG occlusions): no wrong number", sweep_wrong == 0, f"{sweep_wrong} wrong frames in {sweep_runs} runs")

# ---------------------------------------------------------------- 3c. unit checks of the identity rules
def fresh():
    hist = P49.TrackHistory()
    return hist, P49.PhysicalIdentities(hist)


def feed(hist, ident, t, tracks):
    hist.update(t, tracks)
    ident.update(t, tracks)


# a number read ONCE is not shown, read twice on the same pallet it is
hist, ident = fresh()
for i in range(12):
    feed(hist, ident, 10 + i * 0.1, [{"id": 1, "box": box(1300 - 17 * i, 545)}])
ident.assign(4, 1, 10.5)
check("a number read once is NOT shown", ident.number_of(1) is None and ident.claimed_number(1) == 4 and ident.is_provisional(1))
ident.assign(4, 1, 30.5)
check("the same tag read again on the same pallet is shown", ident.number_of(1) == 4 and ident.verified == 1)

# two tracker ids that existed at the same time are two pallets: no number is copied
hist, ident = fresh()
for i in range(15):
    feed(hist, ident, 10 + i * 0.1, [{"id": 1, "box": box(1300 - 17 * i, 545)}, {"id": 2, "box": box(1050 - 17 * i, 545)}])
ident.assign(4, 1, 11.0)
ident.assign(4, 1, 11.2)
feed(hist, ident, 11.6, [{"id": 2, "box": box(1050 - 17 * 15, 545)}])     # id 1 vanishes
feed(hist, ident, 11.7, [{"id": 2, "box": box(1050 - 17 * 16, 545)}])
check("a box that existed before cannot inherit a lost pallet's number", ident.number_of(2) is None)

# a NEW id born where the lost pallet must be inherits it (also after a turn: learned motion)
hist, ident = fresh()
t = 100.0
for lap in range(2):                                    # teach the field: a loop with a turn
    for i in range(60):
        x, y = (1700 - 17 * i, 545.0) if i < 40 else (1020.0, 545.0 + 15 * (i - 40))
        feed(hist, ident, t, [{"id": 100 + lap, "box": box(x, y)}])
        t += 0.1
base = t
for i in range(25):
    x, y = (1700 - 17 * i, 545.0)
    feed(hist, ident, base + i * 0.1, [{"id": 7, "box": box(x, y)}])
ident.assign(2, 7, base + 2.0)
ident.assign(2, 7, base + 2.2)
last = base + 2.4
for k in range(1, 17):                                    # hidden for 1.6 s, then a new id behind the turn
    feed(hist, ident, last + k * 0.1, [])
for i in range(7):
    t_new = last + 1.7 + i * 0.1
    x, y = 1700 - 17 * 24 - 17 * 17, 545.0
    feed(hist, ident, t_new, [{"id": 9, "box": box(max(1020.0, 1700 - 17 * 41 - 17 * i), 545.0)}])
check("flow field is learned from the moving boxes", ident.flow.velocity(1600, 545) is not None)

# the red line: a numbered pallet never jumps to a clearly farther box
hist, ident = fresh()
for i in range(15):
    feed(hist, ident, 10 + i * 0.1, [{"id": 1, "box": box(1300 - 17 * i, 545)}])
ident.assign(3, 1, 11.0)
ident.assign(3, 1, 11.2)
for k in range(1, 8):
    feed(hist, ident, 11.4 + k * 0.1, [])
for i in range(7):
    feed(hist, ident, 12.2 + i * 0.1, [{"id": 5, "box": box(300 - 17 * i, 545)}])    # 600 px away
check("a far-away new box never inherits a number", ident.number_of(5) is None)


# ---------------------------------------------------------------- 5. wiring: main() hooks a (fake) Phase 40
import threading
import time as _time

wire_dir = Path(tempfile.mkdtemp())
(wire_dir / "fake_live.py").write_text(
    """
import time
import numpy as np
import cv2


class Cap:
    def read(self):
        time.sleep(0.04)
        return True, np.zeros((1080, 1920, 3), np.uint8)


def open_camera():
    return Cap(), 1


def main():
    cap, index = open_camera()
    inner_polygon = np.array([[300, 690], [1600, 690], [1600, 770], [300, 770]], np.int32)
    outer_polygon = np.array([[200, 450], [1700, 450], [1700, 1000], [200, 1000]], np.int32)
    for i in range(80):
        ok, frame = cap.read()
        symbol_frame = frame
        confirmed_tracks = [{"track_id": 1, "box": (1200, 480, 1360, 600), "velocity_x": 0.0, "velocity_y": 0.0}]
        visible_track_cache = {
            1: {"box": (1200, 480, 1360, 600), "label": "P1 ACCEPTED NOX:3/3", "missed": 0},
            2: {"box": (300, 800, 500, 950), "label": "P2 CHECKING X:0", "missed": 1},
        }
        cv2.imshow("CREST - FAKE PHASE 40", symbol_frame)
""",
    encoding="utf-8",
)

drawn = []
shown = []
cv2_stub = sys.modules["cv2"]
cv2_stub.imshow = lambda name, image: shown.append(name)
cv2_stub.waitKey = lambda delay=0: -1
for fn in ("polylines", "rectangle", "line"):
    setattr(cv2_stub, fn, lambda *a, **k: None)
cv2_stub.putText = lambda image, text, *a, **k: drawn.append(text)
cv2_stub.FONT_HERSHEY_SIMPLEX = 0

P49.LIVE_PATH = wire_dir / "fake_live.py"
P49.OUT_DIR = wire_dir / "out"
P49.ZONE_FILE = wire_dir / "zone.json"


def fake_reader(events, stop):
    def later(delay, number):
        _time.sleep(delay)
        events.put({"type": "appear", "number": number, "time": _time.monotonic()})
    threading.Thread(target=later, args=(0.4, 3), daemon=True).start()
    threading.Thread(target=later, args=(1.9, 3), daemon=True).start()   # same tag again


P49.rfid1_reader = fake_reader
sys.argv = ["49"]
P49.EVENT_SETTLE_SEC = 0.3
P49.main()
labels = [d for d in drawn if " ACCEPTED" in d or " CHECKING" in d]
check("wiring: main() runs the Phase 40 loop and draws every frame", len(shown) == 80 and len(labels) >= 80, f"{len(shown)} frames")
check("wiring: before the 2nd read the box shows '?'", any(l.startswith("? ACCEPTED") for l in labels))
check("wiring: after the same tag was read twice the box shows '3 ACCEPTED'", any(l.startswith("3 ACCEPTED") for l in labels), sorted(set(labels))[:4])
check("wiring: the held box (another pallet, missed>0) stays '? CHECKING'", any(l.startswith("? CHECKING") for l in labels))
csv_files = list((wire_dir / "out").glob("phase49_*.csv"))
check("wiring: evidence CSV written with both reads", csv_files and len(csv_files[0].read_text().splitlines()) == 3, csv_files)

# two boxes on top of each other (duplicate Phase 40 box) count as ONE candidate
dup_hist = P49.TrackHistory()
dup_ident = P49.PhysicalIdentities(dup_hist)
dup_resolver = P49.EventResolver(dup_hist, dup_ident, P49.ReadZone())
for i in range(6):
    dup_hist.update(500 + i * 0.1, [{"id": 1, "box": box(1288, 545)}, {"id": 2, "box": box(1290, 548)}])
cands = dup_resolver.candidates(500.3, 700.0)
check("duplicate boxes at the same spot are ONE candidate", len(cands) == 1, len(cands))

# ---------------------------------------------------------------- 4. misc
check("label: identified -> '3 ACCEPTED', unknown -> '? ACCEPTED'",
      P49.pallet_label(3, "ACCEPTED") == "3 ACCEPTED" and P49.pallet_label(None, "ACCEPTED") == "? ACCEPTED")
check("label: debug adds the tracker id", P49.pallet_label(2, "EMPTY", 41, True) == "2 EMPTY [T41]")
check("lane midline from polygon", abs(P49.split_from_polygon([(0, 690), (10, 770)], 1) - 730) < 1e-6)

learn_app = P49.App(P49.queue.Queue(), P49.ReadZone(), learn=True, debug=False)
learn_app.split_y = 700.0
for lap in range(3):
    t0 = 2000.0 + lap * 20
    for i in range(30):
        t = t0 + i * 0.11
        cx = 1500 - (t - t0) * 170
        learn_app.ingest(t, [{"id": 1, "box": box(cx, 545)}], (1.0, 1.0))
        if i == 3:
            learn_app.events.put({"type": "appear", "number": 3, "time": t})
    learn_app.ingest(t0 + 3.5, [{"id": 1, "box": box(1000, 545)}], (1.0, 1.0))
zone, changed = learn_app.resolver.learned_zone(P49.ReadZone())
check("learn: one pallet -> the top lane zone is measured, bottom stays unused", changed == ["top"] and zone.bounds["bottom"] is None, (changed, zone.bounds))

invalid = P49.App(P49.queue.Queue(), P49.ReadZone(), learn=True, debug=False)
invalid.split_y = 700.0
for i in range(15):
    invalid.ingest(3000 + i * 0.11, [{"id": 1, "box": box(1300 - i * 5, 545)}, {"id": 2, "box": box(900, 880)}], (1.0, 1.0))
    if i == 1:
        invalid.events.put({"type": "appear", "number": 4, "time": 3000.15})
check("learn: refuses a read while 2 pallets are visible", any(r[3] == "LEARN_INVALID" for r in invalid.resolver.log_rows))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
