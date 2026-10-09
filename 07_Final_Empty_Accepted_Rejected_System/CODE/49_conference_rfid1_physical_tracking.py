"""CREST Phase 49 - RFID1 physical pallet identity tracking.

Phase 40 (YOLO26 detection, tracker, EMPTY/ACCEPTED/REJECTED classification)
is loaded and NOT changed.  This file only adds a separate identity layer:

  * RFID1 only (PLC words 96..99).  READ ONLY.  RFID2 is never touched.
  * No clicking.  No stopping the conveyor.
  * A tag read is turned into a physical pallet only when exactly ONE box is
    inside the RFID read zone at the moment of the read (position interpolated
    to the RFID timestamp).  Anything else is logged and stays "?".
  * Physical identities 1..6 are tracked separately from the temporary Phase 40
    tracker ids (one identity <-> one box, mutual-best geometry, never copied
    along the classification state bridge).
  * Every later read of the same tag is a free physical check: the box in the
    zone must already carry that number (VERIFIED) - otherwise it is logged as
    CONFLICT and RFID wins.

Evidence behind the default read zone (Phase 45, 2 laps, 11 events): the tagged
pallet was found at image x 1250-1310 on the TOP lane for tags 3,4,5,6 and at
x 1080-1285 on the BOTTOM lane for tags 1 and 2 - not at the drawn RFID1 box
(x 1123-1183).  Run with --learn to MEASURE the zone with one pallet instead.

Run:   python -u 49_conference_rfid1_physical_tracking.py
       python -u 49_conference_rfid1_physical_tracking.py --learn   (ONE pallet)
       python -u 49_conference_rfid1_physical_tracking.py --debug
       python -u 49_conference_rfid1_physical_tracking.py --show-provisional

A number is SHOWN only after RFID1 has read the same tag twice on the same physical
pallet (one lap apart); until then the box shows "?".  --debug shows "(?3)" for a
number that was read once.
"""

import argparse
import csv
import importlib.util
import json
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
    r"\07_Final_Empty_Accepted_Rejected_System"
)
LIVE_PATH = ROOT / "CODE" / "40_conference_100_pallet_live.py"
ZONE_FILE = ROOT / "CONFIG" / "rfid1_read_zone.json"
OUT_DIR = ROOT / "RESULTS" / "49_RFID1_Physical_Tracking"

PLC_IP = "192.168.82.231"
RFID_TAGS = [
    f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]" for i in range(96, 100)
]
NO_TAG = (0, 0, 0, 0)
RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}

# Camera frame the zone numbers refer to (positions are scaled if it differs).
REF_W, REF_H = 1920.0, 1080.0

# Read zone: image x range of the tagged box CENTRE at the RFID timestamp, per
# lane.  Derived from the Phase 45 evidence (midpoints between the tagged box
# and the nearest non-tagged boxes); replaced by the measured zone file.
DEFAULT_ZONE = {
    "top": [1160.0, 1400.0],
    "bottom": [1050.0, 1330.0],
    "source": "default: Phase 45 evidence",
}

# Event handling
EVENT_SETTLE_SEC = 0.6       # wait this long after a read: a new box needs ~5 frames to be confirmed
EVENT_RESOLVE_SEC = 1.6      # keep an unresolved read open until this long after the read
OWNER_TRUST_CONF = 0.6       # a box whose number is at least this certain is never "stolen"
REACQ_CONF_FACTOR = 0.8      # every tracker-id change without a new RFID check lowers the certainty
PROVISIONAL_CONF = 0.5       # certainty of a number that no second read has confirmed yet
SHOW_PROVISIONAL = False     # True (--show-provisional): also show numbers read only once
LANE_MEMORY_MIN = 1          # a tag read on ONE lane (never on the other) is looked for on that lane
POSITION_EXTRAP_SEC = 0.8    # how far a track position may be extrapolated
LIVE_RECENT_SEC = 0.6        # a track counts as "visible" if seen this recently

# Physical identity tracking
COAST_MAX_SEC = 2.2          # an identity may coast this long without a box (Phase 40 bridges 2.0 s)
REACQ_GATE = 0.7             # max distance (box widths) between prediction and new box
REACQ_MARGIN = 0.4           # best must beat second best by this (box widths)
REACQ_NUMBERED_GATE = 0.55   # a pallet that already carries a number is rejoined only on a closer match
REACQ_NUMBERED_MARGIN = 0.55
REACQ_BORN_SLACK = 0.25      # only a box BORN after the old one vanished can inherit its number
HISTORY_KEEP_SEC = 8.0
FLOW_CELL_PX = 60.0          # learned motion field: cell size
FLOW_MIN_SPEED = 30.0        # px/s: slower boxes do not teach the field
FLOW_STEP_SEC = 0.1

# Learn mode
LEARN_MIN_EVENTS = 2
LEARN_MARGIN_RATIO = 0.25    # zone margin = this * box width


# ============================================================
# RFID1 READER (read only)
# ============================================================

def rfid1_reader(events, stop):
    """Puts {"type": "appear"|"leave", "number", "time", ...} into `events`.

    A tag counts as read only after two identical valid four-word readings,
    and only after the sensor was seen clear (0,0,0,0) first.
    """

    try:
        from pylogix import PLC
    except ImportError:
        print("ERROR: pylogix missing. Install: python -m pip install pylogix")
        return

    seen_no_tag = False
    absent_streak = 0
    pending_words = None
    first_valid_time = None
    valid_streak = 0
    present_number = None
    last_error = 0.0
    notified_wait = False

    while not stop.is_set():
        try:
            with PLC() as plc:
                plc.IPAddress = PLC_IP
                plc.SocketTimeout = 3
                print("RFID1 ONLY: connected", PLC_IP, "words [96..99]", flush=True)

                while not stop.is_set():
                    words = []
                    for tag in RFID_TAGS:
                        reading = plc.Read(tag)
                        if reading.Status != "Success":
                            raise RuntimeError(f"PLC read {tag}: {reading.Status}")
                        words.append(int(reading.Value))

                    t = time.monotonic()
                    words = tuple(words)

                    if words == NO_TAG:
                        absent_streak += 1
                        valid_streak = 0
                        pending_words = None
                        if absent_streak >= 2:
                            if present_number is not None:
                                events.put(
                                    {"type": "leave", "number": present_number, "time": t}
                                )
                                present_number = None
                            if not seen_no_tag:
                                print("RFID1 READY: sensor clear", flush=True)
                            seen_no_tag = True

                    elif words in RFID_MAP:
                        absent_streak = 0
                        if not seen_no_tag:
                            if not notified_wait:
                                print("RFID1 WAIT: start with a clear sensor", flush=True)
                                notified_wait = True
                            continue

                        if words != pending_words:
                            pending_words = words
                            valid_streak = 1
                            first_valid_time = t
                        else:
                            valid_streak += 1

                        if valid_streak >= 2 and present_number is None:
                            present_number = RFID_MAP[words]
                            events.put(
                                {
                                    "type": "appear",
                                    "number": present_number,
                                    "time": first_valid_time,
                                    "confirmed_at": t,
                                }
                            )
                    else:
                        # incomplete PLC words: never a pallet identity
                        absent_streak = 0
                        valid_streak = 0
                        pending_words = None

                    stop.wait(0.025)

        except Exception as error:
            now = time.monotonic()
            if now - last_error > 3.0:
                print("RFID1 PLC ERROR:", error, flush=True)
                last_error = now
            stop.wait(1.0)


# ============================================================
# TRACK HISTORY (positions at an exact timestamp)
# ============================================================

class TrackHistory:
    """Per Phase 40 tracker id: time-stamped centre/size, interpolated to any time."""

    def __init__(self, keep_sec=HISTORY_KEEP_SEC):
        self.keep_sec = keep_sec
        self.data = {}      # id -> deque of (t, cx, cy, w, h)
        self.last_t = 0.0

    def update(self, t, tracks):
        self.last_t = t
        for tr in tracks:
            x1, y1, x2, y2 = tr["box"]
            row = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1))
            history = self.data.setdefault(int(tr["id"]), deque())
            if history and history[-1][0] >= t:
                continue
            history.append((t,) + row)

        for tid in list(self.data):
            history = self.data[tid]
            while history and t - history[0][0] > self.keep_sec:
                history.popleft()
            if not history:
                del self.data[tid]

    def around(self, tid, t, within=LIVE_RECENT_SEC):
        """True if the track existed at (or within `within` seconds of) time t."""
        history = self.data.get(tid)
        return bool(history) and history[0][0] - within <= t <= history[-1][0] + within

    def span(self, tid):
        history = self.data.get(tid)
        return (history[0][0], history[-1][0]) if history else (None, None)

    def velocity(self, tid):
        """(vx, vy) in px/s from the last observations."""
        history = self.data.get(tid)
        if not history or len(history) < 2:
            return 0.0, 0.0
        last = history[-1]
        for item in reversed(history):
            if last[0] - item[0] >= 0.25:
                dt = last[0] - item[0]
                return (last[1] - item[1]) / dt, (last[2] - item[2]) / dt
        first = history[0]
        dt = max(1e-3, last[0] - first[0])
        return (last[1] - first[1]) / dt, (last[2] - first[2]) / dt

    def position_at(self, tid, t, max_extrap=POSITION_EXTRAP_SEC):
        """(cx, cy, w, h) at time t, or None if the track was not around then."""
        history = self.data.get(tid)
        if not history:
            return None
        rows = list(history)
        t0, tn = rows[0][0], rows[-1][0]

        if t0 <= t <= tn:
            for a, b in zip(rows, rows[1:]):
                if a[0] <= t <= b[0]:
                    f = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
                    return tuple(a[i] + f * (b[i] - a[i]) for i in range(1, 5))
            return rows[-1][1:5]

        vx, vy = self.velocity(tid)
        if t > tn and t - tn <= max_extrap:
            ref = rows[-1]
            return ref[1] + vx * (t - tn), ref[2] + vy * (t - tn), ref[3], ref[4]
        if t < t0 and t0 - t <= max_extrap:
            ref = rows[0]
            return ref[1] + vx * (t - t0), ref[2] + vy * (t - t0), ref[3], ref[4]
        return None


# ============================================================
# READ ZONE + LANES
# ============================================================

class ReadZone:
    def __init__(self, zone=None):
        zone = zone or DEFAULT_ZONE
        self.bounds = {
            "top": tuple(zone["top"]),
            "bottom": tuple(zone["bottom"]),
        }
        self.source = zone.get("source", "?")

    @classmethod
    def load(cls, path=ZONE_FILE):
        try:
            with open(path, encoding="utf-8") as handle:
                return cls(json.load(handle))
        except Exception:
            return cls()

    def save(self, path=ZONE_FILE):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "top": list(self.bounds["top"]),
                    "bottom": list(self.bounds["bottom"]),
                    "source": self.source,
                    "saved": datetime.now().isoformat(timespec="seconds"),
                },
                handle,
                indent=2,
            )

    def contains(self, lane, cx):
        if lane not in self.bounds:
            return False
        low, high = self.bounds[lane]
        return low <= cx <= high


def lane_of(cy, split_y):
    """'top' / 'bottom' relative to the horizontal midline of the conveyor ring."""
    return "top" if cy < split_y else "bottom"


def split_from_polygon(polygon, fallback):
    """Midline = vertical middle of the inner conveyor polygon (dynamic, no constant)."""
    try:
        ys = [float(point[1]) for point in polygon.reshape(-1, 2)]
        return (min(ys) + max(ys)) / 2.0
    except Exception:
        try:
            ys = [float(point[1]) for point in polygon]
            return (min(ys) + max(ys)) / 2.0
        except Exception:
            return fallback


# ============================================================
# LEARNED MOTION FIELD (where does a box go next? works around the turns)
# ============================================================

class FlowField:
    """Average observed velocity per image cell, learned from the moving boxes.

    The conveyor is a closed loop, so every pallet follows the same path; the field
    learns it (no coordinates are typed in).  Used to predict where a pallet that
    vanished for a moment must be now - also behind a turn.
    """

    def __init__(self):
        self.cells = {}   # (i, j) -> [vx, vy, n]

    def _key(self, x, y):
        return int(x // FLOW_CELL_PX), int(y // FLOW_CELL_PX)

    def teach(self, x, y, vx, vy):
        if (vx * vx + vy * vy) ** 0.5 < FLOW_MIN_SPEED:
            return
        cell = self.cells.setdefault(self._key(x, y), [vx, vy, 0])
        weight = 0.25 if cell[2] > 0 else 1.0
        cell[0] += weight * (vx - cell[0])
        cell[1] += weight * (vy - cell[1])
        cell[2] += 1

    def velocity(self, x, y):
        key = self._key(x, y)
        for radius in (0, 1):
            best = None
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    cell = self.cells.get((key[0] + di, key[1] + dj))
                    if cell and cell[2] >= 3 and (best is None or cell[2] > best[2]):
                        best = cell
            if best:
                return best[0], best[1]
        return None

    def predict(self, x, y, vx, vy, dt):
        """Position after dt seconds, following the field (falls back to constant velocity)."""
        remaining = dt
        while remaining > 1e-6:
            step = min(FLOW_STEP_SEC, remaining)
            field = self.velocity(x, y)
            if field is not None:
                vx, vy = field
            x += vx * step
            y += vy * step
            remaining -= step
        return x, y


# ============================================================
# PHYSICAL PALLETS (continuity chains) + NUMBERS 1..6
# ============================================================

class PhysicalIdentities:
    """Follows every physical pallet through tracker-id changes and carries RFID numbers.

    Every Phase 40 track belongs to a CHAIN (one chain = one physical pallet).  When a
    tracker id disappears and a NEW id is born close to where that pallet must be
    (mutual best match, clear margin, born after the old id vanished) the new id joins
    the same chain.  All pallets are followed - also the unnumbered ones - so that
    a neighbour can never be mistaken for a lost pallet.  A number belongs to one
    chain, a chain carries at most one number.  Doubt -> a new anonymous chain ("?").
    """

    def __init__(self, history):
        self.history = history
        self.chains = {}        # chain id -> dict
        self.of_track = {}      # tracker id -> chain id
        self.number_chain = {}  # number -> chain id
        self.next_chain = 1
        self.verified = 0
        self.conflicts = 0
        self.flow = FlowField()

    # ----- queries -------------------------------------------------------
    def number_of(self, tid):
        """The number SHOWN for a box: only a CONFIRMED one (read twice on the same pallet)."""
        cid = self.of_track.get(int(tid))
        if cid is None:
            return None
        if not self.chains[cid]["confirmed"] and not SHOW_PROVISIONAL:
            return None
        return self.chains[cid]["number"]

    def claimed_number(self, tid):
        """Number claimed for a box, confirmed or not (used for exclusion, never shown)."""
        cid = self.of_track.get(int(tid))
        return None if cid is None else self.chains[cid]["number"]

    def is_provisional(self, tid):
        cid = self.of_track.get(int(tid))
        return cid is not None and self.chains[cid]["number"] is not None and not self.chains[cid]["confirmed"]

    def certainty(self, tid):
        """0..1 certainty of the number claimed for a box (0 if none)."""
        cid = self.of_track.get(int(tid))
        if cid is None or self.chains[cid]["number"] is None:
            return 0.0
        return self.chains[cid]["conf"]

    def confirmed_count(self):
        return sum(1 for c in self.chains.values() if c["number"] is not None and c["confirmed"])

    # ----- chains -----------------------------------------------------------
    def _new_chain(self, tid, t, number=None):
        cid = self.next_chain
        self.next_chain += 1
        position = self.history.position_at(tid, t)
        self.chains[cid] = {
            "track": tid, "center": position[:2] if position else None,
            "vel": self.history.velocity(tid), "t_seen": t, "number": number,
            "conf": 1.0, "t_assigned": t, "confirmed": False,
        }
        self.of_track[tid] = cid
        return cid

    def _drop_chain(self, cid):
        chain = self.chains.pop(cid, None)
        if not chain:
            return
        if chain["number"] is not None and self.number_chain.get(chain["number"]) == cid:
            del self.number_chain[chain["number"]]
        for tid in [k for k, v in self.of_track.items() if v == cid]:
            del self.of_track[tid]

    # ----- RFID assignment (authoritative) -------------------------------------
    def assign(self, number, tid, t):
        """RFID says: the box `tid` is pallet `number`.  Returns the outcome text."""
        cid = self.of_track.get(tid)
        if cid is None:
            cid = self._new_chain(tid, t)
        chain = self.chains[cid]

        if chain["number"] == number:
            chain["conf"] = 1.0
            chain["t_assigned"] = t
            chain["confirmed"] = True
            self.verified += 1
            return "VERIFIED"

        outcome = "NEW"
        if chain["number"] is not None:
            if chain["confirmed"]:
                self.conflicts += 1
            outcome = f"CONFLICT (box was {chain['number']})"
            if self.number_chain.get(chain["number"]) == cid:
                del self.number_chain[chain["number"]]
            chain["number"] = None

        other = self.number_chain.get(number)
        if other is not None and other != cid:
            other_chain = self.chains.get(other)
            if other_chain is not None:
                if other_chain["confirmed"] and other_chain["track"] is not None \
                        and self.history.around(other_chain["track"], t, 1.0):
                    self.conflicts += 1
                    if not outcome.startswith("CONFLICT"):
                        outcome = "CONFLICT (number was on another visible box)"
                other_chain["number"] = None
                other_chain["confirmed"] = False

        chain["number"] = number
        chain["conf"] = PROVISIONAL_CONF
        chain["t_assigned"] = t
        chain["confirmed"] = False
        self.number_chain[number] = cid
        return outcome

    # ----- per-frame propagation -------------------------------------------------
    def update(self, t, live_tracks):
        """live_tracks: list of {"id","box"} currently returned by Phase 40."""
        live = {int(tr["id"]) for tr in live_tracks}

        # 1) chains whose track is still there
        for cid, chain in self.chains.items():
            tid = chain["track"]
            if tid is not None and tid in live:
                position = self.history.position_at(tid, t)
                if position:
                    chain["center"] = position[:2]
                    chain["vel"] = self.history.velocity(tid)
                    chain["t_seen"] = t
                    self.flow.teach(position[0], position[1], chain["vel"][0], chain["vel"][1])

        # 2) tracks without a chain, and chains that lost their track
        fresh = [tid for tid in live if tid not in self.of_track]
        coasting = [
            cid for cid, chain in self.chains.items()
            if (chain["track"] is None or chain["track"] not in live)
            and chain["center"] is not None
            and t - chain["t_seen"] <= COAST_MAX_SEC
        ]

        # 3) a fresh track may continue a coasting chain
        costs = {}
        for cid in coasting:
            chain = self.chains[cid]
            dt = t - chain["t_seen"]
            px, py = self.flow.predict(
                chain["center"][0], chain["center"][1], chain["vel"][0], chain["vel"][1], dt
            )
            for tid in fresh:
                born = self.history.span(tid)[0]
                if born is None or born < chain["t_seen"] - REACQ_BORN_SLACK:
                    continue        # existed while the old id was still there: another pallet
                position = self.history.position_at(tid, t)
                if position is None:
                    continue
                distance = ((position[0] - px) ** 2 + (position[1] - py) ** 2) ** 0.5
                costs[(cid, tid)] = distance / max(1.0, position[2])

        def best(pairs):
            ranked = sorted(pairs, key=lambda item: item[0])
            if not ranked:
                return None, 9e9
            return ranked[0], (ranked[1][0] if len(ranked) > 1 else 9e9)

        joined = set()
        for cid in coasting:
            numbered = self.chains[cid]["number"] is not None
            gate = REACQ_NUMBERED_GATE if numbered else REACQ_GATE
            margin = REACQ_NUMBERED_MARGIN if numbered else REACQ_MARGIN
            mine = [(c, tid) for (k, tid), c in costs.items() if k == cid and tid not in joined]
            first, second = best(mine)
            if first is None or first[0] > gate or second - first[0] < margin:
                continue
            tid = first[1]
            theirs = [(c, k) for (k, t2), c in costs.items() if t2 == tid]
            their_first, their_second = best(theirs)
            if their_first[1] != cid or their_second - their_first[0] < margin:
                continue
            chain = self.chains[cid]
            if chain["track"] is not None:
                self.of_track.pop(chain["track"], None)
            chain["track"] = tid
            chain["t_seen"] = t
            if chain["number"] is not None:
                chain["conf"] *= REACQ_CONF_FACTOR
            self.of_track[tid] = cid
            joined.add(tid)

        # 4) every other fresh track is a pallet of its own (anonymous until read)
        for tid in fresh:
            if tid not in joined:
                self._new_chain(tid, t)

        # 5) forget chains that coasted too long (a lost number stays "?" until the next read)
        for cid in [
            cid for cid, chain in self.chains.items()
            if (chain["track"] is None or chain["track"] not in live)
            and t - chain["t_seen"] > COAST_MAX_SEC
        ]:
            self._drop_chain(cid)


# ============================================================
# RFID EVENT -> PHYSICAL PALLET
# ============================================================

class EventResolver:
    def __init__(self, history, identities, zone, learn=False):
        self.history = history
        self.identities = identities
        self.zone = zone
        self.learn = learn
        self.pending = []       # open reads
        self.log_rows = []      # for CSV
        self.lines = []         # console lines of the latest resolutions
        self.learn_rows = {"top": [], "bottom": []}
        self.event_counter = 0
        self.last_read = None   # (number, text, time)
        self.lane_memory = defaultdict(lambda: {"top": 0, "bottom": 0})

    # ----- candidate search -------------------------------------------------------
    def candidates(self, t_event, split_y, frame_scale=(1.0, 1.0)):
        """Every box that existed around the read, position interpolated to the read time."""
        sx, sy = frame_scale
        found = []
        for tid in list(self.history.data):
            position = self.history.position_at(tid, t_event)
            if position is None:
                continue
            first, last = self.history.span(tid)
            cx, cy = position[0] / sx, position[1] / sy
            gap = 0.0 if first <= t_event <= last else min(abs(first - t_event), abs(t_event - last))
            found.append(
                {"id": tid, "cx": cx, "cy": cy, "lane": lane_of(cy * sy, split_y),
                 "in_zone": self.zone.contains(lane_of(cy * sy, split_y), cx),
                 "w": position[2], "gap": gap, "first": first, "last": last}
            )
        return self._dedupe(found)

    def _dedupe(self, found):
        """One pallet that changed tracker id (old id ended, new id confirmed later)
        must count ONCE, otherwise a single pallet looks like two candidates."""
        keep = []
        for item in sorted(found, key=lambda c: (c["gap"], c["id"])):
            twin = None
            for other in keep:
                distance = (
                    (item["cx"] - other["cx"]) ** 2 + (item["cy"] - other["cy"]) ** 2
                ) ** 0.5
                disjoint = item["first"] >= other["last"] - 0.1 or other["first"] >= item["last"] - 0.1
                if disjoint and distance < 0.6 * min(item["w"], other["w"]):
                    twin = other
                    break
            if twin is None:
                keep.append(item)
            elif (
                self.identities.claimed_number(item["id"]) is not None
                and self.identities.claimed_number(twin["id"]) is None
            ):
                keep[keep.index(twin)] = item   # keep the id that carries the number
        return keep

    def add(self, event):
        if event["type"] == "appear":
            self.event_counter += 1
            event = dict(event)
            event["n"] = self.event_counter
            event["deadline"] = event["time"] + EVENT_RESOLVE_SEC
            self.pending.append(event)
        elif event["type"] == "leave" and self.learn:
            self._learn_leave(event)

    # ----- called every frame ---------------------------------------------------------
    def process(self, t_now, split_y, frame_scale=(1.0, 1.0)):
        still = []
        for event in self.pending:
            outcome = self._try(event, split_y, frame_scale)
            if outcome is None and t_now < event["deadline"]:
                still.append(event)
                continue
            if outcome is None:
                outcome = self._final_failure(event, split_y, frame_scale)
            self._finish(event, outcome)
        self.pending = still

    def _try(self, event, split_y, frame_scale):
        # wait until at least one frame after the read exists (a box may only get
        # confirmed by the tracker a few frames later)
        if self.history.last_t < event["time"] + EVENT_SETTLE_SEC:
            return None

        found = self.candidates(event["time"], split_y, frame_scale)
        in_zone = [c for c in found if c["in_zone"]]
        number = event["number"]

        if self.learn:
            visible = [c for c in found if self.history.around(c["id"], event["time"], 0.3)]
            if len(visible) != 1:
                return {"kind": "LEARN_INVALID", "found": found, "chosen": None,
                        "reason": f"{len(visible)} pallets visible - run --learn with ONE pallet"}
            return {"kind": "LEARN", "found": found, "chosen": visible[0], "reason": "single pallet"}

        # a tag that has always been read on ONE lane is only looked for on that lane
        memory = self.lane_memory[number]
        if memory["top"] >= LANE_MEMORY_MIN and memory["bottom"] == 0:
            in_zone = [c for c in in_zone if c["lane"] == "top"]
        elif memory["bottom"] >= LANE_MEMORY_MIN and memory["top"] == 0:
            in_zone = [c for c in in_zone if c["lane"] == "bottom"]

        if len(in_zone) == 1:
            holder = self.identities.claimed_number(in_zone[0]["id"])
            if holder is not None and holder != number:
                certainty = self.identities.certainty(in_zone[0]["id"])
                if certainty >= OWNER_TRUST_CONF:
                    # that box carries another number with good certainty: the
                    # pallet just read is NOT this one (probably not detected)
                    return {"kind": "AMBIGUOUS", "found": found, "chosen": None,
                            "reason": f"only box in zone is pallet {holder} (certainty {certainty:.2f})"}
            return {"kind": "MAP", "found": found, "chosen": in_zone[0], "reason": "one box in zone"}

        if len(in_zone) > 1:
            # exclusion: a box that already carries a DIFFERENT, certain number is
            # not the pallet that was just read
            free = [
                c for c in in_zone
                if self.identities.claimed_number(c["id"]) in (None, number)
                or self.identities.certainty(c["id"]) < OWNER_TRUST_CONF
            ]
            if len(free) == 1:
                return {"kind": "MAP", "found": found, "chosen": free[0],
                        "reason": f"{len(in_zone)} in zone, others carry other numbers"}
            return None  # may resolve once more frames arrive; else AMBIGUOUS

        return None  # nothing in zone (yet)

    def _final_failure(self, event, split_y, frame_scale):
        found = self.candidates(event["time"], split_y, frame_scale)
        in_zone = [c for c in found if c["in_zone"]]
        memory = self.lane_memory[event["number"]]
        if memory["top"] >= LANE_MEMORY_MIN and memory["bottom"] == 0:
            in_zone = [c for c in in_zone if c["lane"] == "top"]
        elif memory["bottom"] >= LANE_MEMORY_MIN and memory["top"] == 0:
            in_zone = [c for c in in_zone if c["lane"] == "bottom"]
        if len(in_zone) > 1:
            return {"kind": "AMBIGUOUS", "found": found, "chosen": None,
                    "reason": f"{len(in_zone)} boxes in zone"}
        return {"kind": "NO_CANDIDATE", "found": found, "chosen": None,
                "reason": "no box in the read zone"}

    def _finish(self, event, outcome):
        number = event["number"]
        chosen = outcome["chosen"]
        text = outcome["kind"]

        if outcome["kind"] == "MAP":
            text = "MAP " + self.identities.assign(number, chosen["id"], event["time"])
            if not text.startswith("MAP CONFLICT"):
                self.lane_memory[number][chosen["lane"]] += 1
        elif outcome["kind"] == "LEARN":
            self.learn_rows[chosen["lane"]].append(
                {"number": number, "cx": chosen["cx"], "w": chosen["w"], "t": event["time"],
                 "id": chosen["id"], "leave_cx": None}
            )
            text = f"LEARN {chosen['lane']} cx={chosen['cx']:.0f}"

        self.last_read = (number, text, time.monotonic())
        near = sorted(outcome["found"], key=lambda c: abs(c["cx"] - 1250))[:4]
        summary = ", ".join(
            f"T{c['id']}:{c['lane'][0]}{c['cx']:.0f}{'*' if c['in_zone'] else ''}" for c in near
        )
        line = (
            f"RFID1 #{event['n']:<3} tag={number}  {text}"
            + (f"  box=T{chosen['id']} ({chosen['lane']} cx={chosen['cx']:.0f})" if chosen else "")
            + f"  [{outcome['reason']}]  seen: {summary}"
        )
        self.lines.append(line)
        print(line, flush=True)

        self.log_rows.append(
            [
                event["n"], number, round(event["time"], 3), outcome["kind"], text,
                chosen["id"] if chosen else "", chosen["lane"] if chosen else "",
                round(chosen["cx"], 1) if chosen else "", outcome["reason"],
                ";".join(
                    f"T{c['id']}/{c['lane']}/{c['cx']:.0f}/{int(c['in_zone'])}"
                    for c in outcome["found"]
                ),
                self.identities.verified, self.identities.conflicts,
            ]
        )

    # ----- learn mode -----------------------------------------------------------------
    def _learn_leave(self, event):
        for lane in ("top", "bottom"):
            for row in reversed(self.learn_rows[lane]):
                if row["number"] == event["number"] and row["leave_cx"] is None:
                    position = self.history.position_at(row["id"], event["time"])
                    row["leave_cx"] = position[0] if position else None
                    return

    def learned_zone(self, base_zone):
        """New zone for every lane with >= LEARN_MIN_EVENTS valid reads."""
        bounds = {k: list(v) for k, v in base_zone.bounds.items()}
        changed = []
        for lane, rows in self.learn_rows.items():
            if len(rows) < LEARN_MIN_EVENTS:
                continue
            xs = [r["cx"] for r in rows]
            margin = LEARN_MARGIN_RATIO * (sum(r["w"] for r in rows) / len(rows))
            bounds[lane] = [min(xs) - margin, max(xs) + margin]
            changed.append(lane)
        result = ReadZone(
            {"top": bounds["top"], "bottom": bounds["bottom"],
             "source": "measured with --learn (" + ",".join(changed) + ")"}
        )
        return result, changed


# ============================================================
# DISPLAY
# ============================================================

def status_of(data):
    label = str(data.get("label", "")).upper()
    for state in ("REJECTED", "ACCEPTED", "EMPTY", "CHECKING"):
        if state in label:
            return state
    return "UNKNOWN"


STATE_COLORS = {
    "EMPTY": (0, 255, 0),
    "ACCEPTED": (255, 255, 0),
    "REJECTED": (0, 0, 255),
    "CHECKING": (0, 255, 255),
    "UNKNOWN": (255, 255, 255),
}


def pallet_label(number, status, track_id=None, debug=False, provisional=None):
    """'3 ACCEPTED' once identified, '? ACCEPTED' otherwise (never a guess).

    A number that no second read has confirmed yet is NOT shown ('?'); with --debug it
    appears as '(3?)' so the build-up can be watched."""
    text = f"{number if number is not None else '?'} {status}"
    if debug and number is None and provisional is not None:
        text = f"(?{provisional}) {status}"
    if debug and track_id is not None:
        text += f" [T{track_id}]"
    return text


class App:
    def __init__(self, event_queue, zone, learn, debug):
        self.events = event_queue
        self.zone = zone
        self.learn = learn
        self.debug = debug
        self.history = TrackHistory()
        self.identities = PhysicalIdentities(self.history)
        self.resolver = EventResolver(self.history, self.identities, zone, learn=learn)
        self.last_image = None
        self.split_y = None

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_file = OUT_DIR / (
            ("learn_" if learn else "phase49_") + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        )
        with self.csv_file.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                ["event", "tag", "read_monotonic", "result", "text", "box_track", "lane",
                 "cx", "reason", "boxes_seen(track/lane/cx/in_zone)", "verified", "conflicts"]
            )
        self.written = 0

    def flush_csv(self):
        rows = self.resolver.log_rows[self.written:]
        if not rows:
            return
        with self.csv_file.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)
        self.written += len(rows)

    def ingest(self, t, live_tracks, scale=(1.0, 1.0)):
        """One frame: history -> RFID events -> identities (no drawing)."""
        self.history.update(t, live_tracks)
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.resolver.add(event)
        self.resolver.process(t, self.split_y, scale)
        self.identities.update(t, live_tracks)
        self.flush_csv()

    def display(self, scope, acquired_t):
        now = time.monotonic()
        t = acquired_t if acquired_t is not None else now

        raw = scope.get("symbol_frame")
        if raw is None:
            raw = scope.get("frame")
        if raw is None:
            raise RuntimeError("Phase40 frame unavailable")
        image = raw.copy()
        height, width = image.shape[:2]
        scale = (width / REF_W, height / REF_H)

        live_tracks = [
            {"id": int(tr["track_id"]), "box": tuple(int(v) for v in tr["box"])}
            for tr in scope.get("confirmed_tracks", [])
        ]

        fallback = height * 0.65
        polygon = scope.get("inner_polygon")
        self.split_y = split_from_polygon(polygon, fallback) if polygon is not None else fallback

        self.ingest(t, live_tracks, scale)

        # draw
        for name, color in (("outer_polygon", (0, 255, 0)), ("inner_polygon", (0, 255, 255))):
            poly = scope.get(name)
            if poly is not None:
                cv2.polylines(image, [poly], True, color, 2)

        if self.debug:
            for lane, (low, high) in self.zone.bounds.items():
                y = int(self.split_y + (-120 if lane == "top" else 120))
                cv2.line(image, (int(low * scale[0]), y), (int(high * scale[0]), y), (255, 0, 255), 3)

        for tid, data in scope.get("visible_track_cache", {}).items():
            x1, y1, x2, y2 = [int(v) for v in data["box"]]
            status = status_of(data)
            color = STATE_COLORS.get(status, (255, 255, 255))
            number = self.identities.number_of(int(tid))
            claim = self.identities.claimed_number(int(tid))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if data.get("missed", 0) else 4)
            cv2.putText(
                image,
                pallet_label(number, status, tid, self.debug, claim if number is None else None),
                (max(0, x1), max(25, y1 - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        if self.learn:
            title = (
                "RFID1 LEARN (ONE pallet on the belt) | top reads "
                f"{len(self.resolver.learn_rows['top'])} | bottom reads "
                f"{len(self.resolver.learn_rows['bottom'])} | Q = save zone and quit"
            )
        else:
            last = self.resolver.last_read
            recent = f"last tag {last[0]}: {last[1]}" if last and now - last[2] < 4.0 else "waiting for tag"
            title = (
                f"RFID1 PHYSICAL TRACKING | numbers confirmed {self.identities.confirmed_count()}/6 "
                f"| verified {self.identities.verified} | conflicts {self.identities.conflicts} | {recent}"
            )
        cv2.rectangle(image, (5, 5), (1500, 52), (0, 0, 0), -1)
        cv2.putText(image, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.last_image = image
        return image


# ============================================================
# MAIN
# ============================================================

class CameraClock:
    """Wraps the existing camera so every frame gets its acquisition time."""

    def __init__(self, original):
        self.original = original
        self.acquired_at = None

    def read(self):
        result = self.original.read()
        self.acquired_at = time.monotonic() if result[0] else None
        return result

    def __getattr__(self, name):
        return getattr(self.original, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn", action="store_true", help="measure the read zone with ONE pallet")
    parser.add_argument("--debug", action="store_true", help="show tracker ids and the read zone")
    parser.add_argument(
        "--show-provisional", action="store_true",
        help="also show numbers that were read only once (faster, less safe)",
    )
    arguments = parser.parse_args()
    global SHOW_PROVISIONAL
    SHOW_PROVISIONAL = arguments.show_provisional

    if not LIVE_PATH.is_file():
        raise FileNotFoundError(LIVE_PATH)

    zone = ReadZone.load()
    print("RFID1 only. RFID2: NOT READ OR USED. PLC: read only.")
    print("Phase 40 detection and classification: UNCHANGED")
    print(f"Read zone ({zone.source}): top {zone.bounds['top']}  bottom {zone.bounds['bottom']}")
    if arguments.learn:
        print("LEARN MODE: put exactly ONE pallet on the conveyor; 2+ laps; Q saves the zone.")

    sys.path.insert(0, str(LIVE_PATH.parent))
    spec = importlib.util.spec_from_file_location("crest_conference_live49", str(LIVE_PATH))
    live = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = live
    spec.loader.exec_module(live)

    events = queue.Queue()
    stop = threading.Event()
    app = App(events, zone, arguments.learn, arguments.debug)

    real_open = live.open_camera
    real_show = cv2.imshow
    clock = {"obj": None}

    def open_camera_wrapper():
        cap, index = real_open()
        clock["obj"] = CameraClock(cap)
        return clock["obj"], index

    def show_wrapper(name, frame):
        if "CREST" not in str(name).upper():
            return real_show(name, frame)
        scope = sys._getframe(1).f_locals
        acquired = clock["obj"].acquired_at if clock["obj"] else None
        return real_show(name, app.display(scope, acquired))

    live.open_camera = open_camera_wrapper
    cv2.imshow = show_wrapper

    worker = threading.Thread(target=rfid1_reader, args=(events, stop), daemon=True)
    worker.start()

    print()
    print("START: let the sensor become clear, then run the conveyor normally.")
    print()

    try:
        live.main()
    finally:
        stop.set()
        worker.join(timeout=2)
        live.open_camera = real_open
        cv2.imshow = real_show
        app.flush_csv()

        if arguments.learn:
            learned, changed = app.resolver.learned_zone(zone)
            if changed:
                learned.save()
                print("Read zone saved:", ZONE_FILE, "lanes:", changed)
                print("  top", learned.bounds["top"], " bottom", learned.bounds["bottom"])
            else:
                print(
                    "Read zone NOT changed: need at least "
                    f"{LEARN_MIN_EVENTS} valid single-pallet reads on a lane."
                )
        print("Evidence CSV:", app.csv_file)


if __name__ == "__main__":
    main()
