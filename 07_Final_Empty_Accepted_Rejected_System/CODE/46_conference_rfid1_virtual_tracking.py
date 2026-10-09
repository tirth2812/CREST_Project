
"""CREST Phase 46: RFID1 virtual gate + physical pallet tracker.

TEST ONLY. Original Conference Phase40 is not edited.
RFID1 PLC words [96..99] only. No PLC writes.
No conveyor stopping or piston commands.

A virtual gate selects one moving pallet at an RFID event.
Physical identities 1-6 are then tracked geometrically,
independently of temporary vision tracker IDs.
"""

import csv
import importlib.util
import math
import queue
import sys
import threading
import time

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
    r"\07_Final_Empty_Accepted_Rejected_System"
)

PHASE40 = (
    ROOT
    / "CODE"
    / "40_conference_100_pallet_live.py"
)

OUT = (
    ROOT
    / "RESULTS"
    / "46_RFID1_Virtual_Gate"
)


# ============================================================
# RFID1 ONLY
# ============================================================

PLC_IP = "192.168.82.231"

TAGS = [
    f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]"
    for i in range(96, 100)
]

TAG_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}

ZERO = (0, 0, 0, 0)


# ============================================================
# VIRTUAL RFID1 STATION
#
# Calibrated approximately from Phase45 observations.
# Match pallet's LEFT bounding-box edge, not center.
# ============================================================

GATE_LEFT_MIN = 1100
GATE_LEFT_MAX = 1275

GATE_CENTER_Y_MIN = 460
GATE_CENTER_Y_MAX = 630

FRAME_TIME_TOLERANCE = 0.38
EVENT_SETTLE_SECONDS = 0.48

MAX_RECOVERY_GAP = 1.65


# ============================================================
# VIRTUAL CONVEYOR PATH
#
# TOP: RIGHT -> LEFT
# LEFT TURN: TOP -> BOTTOM
# BOTTOM: LEFT -> RIGHT
# RIGHT TURN: BOTTOM -> TOP
# ============================================================

PATH = [
    (1600, 548),
    (1360, 548),
    (1100, 550),
    (820, 555),
    (550, 567),
    (370, 594),
    (270, 685),
    (250, 775),
    (380, 851),
    (640, 859),
    (960, 864),
    (1250, 866),
    (1510, 857),
    (1682, 789),
    (1728, 680),
    (1600, 548),
]


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def geometry(box):

    x1, y1, x2, y2 = box

    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
    )


def area(box):

    return (
        max(1, box[2] - box[0])
        * max(1, box[3] - box[1])
    )


def state_from_label(label):

    label = str(label).upper()

    for word in (
        "REJECTED",
        "ACCEPTED",
        "EMPTY",
        "CHECKING",
    ):

        if word in label:
            return word

    return "UNKNOWN"


# ============================================================
# VIRTUAL CONVEYOR PATH PROJECTION
# ============================================================

class BeltPath:

    def __init__(self, points):

        self.segments = []
        self.total = 0.0

        for a, b in zip(
            points[:-1],
            points[1:]
        ):

            length = math.hypot(
                b[0] - a[0],
                b[1] - a[1]
            )

            self.segments.append((
                a,
                b,
                length,
                self.total,
            ))

            self.total += length

    def project(self, box):

        px, py = geometry(box)

        best = None

        for (
            a,
            b,
            length,
            start
        ) in self.segments:

            dx = b[0] - a[0]
            dy = b[1] - a[1]

            fraction = (
                (px - a[0]) * dx
                + (py - a[1]) * dy
            ) / (length * length)

            fraction = max(
                0.0,
                min(1.0, fraction)
            )

            x = a[0] + fraction * dx
            y = a[1] + fraction * dy

            lateral = math.hypot(
                px - x,
                py - y
            )

            if (
                best is None
                or lateral < best[1]
            ):

                position = (
                    start
                    + length * fraction
                ) % self.total

                best = (
                    position,
                    lateral
                )

        return best

    def forward(self, newer, older):

        distance = (
            newer - older
        ) % self.total

        if (
            distance
            > self.total - 28
        ):

            distance -= self.total

        return distance


# ============================================================
# CAMERA TIMESTAMP
# ============================================================

class CameraClock:

    def __init__(self, cap):

        self.cap = cap
        self.acquired = None

    def read(self):

        result = self.cap.read()

        self.acquired = (
            time.monotonic()
            if result[0]
            else None
        )

        return result

    def __getattr__(self, key):

        return getattr(
            self.cap,
            key
        )


# ============================================================
# RFID1 PLC READER
# ============================================================

def read_rfid1(events, stop):

    from pylogix import PLC

    armed = False
    zero_count = 0

    candidate = None
    candidate_since = 0.0
    valid_count = 0

    emitted = False

    while not stop.is_set():

        try:

            with PLC() as plc:

                plc.IPAddress = PLC_IP
                plc.SocketTimeout = 3

                print(
                    "RFID1 PLC: reading "
                    "words 96-99 only",
                    flush=True
                )

                while not stop.is_set():

                    values = []

                    for tag in TAGS:

                        reply = plc.Read(tag)

                        if (
                            reply.Status
                            != "Success"
                        ):

                            raise RuntimeError(
                                f"RFID1 {tag}: "
                                f"{reply.Status}"
                            )

                        values.append(
                            int(reply.Value)
                        )

                    words = tuple(values)

                    now = time.monotonic()

                    # --------------------------------
                    # NO TAG
                    # --------------------------------

                    if words == ZERO:

                        zero_count += 1

                        candidate = None
                        valid_count = 0

                        if zero_count >= 2:

                            armed = True
                            emitted = False

                    # --------------------------------
                    # VALID RFID
                    # --------------------------------

                    elif words in TAG_MAP:

                        zero_count = 0

                        if (
                            not armed
                            or emitted
                        ):

                            stop.wait(0.025)
                            continue

                        if candidate != words:

                            candidate = words

                            candidate_since = now
                            valid_count = 1

                        else:

                            valid_count += 1

                        if valid_count >= 2:

                            events.put({
                                "number":
                                    TAG_MAP[words],
                                "time":
                                    candidate_since,
                                "raw":
                                    words,
                            })

                            emitted = True

                    # --------------------------------
                    # INCOMPLETE RFID WORDS
                    # --------------------------------

                    else:

                        zero_count = 0
                        candidate = None
                        valid_count = 0

                    stop.wait(0.025)

        except Exception as error:

            print(
                "RFID1 PLC retry:",
                error,
                flush=True
            )

            stop.wait(1.0)


# ============================================================
# INDEPENDENT PHYSICAL PALLET TRACKER
#
# Physical IDs are 1,2,3,4,5,6.
# They are NOT temporary YOLO tracker IDs.
# ============================================================

class PhysicalTracker:

    def __init__(self):

        self.path = BeltPath(PATH)

        self.states = {}

        self.current = {}

        self.lost_logged = set()

    def advance(self, tracks, timestamp):

        self.current = {}

        pairs = []

        # ----------------------------------------
        # Compare each known physical pallet
        # against current detected boxes.
        # ----------------------------------------

        for (
            number,
            state
        ) in self.states.items():

            dt = (
                timestamp
                - state["t"]
            )

            if (
                dt < 0
                or dt > MAX_RECOVERY_GAP
            ):
                continue

            for (
                index,
                det
            ) in enumerate(tracks):

                s = det["s"]
                lateral = det["lateral"]

                if lateral > 145:
                    continue

                ratio = (
                    min(
                        area(det["box"]),
                        area(state["box"])
                    )
                    /
                    max(
                        area(det["box"]),
                        area(state["box"])
                    )
                )

                if ratio < 0.55:
                    continue

                delta = (
                    self.path.forward(
                        s,
                        state["s"]
                    )
                )

                # Reject physically implausible
                # motion along the conveyor.

                if (
                    delta < -30
                    or
                    delta > 26 + 350 * dt
                ):
                    continue

                prediction = (
                    min(
                        320,
                        state["speed"]
                    )
                    * dt
                )

                score = (
                    abs(
                        delta - prediction
                    )
                    + 0.12 * lateral
                )

                pairs.append((
                    score,
                    number,
                    index,
                    delta,
                    dt,
                ))

        # ----------------------------------------
        # One-to-one global association.
        # ----------------------------------------

        by_number = {}
        by_detection = {}

        for pair in pairs:

            by_number.setdefault(
                pair[1],
                []
            ).append(pair)

            by_detection.setdefault(
                pair[2],
                []
            ).append(pair)

        groups = (
            list(by_number.values())
            + list(by_detection.values())
        )

        for group in groups:

            group.sort(
                key=lambda p: p[0]
            )

        used_numbers = set()
        used_detections = set()

        for (
            score,
            number,
            index,
            delta,
            dt
        ) in sorted(pairs):

            if (
                number in used_numbers
                or index in used_detections
            ):
                continue

            # Both sides must agree that this
            # is their best match.

            if (
                by_number[number][0][2]
                != index
            ):
                continue

            if (
                by_detection[index][0][1]
                != number
            ):
                continue

            # Reject nearby ambiguous matches.

            if (
                len(by_number[number]) > 1
                and
                by_number[number][1][0]
                - score < 27
            ):
                continue

            if (
                len(by_detection[index]) > 1
                and
                by_detection[index][1][0]
                - score < 27
            ):
                continue

            det = tracks[index]

            state = self.states[number]

            if dt >= 0.02:

                actual_speed = max(
                    0.0,
                    min(
                        340.0,
                        delta / dt
                    )
                )

                state["speed"] = (
                    0.55 * state["speed"]
                    + 0.45 * actual_speed
                )

            state.update({
                "s": det["s"],
                "box": det["box"],
                "t": timestamp,
            })

            self.current[index] = number

            used_numbers.add(number)
            used_detections.add(index)

            self.lost_logged.discard(
                number
            )

        return used_detections

    # ========================================================
    # INITIALIZE IDENTITY FROM RFID1
    # ========================================================

    def anchor(
        self,
        number,
        det,
        timestamp,
        log
    ):

        # Avoid placing two different physical
        # numbers on almost the same location.

        for (
            other_number,
            state
        ) in self.states.items():

            if other_number == number:
                continue

            if (
                timestamp - state["t"]
                > 1.0
            ):
                continue

            distance = abs(
                self.path.forward(
                    det["s"],
                    state["s"]
                )
            )

            if distance < 75:

                log(
                    "IDENTITY_CONFLICT",
                    number,
                    (
                        "near_existing_physical="
                        f"{other_number}; "
                        "separation="
                        f"{distance:.0f}px"
                    )
                )

                return False

        was_known = (
            number in self.states
        )

        self.states[number] = {
            "s": det["s"],
            "box": det["box"],
            "t": timestamp,
            "speed": 180.0,
        }

        self.lost_logged.discard(
            number
        )

        event = (
            "REANCHOR"
            if was_known
            else "MAP"
        )

        log(
            event,
            number,
            (
                "virtual_gate_xleft="
                f"{det['box'][0]} "
                f"box={det['box']}"
            )
        )

        return True


# ============================================================
# RFID1 APPLICATION
# ============================================================

class RFID1App:

    def __init__(self, events):

        self.events = events

        self.physical = (
            PhysicalTracker()
        )

        self.history = deque(
            maxlen=45
        )

        self.pending = deque()

        self.last_image = None

        self.latest_read = None
        self.latest_read_time = 0.0

        filename = (
            "rfid1_virtual_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".csv"
        )

        self.csv_file = (
            OUT / filename
        )

        with self.csv_file.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                "time",
                "event",
                "RFID_number",
                "detail",
            ])

    # ========================================================
    # LOG EVENTS
    # ========================================================

    def log(
        self,
        event,
        number="",
        detail=""
    ):

        message = (
            f"RFID1 {event}: "
            f"pallet={number} "
            f"{detail}"
        )

        print(
            message,
            flush=True
        )

        with self.csv_file.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                event,
                number,
                detail,
            ])

    # ========================================================
    # COLLECT YOLO BOXES
    # ========================================================

    def read_track_boxes(self, scope):

        tracks = []

        for track in scope.get(
            "confirmed_tracks",
            ()
        ):

            box = tuple(
                int(v)
                for v in track["box"]
            )

            s, lateral = (
                self.physical.path.project(
                    box
                )
            )

            tracks.append({
                "box": box,
                "s": s,
                "lateral": lateral,
                "internal_id": int(
                    track["track_id"]
                ),
                "vx": float(
                    track.get(
                        "velocity_x",
                        0
                    )
                ),
            })

        return tracks

    # ========================================================
    # RFID1-TO-VIRTUAL-GATE MATCH
    # ========================================================

    def handle_read(self, event):

        number = event["number"]

        if not self.history:

            self.log(
                "NO_CAMERA_HISTORY",
                number
            )

            return

        # Select one camera snapshot nearest
        # the actual RFID activation time.

        closest = min(
            self.history,
            key=lambda entry: abs(
                entry["time"]
                - event["time"]
            )
        )

        offset = (
            closest["time"]
            - event["time"]
        )

        if (
            abs(offset)
            > FRAME_TIME_TOLERANCE
        ):

            self.log(
                "CAMERA_TIME_GAP",
                number,
                (
                    "nearest_frame_dt="
                    f"{offset:+.3f}s"
                )
            )

            return

        candidates = []

        for det in closest["tracks"]:

            x1, y1, x2, y2 = (
                det["box"]
            )

            _, cy = geometry(
                det["box"]
            )

            # Top conveyor only.
            if not (
                GATE_CENTER_Y_MIN
                <= cy
                <= GATE_CENTER_Y_MAX
            ):
                continue

            # Virtual station checks the
            # leading/left bounding-box edge.
            if not (
                GATE_LEFT_MIN
                <= x1
                <= GATE_LEFT_MAX
            ):
                continue

            # Upper conveyor moves right
            # to left. Allow stopped motion
            # but reject right-moving tracks.
            if det["vx"] > 3.0:
                continue

            candidates.append(det)

        # ----------------------------------------
        # ZERO OR MULTIPLE CANDIDATES
        # ----------------------------------------

        if len(candidates) != 1:

            nearby = sorted(
                (
                    (
                        abs(
                            d["box"][0]
                            - GATE_LEFT_MIN
                        ),
                        d["box"],
                    )
                    for d in closest["tracks"]
                    if (
                        GATE_CENTER_Y_MIN
                        <= geometry(
                            d["box"]
                        )[1]
                        <= GATE_CENTER_Y_MAX
                    )
                ),
                key=lambda pair: pair[0]
            )[:3]

            reason = (
                "NO_GATE_MATCH"
                if not candidates
                else "AMBIGUOUS_GATE"
            )

            self.log(
                reason,
                number,
                (
                    "candidates="
                    f"{[c['box'] for c in candidates]}; "
                    "nearest_top_boxes="
                    f"{nearby}; "
                    "frame_dt="
                    f"{offset:+.3f}s"
                )
            )

            return

        candidate = candidates[0]

        # Physical identity is anchored
        # directly to position along the belt,
        # not to the vision track ID.

        self.physical.anchor(
            number,
            candidate,
            closest["time"],
            self.log
        )

    # ========================================================
    # PROCESS EVERY CAMERA FRAME
    # ========================================================

    def frame(
        self,
        scope,
        acquired_time
    ):

        now = time.monotonic()

        t = (
            acquired_time
            if acquired_time is not None
            else now
        )

        detections = (
            self.read_track_boxes(
                scope
            )
        )

        # Store recent real camera positions.
        self.history.append({
            "time": t,
            "tracks": detections,
        })

        # Follow already-identified physical
        # pallets using motion and geometry.
        self.physical.advance(
            detections,
            t
        )

        # ----------------------------------------
        # RECEIVE RFID1 READINGS
        # ----------------------------------------

        while True:

            try:

                event = (
                    self.events.get_nowait()
                )

            except queue.Empty:
                break

            self.pending.append(event)

            self.latest_read = (
                event["number"]
            )

            self.latest_read_time = now

            self.log(
                "READ",
                event["number"],
                str(event["raw"])
            )

        # Give the camera a short opportunity
        # to capture frames around the read.

        while (
            self.pending
            and
            now - self.pending[0]["time"]
            >= EVENT_SETTLE_SECONDS
        ):

            event = (
                self.pending.popleft()
            )

            self.handle_read(
                event
            )

        # ====================================================
        # DRAW CAMERA DISPLAY
        # ====================================================

        source = scope.get(
            "symbol_frame"
        )

        if source is None:

            source = scope.get(
                "frame"
            )

        if source is None:

            raise RuntimeError(
                "Phase40 camera frame unavailable"
            )

        display = source.copy()

        # Original conveyor boundary.
        for name, color in (
            (
                "outer_polygon",
                (0, 255, 0)
            ),
            (
                "inner_polygon",
                (0, 255, 255)
            ),
        ):

            polygon = scope.get(name)

            if polygon is not None:

                cv2.polylines(
                    display,
                    [polygon],
                    True,
                    color,
                    2
                )

        # ----------------------------------------
        # VIRTUAL RFID1 GATE
        # ----------------------------------------

        cv2.rectangle(
            display,
            (
                GATE_LEFT_MIN,
                GATE_CENTER_Y_MIN
            ),
            (
                GATE_LEFT_MAX,
                GATE_CENTER_Y_MAX
            ),
            (0, 165, 255),
            2
        )

        cv2.putText(
            display,
            "RFID1 VIRTUAL GATE",
            (
                GATE_LEFT_MIN - 5,
                GATE_CENTER_Y_MIN - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (0, 165, 255),
            2
        )

        # ----------------------------------------
        # PHYSICAL PALLET LABELS
        # ----------------------------------------

        cache = scope.get(
            "visible_track_cache",
            {}
        )

        mapped = 0

        for (
            index,
            det
        ) in enumerate(detections):

            internal_id = (
                det["internal_id"]
            )

            data = cache.get(
                internal_id,
                {}
            )

            status = (
                state_from_label(
                    data.get(
                        "label",
                        ""
                    )
                )
            )

            physical_id = (
                self.physical.current.get(
                    index
                )
            )

            if physical_id is not None:
                mapped += 1

            identity = (
                str(physical_id)
                if physical_id is not None
                else "?"
            )

            label = (
                f"{identity} {status}"
            )

            color = data.get(
                "color",
                (255, 255, 255)
            )

            x1, y1, x2, y2 = (
                det["box"]
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                display,
                label,
                (
                    max(0, x1),
                    max(24, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                color,
                2
            )

        # ----------------------------------------
        # HELD MODEL BOXES
        #
        # No physical identity assigned to
        # missing/held detections.
        # ----------------------------------------

        current_ids = {
            det["internal_id"]
            for det in detections
        }

        for (
            track_id,
            data
        ) in cache.items():

            if int(track_id) in current_ids:
                continue

            x1, y1, x2, y2 = (
                int(value)
                for value in data["box"]
            )

            color = data.get(
                "color",
                (255, 255, 255)
            )

            status = state_from_label(
                data.get("label", "")
            )

            label = f"? {status}"

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                1
            )

            cv2.putText(
                display,
                label,
                (
                    max(0, x1),
                    max(24, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                color,
                1
            )

        # ----------------------------------------
        # STATUS PANEL
        # ----------------------------------------

        cv2.rectangle(
            display,
            (7, 7),
            (980, 83),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            (
                "RFID1 VIRTUAL TEST | "
                f"MAPPED {mapped}/{len(detections)}"
            ),
            (18, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.73,
            (255, 255, 255),
            2
        )

        latest = (
            str(self.latest_read)
            if (
                self.latest_read is not None
                and
                now - self.latest_read_time
                < 3.0
            )
            else "-"
        )

        cv2.putText(
            display,
            (
                f"RFID1 LAST: {latest} "
                "| ? = NO VERIFIED MATCH "
                "| S save | Q quit"
            ),
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        self.last_image = display

        return display


# ============================================================
# MAIN
# ============================================================

def main():

    if not PHASE40.is_file():

        raise FileNotFoundError(
            "Missing original Phase40: "
            f"{PHASE40}"
        )

    from pylogix import PLC

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    sys.path.insert(
        0,
        str(PHASE40.parent)
    )

    spec = (
        importlib.util.spec_from_file_location(
            "crest_phase40_rfid_virtual",
            str(PHASE40)
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Could not import Conference Phase40"
        )

    live = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[spec.name] = live

    spec.loader.exec_module(
        live
    )

    events = queue.Queue()
    stop = threading.Event()

    app = RFID1App(events)

    real_open = live.open_camera
    real_show = cv2.imshow
    real_key = cv2.waitKey

    clock = None

    # ----------------------------------------
    # CAMERA TIMESTAMP WRAPPER
    # ----------------------------------------

    def open_wrapper():

        nonlocal clock

        capture, index = (
            real_open()
        )

        clock = CameraClock(
            capture
        )

        return clock, index

    # ----------------------------------------
    # LIVE DISPLAY WRAPPER
    # ----------------------------------------

    def show_wrapper(
        window,
        frame
    ):

        if (
            "CREST"
            not in str(window).upper()
        ):

            return real_show(
                window,
                frame
            )

        variables = (
            sys._getframe(1).f_locals
        )

        acquired = (
            clock.acquired
            if clock is not None
            else None
        )

        return real_show(
            window,
            app.frame(
                variables,
                acquired
            )
        )

    # ----------------------------------------
    # SCREENSHOT HANDLING
    # ----------------------------------------

    def key_wrapper(delay=0):

        key = real_key(delay)

        if (
            key & 0xFF == ord("s")
            and
            app.last_image is not None
        ):

            filename = (
                "RFID1_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                + ".png"
            )

            path = OUT / filename

            cv2.imwrite(
                str(path),
                app.last_image
            )

            print(
                "Saved:",
                path,
                flush=True
            )

            return 255

        return key

    # ----------------------------------------
    # TEMPORARY WRAPPERS
    # ----------------------------------------

    live.open_camera = (
        open_wrapper
    )

    cv2.imshow = (
        show_wrapper
    )

    cv2.waitKey = (
        key_wrapper
    )

    worker = threading.Thread(
        target=read_rfid1,
        args=(events, stop),
        daemon=True
    )

    worker.start()

    print("=" * 58)
    print(
        "CREST PHASE 46 - "
        "RFID1 VIRTUAL MOVING PALLET TEST"
    )
    print("=" * 58)
    print(
        "RFID1 only; RFID2 not used"
    )
    print(
        "Phase40 vision and classification: "
        "UNCHANGED"
    )
    print(
        "Physical IDs are independent "
        "of temporary track IDs"
    )
    print(
        "No PLC writes or conveyor "
        "stop requests"
    )
    print(
        "Q quit | S screenshot | "
        "R original model recheck"
    )
    print("=" * 58, flush=True)

    try:

        live.main()

    finally:

        stop.set()

        worker.join(
            timeout=2.0
        )

        live.open_camera = real_open
        cv2.imshow = real_show
        cv2.waitKey = real_key

        print(
            "RFID1 events log:",
            app.csv_file
        )


if __name__ == "__main__":
    main()
