
"""CREST Phase 47: RFID1 lower-lane virtual-gate identity test.

Runs the original Conference Phase40 system unchanged.

RFID1 only: PLC words [96..99].
RFID2 is not accessed.
PLC READ ONLY. No conveyor control or piston commands.

Physical IDs 1..6 are tracked geometrically.
No RFID identity is copied through temporary vision IDs.

All physical mappings remain experimental.
Unknown physical identity is displayed as ?.
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

LIVE_FILE = (
    ROOT
    / "CODE"
    / "40_conference_100_pallet_live.py"
)

OUT_DIR = (
    ROOT
    / "RESULTS"
    / "47_RFID1_Lower_Lane_Test"
)


# ============================================================
# RFID1 CONFIGURATION
# ============================================================

PLC_IP = "192.168.82.231"

PLC_TAGS = [
    (
        "IO_Link_Master_ST1_to_ST4:"
        f"I1.Data[{i}]"
    )
    for i in range(96, 100)
]

ZERO = (0, 0, 0, 0)

TAG_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}


# ============================================================
# NEW VIRTUAL RFID1 GATE: LOWER CONVEYOR
#
# Previous Phase46 gate: Y = 460..630 (WRONG LANE).
# New Phase47 gate:     Y = 720..970 (LOWER LANE).
#
# Lower conveyor moves LEFT TO RIGHT.
#
# X=1153 is provisional, based on the RFID1
# reference position, and requires physical validation.
# ============================================================

GATE_X = 1153

GATE_Y_MIN = 720
GATE_Y_MAX = 970

GATE_EDGE_TOLERANCE = 48

TIME_TOLERANCE = 0.48
EVENT_WAIT = 0.54

MAX_TRACK_GAP = 1.30


# ============================================================
# CONVEYOR PATH
# ============================================================

BELT_POINTS = [
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
# HELPERS
# ============================================================

def center(box):

    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def status(data):

    label = str(
        data.get("label", "")
    ).upper()

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
# CAMERA TIMESTAMPS
# ============================================================

class TimestampCamera:

    def __init__(self, camera):

        self.camera = camera
        self.acquired = None

    def read(self):

        result = self.camera.read()

        self.acquired = (
            time.monotonic()
            if result[0]
            else None
        )

        return result

    def __getattr__(self, attribute):

        return getattr(
            self.camera,
            attribute
        )


# ============================================================
# RFID1-ONLY READER
# ============================================================

def poll_rfid1(events, stop):

    from pylogix import PLC

    armed = False
    zeros = 0

    candidate = None
    consecutive = 0
    onset = 0.0

    emitted = False
    last_error = 0.0

    while not stop.is_set():

        try:

            with PLC() as plc:

                plc.IPAddress = PLC_IP
                plc.SocketTimeout = 3

                print(
                    "RFID1 ONLY | PLC",
                    PLC_IP,
                    "| words 96-99",
                    flush=True
                )

                while not stop.is_set():

                    parts = []

                    for tag in PLC_TAGS:

                        reply = plc.Read(tag)

                        if (
                            reply.Status
                            != "Success"
                        ):

                            raise RuntimeError(
                                f"{tag}: {reply.Status}"
                            )

                        parts.append(
                            int(reply.Value)
                        )

                    raw = tuple(parts)
                    now = time.monotonic()

                    if raw == ZERO:

                        zeros += 1
                        candidate = None
                        consecutive = 0

                        if zeros >= 2:

                            armed = True
                            emitted = False

                    elif raw in TAG_MAP:

                        zeros = 0

                        if armed and not emitted:

                            if raw != candidate:

                                candidate = raw
                                consecutive = 1
                                onset = now

                            else:

                                consecutive += 1

                            if consecutive >= 2:

                                events.put({
                                    "number":
                                        TAG_MAP[raw],
                                    "t": onset,
                                    "raw": raw,
                                })

                                emitted = True

                    else:

                        zeros = 0
                        candidate = None
                        consecutive = 0

                    stop.wait(0.025)

        except Exception as exc:

            if (
                time.monotonic()
                - last_error > 3.0
            ):

                print(
                    "RFID1 reconnect:",
                    exc,
                    flush=True
                )

                last_error = (
                    time.monotonic()
                )

            stop.wait(1.0)


# ============================================================
# CONVEYOR PATH COORDINATES
# ============================================================

class ConveyorPath:

    def __init__(self):

        self.segments = []
        distance = 0.0

        for a, b in zip(
            BELT_POINTS,
            BELT_POINTS[1:]
        ):

            length = math.hypot(
                b[0] - a[0],
                b[1] - a[1]
            )

            self.segments.append(
                (a, b, length, distance)
            )

            distance += length

        self.length = distance

    def project(self, box):

        x, y = center(box)

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
                (x - a[0]) * dx
                + (y - a[1]) * dy
            ) / (length * length)

            fraction = max(
                0.0,
                min(1.0, fraction)
            )

            px = (
                a[0] + fraction * dx
            )

            py = (
                a[1] + fraction * dy
            )

            lateral = math.hypot(
                x - px,
                y - py
            )

            if (
                best is None
                or lateral < best[1]
            ):

                position = (
                    start + fraction * length
                ) % self.length

                best = (
                    position,
                    lateral
                )

        return best

    def delta(self, later, earlier):

        result = (
            later - earlier
        ) % self.length

        if result > self.length / 2:

            result -= self.length

        return result


# ============================================================
# INDEPENDENT PHYSICAL PALLET TRACKER
# ============================================================

class PhysicalTracker:

    def __init__(self):

        self.path = ConveyorPath()

        self.states = {}

        self.visible = {}

    def follow(self, boxes, t):

        self.visible = {}

        candidates = []

        for (
            number,
            state
        ) in self.states.items():

            dt = t - state["t"]

            if (
                dt <= 0
                or dt > MAX_TRACK_GAP
            ):
                continue

            for (
                index,
                obj
            ) in enumerate(boxes):

                if obj["lateral"] > 145:
                    continue

                old_box = state["box"]
                new_box = obj["box"]

                old_area = max(
                    1,
                    (old_box[2] - old_box[0])
                    * (old_box[3] - old_box[1])
                )

                new_area = max(
                    1,
                    (new_box[2] - new_box[0])
                    * (new_box[3] - new_box[1])
                )

                size_ratio = (
                    min(old_area, new_area)
                    / max(old_area, new_area)
                )

                if size_ratio < 0.58:
                    continue

                movement = (
                    self.path.delta(
                        obj["s"],
                        state["s"]
                    )
                )

                if (
                    movement < -28
                    or movement > 35 + 350 * dt
                ):
                    continue

                prediction = (
                    state["speed"] * dt
                )

                score = (
                    abs(movement - prediction)
                    + 0.10 * obj["lateral"]
                )

                candidates.append((
                    score,
                    number,
                    index,
                    movement,
                    dt,
                ))

        by_number = {}
        by_index = {}

        for item in candidates:

            by_number.setdefault(
                item[1], []
            ).append(item)

            by_index.setdefault(
                item[2], []
            ).append(item)

        for choices in (
            *by_number.values(),
            *by_index.values()
        ):

            choices.sort()

        used_numbers = set()
        used_indices = set()

        for (
            score,
            number,
            index,
            movement,
            dt
        ) in sorted(candidates):

            if (
                number in used_numbers
                or index in used_indices
            ):
                continue

            if (
                by_number[number][0][2]
                != index
            ):
                continue

            if (
                by_index[index][0][1]
                != number
            ):
                continue

            if (
                len(by_number[number]) > 1
                and
                by_number[number][1][0]
                - score < 30
            ):
                continue

            if (
                len(by_index[index]) > 1
                and
                by_index[index][1][0]
                - score < 30
            ):
                continue

            obj = boxes[index]
            state = self.states[number]

            speed = min(
                340.0,
                max(
                    0.0,
                    movement / dt
                )
            )

            state["speed"] = (
                0.6 * state["speed"]
                + 0.4 * speed
            )

            state["s"] = obj["s"]
            state["box"] = obj["box"]
            state["t"] = t

            self.visible[index] = number

            used_numbers.add(number)
            used_indices.add(index)

    def anchor(
        self,
        number,
        candidate,
        event_t,
        log
    ):

        for (
            other,
            state
        ) in self.states.items():

            if (
                other == number
                or
                abs(event_t - state["t"]) > 1.0
            ):
                continue

            separation = abs(
                self.path.delta(
                    candidate["s"],
                    state["s"]
                )
            )

            if separation < 110:

                log(
                    "CONFLICT",
                    number,
                    (
                        "near physical pallet "
                        f"{other}; not assigned"
                    )
                )

                return False

        prior = self.states.get(number)

        if prior:

            dt = event_t - prior["t"]

            if 0 < dt < 3.0:

                distance = abs(
                    self.path.delta(
                        candidate["s"],
                        prior["s"]
                    )
                )

                if distance > 350:

                    log(
                        "CONFLICT",
                        number,
                        (
                            "existing identity too "
                            "far away; not assigned"
                        )
                    )

                    return False

        self.states[number] = {
            "s": candidate["s"],
            "box": candidate["box"],
            "t": event_t,
            "speed": 175.0,
        }

        action = (
            "REANCHOR"
            if prior
            else "MAP"
        )

        log(
            action,
            number,
            (
                f"box={candidate['box']} "
                "| RFID1 lower lane "
                "| provisional"
            )
        )

        return True


# ============================================================
# MAIN RFID1 APPLICATION
# ============================================================

class App:

    def __init__(self, events):

        self.events = events

        self.tracker = PhysicalTracker()

        self.history = deque(
            maxlen=45
        )

        self.pending = deque()

        self.last_frame = None

        self.latest_tag = None
        self.latest_time = 0.0

        self.sequence = 0

        self.logfile = (
            OUT_DIR
            / (
                "rfid1_lower_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".csv"
            )
        )

        with self.logfile.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as handle:

            csv.writer(handle).writerow([
                "time",
                "event",
                "physical_pallet",
                "details",
            ])

    def log(
        self,
        name,
        number="",
        details=""
    ):

        print(
            f"RFID1 {name}: "
            f"number={number} "
            f"{details}",
            flush=True
        )

        with self.logfile.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as handle:

            csv.writer(handle).writerow([
                datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                name,
                number,
                details,
            ])

    # ========================================================
    # READ ORIGINAL YOLO DETECTIONS
    # ========================================================

    def detections(self, scope):

        result = []

        for entry in scope.get(
            "confirmed_tracks",
            ()
        ):

            box = tuple(
                int(x)
                for x in entry["box"]
            )

            s, lateral = (
                self.tracker.path.project(
                    box
                )
            )

            result.append({
                "box": box,
                "s": s,
                "lateral": lateral,
                "vx": float(
                    entry.get(
                        "velocity_x",
                        0.0
                    )
                ),
                "vision_id": int(
                    entry["track_id"]
                ),
            })

        return result

    # ========================================================
    # MATCH RFID1 AGAINST LOWER-LANE BOXES ONLY
    # ========================================================

    def handle_event(self, event):

        number = event["number"]

        if not self.history:

            self.log(
                "NO_CAMERA",
                number
            )

            return

        nearest = min(
            self.history,
            key=lambda snap: abs(
                snap["t"] - event["t"]
            )
        )

        dt = (
            nearest["t"]
            - event["t"]
        )

        all_bottom = []
        candidates = []

        for obj in nearest["boxes"]:

            x1, y1, x2, y2 = (
                obj["box"]
            )

            _, cy = center(
                obj["box"]
            )

            # LOWER conveyor ONLY.
            if not (
                GATE_Y_MIN
                <= cy
                <= GATE_Y_MAX
            ):
                continue

            all_bottom.append(obj)

            # Lower conveyor moves
            # LEFT TO RIGHT.
            if obj["vx"] < -3.0:
                continue

            # Require box to intersect
            # the virtual sensing line,
            # allowing a small uncertainty.
            if (
                x1 - GATE_EDGE_TOLERANCE
                <= GATE_X
                <= x2 + GATE_EDGE_TOLERANCE
            ):

                candidates.append(obj)

        # ====================================================
        # AUTOMATIC EVIDENCE IMAGE
        # ====================================================

        sample = (
            nearest["image"].copy()
        )

        cv2.rectangle(
            sample,
            (
                GATE_X - 12,
                GATE_Y_MIN
            ),
            (
                GATE_X + 12,
                GATE_Y_MAX
            ),
            (0, 165, 255),
            3
        )

        cv2.putText(
            sample,
            (
                f"RFID1={number} "
                f"| dt={dt:+.2f}s "
                f"| candidates={len(candidates)}"
            ),
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 220, 255),
            2
        )

        for obj in candidates:

            x1, y1, x2, y2 = (
                obj["box"]
            )

            cv2.rectangle(
                sample,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                4
            )

        self.sequence += 1

        path = (
            OUT_DIR
            / (
                f"RFID1_P{number}_"
                f"E{self.sequence:03d}_LOWER.png"
            )
        )

        cv2.imwrite(
            str(path),
            sample
        )

        diagnostic = [
            (
                obj["box"],
                round(obj["vx"], 1)
            )
            for obj in sorted(
                all_bottom,
                key=lambda obj: abs(
                    center(
                        obj["box"]
                    )[0] - GATE_X
                )
            )[:4]
        ]

        details = (
            f"frame_dt={dt:+.3f}s "
            f"| candidates={len(candidates)} "
            f"| nearest_bottom={diagnostic} "
            f"| image={path.name}"
        )

        # ====================================================
        # CONFIDENCE / AMBIGUITY CHECKS
        # ====================================================

        if abs(dt) > TIME_TOLERANCE:

            self.log(
                "CAMERA_DELAY_NO_MAP",
                number,
                details
            )

        elif len(candidates) == 0:

            self.log(
                "NO_LOWER_GATE_MATCH",
                number,
                details
            )

        elif len(candidates) > 1:

            self.log(
                "AMBIGUOUS_NO_MAP",
                number,
                details
            )

        else:

            self.tracker.anchor(
                number,
                candidates[0],
                nearest["t"],
                self.log
            )

            self.log(
                "EVIDENCE",
                number,
                details
            )

    # ========================================================
    # LIVE DISPLAY
    # ========================================================

    def render(
        self,
        scope,
        captured
    ):

        t = (
            captured
            if captured is not None
            else time.monotonic()
        )

        now = time.monotonic()

        src = scope.get(
            "symbol_frame"
        )

        if src is None:

            src = scope.get(
                "frame"
            )

        if src is None:

            raise RuntimeError(
                "Camera frame not found in Phase40"
            )

        image = src.copy()

        boxes = self.detections(scope)

        # Follow physical pallet positions.
        self.tracker.follow(
            boxes,
            t
        )

        # Original classification decisions.
        cache = scope.get(
            "visible_track_cache",
            {}
        )

        # Original conveyor boundaries.
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

            polygon = scope.get(
                name
            )

            if polygon is not None:

                cv2.polylines(
                    image,
                    [polygon],
                    True,
                    color,
                    2
                )

        # ====================================================
        # DISPLAY PHYSICAL IDs
        # ====================================================

        for (
            index,
            obj
        ) in enumerate(boxes):

            x1, y1, x2, y2 = (
                obj["box"]
            )

            saved = cache.get(
                obj["vision_id"],
                {}
            )

            result = status(saved)

            number = (
                self.tracker.visible.get(
                    index
                )
            )

            identity = (
                str(number)
                if number is not None
                else "?"
            )

            label = (
                f"{identity} {result}"
            )

            color = saved.get(
                "color",
                (255, 255, 255)
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                image,
                label,
                (
                    max(0, x1),
                    max(25, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                color,
                2
            )

        # ====================================================
        # RFID1 LOWER VIRTUAL GATE
        # ====================================================

        cv2.rectangle(
            image,
            (
                GATE_X - 12,
                GATE_Y_MIN
            ),
            (
                GATE_X + 12,
                GATE_Y_MAX
            ),
            (0, 165, 255),
            2
        )

        cv2.putText(
            image,
            "RFID1 LOWER VIRTUAL GATE",
            (
                GATE_X - 180,
                GATE_Y_MIN - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            (0, 165, 255),
            2
        )

        # ====================================================
        # STATUS PANEL
        # ====================================================

        cv2.rectangle(
            image,
            (6, 6),
            (1040, 82),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            (
                "PHASE47 RFID1 LOWER LANE "
                "| PROVISIONAL IDS"
            ),
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            (
                "? = UNMAPPED "
                "| Orange = lower RFID1 gate "
                "| Q quit | S save"
            ),
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        self.history.append({
            "t": t,
            "boxes": boxes,
            "image": image.copy(),
        })

        self.last_frame = image

        # ====================================================
        # RFID1 READINGS
        # ====================================================

        while True:

            try:

                event = (
                    self.events.get_nowait()
                )

            except queue.Empty:
                break

            self.latest_tag = (
                event["number"]
            )

            self.latest_time = now

            self.pending.append(
                event
            )

            self.log(
                "READ",
                event["number"],
                f"raw={event['raw']}"
            )

        # Wait briefly for camera frames
        # surrounding the RFID1 activation.
        while (
            self.pending
            and
            now - self.pending[0]["t"]
            >= EVENT_WAIT
        ):

            self.handle_event(
                self.pending.popleft()
            )

        return image


# ============================================================
# MAIN
# ============================================================

def main():

    if not LIVE_FILE.is_file():

        raise FileNotFoundError(
            "Original Conference file missing: "
            f"{LIVE_FILE}"
        )

    import pylogix

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sys.path.insert(
        0,
        str(LIVE_FILE.parent)
    )

    spec = (
        importlib.util.spec_from_file_location(
            "crest_phase40_phase47",
            LIVE_FILE
        )
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

    app = App(events)

    worker = threading.Thread(
        target=poll_rfid1,
        args=(events, stop),
        daemon=True
    )

    old_open = live.open_camera
    old_show = cv2.imshow
    old_key = cv2.waitKey

    clock = None

    # ========================================================
    # TEMPORARY CAMERA WRAPPER
    # ========================================================

    def open_wrapper():

        nonlocal clock

        camera, index = (
            old_open()
        )

        clock = TimestampCamera(
            camera
        )

        return clock, index

    # ========================================================
    # TEMPORARY DISPLAY WRAPPER
    # ========================================================

    def show_wrapper(
        window,
        image
    ):

        if (
            "CREST"
            not in str(window).upper()
        ):

            return old_show(
                window,
                image
            )

        scope = (
            sys._getframe(1).f_locals
        )

        acquired = (
            clock.acquired
            if clock is not None
            else None
        )

        return old_show(
            window,
            app.render(
                scope,
                acquired
            )
        )

    # ========================================================
    # SAVE SCREENSHOT
    # ========================================================

    def key_wrapper(delay=0):

        key = old_key(delay)

        if (
            (key & 255) == ord("s")
            and app.last_frame is not None
        ):

            filename = (
                "manual_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                + ".png"
            )

            file = (
                OUT_DIR / filename
            )

            cv2.imwrite(
                str(file),
                app.last_frame
            )

            print(
                "Screenshot saved:",
                file,
                flush=True
            )

            return 255

        return key

    # ========================================================
    # RUN ORIGINAL PHASE40
    # ========================================================

    live.open_camera = open_wrapper
    cv2.imshow = show_wrapper
    cv2.waitKey = key_wrapper

    worker.start()

    print()
    print("=" * 60)
    print("CREST PHASE47 - RFID1 LOWER LANE ONLY")
    print("=" * 60)
    print(
        "Virtual sensor X=1153, "
        "Y=720..970"
    )
    print(
        "LOWER conveyor: LEFT -> RIGHT"
    )
    print(
        "RFID2: NOT USED"
    )
    print(
        "Phase40 vision: UNCHANGED"
    )
    print(
        "No RFID identity transfer "
        "through vision tracker IDs"
    )
    print(
        "PLC: READ ONLY"
    )
    print("=" * 60)
    print()

    try:

        live.main()

    finally:

        stop.set()

        worker.join(
            timeout=2
        )

        live.open_camera = old_open

        cv2.imshow = old_show
        cv2.waitKey = old_key

        print(
            "RFID1 evidence:",
            OUT_DIR
        )

        print(
            "CSV:",
            app.logfile
        )


if __name__ == "__main__":
    main()
