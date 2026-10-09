
"""CREST Phase 48: RFID1-only interactive virtual gate and identity test.

Original Conference Phase40 models/tracker/classification are never edited.
RFID1 only, read-only PLC. No conveyor/robot/piston commands.

Left click: position virtual gate, A: auto test on/off, G: save gate.
Right click physical pallet box while RFID1 tag is ACTIVE: manual verify.
Number without suffix: user-verified; e.g. 4~: provisional auto-match.
"""

import csv
import importlib.util
import json
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

BASE = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
    r"\07_Final_Empty_Accepted_Rejected_System"
)

PHASE40 = (
    BASE
    / "CODE"
    / "40_conference_100_pallet_live.py"
)

OUTPUT = (
    BASE
    / "RESULTS"
    / "48_RFID1_Click_Calibration"
)

CONFIG = (
    BASE
    / "CONFIG"
    / "rfid1_phase48_gate.json"
)


# ============================================================
# RFID1 ONLY
# ============================================================

PLC_IP = "192.168.82.231"

WORDS = [
    (
        "IO_Link_Master_ST1_to_ST4:"
        f"I1.Data[{i}]"
    )
    for i in range(96, 100)
]

NO_TAG = (0, 0, 0, 0)

RFID_NUMBERS = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}


# ============================================================
# CALIBRATION SETTINGS
# ============================================================

GATE_W = 240
GATE_H = 190

EVENT_DELAY = 0.45
FRAME_MAX_AGE = 0.40

FOLLOW_GAP = 1.8


# ============================================================
# CONVEYOR PATH
#
# Upper lane: RIGHT -> LEFT
# Lower lane: LEFT -> RIGHT
# ============================================================

BELT = [
    (1610, 535),
    (1360, 540),
    (1120, 547),
    (820, 552),
    (550, 565),
    (350, 620),
    (255, 715),
    (295, 795),
    (510, 858),
    (820, 868),
    (1150, 868),
    (1450, 859),
    (1630, 810),
    (1720, 705),
    (1720, 602),
    (1610, 535),
]


# ============================================================
# GEOMETRY
# ============================================================

def center(box):

    return (
        (box[0] + box[2]) * 0.5,
        (box[1] + box[3]) * 0.5,
    )


def get_status(info):

    label = str(
        info.get("label", "")
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
# CONVEYOR PATH COORDINATES
# ============================================================

class PathCoordinate:

    def __init__(self):

        self.parts = []
        length_sum = 0.0

        for start, end in zip(
            BELT[:-1],
            BELT[1:]
        ):

            segment_len = math.dist(
                start,
                end
            )

            self.parts.append((
                start,
                end,
                segment_len,
                length_sum,
            ))

            length_sum += segment_len

        self.length = length_sum

    def project(self, box):

        x, y = center(box)

        closest = None

        for (
            a,
            b,
            length,
            start
        ) in self.parts:

            dx = b[0] - a[0]
            dy = b[1] - a[1]

            numerator = (
                (x - a[0]) * dx
                + (y - a[1]) * dy
            )

            r = max(
                0.0,
                min(
                    1.0,
                    numerator / (length * length)
                )
            )

            px = a[0] + dx * r
            py = a[1] + dy * r

            side = math.hypot(
                x - px,
                y - py
            )

            if (
                closest is None
                or side < closest[1]
            ):

                position = (
                    start + r * length
                ) % self.length

                closest = (
                    position,
                    side
                )

        return closest

    def forward(
        self,
        newer,
        earlier
    ):

        d = (
            newer - earlier
        ) % self.length

        if d > 0.5 * self.length:
            d -= self.length

        return d


# ============================================================
# RFID1 PLC READER
# ============================================================

def rfid1_worker(events, stop):

    from pylogix import PLC

    armed = False
    emitted = False

    zeros = 0

    last_valid = None
    count = 0
    first_time = None

    while not stop.is_set():

        try:

            with PLC() as plc:

                plc.IPAddress = PLC_IP
                plc.SocketTimeout = 3

                print(
                    "RFID1 polling only:",
                    PLC_IP,
                    "tags 96-99",
                    flush=True
                )

                while not stop.is_set():

                    read_values = []

                    for tag in WORDS:

                        response = plc.Read(tag)

                        if (
                            response.Status
                            != "Success"
                        ):

                            raise RuntimeError(
                                f"RFID1 {tag}: "
                                f"{response.Status}"
                            )

                        read_values.append(
                            int(response.Value)
                        )

                    raw = tuple(read_values)

                    now = time.monotonic()

                    # --------------------------------
                    # ZERO / NO TAG
                    # --------------------------------

                    if raw == NO_TAG:

                        zeros += 1

                        count = 0
                        last_valid = None

                        if zeros >= 2:

                            armed = True

                            if emitted:

                                events.put({
                                    "kind": "CLEAR",
                                    "time": now,
                                })

                            emitted = False

                    # --------------------------------
                    # VALID RFID TAG
                    # --------------------------------

                    elif raw in RFID_NUMBERS:

                        zeros = 0

                        if armed and not emitted:

                            if raw != last_valid:

                                last_valid = raw
                                count = 1
                                first_time = now

                            else:

                                count += 1

                            if count >= 2:

                                events.put({
                                    "kind": "READ",
                                    "number":
                                        RFID_NUMBERS[raw],
                                    "time": first_time,
                                    "raw": raw,
                                })

                                emitted = True

                    # --------------------------------
                    # INCOMPLETE RFID DATA
                    # --------------------------------

                    else:

                        zeros = 0
                        last_valid = None
                        count = 0

                    stop.wait(0.025)

        except Exception as exc:

            print(
                "RFID1 reconnect:",
                exc,
                flush=True
            )

            stop.wait(1.0)


# ============================================================
# CAMERA ACQUISITION TIMESTAMP
# ============================================================

class ClockCamera:

    def __init__(self, camera):

        self.camera = camera
        self.t = None

    def read(self):

        result = self.camera.read()

        self.t = (
            time.monotonic()
            if result[0]
            else None
        )

        return result

    def __getattr__(self, name):

        return getattr(
            self.camera,
            name
        )


# ============================================================
# PHYSICAL PALLET TRACKING
#
# Independent of YOLO software tracker IDs.
# ============================================================

class PhysicalTracking:

    def __init__(self):

        self.path = PathCoordinate()

        # Physical pallet number -> state
        self.states = {}

        # Current detection index -> pallet
        self.visible = {}

    def clear_provisional(self):

        self.states = {
            number: state
            for number, state in self.states.items()
            if state["verified"]
        }

        self.visible.clear()

    # ========================================================
    # ANCHOR PHYSICAL RFID IDENTITY
    # ========================================================

    def assign(
        self,
        number,
        detection,
        t,
        verified,
        log
    ):

        target_x, target_y = center(
            detection["box"]
        )

        for (
            old_num,
            state
        ) in self.states.items():

            if old_num == number:
                continue

            if t - state["t"] < 1.5:

                old_x, old_y = center(
                    state["box"]
                )

                distance = math.hypot(
                    target_x - old_x,
                    target_y - old_y
                )

                if distance < 95:

                    log(
                        "CONFLICT",
                        number,
                        (
                            "near numbered pallet "
                            f"{old_num}; "
                            "no assignment"
                        )
                    )

                    return False

        previous = (
            self.states.get(number)
        )

        if (
            previous
            and previous["verified"]
            and not verified
        ):

            log(
                "SKIP_AUTO",
                number,
                "already operator-verified"
            )

            return False

        self.states[number] = {
            "box": detection["box"],
            "s": detection["s"],
            "t": t,
            "speed": (
                previous["speed"]
                if previous
                else 175.0
            ),
            "verified": (
                verified
                or bool(
                    previous
                    and previous["verified"]
                )
            ),
        }

        action = (
            "VERIFY"
            if verified
            else "AUTO_PROVISIONAL"
        )

        method = (
            "RIGHT_CLICK"
            if verified
            else "GATE"
        )

        log(
            action,
            number,
            (
                f"box={detection['box']} "
                f"method={method}"
            )
        )

        return True

    # ========================================================
    # FOLLOW MOVING PALLETS
    # ========================================================

    def advance(self, dets, t):

        self.visible.clear()

        pairs = []

        for (
            number,
            state
        ) in self.states.items():

            dt = t - state["t"]

            if not (
                0.005 <= dt <= FOLLOW_GAP
            ):
                continue

            previous_box = (
                state["box"]
            )

            old_area = max(
                1,
                (
                    previous_box[2]
                    - previous_box[0]
                )
                * (
                    previous_box[3]
                    - previous_box[1]
                )
            )

            for (
                index,
                det
            ) in enumerate(dets):

                if det["lateral"] > 130:
                    continue

                box = det["box"]

                size = max(
                    1,
                    (box[2] - box[0])
                    * (box[3] - box[1])
                )

                ratio = (
                    min(size, old_area)
                    / max(size, old_area)
                )

                if ratio < 0.55:
                    continue

                distance = (
                    self.path.forward(
                        det["s"],
                        state["s"]
                    )
                )

                # Reject impossible movement.
                if (
                    distance < -25
                    or
                    distance > 35 + 340 * dt
                ):
                    continue

                predicted = (
                    state["speed"] * dt
                )

                score = (
                    abs(distance - predicted)
                    + det["lateral"] * 0.12
                )

                pairs.append((
                    score,
                    number,
                    index,
                    distance,
                    dt,
                ))

        # ----------------------------------------
        # ONE-TO-ONE MATCHING
        # ----------------------------------------

        by_number = {}
        by_det = {}

        for pair in pairs:

            by_number.setdefault(
                pair[1], []
            ).append(pair)

            by_det.setdefault(
                pair[2], []
            ).append(pair)

        groups = (
            list(by_number.values())
            + list(by_det.values())
        )

        for group in groups:
            group.sort()

        assigned_numbers = set()
        assigned_indices = set()

        for (
            score,
            number,
            index,
            distance,
            dt
        ) in sorted(pairs):

            if (
                number in assigned_numbers
                or index in assigned_indices
            ):
                continue

            group_a = by_number[number]
            group_b = by_det[index]

            if group_a[0][2] != index:
                continue

            if group_b[0][1] != number:
                continue

            # Reject ambiguous neighboring
            # pallet associations.

            if (
                len(group_a) > 1
                and group_a[1][0] - score < 25
            ):
                continue

            if (
                len(group_b) > 1
                and group_b[1][0] - score < 25
            ):
                continue

            state = self.states[number]

            speed = max(
                0.0,
                min(
                    320.0,
                    distance / dt
                )
            )

            state["speed"] = (
                0.65 * state["speed"]
                + 0.35 * speed
            )

            state["box"] = (
                dets[index]["box"]
            )

            state["s"] = (
                dets[index]["s"]
            )

            state["t"] = t

            self.visible[index] = number

            assigned_numbers.add(
                number
            )

            assigned_indices.add(
                index
            )


# ============================================================
# RFID1 APPLICATION
# ============================================================

class Application:

    def __init__(self, events):

        self.events = events

        self.tracker = PhysicalTracking()

        # (center_x, center_y, width, height)
        self.gate = None

        # Auto mapping starts DISABLED.
        self.auto = False

        self.history = deque(
            maxlen=90
        )

        self.pending = deque()

        self.active_number = None

        self.latest_read = None
        self.latest_read_time = 0

        self.now_detections = []
        self.now_t = 0

        self.last_image = None

        self.counter = 0

        self.log_path = (
            OUTPUT
            / (
                "phase48_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".csv"
            )
        )

        with self.log_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                "time",
                "event",
                "rfid_number",
                "details",
            ])

        # ----------------------------------------
        # LOAD PREVIOUSLY SAVED GATE
        # ----------------------------------------

        if CONFIG.is_file():

            try:

                with CONFIG.open(
                    encoding="utf-8"
                ) as file:

                    values = json.load(
                        file
                    )

                self.gate = tuple(
                    int(values[key])
                    for key in (
                        "cx",
                        "cy",
                        "width",
                        "height"
                    )
                )

                print(
                    "Saved RFID1 virtual gate loaded:",
                    self.gate
                )

            except (
                OSError,
                ValueError,
                KeyError,
                TypeError
            ) as exc:

                print(
                    "Could not load old gate:",
                    exc
                )

    # ========================================================
    # LOGGING
    # ========================================================

    def log(
        self,
        event,
        number="",
        details=""
    ):

        print(
            f"RFID1 {event}: "
            f"pallet={number} "
            f"{details}",
            flush=True
        )

        with self.log_path.open(
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
                details,
            ])

    # ========================================================
    # READ EXISTING YOLO BOXES
    # ========================================================

    def get_detections(self, scope):

        result = []

        for track in scope.get(
            "confirmed_tracks",
            ()
        ):

            box = tuple(
                int(v)
                for v in track["box"]
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

                # Diagnostic only.
                "tid": int(
                    track["track_id"]
                ),
            })

        return result

    # ========================================================
    # VIRTUAL GATE CHECK
    # ========================================================

    def in_gate(self, det):

        if self.gate is None:
            return False

        cx, cy, width, height = (
            self.gate
        )

        x, y = center(
            det["box"]
        )

        return (
            abs(x - cx) <= width / 2
            and
            abs(y - cy) <= height / 2
        )

    # ========================================================
    # MOUSE CONTROL
    # ========================================================

    def on_mouse(
        self,
        event,
        x,
        y,
        flags,
        userdata
    ):

        # ----------------------------------------
        # LEFT CLICK: MOVE VIRTUAL GATE
        # ----------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:

            width = (
                self.gate[2]
                if self.gate
                else GATE_W
            )

            height = (
                self.gate[3]
                if self.gate
                else GATE_H
            )

            self.gate = (
                x,
                y,
                width,
                height
            )

            # Prevent automatic assignments
            # when the gate is repositioned.
            self.auto = False

            self.tracker.clear_provisional()

            self.log(
                "MOVE_GATE",
                "",
                (
                    f"center=({x},{y}) "
                    f"size={width}x{height}; "
                    "AUTO OFF"
                )
            )

        # ----------------------------------------
        # RIGHT CLICK: VERIFY CORRECT PALLET
        # ----------------------------------------

        elif event == cv2.EVENT_RBUTTONDOWN:

            if self.active_number is None:

                self.log(
                    "NO_ACTIVE_TAG",
                    "",
                    (
                        "right click requires "
                        "active RFID1 reading"
                    )
                )

                return

            possible = []

            for det in self.now_detections:

                x1, y1, x2, y2 = (
                    det["box"]
                )

                if (
                    x1 <= x <= x2
                    and
                    y1 <= y <= y2
                ):

                    possible.append(det)

            if len(possible) != 1:

                self.log(
                    "CLICK_AMBIGUOUS",
                    self.active_number,
                    (
                        "boxes_under_click="
                        f"{len(possible)}"
                    )
                )

                return

            self.tracker.assign(
                self.active_number,
                possible[0],
                self.now_t,
                True,
                self.log
            )

    # ========================================================
    # SAVE CALIBRATED GATE
    # ========================================================

    def save_gate(self):

        if self.gate is None:

            self.log(
                "GATE_MISSING",
                "",
                (
                    "left click on the actual "
                    "sensing position first"
                )
            )

            return

        CONFIG.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        values = dict(
            zip(
                (
                    "cx",
                    "cy",
                    "width",
                    "height"
                ),
                self.gate
            )
        )

        with CONFIG.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                values,
                file,
                indent=2
            )

        self.log(
            "GATE_SAVED",
            "",
            str(CONFIG)
        )

    # ========================================================
    # PROCESS RFID1 READ EVENT
    # ========================================================

    def handle_read(self, reading):

        self.counter += 1

        number = (
            reading["number"]
        )

        if not self.history:

            self.log(
                "NO_CAMERA_HISTORY",
                number
            )

            return

        # Find camera frame nearest
        # RFID activation timestamp.

        snapshot = min(
            self.history,
            key=lambda item: abs(
                item["t"] - reading["time"]
            )
        )

        dt = (
            snapshot["t"]
            - reading["time"]
        )

        valid = [
            det
            for det in snapshot["dets"]
            if self.in_gate(det)
        ]

        self.log(
            "GATE_OBSERVE",
            number,
            (
                f"dt={dt:+.3f}s "
                f"gate={self.gate} "
                f"candidates={len(valid)} "
                "boxes="
                f"{[d['box'] for d in valid]}"
            )
        )

        # ----------------------------------------
        # SAVE CAMERA EVIDENCE
        # ----------------------------------------

        image = (
            snapshot["image"].copy()
        )

        cv2.putText(
            image,
            (
                f"RFID1={number} "
                f"dt={dt:+.2f}s "
                f"candidates={len(valid)}"
            ),
            (18, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        for det in valid:

            x1, y1, x2, y2 = (
                det["box"]
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                4
            )

        filename = (
            f"RFID1_{number}_"
            f"E{self.counter:03d}.png"
        )

        image_path = (
            OUTPUT / filename
        )

        cv2.imwrite(
            str(image_path),
            image
        )

        # ----------------------------------------
        # AUTO MODE DISABLED
        # ----------------------------------------

        if not self.auto:

            self.log(
                "AUTO_OFF",
                number,
                f"evidence={filename}"
            )

            return

        # ----------------------------------------
        # CHECK SAFE AUTO ASSOCIATION
        # ----------------------------------------

        if (
            self.gate is None
            or abs(dt) > FRAME_MAX_AGE
            or len(valid) != 1
        ):

            self.log(
                "NO_SAFE_AUTO_MATCH",
                number,
                (
                    f"dt={dt:+.3f}s "
                    f"candidates={len(valid)}"
                )
            )

            return

        # Require a second camera observation.
        # Compare geometry, not YOLO IDs.

        confirmations = [
            item
            for item in self.history
            if (
                0.06
                <= abs(
                    item["t"] - snapshot["t"]
                )
                <= 0.45
            )
        ]

        selected_center = center(
            valid[0]["box"]
        )

        confirmed = False

        for item in confirmations:

            choices = [
                det
                for det in item["dets"]
                if self.in_gate(det)
            ]

            if len(choices) != 1:
                continue

            px, py = center(
                choices[0]["box"]
            )

            distance = math.hypot(
                px - selected_center[0],
                py - selected_center[1]
            )

            if distance <= 95:

                confirmed = True
                break

        if not confirmed:

            self.log(
                "NO_SECOND_FRAME",
                number,
                "not mapped"
            )

            return

        # Auto mappings are PROVISIONAL.
        self.tracker.assign(
            number,
            valid[0],
            snapshot["t"],
            False,
            self.log
        )

    # ========================================================
    # CAMERA FRAME PROCESSING
    # ========================================================

    def process(
        self,
        scope,
        acquired
    ):

        now = time.monotonic()

        t = (
            acquired
            if acquired is not None
            else now
        )

        frame = scope.get(
            "symbol_frame"
        )

        if frame is None:
            frame = scope.get("frame")

        if frame is None:

            raise RuntimeError(
                "Original Phase40 camera frame missing"
            )

        image = frame.copy()

        dets = self.get_detections(
            scope
        )

        self.now_detections = dets
        self.now_t = t

        # Update physical pallet positions.
        self.tracker.advance(
            dets,
            t
        )

        cache = scope.get(
            "visible_track_cache",
            {}
        )

        # ----------------------------------------
        # ORIGINAL CONVEYOR BOUNDARIES
        # ----------------------------------------

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

        # ----------------------------------------
        # DETECTED PALLETS
        # ----------------------------------------

        for (
            index,
            det
        ) in enumerate(dets):

            info = cache.get(
                det["tid"],
                {}
            )

            state = get_status(
                info
            )

            number = (
                self.tracker.visible.get(
                    index
                )
            )

            if number is None:

                label = f"? {state}"

            elif self.tracker.states[
                number
            ]["verified"]:

                label = (
                    f"{number} {state}"
                )

            else:

                # ~ means provisional auto-match,
                # not independently verified.
                label = (
                    f"{number}~ {state}"
                )

            x1, y1, x2, y2 = (
                det["box"]
            )

            color = info.get(
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
                    max(25, y1 - 9)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

        # ----------------------------------------
        # ORIGINAL HELD PALLET BOXES
        # ----------------------------------------

        current_ids = {
            det["tid"]
            for det in dets
        }

        for (
            track_id,
            info
        ) in cache.items():

            if int(track_id) in current_ids:
                continue

            x1, y1, x2, y2 = (
                int(v)
                for v in info["box"]
            )

            color = info.get(
                "color",
                (255, 255, 255)
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                1
            )

            cv2.putText(
                image,
                f"? {get_status(info)}",
                (
                    max(0, x1),
                    max(25, y1 - 9)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1
            )

        # ----------------------------------------
        # ADJUSTABLE RFID1 VIRTUAL GATE
        # ----------------------------------------

        if self.gate is not None:

            gx, gy, gw, gh = (
                self.gate
            )

            cv2.rectangle(
                image,
                (
                    gx - gw // 2,
                    gy - gh // 2
                ),
                (
                    gx + gw // 2,
                    gy + gh // 2
                ),
                (0, 165, 255),
                2
            )

            cv2.putText(
                image,
                "RFID1 VIRTUAL GATE",
                (
                    max(0, gx - gw // 2),
                    max(20, gy - gh // 2 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (0, 165, 255),
                2
            )

        # ----------------------------------------
        # STATUS PANEL
        # ----------------------------------------

        cv2.rectangle(
            image,
            (5, 5),
            (1100, 90),
            (0, 0, 0),
            -1
        )

        mode = (
            "ON (PROVISIONAL)"
            if self.auto
            else "OFF"
        )

        tag = (
            str(self.active_number)
            if self.active_number is not None
            else "NO_TAG"
        )

        cv2.putText(
            image,
            (
                f"PHASE 48 | RFID1={tag} "
                f"| AUTO={mode}"
            ),
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            (
                "LEFT: move gate | "
                "RIGHT: verify RFID pallet | "
                "G: save | A: auto | "
                "C: clear | S: save | Q: quit"
            ),
            (18, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        self.last_image = (
            image.copy()
        )

        # ----------------------------------------
        # SAVE TIMESTAMPED CAMERA OBSERVATIONS
        # ----------------------------------------

        self.history.append({
            "t": t,
            "dets": dets,
            "image": image.copy(),
        })

        # ----------------------------------------
        # RECEIVE RFID1 EVENTS
        # ----------------------------------------

        while True:

            try:

                event = (
                    self.events.get_nowait()
                )

            except queue.Empty:
                break

            if event["kind"] == "CLEAR":

                self.active_number = None

                self.log(
                    "CLEAR",
                    "",
                    (
                        "PLC returned to "
                        "(0,0,0,0)"
                    )
                )

            else:

                self.active_number = (
                    event["number"]
                )

                self.latest_read = (
                    event["number"]
                )

                self.latest_read_time = (
                    event["time"]
                )

                self.pending.append(
                    event
                )

                self.log(
                    "READ",
                    event["number"],
                    f"raw={event['raw']}"
                )

        # ----------------------------------------
        # PROCESS RFID1 ACTIVATION
        # ----------------------------------------

        while (
            self.pending
            and
            now - self.pending[0]["time"]
            >= EVENT_DELAY
        ):

            event = (
                self.pending.popleft()
            )

            self.handle_read(
                event
            )

        return image


# ============================================================
# MAIN
# ============================================================

def main():

    if not PHASE40.is_file():

        raise FileNotFoundError(
            "Original Conference Phase40 missing: "
            f"{PHASE40}"
        )

    from pylogix import PLC

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    sys.path.insert(
        0,
        str(PHASE40.parent)
    )

    spec = (
        importlib.util.spec_from_file_location(
            "crest_phase40_phase48",
            str(PHASE40)
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Cannot import Phase40"
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

    app = Application(events)

    reader = threading.Thread(
        target=rfid1_worker,
        args=(events, stop),
        daemon=True
    )

    original_open = live.open_camera
    original_show = cv2.imshow
    original_wait = cv2.waitKey

    camera_clock = None

    window_hooked = set()

    # ========================================================
    # CAMERA WRAPPER
    # ========================================================

    def open_wrapper():

        nonlocal camera_clock

        camera, index = (
            original_open()
        )

        camera_clock = (
            ClockCamera(camera)
        )

        return camera_clock, index

    # ========================================================
    # DISPLAY WRAPPER
    # ========================================================

    def show_wrapper(
        name,
        image
    ):

        if (
            "CREST"
            not in str(name).upper()
        ):

            return original_show(
                name,
                image
            )

        scope = (
            sys._getframe(1).f_locals
        )

        acquired = (
            camera_clock.t
            if camera_clock is not None
            else None
        )

        display = app.process(
            scope,
            acquired
        )

        original_show(
            name,
            display
        )

        if name not in window_hooked:

            cv2.setMouseCallback(
                name,
                app.on_mouse
            )

            window_hooked.add(
                name
            )

    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    def wait_wrapper(delay=0):

        key = original_wait(delay)

        letter = key & 255

        # ----------------------------------------
        # A: TOGGLE AUTOMATIC ASSOCIATION
        # ----------------------------------------

        if letter == ord("a"):

            if app.gate is None:

                app.log(
                    "AUTO_REJECTED",
                    "",
                    (
                        "first set gate "
                        "with LEFT click"
                    )
                )

            else:

                app.auto = (
                    not app.auto
                )

                app.tracker.clear_provisional()

                app.log(
                    "AUTO",
                    "",
                    (
                        "ON (provisional)"
                        if app.auto
                        else "OFF"
                    )
                )

            return 255

        # ----------------------------------------
        # G: SAVE GATE
        # ----------------------------------------

        if letter == ord("g"):

            app.save_gate()

            return 255

        # ----------------------------------------
        # C: RESET ALL RFID IDENTITIES
        # ----------------------------------------

        if letter == ord("c"):

            app.tracker.states.clear()
            app.tracker.visible.clear()

            app.log(
                "RESET_IDS",
                "",
                (
                    "all RFID identities "
                    "cleared"
                )
            )

            return 255

        # ----------------------------------------
        # S: SAVE SCREENSHOT
        # ----------------------------------------

        if (
            letter == ord("s")
            and app.last_image is not None
        ):

            filename = (
                "manual_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                + ".png"
            )

            destination = (
                OUTPUT / filename
            )

            cv2.imwrite(
                str(destination),
                app.last_image
            )

            app.log(
                "SCREENSHOT",
                "",
                str(destination)
            )

            return 255

        return key

    # ========================================================
    # TEMPORARY WRAPPERS
    # ========================================================

    live.open_camera = open_wrapper

    cv2.imshow = show_wrapper
    cv2.waitKey = wait_wrapper

    reader.start()

    print()
    print("=" * 65)
    print(
        "CREST PHASE 48: "
        "RFID1 CLICK-CALIBRATION TEST"
    )
    print("=" * 65)
    print(
        "RFID2: NOT ACCESSED"
    )
    print(
        "PLC: READ ONLY"
    )
    print(
        "Phase40 vision: UNCHANGED"
    )
    print(
        "LEFT CLICK: move virtual gate"
    )
    print(
        "RIGHT CLICK: verify box "
        "during active RFID1 tag"
    )
    print(
        "G: save gate"
    )
    print(
        "A: toggle automatic provisional mapping"
    )
    print(
        "C: clear identities"
    )
    print(
        "S: screenshot | Q: quit"
    )
    print(
        "Auto mapping STARTS OFF"
    )
    print("=" * 65)
    print()

    try:

        live.main()

    finally:

        stop.set()

        reader.join(
            timeout=2.0
        )

        live.open_camera = (
            original_open
        )

        cv2.imshow = (
            original_show
        )

        cv2.waitKey = (
            original_wait
        )

        print(
            "RFID1 log:",
            app.log_path
        )

        print(
            "RFID1 evidence:",
            OUTPUT
        )


if __name__ == "__main__":
    main()
