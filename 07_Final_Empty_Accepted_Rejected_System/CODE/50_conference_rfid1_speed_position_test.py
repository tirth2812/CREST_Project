
"""CREST Phase 50: read-only RFID1 / vision speed and timing experiment.

Loads Phase 40 unchanged; does NOT claim physical RFID-to-box matches.
Creates timestamped, motion-predicted RFID candidate evidence for later calibration.
"""
from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import time

import cv2

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(__file__).resolve().parent / "40_conference_100_pallet_live.py"
OUT = ROOT / "RESULTS" / "50_RFID1_Speed_Prediction"
PLC_IP = "192.168.82.231"
TAG_BASE = "IO_Link_Master_ST1_to_ST4:I1.Data"
TAG_WORDS = [f"{TAG_BASE}[{i}]" for i in range(96, 100)]

RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}
ZERO = (0, 0, 0, 0)

POLL_INTERVAL_SEC = 0.10
MAX_FRAME_EXTRAPOLATION_SEC = 0.45
MAX_INTERPOLATION_GAP_SEC = 2.0
EVENT_FRAME_WAIT_SEC = 0.80
EVENT_MAX_WAIT_SEC = 4.0
CHAIN_MAX_GAP_SEC = 2.0
CHAIN_HISTORY_SEC = 14.0


def signed_16(value):
    value = int(value)
    return value - 65536 if 32768 <= value <= 65535 else value


@dataclass
class Chain:
    number: int
    samples: deque = field(default_factory=lambda: deque(maxlen=160))
    vx: float = 0.0
    vy: float = 0.0
    raw_id: int | None = None
    label: str = "UNKNOWN"
    box: tuple = (0, 0, 0, 0)

    def predict_now(self, timestamp):
        if not self.samples:
            return None
        t, x, y = self.samples[-1]
        dt = timestamp - t
        return (x + self.vx * dt, y + self.vy * dt)

    def add(self, timestamp, center, raw_id, box, label):
        if self.samples:
            t, x, y = self.samples[-1]
            dt = timestamp - t
            if dt > 0.001:
                nx = (center[0] - x) / dt
                ny = (center[1] - y) / dt
                if math.hypot(nx, ny) < 1200:
                    if len(self.samples) == 1:
                        self.vx, self.vy = nx, ny
                    else:
                        self.vx = 0.50 * nx + 0.50 * self.vx
                        self.vy = 0.50 * ny + 0.50 * self.vy
        self.samples.append((timestamp, center[0], center[1]))
        self.raw_id = raw_id
        self.box = box
        self.label = label
        while self.samples and timestamp - self.samples[0][0] > CHAIN_HISTORY_SEC:
            self.samples.popleft()

    def position_at(self, timestamp):
        samples = list(self.samples)
        if not samples:
            return None
        for before, after in zip(samples, samples[1:]):
            if before[0] <= timestamp <= after[0]:
                dt = after[0] - before[0]
                if 0.001 < dt <= MAX_INTERPOLATION_GAP_SEC:
                    frac = (timestamp - before[0]) / dt
                    return (
                        before[1] + frac * (after[1] - before[1]),
                        before[2] + frac * (after[2] - before[2]),
                        "INTERPOLATED",
                        round(dt, 3),
                    )
        nearest = min(samples, key=lambda s: abs(s[0] - timestamp))
        delta = timestamp - nearest[0]
        if abs(delta) > MAX_FRAME_EXTRAPOLATION_SEC or len(samples) < 2:
            return None
        if abs(nearest[0] - samples[-1][0]) < 0.001:
            vx, vy = self.vx, self.vy
        else:
            index = samples.index(nearest)
            a, b = (samples[index - 1], nearest) if index else (nearest, samples[1])
            dt = b[0] - a[0]
            if dt < 0.001:
                return None
            vx, vy = (b[1] - a[1]) / dt, (b[2] - a[2]) / dt
        return (
            nearest[1] + vx * delta,
            nearest[2] + vy * delta,
            "EXTRAPOLATED",
            round(abs(delta), 3),
        )


class VisionChains:
    def __init__(self):
        self.chains = {}
        self.next_id = 1
        self.last_frame_time = -1

    def update(self, timestamp, confirmed_tracks, visible_cache):
        if timestamp <= self.last_frame_time:
            return []
        self.last_frame_time = timestamp
        observations = []
        for track in confirmed_tracks:
            try:
                raw = int(track["track_id"])
                box = tuple(int(round(v)) for v in track["box"])
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                label = str(visible_cache.get(raw, {}).get("label", "UNKNOWN"))
                observations.append((raw, box, (cx, cy), label))
            except (KeyError, TypeError, ValueError):
                continue

        # One-to-one matching. Reuse the same raw vision ID when geometrically
        # plausible; otherwise require an unambiguous motion-based continuation.
        edges = []
        for i, (raw, box, center, label) in enumerate(observations):
            width = max(1, box[2] - box[0])
            height = max(1, box[3] - box[1])
            for chain_id, chain in self.chains.items():
                if not chain.samples:
                    continue
                gap = timestamp - chain.samples[-1][0]
                if not 0 < gap <= CHAIN_MAX_GAP_SEC:
                    continue
                predicted = chain.predict_now(timestamp)
                distance = math.hypot(center[0] - predicted[0], center[1] - predicted[1])
                last_w = max(1, chain.box[2] - chain.box[0])
                last_h = max(1, chain.box[3] - chain.box[1])
                if min(width / last_w, last_w / width) < 0.55:
                    continue
                if min(height / last_h, last_h / height) < 0.55:
                    continue
                limit = min(150.0, max(45.0, 0.55 * max(width, height) + 50.0 * gap))
                if distance > limit:
                    continue
                penalty = 0 if chain.raw_id == raw else 25
                edges.append((distance + penalty, i, chain_id))

        edges.sort()
        matches = {}
        occupied = set()
        for cost, obs_i, chain_id in edges:
            if obs_i in matches or chain_id in occupied:
                continue
            # Avoid an ID swap if the best and second-best candidates are close.
            alternatives = [c for c, oi, ci in edges if oi == obs_i and ci != chain_id and ci not in occupied]
            if alternatives and min(alternatives) - cost < 20:
                continue
            matches[obs_i] = chain_id
            occupied.add(chain_id)

        result = []
        for i, (raw, box, center, label) in enumerate(observations):
            chain_id = matches.get(i)
            if chain_id is None:
                chain_id = self.next_id
                self.next_id += 1
                self.chains[chain_id] = Chain(number=chain_id)
            chain = self.chains[chain_id]
            chain.add(timestamp, center, raw, box, label)
            result.append(chain)

        # Do not retain unlimited retired visual track candidates.
        old = [cid for cid, ch in self.chains.items()
               if ch.samples and timestamp - ch.samples[-1][0] > CHAIN_HISTORY_SEC]
        for cid in old:
            del self.chains[cid]
        return result

    def candidates_at(self, timestamp):
        candidates = []
        for chain_id, ch in sorted(self.chains.items()):
            pos = ch.position_at(timestamp)
            if pos is None:
                continue
            x, y, method, interval = pos
            candidates.append({
                "visual_chain": chain_id,
                "last_yolo_track": ch.raw_id,
                "x": round(x, 1),
                "y": round(y, 1),
                "speed_px_s": round(math.hypot(ch.vx, ch.vy), 1),
                "method": method,
                "time_uncertainty_s": interval,
                "vision_state": ch.label,
                "latest_box": list(ch.box),
            })
        return candidates


class RFID1Reader(Thread):
    def __init__(self, events, stop):
        super().__init__(daemon=True)
        self.events = events
        self.stop_signal = stop

    def run(self):
        try:
            from pylogix import PLC
        except ImportError:
            print("RFID1 ERROR: pylogix is missing; no RFID data will be collected.")
            return
        plc = PLC()
        plc.IPAddress = PLC_IP
        armed = False
        zeros = 0
        last_valid = None
        valid_count = 0
        first_valid_at = 0.0
        active = False
        last_error_print = 0.0
        try:
            while not self.stop_signal.is_set():
                try:
                    values = []
                    replies = plc.Read(TAG_WORDS)  # READ ONLY; batch where supported
                    if not isinstance(replies, (list, tuple)) or len(replies) != 4:
                        replies = [plc.Read(tag) for tag in TAG_WORDS]
                    for tag, read in zip(TAG_WORDS, replies):
                        if read.Value is None or str(read.Status).lower() != "success":
                            raise RuntimeError(f"{tag}: {read.Status}")
                        values.append(signed_16(read.Value))
                    raw = tuple(values)
                    now = time.monotonic()
                    if raw == ZERO:
                        zeros += 1
                        if zeros >= 2:
                            armed = True
                            active = False
                            last_valid = None
                            valid_count = 0
                    else:
                        zeros = 0
                        pallet = RFID_MAP.get(raw)
                        if pallet is None:
                            last_valid = None
                            valid_count = 0
                        elif armed and not active:
                            if raw == last_valid:
                                valid_count += 1
                            else:
                                last_valid = raw
                                valid_count = 1
                                first_valid_at = now
                            if valid_count >= 2:
                                self.events.put({
                                    "pallet": pallet,
                                    "words": raw,
                                    "first_time": first_valid_at,
                                    "confirm_time": now,
                                })
                                active = True
                                armed = False
                except Exception as exc:
                    if time.monotonic() - last_error_print > 5:
                        print(f"RFID1 poll warning: {exc}")
                        last_error_print = time.monotonic()
                self.stop_signal.wait(POLL_INTERVAL_SEC)
        finally:
            plc.Close()


class CameraWithTimestamps:
    def __init__(self, cap, owner):
        self.cap = cap
        self.owner = owner

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.owner.camera_time = time.monotonic()
        return ok, frame

    def __getattr__(self, name):
        return getattr(self.cap, name)


class Experiment:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_file = OUT / f"phase50_{stamp}.csv"
        self.handle = self.csv_file.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=[
            "wall_clock", "rfid_number", "raw_words", "rfid_first_monotonic",
            "rfid_confirm_monotonic", "read_gap_s", "candidate_count",
            "candidate_tracks_json", "result",
        ])
        self.writer.writeheader()
        self.events = Queue()
        self.pending = deque()
        self.stop_signal = Event()
        self.reader = RFID1Reader(self.events, self.stop_signal)
        self.graph = VisionChains()
        self.camera_time = None
        self.last_read_time = None
        self.recent_read = "none"
        self.display_count = 0
        self.missing_interface_warned = False

    def start(self):
        self.reader.start()
        print(f"PHASE 50 result file: {self.csv_file}")
        print("Mode: RFID1 READ ONLY, motion evidence; identity labels NOT assigned.")
        print("Press Q in the camera window to exit.")

    def collect_events(self):
        while True:
            try:
                ev = self.events.get_nowait()
            except Empty:
                break
            self.pending.append(ev)
            self.recent_read = f"RFID1 {ev['pallet']}"
            print(f"RFID1 READ pallet={ev['pallet']} first={ev['first_time']:.3f}")

    def write_event(self, ev, result):
        t = ev["first_time"]
        candidates = self.graph.candidates_at(t)
        gap = None if self.last_read_time is None else t - self.last_read_time
        self.last_read_time = t
        self.writer.writerow({
            "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rfid_number": ev["pallet"],
            "raw_words": repr(ev["words"]),
            "rfid_first_monotonic": f"{t:.6f}",
            "rfid_confirm_monotonic": f"{ev['confirm_time']:.6f}",
            "read_gap_s": "" if gap is None else f"{gap:.3f}",
            "candidate_count": len(candidates),
            "candidate_tracks_json": json.dumps(candidates, separators=(",", ":")),
            "result": result if candidates else "NO_SAMPLED_POSITION",
        })
        self.handle.flush()
        print(f"RFID1 EVIDENCE tag={ev['pallet']} gap={gap} candidates={len(candidates)} result=UNVERIFIED")

    def on_imshow(self, window_name, original_image, original_imshow):
        # cv2.imshow -> show_hook -> on_imshow: walk to Phase 40 main().
        caller = inspect.currentframe().f_back
        local = {}
        for _ in range(5):
            if caller is None:
                break
            if "confirmed_tracks" in caller.f_locals:
                local = caller.f_locals
                break
            caller = caller.f_back
        tracks = local.get("confirmed_tracks")
        visible = local.get("visible_track_cache", {})
        frame_time = self.camera_time
        self.collect_events()
        image = original_image.copy()
        live_chains = []
        if isinstance(tracks, (list, tuple)) and frame_time is not None:
            live_chains = self.graph.update(frame_time, tracks, visible)
        elif not self.missing_interface_warned:
            print("WARNING: Phase 40 did not expose confirmed_tracks at imshow; check version.")
            self.missing_interface_warned = True

        while self.pending and (
            (frame_time is not None and frame_time >= self.pending[0]["first_time"] + EVENT_FRAME_WAIT_SEC)
            or time.monotonic() >= self.pending[0]["first_time"] + EVENT_MAX_WAIT_SEC
        ):
            self.write_event(self.pending.popleft(), "CANDIDATES_ONLY_NO_RFID_ANCHOR")

        for chain in live_chains:
            speed = math.hypot(chain.vx, chain.vy)
            box = chain.box
            # V is a local motion-chain ID, NEVER a physical RFID number.
            cv2.putText(image, f"V{chain.number} {speed:.0f}px/s",
                        (box[0], min(image.shape[0] - 5, box[3] + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.rectangle(image, (6, max(0, image.shape[0] - 72)),
                      (750, image.shape[0] - 6), (0, 0, 0), -1)
        cv2.putText(image, f"PHASE 50 SPEED / RFID1: {self.recent_read}",
                    (16, image.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (255, 255, 255), 2)
        cv2.putText(image, "V=visual chain, not physical ID | RFID association UNVERIFIED",
                    (16, image.shape[0] - 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 255, 255), 1)
        original_imshow(window_name, image)

    def close(self):
        self.stop_signal.set()
        if self.reader.is_alive():
            self.reader.join(timeout=3)
        self.collect_events()
        while self.pending:
            self.write_event(self.pending.popleft(), "SESSION_ENDED")
        self.handle.close()
        print(f"Results saved: {self.csv_file}")


def main():
    if not BASE.is_file():
        raise FileNotFoundError(f"Original Phase 40 not found: {BASE}")
    exp = Experiment()
    spec = importlib.util.spec_from_file_location("crest_conference_phase40", str(BASE))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load original Phase 40 script.")
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    original_open = live.open_camera
    original_imshow = cv2.imshow

    def camera_hook():
        cap, index = original_open()
        return CameraWithTimestamps(cap, exp), index

    def show_hook(window_name, frame):
        return exp.on_imshow(window_name, frame, original_imshow)

    live.open_camera = camera_hook
    cv2.imshow = show_hook
    try:
        exp.start()
        live.main()
    finally:
        cv2.imshow = original_imshow
        live.open_camera = original_open
        exp.close()


if __name__ == "__main__":
    main()
