"""CREST Phase 52: six-pallet RFID1 timing-pattern-to-visual-track matching.

EXPERIMENTAL: displayed RFID~N labels are unverified hypotheses, not certified
physical IDs. The script NEVER writes PLC tags. Phase 40 is imported unchanged.
"""
from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import math
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = CODE_DIR.parent
BASE_FILE = CODE_DIR / "40_conference_100_pallet_live.py"
BOUNDARY_FILE = SYSTEM_DIR / "CONFIG" / "conveyor_boundary.json"
RESULT_DIR = SYSTEM_DIR / "RESULTS" / "52_RFID1_Pattern_Mapping"
PLC_IP = "192.168.82.231"
RFID_WORDS = [
    f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]"
    for i in range(96, 100)
]
TAG_TO_ID = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}

# Conservative experiment parameters, not established conveyor calibration.
POLL_SEC = 0.08
TRACK_MAX_GAP_SEC = 1.6
TRACK_MAX_TRAVEL_PX = 170.0
MIN_TRACK_OBSERVATIONS = 3  # 3 distinct camera observations, not 1.2 s old UID
MATCH_EVERY_SEC = 0.30
MIN_PATH_MOTION_SAMPLES = 150
MAX_PATTERN_RMS_ERROR = 0.065
MIN_SECOND_BEST_MARGIN = 0.025
MIN_REPEAT_AGREEMENTS = 6
MAX_VIEW_UNSEEN_SEC = 0.30
MAX_MAPPING_STALE_SEC = 1.5


def signed16(value):
    n = int(value)
    return n - 65536 if n >= 32768 else n


class ConveyorPath:
    """Derive a closed approximate centerline from the saved ring polygons."""

    def __init__(self, filename, image_width=1920, image_height=1080):
        data = json.loads(filename.read_text(encoding="utf-8-sig"))
        outer = np.asarray(data["outer_polygon"], dtype=float)
        inner = np.asarray(data["inner_polygon"], dtype=float)
        if data.get("coordinate_type") == "normalized":
            outer *= [image_width, image_height]
            inner *= [image_width, image_height]
        all_outer_min = outer.min(axis=0)
        all_outer_max = outer.max(axis=0)
        self.center = (all_outer_min + all_outer_max) / 2.0
        samples = []
        for theta in np.linspace(0, 2 * math.pi, 361)[:-1]:
            direction = np.array([math.cos(theta), math.sin(theta)])
            outer_r = self._ray_distance(outer, direction)
            inner_r = self._ray_distance(inner, direction)
            if outer_r is None or inner_r is None:
                raise ValueError(
                    "Conveyor ring geometry cannot be converted to a "
                    "centerline. Stop; do not guess the path."
                )
            if inner_r >= outer_r:
                raise ValueError("Conveyor inner/outer polygons overlap on ray")
            samples.append(self.center + direction * (outer_r + inner_r) / 2)
        self.points = np.asarray(samples, dtype=float)
        self.next_points = np.roll(self.points, -1, axis=0)
        self.segments = self.next_points - self.points
        self.segment_lengths = np.linalg.norm(self.segments, axis=1)
        self.starts = np.concatenate([[0], np.cumsum(self.segment_lengths)[:-1]])
        self.length = float(self.segment_lengths.sum())
        if self.length < 500:
            raise ValueError("Invalid conveyor path length")

    def _ray_distance(self, polygon, direction):
        # Ray starting from the center and intersections with polygon edges.
        distances = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end - start
            relative = start - self.center
            denominator = direction[0] * edge[1] - direction[1] * edge[0]
            if abs(denominator) < 1e-9:
                continue
            t = (relative[0] * edge[1] - relative[1] * edge[0]) / denominator
            u = (relative[0] * direction[1] - relative[1] * direction[0]) / denominator
            if t > 0 and -1e-7 <= u <= 1.0000001:
                distances.append(float(t))
        return max(distances) if distances else None

    def project(self, point):
        p = np.asarray(point, dtype=float)
        v = p - self.points
        denominators = np.maximum(self.segment_lengths ** 2, 1e-6)
        frac = np.clip(np.sum(v * self.segments, axis=1) / denominators, 0, 1)
        closest = self.points + self.segments * frac[:, None]
        errors = np.sum((closest - p) ** 2, axis=1)
        index = int(np.argmin(errors))
        s = float(self.starts[index] + frac[index] * self.segment_lengths[index])
        return s % self.length, math.sqrt(float(errors[index]))


class Track:
    def __init__(self, uid, now, raw_id, box, s):
        self.uid = uid
        self.created = now
        self.seen = now
        self.raw_id = raw_id
        self.box = box
        self.s = s
        self.speed = None  # signed path speed, pixels per second
        self.observation_count = 1
        self.recent_steps = deque(maxlen=8)
        self.active = True

    def predict(self, now, loop_length):
        dt = max(0.0, now - self.seen)
        return (self.s + (self.speed or 0) * dt) % loop_length

    def update(self, now, raw_id, box, s, loop_length):
        dt = now - self.seen
        step = ((s - self.s + loop_length / 2) % loop_length) - loop_length / 2
        if 0.015 <= dt <= TRACK_MAX_GAP_SEC:
            new_speed = step / dt
            if abs(new_speed) < 1400:
                self.recent_steps.append((self.s, new_speed))
                self.speed = (new_speed if self.speed is None
                              else 0.7 * self.speed + 0.3 * new_speed)
        self.seen = now
        self.raw_id = raw_id
        self.box = box
        self.s = s
        self.observation_count += 1
        self.active = True


class VisualTracker:
    def __init__(self, path):
        self.path = path
        self.tracks = {}
        self.next_uid = 1
        self.motion_samples = deque(maxlen=12000)
        self.last_update = -1
        self.calibration_bins = 0

    def update(self, now, confirmed_tracks):
        if now <= self.last_update:
            return []
        self.last_update = now
        observations = []
        for item in confirmed_tracks:
            try:
                raw_id = int(item["track_id"])
                box = tuple(int(round(x)) for x in item["box"])
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                s, error = self.path.project(center)
                if error > 165:
                    continue  # not close enough to the learned conveyor ring
                observations.append((raw_id, box, s))
            except (KeyError, ValueError, TypeError):
                continue

        previous = {
            uid: t for uid, t in self.tracks.items()
            if now - t.seen <= TRACK_MAX_GAP_SEC
        }
        edges = []
        for obs_idx, (raw_id, box, s) in enumerate(observations):
            for uid, t in previous.items():
                predicted = t.predict(now, self.path.length)
                dist = min((s - predicted) % self.path.length,
                           (predicted - s) % self.path.length)
                gap = now - t.seen
                max_dist = min(TRACK_MAX_TRAVEL_PX, 45 + 90 * gap)
                if dist > max_dist:
                    continue
                old_w = max(1, t.box[2] - t.box[0])
                old_h = max(1, t.box[3] - t.box[1])
                w = max(1, box[2] - box[0])
                h = max(1, box[3] - box[1])
                if min(old_w / w, w / old_w, old_h / h, h / old_h) < 0.5:
                    continue
                cost = dist + (0 if raw_id == t.raw_id else 15)
                edges.append((cost, obs_idx, uid))

        edges.sort()
        assignments = {}
        used = set()
        for cost, idx, uid in edges:
            if idx in assignments or uid in used:
                continue
            alternatives = [c for c, i, other in edges
                            if i == idx and other != uid and other not in used]
            if alternatives and min(alternatives) - cost < 14:
                continue
            assignments[idx] = uid
            used.add(uid)

        for t in self.tracks.values():
            t.active = False
        visible = []
        for idx, (raw_id, box, s) in enumerate(observations):
            uid = assignments.get(idx)
            if uid is None:
                uid = self.next_uid
                self.next_uid += 1
                self.tracks[uid] = Track(uid, now, raw_id, box, s)
            else:
                t = self.tracks[uid]
                t.update(now, raw_id, box, s, self.path.length)
                if t.recent_steps:
                    start_s, rate = t.recent_steps[-1]
                    if 8 <= abs(rate) <= 1000:
                        self.motion_samples.append((now, start_s, rate))
            visible.append(self.tracks[uid])
        for uid, t in list(self.tracks.items()):
            if now - t.seen > TRACK_MAX_GAP_SEC:
                del self.tracks[uid]
        return visible

    def direction(self):
        samples = [v for _, _, v in list(self.motion_samples)[-300:]]
        if len(samples) < 30:
            return 0
        avg = float(np.median(samples))
        return 1 if avg > 5 else -1 if avg < -5 else 0

    def travel_coordinate(self, positions):
        """Warp path distance to travel time using measured local image speed."""
        samples = list(self.motion_samples)
        if len(samples) < MIN_PATH_MOTION_SAMPLES:
            return None
        n_bins = 72
        grouped = [[] for _ in range(n_bins)]
        for _, s, v in samples:
            k = min(n_bins - 1, int(s / self.path.length * n_bins))
            grouped[k].append(abs(v))
        observed = sum(bool(g) for g in grouped)
        self.calibration_bins = observed
        if observed < int(n_bins * 0.55):
            return None
        speeds = np.array(
            [float(np.median(g)) if g else float('nan') for g in grouped]
        )
        available = np.where(np.isfinite(speeds))[0]
        extended_x = np.concatenate([available - n_bins, available, available + n_bins])
        extended_y = np.tile(speeds[available], 3)
        speeds = np.interp(np.arange(n_bins), extended_x, extended_y)
        speeds = np.maximum(speeds, 5.0)
        times = self.path.length / n_bins / speeds
        cumulative = np.concatenate([[0.0], np.cumsum(times)])
        total = float(cumulative[-1])
        if total <= 0:
            return None
        coords = []
        for s in positions:
            index_float = (s % self.path.length) / self.path.length * n_bins
            index = min(n_bins - 1, int(index_float))
            fraction = index_float - index
            coords.append(float((cumulative[index] + fraction * times[index]) / total))
        return coords


class RFIDReader(Thread):
    def __init__(self, queue, stop):
        super().__init__(daemon=True)
        self.queue = queue
        self.stop_signal = stop

    def run(self):
        try:
            from pylogix import PLC
        except ImportError:
            print("ERROR: pylogix is not installed; RFID1 not available.")
            return
        plc = PLC()
        plc.IPAddress = PLC_IP
        zero_reads = 0
        armed = False
        candidate = None
        count = 0
        start_time = 0
        last_error = 0
        try:
            while not self.stop_signal.is_set():
                try:
                    replies = plc.Read(RFID_WORDS)  # READ ONLY
                    if not isinstance(replies, (list, tuple)) or len(replies) != 4:
                        replies = [plc.Read(name) for name in RFID_WORDS]
                    words = []
                    for result in replies:
                        if result.Value is None or str(result.Status).lower() != 'success':
                            raise RuntimeError(f"RFID PLC read failure: {result.Status}")
                        words.append(signed16(result.Value))
                    raw = tuple(words)
                    now = time.monotonic()
                    if raw == (0, 0, 0, 0):
                        zero_reads += 1
                        if zero_reads >= 2:
                            armed = True
                            candidate = None
                            count = 0
                    else:
                        zero_reads = 0
                        number = TAG_TO_ID.get(raw)
                        if armed and number is not None:
                            if candidate == number:
                                count += 1
                            else:
                                candidate = number
                                count = 1
                                start_time = now
                            if count == 2:
                                self.queue.put((start_time, now, number, raw))
                                armed = False
                        elif number is None:
                            candidate = None
                            count = 0
                except Exception as exc:
                    if time.monotonic() - last_error > 5:
                        print('RFID1 WARNING:', exc)
                        last_error = time.monotonic()
                self.stop_signal.wait(POLL_SEC)
        finally:
            plc.Close()


class PatternMatcher:
    def __init__(self):
        self.events = deque(maxlen=60)
        self.mapping = {}  # visual-chain UID -> unverified physical RFID number
        self.proposal = None
        self.agreements = 0
        self.last_match_time = -100
        self.last_good_time = -100
        self.status = 'COLLECTING'
        self.best_score = None

    def add_event(self, event):
        self.events.append(event)
        # Preserve a previously matching assignment while collecting the
        # next valid event; the following frame must re-check it.
        self.status = 'NEW_RFID_EVENT'

    def two_lap_pattern(self):
        if len(self.events) < 13:
            return None
        events = list(self.events)[-13:]
        ids = [item[2] for item in events]
        first = ids[:6]
        if sorted(first) != [1, 2, 3, 4, 5, 6]:
            return None
        if ids[6:] != first + [first[0]]:
            return None
        gaps = np.diff([event[0] for event in events])
        if np.any(gaps < 0.25):
            return None
        first_lap = gaps[:6] / sum(gaps[:6])
        second_lap = gaps[6:] / sum(gaps[6:])
        if float(np.sqrt(np.mean((first_lap - second_lap) ** 2))) > 0.045:
            return None
        return first, (first_lap + second_lap) / 2

    def update(self, now, live, tracker):
        if now - self.last_match_time < MATCH_EVERY_SEC:
            return
        self.last_match_time = now
        pattern = self.two_lap_pattern()
        if pattern is None:
            self.status = 'WAIT_2_COMPLETE_RFID_LAPS'
            self.mapping.clear()
            return
        # Phase 51 used `now - track.created >= 1.2`, which repeatedly
        # rejected all-six-box frames when existing tracks got new UIDs.
        # Require direct repeated observations instead, and KEEP the six-way
        # agreement test below as the main anti-ID-swap protection.
        qualified = [t for t in live
                     if t.observation_count >= MIN_TRACK_OBSERVATIONS
                     and now - t.seen <= MAX_VIEW_UNSEEN_SEC]
        if len(live) != 6:
            self.status = f'WAIT_6_VISIBLE visible={len(live)} stable={len(qualified)}'
            if now - self.last_good_time > MAX_MAPPING_STALE_SEC:
                self.mapping.clear()
            return
        if len(qualified) != 6:
            counts = ','.join(str(t.observation_count) for t in live)
            self.status = f'WAIT_TRACK_STABILITY stable={len(qualified)}/6 counts={counts}'
            if now - self.last_good_time > MAX_MAPPING_STALE_SEC:
                self.mapping.clear()
            return
        direction = tracker.direction()
        if direction == 0:
            self.status = 'WAIT_DIRECTION'
            self.mapping.clear()
            return
        ordered = sorted(qualified, key=lambda t: direction * t.s)
        coord = tracker.travel_coordinate([t.s for t in ordered])
        if coord is None:
            self.status = f'CALIBRATING_MOTION samples={len(tracker.motion_samples)} bins={tracker.calibration_bins}/72'
            self.mapping.clear()
            return
        # Normalized estimated travel-time gaps, in direction of movement.
        visual_gaps = np.array([
            (coord[(i + 1) % 6] - coord[i]) % 1 if direction > 0
            else (coord[i] - coord[(i + 1) % 6]) % 1
            for i in range(6)
        ])
        if abs(float(sum(visual_gaps)) - 1) > 0.05:
            self.status = 'BAD_VISUAL_CYCLE'
            self.mapping.clear()
            return
        rfid_numbers, rfid_gaps = pattern
        scored = []
        for shift in range(6):
            reference = np.array([rfid_gaps[(i + shift) % 6]
                                  for i in range(6)])
            score = float(np.sqrt(np.mean((visual_gaps - reference) ** 2)))
            assignment = tuple(sorted(
                (t.uid, rfid_numbers[(i + shift) % 6])
                for i, t in enumerate(ordered)
            ))
            scored.append((score, shift, assignment))
        scored.sort(key=lambda a: a[0])
        best, second = scored[:2]
        self.best_score = round(best[0], 4)
        if (best[0] > MAX_PATTERN_RMS_ERROR or
                second[0] - best[0] < MIN_SECOND_BEST_MARGIN):
            self.status = f'AMBIGUOUS {best[0]:.3f}/{second[0]:.3f}'
            self.mapping.clear()
            self.proposal = None
            self.agreements = 0
            return
        if self.proposal == best[2]:
            self.agreements += 1
        else:
            self.proposal = best[2]
            self.agreements = 1
            self.mapping.clear()
        self.last_good_time = now
        if self.agreements >= MIN_REPEAT_AGREEMENTS:
            self.mapping = dict(best[2])
            self.status = f'PATTERN_MATCH_UNVERIFIED error={best[0]:.3f}'
        else:
            self.status = f'CHECKING_ALIGNMENT {self.agreements}/{MIN_REPEAT_AGREEMENTS}'


class CameraProxy:
    def __init__(self, real_camera, experiment):
        self.real_camera = real_camera
        self.experiment = experiment

    def read(self):
        ok, frame = self.real_camera.read()
        if ok:
            self.experiment.frame_time = time.monotonic()
        return ok, frame

    def __getattr__(self, name):
        return getattr(self.real_camera, name)


class Experiment:
    def __init__(self):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.file = RESULT_DIR / f'phase52_{stamp}.csv'
        self.handle = self.file.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.handle)
        self.writer.writerow(['time', 'type', 'rfid', 'raw_words',
                              'rfid_first_mono', 'rfid_confirm_mono',
                              'mapping_json', 'status', 'error'])
        self.queue = Queue()
        self.stop = Event()
        self.reader = RFIDReader(self.queue, self.stop)
        self.path = ConveyorPath(BOUNDARY_FILE)
        self.visual = VisualTracker(self.path)
        self.matcher = PatternMatcher()
        self.frame_time = None
        self.warned = False
        self.last_report = 0.0
        self.last_status = None

    def start(self):
        self.reader.start()
        print('PHASE 52 RFID1 ONLY / PLC READ ONLY')
        print("PHASE 52 ACTIVE: observation-based stability gate | file:", Path(__file__).resolve())
        print('~N = UNVERIFIED predicted RFID number, not ground truth')
        print('Need 13 sequential RFID reads (2 complete laps) and 6 tracks')
        print('CSV:', self.file)

    def collect(self):
        while True:
            try:
                item = self.queue.get_nowait()
            except Empty:
                return
            self.matcher.add_event(item)
            t0, t1, number, words = item
            self.writer.writerow([time.strftime('%H:%M:%S'), 'RFID', number,
                                  repr(words), f'{t0:.6f}', f'{t1:.6f}',
                                  '', self.matcher.status, ''])
            self.handle.flush()
            print(f'RFID1: {number}  reads={len(self.matcher.events)}')

    def show(self, title, frame, original_imshow):
        caller = inspect.currentframe().f_back
        variables = {}
        for _ in range(7):
            if caller is None:
                break
            if 'confirmed_tracks' in caller.f_locals:
                variables = caller.f_locals
                break
            caller = caller.f_back
        self.collect()
        tracks = variables.get('confirmed_tracks')
        now = self.frame_time
        picture = frame.copy()
        live = []
        if isinstance(tracks, (list, tuple)) and now is not None:
            live = self.visual.update(now, tracks)
            self.matcher.update(now, live, self.visual)
        elif not self.warned:
            print('WARNING: Phase 40 confirmed_tracks interface unavailable')
            self.warned = True
        visible = variables.get('visible_track_cache', {})
        for track in live:
            number = self.matcher.mapping.get(track.uid)
            # Do not present pattern prediction as verified physical ID.
            tag = f'RFID~{number}' if number is not None else 'RFID ?'
            data = visible.get(track.raw_id, {}) if isinstance(visible, dict) else {}
            label = str(data.get('label', ''))
            x1, y1, x2, y2 = track.box
            position = (max(0, x1), min(picture.shape[0] - 12, y2 + 28))
            cv2.putText(picture, f'{tag} {label}', position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 255) if number is not None else (230, 230, 230), 2)
        status_text = self.matcher.status
        if status_text != self.last_status:
            self.last_status = status_text
            if status_text.startswith(('AMBIGUOUS', 'BAD_VISUAL_CYCLE',
                                       'PATTERN_MATCH_UNVERIFIED', 'WAIT_DIRECTION')):
                print('PHASE 52:', status_text)
        cv2.rectangle(picture, (5, picture.shape[0] - 80),
                      (min(1080, picture.shape[1] - 5), picture.shape[0] - 5),
                      (0, 0, 0), -1)
        cv2.putText(picture, 'PHASE 52: RFID1 TIMING / SIX-BOX PATTERN',
                    (17, picture.shape[0] - 53), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(picture, f'{status_text} | ~ = predicted, NOT VERIFIED',
                    (17, picture.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        if now is not None and now - self.last_report >= 4:
            self.last_report = now
            mapping = dict(self.matcher.mapping)
            self.writer.writerow([time.strftime('%H:%M:%S'), 'STATE', '', '',
                                  '', '', json.dumps(mapping), status_text,
                                  self.matcher.best_score])
            self.handle.flush()
            print(f'PATTERN: {status_text} mapped={mapping}')
        original_imshow(title, picture)

    def close(self):
        self.stop.set()
        if self.reader.is_alive():
            self.reader.join(timeout=3)
        self.collect()
        self.handle.close()
        print('Phase 52 CSV saved:', self.file)


def main():
    if not BASE_FILE.exists():
        raise FileNotFoundError(f'Missing Phase 40 baseline: {BASE_FILE}')
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(f'Missing conveyor config: {BOUNDARY_FILE}')
    spec = importlib.util.spec_from_file_location('crest_phase40_reference', BASE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import Phase 40')
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    if not hasattr(live, 'open_camera') or not hasattr(live, 'main'):
        raise RuntimeError('Phase 40 camera interface differs; do not modify it')
    experiment = Experiment()
    original_open = live.open_camera
    original_show = cv2.imshow

    def open_hook():
        cap, index = original_open()
        return CameraProxy(cap, experiment), index

    def show_hook(title, frame):
        experiment.show(title, frame, original_show)

    live.open_camera = open_hook
    cv2.imshow = show_hook
    try:
        experiment.start()
        live.main()
    finally:
        live.open_camera = original_open
        cv2.imshow = original_show
        experiment.close()


if __name__ == '__main__':
    main()
