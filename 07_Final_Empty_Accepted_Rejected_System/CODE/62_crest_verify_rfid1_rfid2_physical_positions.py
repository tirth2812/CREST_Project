"""CREST Phase 62: identify which PLC RFID reader is physically TOP/BOTTOM.

This is an independent read-only diagnostic, NOT a replacement for Phase 40.
Never writes PLC tags, controls actuators, changes YOLO models or edits files.
PLC Reader A = RFID1 words [96:99]; Reader B = RFID2 words [128:131].
The existing RFID station-box names are treated as HISTORICAL guesses only.

Usage:
  python -u 62_crest_verify_rfid1_rfid2_physical_positions.py --self-test
  python -u 62_crest_verify_rfid1_rfid2_physical_positions.py
  python -u 62_crest_verify_rfid1_rfid2_physical_positions.py --console-only

One physical tagged pallet passing each RFID station establishes which PLC
word group belongs to which physical location. Compare printed event/time
with the pallet's actual TOP/BOTTOM position; do not infer from config names.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference")
STAGE = ROOT / "07_Final_Empty_Accepted_Rejected_System"
CONFIG = STAGE / "CONFIG" / "rfid_stations.json"
OUTPUT = STAGE / "RESULTS" / "62_RFID_Station_Validation_Readonly"
PLC_IP = "192.168.82.231"
PREFIX = "IO_Link_Master_ST1_to_ST4:I1.Data"
READER_TAGS = {
    "RFID1_W96_99": [f"{PREFIX}[{i}]" for i in range(96, 100)],
    "RFID2_W128_131": [f"{PREFIX}[{i}]" for i in range(128, 132)],
}
RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}
NO_TAG = (0, 0, 0, 0)
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
CAMERA_INDEX = 1
# Coordinates verified as ROIs in the earlier camera setup, but the names
# previously associated with these ROIs have NOT been verified electrically.
HISTORICAL_TOP = (1123, 578, 1183, 675)
HISTORICAL_BOTTOM = (823, 734, 884, 815)


def load_camera_rois():
    top, bottom = HISTORICAL_TOP, HISTORICAL_BOTTOM
    if not CONFIG.is_file():
        return top, bottom
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        image_w = int(data.get("image_width", FRAME_WIDTH))
        image_h = int(data.get("image_height", FRAME_HEIGHT))
        found = []
        for key in ("rfid1", "rfid2"):
            info = data[key]["box"]
            raw = [int(info[k]) for k in ("x1", "y1", "x2", "y2")]
            found.append(tuple([
                int(raw[0] * FRAME_WIDTH / image_w),
                int(raw[1] * FRAME_HEIGHT / image_h),
                int(raw[2] * FRAME_WIDTH / image_w),
                int(raw[3] * FRAME_HEIGHT / image_h),
            ]))
        found.sort(key=lambda b: (b[1] + b[3]) / 2)
        top, bottom = found
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print("Warning: station ROI configuration unreadable; using historical coordinates:", exc)
    return top, bottom


def read_words(plc, tags):
    """Read exactly four input words, first batch, then individual fallback."""
    try:
        results = plc.Read(tags)
        if not isinstance(results, (tuple, list)):
            raise RuntimeError(f"Unexpected batch type {type(results).__name__}")
        if len(results) != len(tags):
            raise RuntimeError(f"Unexpected batch length {len(results)}")
        if not all(getattr(r, "Status", "") == "Success" for r in results):
            raise RuntimeError("PLC batch status failed")
        return tuple(int(r.Value) for r in results)
    except Exception:
        vals = []
        for tag in tags:
            result = plc.Read(tag)
            if getattr(result, "Status", "") != "Success" or result.Value is None:
                raise RuntimeError(f"{tag}: {getattr(result, 'Status', 'no response')}")
            vals.append(int(result.Value))
        return tuple(vals)


def plc_poll_thread(stop, events, status):
    try:
        from pylogix import PLC
    except ImportError:
        status["error"] = "pylogix is missing. Run: python -m pip install pylogix"
        return
    previous = {reader: NO_TAG for reader in READER_TAGS}
    last_error = 0.0
    try:
        with PLC() as comm:
            comm.IPAddress = PLC_IP
            comm.SocketTimeout = 2.0
            while not stop.is_set():
                any_online = False
                for reader, tags in READER_TAGS.items():
                    if stop.is_set():
                        break
                    try:
                        words = read_words(comm, tags)
                        any_online = True
                        status[reader] = "ONLINE"
                        status["error"] = ""
                    except Exception as exc:
                        status[reader] = "READ ERROR"
                        status["error"] = f"{reader}: {exc}"
                        if time.monotonic() - last_error > 5:
                            print("PLC read error:", status["error"], flush=True)
                            last_error = time.monotonic()
                        previous[reader] = NO_TAG
                        continue
                    if words != NO_TAG and words != previous[reader]:
                        event = {
                            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "reader": reader,
                            "pallet": RFID_MAP.get(words),
                            "raw": words,
                            "monotonic": time.monotonic(),
                        }
                        events.put(event)
                        print(f"{event['timestamp']}  {reader}  "
                              f"PALLET={event['pallet'] or 'UNRECOGNIZED'}  "
                              f"RAW={words}", flush=True)
                    previous[reader] = words
                if any_online:
                    status["connected"] = True
                stop.wait(0.07)
    except Exception as exc:
        status["error"] = f"PLC session error: {exc}"
        print(status["error"], flush=True)


def draw_label(frame, message, origin, color, size=0.85):
    import cv2
    cv2.putText(frame, message, origin, cv2.FONT_HERSHEY_SIMPLEX,
                size, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, message, origin, cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 2, cv2.LINE_AA)


def annotate_frame(frame, top, bottom, last, status):
    import cv2
    height, width = frame.shape[:2]
    sx, sy = width / FRAME_WIDTH, height / FRAME_HEIGHT
    for region, name, color in ((top, "TOP CAMERA ROI", (0, 255, 255)),
                                (bottom, "BOTTOM CAMERA ROI", (255, 255, 0))):
        x1, y1, x2, y2 = region
        scaled = (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
        cv2.rectangle(frame, scaled[:2], scaled[2:], color, 3)
        draw_label(frame, name, (max(5, scaled[0] - 90),
                      max(90, scaled[1] - 14)), color, 0.68)
    cv2.rectangle(frame, (0, 0), (width, 118), (0, 0, 0), -1)
    draw_label(frame, "CREST READ-ONLY RFID STATION CHECK - Q exit / S snapshot",
               (18, 35), (255, 255, 255), 0.72)
    draw_label(frame, "RFID1 = PLC [96-99]  |  RFID2 = PLC [128-131]",
               (18, 70), (255, 255, 255), 0.68)
    if last:
        seconds = time.monotonic() - last["monotonic"]
        if seconds < 4.0:
            message = f"LATEST: {last['reader']} = PALLET {last['pallet']}  ({seconds:.1f}s ago)"
            draw_label(frame, message, (18, 103), (0, 255, 0), 0.73)
        else:
            draw_label(frame, "Waiting for new tag event...", (18, 103), (200, 200, 200), 0.7)
    else:
        draw_label(frame, "Waiting for new tag event...", (18, 103), (200, 200, 200), 0.7)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--console-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert len(RFID_MAP) == 6
        assert len(set(RFID_MAP.values())) == 6
        assert set(READER_TAGS) == {"RFID1_W96_99", "RFID2_W128_131"}
        assert all(len(x) == 4 for x in READER_TAGS.values())
        assert RFID_MAP[(-8188, 336, -18499, -14824)] == 1
        print("SELF-TEST PASSED. Hardware placement still requires live verification.")
        return
    run = OUTPUT / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    log_path = run / "rfid1_rfid2_events.csv"
    top, bottom = load_camera_rois()
    print("CREST Phase 62: verify actual RFID1/RFID2 physical placement", flush=True)
    print("RFID1: read-only PLC input words 96-99", flush=True)
    print("RFID2: read-only PLC input words 128-131", flush=True)
    print("Historical TOP/BOTTOM camera rectangles do not prove wiring.")
    print("Top ROI:", top, " Bottom ROI:", bottom)
    print("No PLC writes. No YOLO/model calls. Phase 40 unchanged.")
    print("Log:", log_path, flush=True)
    stop = threading.Event()
    events = queue.Queue()
    status = {"connected": False, "error": "", "RFID1_W96_99": "WAITING",
              "RFID2_W128_131": "WAITING"}
    capture = None
    window = "CREST RFID station identification (read only)"
    last = None
    try:
        if not args.console_only:
            import cv2
            capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(CAMERA_INDEX)
            if not capture.isOpened():
                raise RuntimeError("Camera index 1 unavailable. Close Phase 40, then retry; or use --console-only")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 720)
        thread = threading.Thread(target=plc_poll_thread, args=(stop, events, status), daemon=True)
        thread.start()
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "plc_word_group", "physical_pallet_id",
                             "raw_four_words", "snapshot"])
            while True:
                ok, frame = (True, None) if args.console_only else capture.read()
                if not ok:
                    print("Camera frame unavailable; close this diagnostic with Q or Ctrl+C.")
                    break
                display = annotate_frame(frame.copy(), top, bottom, last, status) if frame is not None else None
                while not events.empty():
                    current = events.get_nowait()
                    last = current
                    snapshot = ""
                    if frame is not None:
                        display = annotate_frame(frame.copy(), top, bottom, current, status)
                        snapshot = (f"{current['timestamp'].replace(':', '').replace('-', '')}_"
                                    f"{current['reader']}_P{current['pallet']}.jpg")
                        cv2.imwrite(str(run / snapshot), display)
                    writer.writerow([current["timestamp"], current["reader"], current["pallet"],
                                     repr(current["raw"]), snapshot])
                    handle.flush()
                if args.console_only:
                    time.sleep(0.1)
                    continue
                display = annotate_frame(frame.copy(), top, bottom, last, status)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 255
                if key == ord("q") or key == 27:
                    break
                if key == ord("s"):
                    snapshot = "manual_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                    cv2.imwrite(str(run / snapshot), display)
                    print("Saved:", run / snapshot)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if capture is not None:
            capture.release()
            cv2.destroyAllWindows()
        print("Saved RFID events:", log_path, flush=True)
        print("Do not infer top/bottom from old file names; verify against event snapshots.")


if __name__ == "__main__":
    main()
