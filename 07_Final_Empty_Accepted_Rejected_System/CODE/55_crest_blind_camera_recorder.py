"""CREST Phase 55: camera-only, independent held-out recording.
No YOLO models, PLC, RFID, robot control, or training-file changes.
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

CAMERA_INDEX = 1
WIDTH, HEIGHT = 1920, 1080
DURATION_SECONDS = 180
VIDEO_FPS = 12.0
TEST_FRAME_INTERVAL = 4.0
OUTPUT_ROOT = Path(
    r"C:\Do_Not_Delete_PLC\Conference\09_Independent_Blind_Test_Captures"
)


def open_writer(folder):
    for filename, codec in [
        ("raw_camera.mp4", "mp4v"),
        ("raw_camera.avi", "XVID"),
        ("raw_camera_mjpg.avi", "MJPG"),
    ]:
        path = folder / filename
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec),
            VIDEO_FPS, (WIDTH, HEIGHT)
        )
        if writer.isOpened():
            return writer, path
        writer.release()
        if path.exists():
            path.unlink()
    raise RuntimeError("No working video codec (MP4/AVI) was found.")


def main():
    print("CREST PHASE 55 - FRESH, READ-ONLY CAMERA TEST")
    print("Close Phase 40 before running; show only moving empty pallets.")
    print("Recording 180 seconds. Press Q or ESC in preview to end early.")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Camera index 1 unavailable. Close other camera apps.")

    writer = None
    timestamp_file = None
    session = None
    start = None
    video_frames = 0
    test_frames = 0
    frames_read = 0
    finish_reason = "completed"
    started_at = datetime.now().astimezone().isoformat()
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("Camera opened but did not supply a frame.")
        h, w = frame.shape[:2]
        print("Actual camera resolution: %d x %d" % (w, h))
        if (w, h) != (WIDTH, HEIGHT):
            raise RuntimeError("Expected 1920x1080; no test data was collected.")

        session = OUTPUT_ROOT / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        if session.exists():
            raise RuntimeError("Session folder exists. Retry in one second.")
        (session / "test_frames").mkdir(parents=True)
        writer, video_path = open_writer(session)
        timestamp_file = open(
            session / "capture_timestamps.csv", "w", newline="", encoding="utf-8"
        )
        log = csv.writer(timestamp_file)
        log.writerow(["video_frame", "elapsed_seconds", "local_timestamp", "test_png"])
        print("OUTPUT FOLDER: " + str(session), flush=True)
        start = time.monotonic()
        next_video = 0.0
        next_test = 0.0
        next_progress = 20.0

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= DURATION_SECONDS:
                break
            frames_read += 1
            if elapsed >= next_video:
                test_name = ""
                if elapsed >= next_test:
                    test_frames += 1
                    test_name = "test_%04d.png" % test_frames
                    saved = cv2.imwrite(
                        str(session / "test_frames" / test_name),
                        frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    )
                    if not saved:
                        raise RuntimeError("Could not save " + test_name)
                    next_test += TEST_FRAME_INTERVAL

                writer.write(frame)
                video_frames += 1
                log.writerow([
                    video_frames, "%.6f" % elapsed,
                    datetime.now().astimezone().isoformat(), test_name
                ])
                next_video = max(
                    next_video + 1.0 / VIDEO_FPS,
                    elapsed + 1.0 / VIDEO_FPS
                )
                preview = cv2.resize(frame, (960, 540))
                cv2.putText(
                    preview, "RECORDING %ds / %ds | TEST IMAGES %d" % (
                        elapsed, DURATION_SECONDS, test_frames
                    ), (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2
                )
                cv2.imshow("CREST Independent Test - Q to stop", preview)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                finish_reason = "user_stopped_early"
                break
            if elapsed >= next_progress:
                print("Elapsed %ds | test frames %d | video frames %d" % (
                    elapsed, test_frames, video_frames
                ), flush=True)
                next_progress += 20

            ok, frame = camera.read()
            if not ok or frame is None:
                finish_reason = "camera_disconnected"
                print("WARNING: Camera stopped returning frames.")
                break
            if frame.shape[:2] != (HEIGHT, WIDTH):
                finish_reason = "camera_resolution_changed"
                print("WARNING: Camera resolution changed; recording stopped.")
                break
    finally:
        duration = time.monotonic() - start if start is not None else 0
        camera.release()
        if writer is not None:
            writer.release()
        if timestamp_file is not None:
            timestamp_file.close()
        cv2.destroyAllWindows()
        if session is not None:
            manifest = {
                "session_started_local": started_at,
                "session_finished_local": datetime.now().astimezone().isoformat(),
                "camera_index": CAMERA_INDEX,
                "resolution": [WIDTH, HEIGHT],
                "duration_seconds": round(duration, 3),
                "requested_camera_fps": 30,
                "nominal_video_fps": VIDEO_FPS,
                "measured_video_frames_per_second": round(
                    video_frames / duration, 3
                ) if duration > 0 else None,
                "captured_frames_read": frames_read,
                "video_frames_saved": video_frames,
                "test_images_saved": test_frames,
                "test_frame_interval_seconds": TEST_FRAME_INTERVAL,
                "video_path": str(video_path) if writer is not None else None,
                "stop_reason": finish_reason,
                "models_or_plc_used": False,
                "ground_truth_labeled": False,
                "note": "Fresh session, fixed-interval PNG sampling. Do not use in "
                        "training or threshold tuning. Images from recurring "
                        "pallet passes are correlated, not independent trials."
            }
            (session / "session_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print("\nRECORDING SAVED TO: " + str(session))
            print("VIDEO FRAMES: %d | TEST IMAGES: %d" % (
                video_frames, test_frames
            ))
            print("Send this CMD summary back. Do not train on this test data.")


if __name__ == "__main__":
    main()
