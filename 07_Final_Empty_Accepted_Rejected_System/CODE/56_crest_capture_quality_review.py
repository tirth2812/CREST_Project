"""CREST Phase 56: read-only review of newly recorded detector test frames.

Does not load models or access the camera, PLC, or training datasets.
Does not delete, rename, or exclude any image. QC flags are only suggestions.
"""
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get(
    "CREST_CAPTURE_ROOT",
    r"C:\Do_Not_Delete_PLC\Conference\09_Independent_Blind_Test_Captures",
))
OUT = ROOT / "_56_Capture_Quality_Review"
EXPECTED_SIZE = (1920, 1080)
COLUMNS = 4
ROWS = 5
THUMB_W = 320
THUMB_H = 180
HEADER_H = 37


def captioned_thumbnail(image, title, flags):
    tile = np.full((THUMB_H + HEADER_H, THUMB_W, 3), 32, dtype=np.uint8)
    if image is not None:
        h, w = image.shape[:2]
        scale = min(THUMB_W / w, THUMB_H / h)
        preview = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        ph, pw = preview.shape[:2]
        tile[HEADER_H + (THUMB_H - ph) // 2:HEADER_H + (THUMB_H - ph) // 2 + ph,
             (THUMB_W - pw) // 2:(THUMB_W - pw) // 2 + pw] = preview
    cv2.putText(tile, title[:45], (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.37, (240, 240, 240), 1, cv2.LINE_AA)
    status = ",".join(flags) if flags else "OK - review visually"
    cv2.putText(tile, status[:49], (6, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, (40, 200, 240) if flags else (100, 240, 100),
                1, cv2.LINE_AA)
    return tile


def main():
    print("CREST Phase 56 - FRESH TEST IMAGE QUALITY REVIEW (READ ONLY)", flush=True)
    if not ROOT.is_dir():
        raise SystemExit("Capture folder not found: " + str(ROOT))
    paths = sorted(
        p for p in ROOT.glob("session_*/test_frames/*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise SystemExit("No test images found in session_*/test_frames/")
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    thumb_pages = []
    counter = Counter()
    last_by_session = {}
    seen_pixel_hashes = {}

    for number, path in enumerate(paths, 1):
        session = path.parent.parent.name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        flags = []
        width = height = mean_brightness = sharpness = None
        previous_frame_difference = None
        duplicate_of = ""
        if image is None:
            flags.append("UNREADABLE")
        else:
            height, width = image.shape[:2]
            if (width, height) != EXPECTED_SIZE:
                flags.append("WRONG_RESOLUTION")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(
                cv2.resize(gray, (640, 360)), cv2.CV_64F).var())
            if mean_brightness < 15:
                flags.append("VERY_DARK")
            if sharpness < 12:
                flags.append("CHECK_SHARPNESS")
            # Identical decoded pixels across any sessions, not merely similar backgrounds.
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen_pixel_hashes:
                duplicate_of = seen_pixel_hashes[digest]
                flags.append("EXACT_PIXEL_REPEAT")
            else:
                seen_pixel_hashes[digest] = str(path)
            # A nearly motionless *entire* image can indicate frozen camera output.
            small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
            if session in last_by_session:
                prev = last_by_session[session]
                previous_frame_difference = float(np.mean(
                    cv2.absdiff(small, prev)))
                if previous_frame_difference < 0.20:
                    flags.append("POSSIBLE_FROZEN")
            last_by_session[session] = small
        for flag in flags:
            counter[flag] += 1
        records.append({
            "number": number,
            "session": session,
            "filename": path.name,
            "path": str(path),
            "width": width if width is not None else "",
            "height": height if height is not None else "",
            "mean_brightness": f"{mean_brightness:.2f}" if mean_brightness is not None else "",
            "sharpness_heuristic": f"{sharpness:.2f}" if sharpness is not None else "",
            "difference_from_previous_in_session": (
                f"{previous_frame_difference:.3f}"
                if previous_frame_difference is not None else ""),
            "exact_pixel_repeat_of": duplicate_of,
            "flags": ";".join(flags),
        })
        thumb_pages.append(captioned_thumbnail(
            image, f"{number:03d} {session} {path.name}", flags))
        if number % 20 == 0:
            print("Checked %d / %d images" % (number, len(paths)), flush=True)

    csv_path = OUT / "image_quality_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    images_per_page = COLUMNS * ROWS
    n_pages = (len(thumb_pages) + images_per_page - 1) // images_per_page
    for page_index in range(n_pages):
        canvas = np.full((ROWS * (THUMB_H + HEADER_H), COLUMNS * THUMB_W, 3),
                         25, dtype=np.uint8)
        start = page_index * images_per_page
        for k, thumbnail in enumerate(thumb_pages[start:start + images_per_page]):
            row, col = divmod(k, COLUMNS)
            y, x = row * (THUMB_H + HEADER_H), col * THUMB_W
            canvas[y:y + THUMB_H + HEADER_H, x:x + THUMB_W] = thumbnail
        cv2.imwrite(str(OUT / f"contact_sheet_{page_index + 1:02d}.jpg"),
                    canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    session_counts = Counter(row["session"] for row in records)
    report_lines = [
        "CREST PHASE 56 - CAPTURE QUALITY REPORT",
        "Read-only quality screening, NOT an accuracy evaluation.",
        f"Total images scanned: {len(paths)}",
        f"Unreadable: {counter['UNREADABLE']}",
        f"Wrong resolution: {counter['WRONG_RESOLUTION']}",
        f"Very dark: {counter['VERY_DARK']}",
        f"Check sharpness manually: {counter['CHECK_SHARPNESS']}",
        f"Exact repeated pixels: {counter['EXACT_PIXEL_REPEAT']}",
        f"Possible frozen frame pairs: {counter['POSSIBLE_FROZEN']}",
        "",
        "Images per session:",
        *[f"  {key}: {value}" for key, value in sorted(session_counts.items())],
        "",
        "Flags are screening hints, not permission to discard hard images.",
        "Inspect all contact sheets, including images flagged OK.",
        "The collected frames are not necessarily independent physical pallet passes.",
        "Do not train, tune, or relabel from predictions on this held-out set.",
        f"CSV: {csv_path}",
        f"Contact sheets: {n_pages}",
    ]
    summary = OUT / "quality_summary.txt"
    summary.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines), flush=True)
    print("\nREVIEW FOLDER: " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
