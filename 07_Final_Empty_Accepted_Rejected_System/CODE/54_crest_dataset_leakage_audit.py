"""CREST Phase 54: READ-ONLY, three-dataset split-overlap audit.

This script does not change datasets, checkpoints, Phase 40 or any PLC tag.
Exact duplicates are objective; visual/session matches are REVIEW flags, not proof.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference")
DATASETS = {
    "PALLET_100": ROOT / "00_Pallet_100_YOLO26_Dataset_v2",
    "EMPTY_OBJECT": ROOT / "03_Empty_Object_Classifier" / "DATASET" / "prepared_yolo26",
    "X_NO_X": ROOT / "00_Current_Working_X_NO_X_Dataset",
}
EXPECTED = {
    "PALLET_100": (80, 10, 10),
    "EMPTY_OBJECT": (114, 30, 4),
    "X_NO_X": (168, 30, 24),
}
OUT = (ROOT / "07_Final_Empty_Accepted_Rejected_System" /
       "RESULTS" / "54_Dataset_Leakage_Audit")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}
# These are screening thresholds, NOT a proof of shared source.
DHASH_DISTANCE_LIMIT = 7
THUMBNAIL_MAE_LIMIT = 18.0


def split_for(path: Path, root: Path):
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts[:-1]):
        key = SPLITS.get(part.lower())
        if key:
            label = parts[index + 1] if index + 2 < len(parts) else ""
            if label.lower() in {"images", "labels"}:
                label = ""
            return key, label
    return "UNKNOWN", ""


def hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path):
    # cv2.imdecode + np.fromfile supports non-ASCII Windows paths.
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Image could not be opened")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).reshape(-1):
        value = (value << 1) | int(bit)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return value, thumb, bgr.shape[1], bgr.shape[0]


def recording_hint(filename: str):
    # Filename evidence only, not proof of a recording session.
    match = re.search(r"(?<!\d)20\d{6}[_-]\d{6}(?!\d)", filename)
    if not match:
        return ""
    return filename[:match.end()].lower()


def save_csv(path, fields, records):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    for model, root in DATASETS.items():
        if not root.is_dir():
            failures.append((model, str(root), "DATASET_DIRECTORY_MISSING"))
            continue
        images = sorted(p for p in root.rglob("*")
                        if p.is_file() and p.suffix.lower() in EXTS)
        for path in images:
            split, label = split_for(path, root)
            try:
                dhash, thumbnail, width, height = fingerprint(path)
                records.append({
                    "dataset": model, "split": split, "class_folder": label,
                    "filename": path.name, "path": str(path),
                    "width": width, "height": height,
                    "sha256": hash_file(path),
                    "dhash64": f"{dhash:016x}",
                    "session_hint": recording_hint(path.name),
                    "_dhash": dhash, "_thumb": thumbnail,
                })
            except (OSError, ValueError, cv2.error) as exc:
                failures.append((model, str(path), str(exc)))

    fields = ["dataset", "split", "class_folder", "filename", "path",
              "width", "height", "sha256", "dhash64", "session_hint"]
    save_csv(OUT / "image_manifest.csv", fields, records)

    suspects = []
    by_model = defaultdict(list)
    for rec in records:
        by_model[rec["dataset"]].append(rec)

    for model, entries in by_model.items():
        for i, left in enumerate(entries):
            for right in entries[i + 1:]:
                if left["split"] == right["split"]:
                    continue
                if "UNKNOWN" in (left["split"], right["split"]):
                    continue
                exact = left["sha256"] == right["sha256"]
                distance = (left["_dhash"] ^ right["_dhash"]).bit_count()
                mae = float(np.mean(np.abs(
                    left["_thumb"].astype(np.int16) -
                    right["_thumb"].astype(np.int16))))
                near = distance <= DHASH_DISTANCE_LIMIT and mae <= THUMBNAIL_MAE_LIMIT
                same_name = left["filename"].lower() == right["filename"].lower()
                session = (bool(left["session_hint"]) and
                           left["session_hint"] == right["session_hint"])
                flags = []
                if exact:
                    flags.append("EXACT_FILE_DUPLICATE")
                if near and not exact:
                    flags.append("SIMILAR_IMAGE_REVIEW")
                if same_name and not exact:
                    flags.append("SAME_FILENAME_REVIEW")
                if session:
                    flags.append("SAME_TIMESTAMP_PREFIX_REVIEW")
                if flags:
                    suspects.append({
                        "dataset": model, "flags": ";".join(flags),
                        "split_a": left["split"], "split_b": right["split"],
                        "class_a": left["class_folder"],
                        "class_b": right["class_folder"],
                        "file_a": left["path"], "file_b": right["path"],
                        "dhash_distance": distance,
                        "thumbnail_gray_mae": f"{mae:.2f}",
                        "session_hint": left["session_hint"] if session else "",
                    })
    suspect_cols = ["dataset", "flags", "split_a", "split_b", "class_a",
                    "class_b", "file_a", "file_b", "dhash_distance",
                    "thumbnail_gray_mae", "session_hint"]
    suspects.sort(key=lambda x: (
        x["dataset"],
        0 if "EXACT_FILE_DUPLICATE" in x["flags"] else 1,
        int(x["dhash_distance"]),
    ))
    save_csv(OUT / "cross_split_suspects.csv", suspect_cols, suspects)
    save_csv(OUT / "read_errors.csv", ["dataset", "path", "error"],
             [{"dataset": d, "path": p, "error": e} for d, p, e in failures])

    report = ["CREST DATASET SPLIT AUDIT — READ ONLY",
              f"Generated: {datetime.now().isoformat(timespec='seconds')}",
              "Scope: final 100-image detector and two deployed classifiers",
              "", "SPLIT INVENTORY"]
    print("\nCREST Phase 54 — dataset leakage audit (READ ONLY)")
    for model in DATASETS:
        counts = Counter(r["split"] for r in by_model[model])
        counts_txt = (f"train={counts['train']}, val={counts['val']}, "
                      f"test={counts['test']}, unknown={counts['UNKNOWN']}")
        expected = EXPECTED[model]
        expected_txt = f"expected={expected[0]}/{expected[1]}/{expected[2]}"
        report.append(f"{model}: {counts_txt}; {expected_txt}")
        print(f"  {model}: {counts_txt}")
        if tuple(counts[s] for s in ("train", "val", "test")) != expected:
            report.append("  WARNING: split inventory differs from October 8 report")
    count_flags = Counter(flag for row in suspects for flag in row["flags"].split(";"))
    report += ["", "CROSS-SPLIT FLAGS (pair counts, not image counts)"]
    for key in ("EXACT_FILE_DUPLICATE", "SIMILAR_IMAGE_REVIEW",
                "SAME_FILENAME_REVIEW", "SAME_TIMESTAMP_PREFIX_REVIEW"):
        report.append(f"{key}: {count_flags[key]}")
        print(f"  {key}: {count_flags[key]}")
    report += ["", f"Unreadable / missing items: {len(failures)}",
               "", "INTERPRETATION",
               "EXACT_FILE_DUPLICATE in train/test is direct file-level overlap.",
               "Visual similarity is a REVIEW FLAG, not proof of data leakage.",
               "A common background can make different conveyor frames look similar.",
               "No flagged images does NOT establish session independence.",
               "Filename timestamp groups are hints, not verified physical passes.",
               "This script CANNOT audit historical checkpoint ancestry.",
               "Do not change datasets, remove examples, or edit reported accuracy.",
               "Open cross_split_suspects.csv and inspect source sessions.",
               "For defensible final performance, use a new untouched test session",
               "if existing sources cannot be independently verified."]
    (OUT / "audit_summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nOutput:", OUT)
    print("Open audit_summary.txt and cross_split_suspects.csv")


if __name__ == "__main__":
    run()
