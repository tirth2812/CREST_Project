# dynamic_live_pallet_detection.py

**Stage:** Final live inspection and persistent RFID identity
**Source path in package:** `07_Final_Empty_Accepted_Rejected_System/CODE/dynamic_live_pallet_detection.py`

## Purpose

Required dynamic tracking module loaded by Phase 40 and Phase 65 at runtime.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\dynamic_live_pallet_detection.py`. This README is generated; the Python file is unchanged.

## Required at runtime

The final scripts load this module from their `CODE` directory. Keep this file alongside Phase 40 and Phase 65 in the archived Stage 07 `CODE/` folder.

The final application uses tracker objects and boundary helpers from this module; do not modify it during documentation cleanup.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Classes

| Class | Line | Responsibility |
|---|---:|---|
| `DynamicTracker` | 908 | Implements `DynamicTracker`; see the indicated code section. |

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `write_readme()` | 52 | Handles write readme; precise behavior is in the source. |
| `point_distance()` | 242 | Handles point distance; precise behavior is in the source. |
| `normalize_points()` | 249 | Handles normalize points; precise behavior is in the source. |
| `denormalize_points()` | 259 | Handles denormalize points; precise behavior is in the source. |
| `clean_polygon()` | 273 | Handles clean polygon; precise behavior is in the source. |
| `save_boundary()` | 307 | Handles save boundary; precise behavior is in the source. |
| `load_boundary()` | 332 | Handles load boundary; precise behavior is in the source. |
| `create_ring_mask()` | 371 | Handles create ring mask; precise behavior is in the source. |
| `calibration_screen()` | 406 | Handles calibration screen; precise behavior is in the source. |
| `box_iou()` | 758 | Calculate intersection-over-union between boxes. |
| `ring_overlap_ratio()` | 807 | Handles ring overlap ratio; precise behavior is in the source. |
| `box_center_inside_ring()` | 868 | Handles box center inside ring; precise behavior is in the source. |
| `get_yolo_detections()` | 1056 | Handles get yolo detections; precise behavior is in the source. |
| `filter_by_ring()` | 1111 | Handles filter by ring; precise behavior is in the source. |
| `draw_live_frame()` | 1152 | Handles draw live frame; precise behavior is in the source. |
| `main()` | 1336 | Run the program's main processing loop. |

## Imported Python modules

`cv2`, `json`, `math`, `numpy`, `pathlib`, `time`, `ultralytics`

## Model filenames visible in source

`pallet_yolo26n_best.pt`

## Preservation

Only this Markdown document was generated; the program source has not been edited.
