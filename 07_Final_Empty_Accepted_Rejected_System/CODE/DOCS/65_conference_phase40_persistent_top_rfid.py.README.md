# 65_conference_phase40_persistent_top_rfid.py

**Stage:** Final live inspection and persistent RFID identity
**Source path in package:** `07_Final_Empty_Accepted_Rejected_System/CODE/65_conference_phase40_persistent_top_rfid.py`

## Purpose

Final live application: Phase 40 inspection plus persistent, read-only TOP RFID2 physical-pallet association.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\65_conference_phase40_persistent_top_rfid.py`. This README is generated; the Python file is unchanged.

## How Phase 65 works

1. Open the camera and load the detector, EMPTY/OBJECT classifier, and X/NO_X classifier.
2. Use the separate dynamic tracking module to obtain pallet IDs; suppress duplicate boxes and bridge short tracking interruptions.
3. Determine EMPTY or OBJECT_PRESENT, then collect permitted X/NO_X observations for ACCEPTED or REJECTED decisions.
4. Read **TOP RFID2** through PLC input words **128–131** on a background thread (read-only). Recognized four-word values identify physical pallets **1–6**.
5. Associate a recent RFID event with a unique eligible track entering the calibrated top zone. Keep the physical identity in a separate in-session ledger even if the vision track expires.
6. Recover labels only when geometric/state continuity is sufficiently unambiguous; a later matching RFID scan verifies or corrects an association.
7. Draw the physical pallet number alongside the Phase 40 inspection display and write RFID/audit logs.

**Important:** Physical identity memory does not guarantee the box stays visible. If there is no safe match, an unknown track remains unlabeled until reconnection or RFID verification. A restart begins a new in-memory learning session.

## Hardware and input files

- Camera: 1920 × 1080 requested by source; available capture hardware must be verified.
- PLC: `192.168.82.231`, read-only via `pylogix`; TOP RFID2 words `IO_Link_Master_ST1_to_ST4:I1.Data[128..131]`.
- Tracker: `CODE/dynamic_live_pallet_detection.py` (required).
- Pallet detector: Conference Stage 00 checkpoint `00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights/best.pt` (original absolute path in source).
- EMPTY classifier: `MODELS/empty_object_yolo26n_cls_best.pt`.
- X/NO_X classifier: `MODELS/x_no_x_yolo26n_medium_focus_final_hard_best.pt`.
- Conveyor boundary: `CONFIG/conveyor_boundary.json`.
- Saved top RFID zone: `CONFIG/63_top_rfid_center_zone.json`; press **Z** to redraw it if necessary.

## Run on the original Conference PC

Activate the environment:

```bat
call C:\VLM_Finetune\vlm_env\Scripts\activate.bat
```

Run the hardware-free identity test:

```bat
python -u "C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\65_conference_phase40_persistent_top_rfid.py" --self-test
```

Run the live hardware application:

```bat
python -u "C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\65_conference_phase40_persistent_top_rfid.py"
```

**Keys:** `Z` edit top zone; `M` clear/relearn RFID identity after manual pallet rearrangement; `R` recheck classification after intentionally changing a stationary pallet; `S` save screenshot; `Q` or `Esc` quit.

## Results and constraints

- RFID CSV logs: `RESULTS/65_Top_RFID_Persistent_Readonly/` in the original Conference runtime.
- Inspection logs and screenshots: `RESULTS/65_Phase40_Top_RFID_Persistent/`.
- The copied archive is for provenance and version control; source paths refer to the original installed directory. Cloning alone is **not** a hardware-portability guarantee.
- Do not alter Phase 40's detection, box persistence, or classification logic to improve documentation.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Classes

| Class | Line | Responsibility |
|---|---:|---|
| `TopRFIDReadOnly` | 3621 | Poll PLC RFID2 inputs in a background thread without PLC writes. |
| `TopRFIDAssociation` | 3692 | Maintain physical ID memory, pending RFID events, spatial reconnection, and identity corrections. |

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `log_event()` | 272 | Append event text to the existing decision log. |
| `load_stage08()` | 456 | Load the standalone dynamic tracker module and synchronize its conveyor boundary path. |
| `open_camera()` | 488 | Try configured camera inputs and initialize video capture. |
| `make_pallet_frame()` | 570 | Prepare camera input for pallet detection. |
| `make_symbol_frame()` | 624 | Prepare image input for symbol classification. |
| `crop_pallet()` | 717 | Extract a pallet region from an image. |
| `box_iou()` | 791 | Calculate intersection-over-union between boxes. |
| `smaller_box_overlap_ratio()` | 878 | Measure overlap relative to the smaller box. |
| `vertical_overlap_ratio()` | 961 | Measure the vertical overlap of two boxes. |
| `box_size_ratio()` | 1010 | Compare the size of two boxes. |
| `center_ratios()` | 1060 | Compare box-center positions relative to their sizes. |
| `merge_pallet_detections()` | 1144 | Merge competing detections for likely duplicate pallets. |
| `current_boxes_are_duplicate()` | 1222 | Check whether simultaneous boxes represent the same pallet. |
| `filter_current_duplicate_tracks()` | 1281 | Suppress simultaneous duplicate track candidates. |
| `old_new_tracks_match()` | 1361 | Test geometric continuity between lost and new tracking IDs. |
| `state_bridge_score()` | 1416 | Score an old/new tracking-state match using geometric constraints. |
| `has_meaningful_track_state()` | 1484 | Check whether a track has state worth preserving. |
| `move_track_state()` | 1505 | Move inspection state from an old tracker ID to its verified replacement. |
| `track_can_receive_state()` | 1580 | Check whether a new track is eligible to inherit state. |
| `transfer_bridged_state()` | 1598 | Transfer state after a qualifying bridge match. |
| `bridge_reacquired_tracks()` | 1646 | Find safe one-to-one matches when tracker IDs change briefly. |
| `normalize_object_class()` | 1874 | Normalize classifier labels for EMPTY/OBJECT_PRESENT. |
| `normalize_symbol_class()` | 1904 | Normalize classifier labels for X/NO_X. |
| `classify_empty_object()` | 1937 | Run EMPTY/OBJECT_PRESENT inference on a crop. |
| `update_object_decision()` | 1976 | Update temporal EMPTY/OBJECT_PRESENT decision state. |
| `prepare_medium_focus()` | 2172 | Prepare a medium-focus image region for X/NO_X verification. |
| `classify_medium_focus_symbol()` | 2348 | Run the symbol classifier on a prepared focus region. |
| `box_center_and_diagonal()` | 2423 | Handles box center and diagonal; precise behavior is in the source. |
| `is_left_turn_hold_zone()` | 2465 | Identify unstable turn-region observations. |
| `update_dynamic_symbol_gate()` | 2491 | Determine whether a tracked symbol view is suitable for voting. |
| `should_take_symbol_sample()` | 2662 | Check if another symbol observation should be sampled. |
| `commit_sample_position()` | 2773 | Record sampling position to manage duplicate/closely spaced views. |
| `clear_symbol_state()` | 2801 | Clear cached X/NO_X voting state for a track. |
| `_audit_float()` | 2826 | Handles  audit float; precise behavior is in the source. |
| `audit_note_tracks()` | 2833 | Handles audit note tracks; precise behavior is in the source. |
| `audit_note_frame()` | 2843 | Remember this frame's gate result + box-stability numbers (read only). |
| `audit_record_sample()` | 2882 | Record individual symbol evidence for decision auditing. |
| `audit_commit()` | 2927 | Write audit details for a completed inspection decision. |
| `should_store_symbol_sample()` | 3090 | Filter X/NO_X evidence using view quality. |
| `note_deferred_no_x()` | 3108 | Record a NO_X result deferred because the view was not sufficiently clean. |
| `log_bridge_diag()` | 3120 | Handles log bridge diag; precise behavior is in the source. |
| `diagnose_unbridged_tracks()` | 3128 | Log why a candidate tracker state was not bridged. |
| `update_three_view_decision()` | 3256 | Update multi-view symbol decision voting. |
| `draw_pallet()` | 3394 | Draw inspection labels and bounding boxes on the output frame. |
| `advance_held_box()` | 3443 | Predict short box motion during detector misses, when physically plausible. |
| `build_visible_track_cache()` | 3509 | Combine current and briefly held boxes for display. |
| `top_rfid_validate_zone()` | 3609 | Validate operator-drawn RFID mapping-zone coordinates. |
| `phase65_self_test()` | 4161 | Run synthetic identity-persistence checks without camera or PLC. |
| `main()` | 4239 | Run the program's main processing loop. |

## Imported Python modules

`__future__`, `collections`, `csv`, `cv2`, `importlib.util`, `json`, `numpy`, `pathlib`, `queue`, `threading`, `time`, `ultralytics`

## Model filenames visible in source

`/Do_Not_Delete_PLC/Conference/00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights/best.pt`, `empty_object_yolo26n_cls_best.pt`, `x_no_x_yolo26n_medium_focus_final_hard_best.pt`

## CLI strings visible in source

`--self-test`

Other than the documented Phase 65 self-test, these text matches are not verified command-line options.

## Preservation

Only this Markdown document was generated; the program source has not been edited.
