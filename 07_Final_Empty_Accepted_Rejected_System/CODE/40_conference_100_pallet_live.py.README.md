# 40_conference_100_pallet_live.py

**Stage:** Final live inspection and persistent RFID identity
**Source path in package:** `07_Final_Empty_Accepted_Rejected_System/CODE/40_conference_100_pallet_live.py`

## Purpose

Protected working Conference live inspection baseline; no RFID persistence changes should be applied here.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\40_conference_100_pallet_live.py`. This README is generated; the Python file is unchanged.

## How it fits

This is the frozen, operator-tested baseline for detection, tracking, box holding, EMPTY/OBJECT_PRESENT and X/NO_X decisions. Phase 65 builds its independent RFID association around this behavior.

Its camera, model, boundary and tracking dependencies must be checked before running on any new machine.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `log_event()` | 267 | Append event text to the existing decision log. |
| `load_stage08()` | 451 | Load the standalone dynamic tracker module and synchronize its conveyor boundary path. |
| `open_camera()` | 483 | Try configured camera inputs and initialize video capture. |
| `make_pallet_frame()` | 565 | Prepare camera input for pallet detection. |
| `make_symbol_frame()` | 619 | Prepare image input for symbol classification. |
| `crop_pallet()` | 712 | Extract a pallet region from an image. |
| `box_iou()` | 786 | Calculate intersection-over-union between boxes. |
| `smaller_box_overlap_ratio()` | 873 | Measure overlap relative to the smaller box. |
| `vertical_overlap_ratio()` | 956 | Measure the vertical overlap of two boxes. |
| `box_size_ratio()` | 1005 | Compare the size of two boxes. |
| `center_ratios()` | 1055 | Compare box-center positions relative to their sizes. |
| `merge_pallet_detections()` | 1139 | Merge competing detections for likely duplicate pallets. |
| `current_boxes_are_duplicate()` | 1217 | Check whether simultaneous boxes represent the same pallet. |
| `filter_current_duplicate_tracks()` | 1276 | Suppress simultaneous duplicate track candidates. |
| `old_new_tracks_match()` | 1356 | Test geometric continuity between lost and new tracking IDs. |
| `state_bridge_score()` | 1411 | Score an old/new tracking-state match using geometric constraints. |
| `has_meaningful_track_state()` | 1479 | Check whether a track has state worth preserving. |
| `move_track_state()` | 1500 | Move inspection state from an old tracker ID to its verified replacement. |
| `track_can_receive_state()` | 1571 | Check whether a new track is eligible to inherit state. |
| `transfer_bridged_state()` | 1589 | Transfer state after a qualifying bridge match. |
| `bridge_reacquired_tracks()` | 1637 | Find safe one-to-one matches when tracker IDs change briefly. |
| `normalize_object_class()` | 1865 | Normalize classifier labels for EMPTY/OBJECT_PRESENT. |
| `normalize_symbol_class()` | 1895 | Normalize classifier labels for X/NO_X. |
| `classify_empty_object()` | 1928 | Run EMPTY/OBJECT_PRESENT inference on a crop. |
| `update_object_decision()` | 1967 | Update temporal EMPTY/OBJECT_PRESENT decision state. |
| `prepare_medium_focus()` | 2163 | Prepare a medium-focus image region for X/NO_X verification. |
| `classify_medium_focus_symbol()` | 2339 | Run the symbol classifier on a prepared focus region. |
| `box_center_and_diagonal()` | 2414 | Handles box center and diagonal; precise behavior is in the source. |
| `is_left_turn_hold_zone()` | 2456 | Identify unstable turn-region observations. |
| `update_dynamic_symbol_gate()` | 2482 | Determine whether a tracked symbol view is suitable for voting. |
| `should_take_symbol_sample()` | 2653 | Check if another symbol observation should be sampled. |
| `commit_sample_position()` | 2764 | Record sampling position to manage duplicate/closely spaced views. |
| `clear_symbol_state()` | 2792 | Clear cached X/NO_X voting state for a track. |
| `_audit_float()` | 2817 | Handles  audit float; precise behavior is in the source. |
| `audit_note_tracks()` | 2824 | Handles audit note tracks; precise behavior is in the source. |
| `audit_note_frame()` | 2834 | Remember this frame's gate result + box-stability numbers (read only). |
| `audit_record_sample()` | 2873 | Record individual symbol evidence for decision auditing. |
| `audit_commit()` | 2918 | Write audit details for a completed inspection decision. |
| `should_store_symbol_sample()` | 3081 | Filter X/NO_X evidence using view quality. |
| `note_deferred_no_x()` | 3099 | Record a NO_X result deferred because the view was not sufficiently clean. |
| `log_bridge_diag()` | 3111 | Handles log bridge diag; precise behavior is in the source. |
| `diagnose_unbridged_tracks()` | 3119 | Log why a candidate tracker state was not bridged. |
| `update_three_view_decision()` | 3247 | Update multi-view symbol decision voting. |
| `draw_pallet()` | 3385 | Draw inspection labels and bounding boxes on the output frame. |
| `advance_held_box()` | 3434 | Predict short box motion during detector misses, when physically plausible. |
| `build_visible_track_cache()` | 3500 | Combine current and briefly held boxes for display. |
| `main()` | 3553 | Run the program's main processing loop. |

## Imported Python modules

`__future__`, `collections`, `cv2`, `importlib.util`, `numpy`, `pathlib`, `time`, `ultralytics`

## Model filenames visible in source

`/Do_Not_Delete_PLC/Conference/00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights/best.pt`, `empty_object_yolo26n_cls_best.pt`, `x_no_x_yolo26n_medium_focus_final_hard_best.pt`

## Preservation

Only this Markdown document was generated; the program source has not been edited.
