# CREST Conference Staging Package Audit

This checks file presence and compares the two most important programs with their original local copies. It does not test camera/PLC operations or guarantee a portable deployment.

## Required dependencies

| Item | Staged path | Status |
|---|---|---|
| Final code | `07_Final_Empty_Accepted_Rejected_System/CODE/65_conference_phase40_persistent_top_rfid.py` | **FOUND** |
| Protected baseline | `07_Final_Empty_Accepted_Rejected_System/CODE/40_conference_100_pallet_live.py` | **FOUND** |
| Dynamic tracker helper | `07_Final_Empty_Accepted_Rejected_System/CODE/dynamic_live_pallet_detection.py` | **FOUND** |
| Conveyor boundary | `07_Final_Empty_Accepted_Rejected_System/CONFIG/conveyor_boundary.json` | **FOUND** |
| Conference detector checkpoint | `00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights/best.pt` | **FOUND** |
| EMPTY classifier checkpoint | `07_Final_Empty_Accepted_Rejected_System/MODELS/empty_object_yolo26n_cls_best.pt` | **FOUND** |
| X/NO_X classifier checkpoint | `07_Final_Empty_Accepted_Rejected_System/MODELS/x_no_x_yolo26n_medium_focus_final_hard_best.pt` | **FOUND** |

## RFID calibration

`07_Final_Empty_Accepted_Rejected_System/CONFIG/63_top_rfid_center_zone.json`: NOT INCLUDED — draw zone using Z on live hardware

## Working-source integrity

| File | Source vs staged SHA-256 |
|---|---|
| `40_conference_100_pallet_live.py` | MATCH |
| `65_conference_phase40_persistent_top_rfid.py` | MATCH |
| `dynamic_live_pallet_detection.py` | MATCH |

## Caveats

- The detector checkpoint is referenced by an original absolute Windows path in Phase 65. It is archived here but not automatically repointed by copying.
- A frozen checkpoint alone is insufficient to reproduce a training run without exact dependency versions, train parameters and datasets.
- Phase 65 keeps physical pallet identity in memory; it is not guaranteed persistent across restarting the program.
- All PLC integration in the final code is read-only and should remain so during archive validation.
