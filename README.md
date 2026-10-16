# CREST Conference â€” Pallet Inspection and RFID Identity

A stage-by-stage record of the Conference conveyor inspection project. The package preserves the detector/classifier training data, corrected annotations, live vision stages, frozen models, RFID integration and selected independent tests.

**Package type:** a documented copy of the local development artifacts, assembled outside the running Conference Git repository. The original lab-PC installation remains the known working runtime; the copied source is not automatically path-portable.

## System at a glance

```text
Live camera
    |
YOLO26 pallet detector (Conference 100-image checkpoint)
    |
Dynamic pallet tracking + duplicate suppression + short box persistence
    |
EMPTY versus OBJECT_PRESENT classifier
    |
X versus NO_X classifier + multi-view evidence gate
    |
Decision: ACCEPTED / REJECTED / CHECKING
    |
Display + inspection logs

TOP RFID2 (read-only PLC data words 128-131)
    |
Physical Pallet ID 1-6 -> associate with a unique eligible vision track
    |
Persist identity across tracker loss; verify/correct on a fresh RFID read
```

**Interpretation:** `X` means REJECTED and `NO_X` means ACCEPTED when sufficient symbol evidence is available. `EMPTY` is a separate pallet/object classification state. Physical RFID numbers are different from temporary vision tracker IDs.

## Project stages

| Stage | Folder | Purpose |
|---|---|---|
| 00 | [00_Pallet_100_YOLO26_Training](00_Pallet_100_YOLO26_Training/README.md) | Build and evaluate the revised pallet-localization detector used by the Conference live pipeline. |
| 01 | [01_Pallet_Annotation](01_Pallet_Annotation/README.md) | Prepare reviewed pallet bounding boxes and annotation tools before model training. |
| 02 | [02_YOLO_Pallet_Detection](02_YOLO_Pallet_Detection/README.md) | Document the original detector baseline before the Conference 100-image retraining. |
| 03 | [03_Empty_Object_Classifier](03_Empty_Object_Classifier/README.md) | Classify detected pallet crops as EMPTY or OBJECT_PRESENT using a separate image classifier. |
| 04 | [04_Live_Empty_Object_System](04_Live_Empty_Object_System/README.md) | Bring camera ingestion, pallet localization and empty/object classification together. |
| 05 | [05_Dynamic_Pallet_Detection](05_Dynamic_Pallet_Detection/README.md) | Follow moving pallets over frames and retain useful track information as the conveyor moves. |
| 06 | [06_Live_Empty_Object_Detection_Final](06_Live_Empty_Object_Detection_Final/README.md) | Preserve the working live EMPTY/OBJECT pipeline before integrating X/NO_X decisions and RFID. |
| 07 | [07_Final_Empty_Accepted_Rejected_System](07_Final_Empty_Accepted_Rejected_System/README.md) | Run the complete pallet inspection pipeline and link temporary vision tracks to physical pallet IDs 1â€“6. |

## Actual dataset and checkpoint sources

| Asset | Source in the local project | Location in this package |
|---|---|---|
| Conference final 100-image pallet detector dataset | `Conference/00_Pallet_100_YOLO26_Dataset_v2` | `00_Pallet_100_YOLO26_Training/DATASET/100_IMAGE_DETECTOR` |
| Final pallet detector `best.pt` | `Conference/00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights` | `00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights` |
| EMPTY/OBJECT training images | `Conference/03_Empty_Object_Classifier/DATASET` | `03_Empty_Object_Classifier/DATASET` |
| Conference X/NO_X original train/val/test split | `Conference/00_Current_Working_X_NO_X_Dataset` | `07_Final_Empty_Accepted_Rejected_System/DATASET/CONFERENCE_X_NO_X_ORIGINAL` |
| Additional augmented X/NO_X training data | `CREST_YOLO26/07_Final_Empty_Accepted_Rejected_System/DATASET/yolo26_medium_focus_final_hard_augmented` | `07_Final_Empty_Accepted_Rejected_System/DATASET/CREST_MEDIUM_FOCUS_AUGMENTED` |
| Final EMPTY and X/NO_X classifier checkpoints | `Conference/07_Final_Empty_Accepted_Rejected_System/MODELS` | `07_Final_Empty_Accepted_Rejected_System/MODELS` |

**Provenance caution:** The additional CREST augmented dataset is recorded separately; its inclusion does not prove it was the exact Conference training input. Review saved training configurations before making a reproduction claim.

## Final files and how to run

- Protected reference: [`40_conference_100_pallet_live.py`](07_Final_Empty_Accepted_Rejected_System/CODE/40_conference_100_pallet_live.py)
- Final live application: [`65_conference_phase40_persistent_top_rfid.py`](07_Final_Empty_Accepted_Rejected_System/CODE/65_conference_phase40_persistent_top_rfid.py)
- Required tracker: [`dynamic_live_pallet_detection.py`](07_Final_Empty_Accepted_Rejected_System/CODE/dynamic_live_pallet_detection.py)
- Detailed operation: [Phase 65 README](07_Final_Empty_Accepted_Rejected_System/CODE/DOCS/65_conference_phase40_persistent_top_rfid.py.README.md)

Run the self-test on the original Conference machine after activating the existing `vlm_env`:

```bat
python -u "C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\65_conference_phase40_persistent_top_rfid.py" --self-test
```

The live run uses the same command without `--self-test`. Confirm the camera, model paths, conveyor boundary, saved RFID zone and read-only PLC access before running.

## Documentation map

- Every meaningful folder has a `README.md` describing its purpose and contents.
- Each included code file has a separate `DOCS/<filename>.README.md` based on its source, with function/class inventory and run guidance where verified.
- `PACKAGE_AUDIT.md` reports whether final runtime dependencies are present.

## Repository safety

- Stage documentation is generated **outside** the active Conference Git working tree.
- No working Phase 40/65 Python source, dataset or model checkpoint is rewritten.
- Do not run `git add -A` in the active Conference repository while its live RESULTS images/logs are modified.
- After reviewing the staged content and GitHub file limits, commit the curated package on an isolated branch; do not force-push or delete originals.
- Check captured lab images and network configuration for material that should not be published publicly.

**Dependency-audit state:** Required staged runtime files found â€” see [PACKAGE_AUDIT.md](PACKAGE_AUDIT.md).
