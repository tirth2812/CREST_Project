# A Vision-Driven Digital Twin for Real-Time Conveyor Pallet Inspection in Smart Manufacturing

This project develops a camera-based conveyor inspection system for smart manufacturing. It detects and tracks moving pallets, determines whether each pallet is empty or contains an object, evaluates the object's marking, and associates the inspection result with the pallet's physical RFID number. The research was conducted in the CREST laboratory; CREST is the lab name, not the project name.

This repository documents the project from image annotation and model development through live testing and dual-RFID integration. It is organized by development stage so a reader can understand **what was built, which files belong to each stage, and how those stages connect**.

> **Repository branch:** `conveyor-upload`  
> **Important:** The repository is a documented project copy. Some scripts use model paths, camera settings, and PLC connections from the lab computer; cloning the repository does not automatically make the full hardware setup runnable elsewhere.

## 1. How the system works

```text
RGB camera (overhead conveyor view)
         |
         v
YOLO26 pallet detection
         |
         v
Conveyor boundary filtering and pallet tracking
         |
         v
EMPTY / OBJECT_PRESENT classification
         |
         +---- EMPTY ------------------------------> EMPTY
         |
         +---- OBJECT_PRESENT
                    |
                    v
              X / NO_X classification
                    |
                    v
              Multi-view voting
                    |
                    +---- X -----------------------> REJECTED
                    +---- NO_X --------------------> ACCEPTED
                    +---- Not enough evidence -----> CHECKING

TOP RFID2 + BOTTOM RFID1 (read-only PLC inputs)
         |
         v
Map physical RFID pallet numbers 1–6 to eligible vision tracks
         |
         v
Show pallet number and inspection status; record events
```

The three machine-learning models have separate responsibilities: **finding a pallet**, **detecting whether it is occupied**, and **checking the visible marking**. Tracking and multi-view voting help the system maintain a decision while pallets move, rotate, or are briefly obscured.

A number such as **P3** means *physical pallet 3* when confirmed or recovered by RFID association. It is not simply the temporary ID assigned by the image tracker. If the association is uncertain, the system should not guess a physical number.

## 2. Explore the project, stage by stage

**Start with a stage's `README.md`**, then open its `CODE` folder to see the Python scripts and their file-specific explanations. The detailed code documentation was moved next to the related scripts so it is easy to find.

| Stage | What it covers | Start here |
|---|---|---|
| **00 — Revised pallet detector** | Final 100-image detector dataset, training run, and checkpoint used by the conference pipeline. | [Stage 00 overview](00_Pallet_100_YOLO26_Training/README.md) |
| **01 — Pallet annotation** | Preparing images and manually defining pallet bounding boxes. | [Stage 01 overview](01_Pallet_Annotation/README.md) · [Annotation code](01_Pallet_Annotation/CODE/pallet_annotator.py) · [Code guide](01_Pallet_Annotation/CODE/pallet_annotator.py.README.md) |
| **02 — YOLO pallet detection** | Original pallet-detection development, training, and evaluation. | [Stage 02 overview](02_YOLO_Pallet_Detection/README.md) · [Training guide](02_YOLO_Pallet_Detection/CODE/train_yolo26_pallet_localizer.py.README.md) · [Evaluation guide](02_YOLO_Pallet_Detection/CODE/evaluate_yolo_pallet_localizer_test.py.README.md) |
| **03 — Empty/object classifier** | Training a classifier to distinguish `EMPTY` from `OBJECT_PRESENT`. | [Stage 03 overview](03_Empty_Object_Classifier/README.md) · [Training script](03_Empty_Object_Classifier/CODE/train_yolo26_empty_object_classifier.py) · [Code guide](03_Empty_Object_Classifier/CODE/train_yolo26_empty_object_classifier.py.README.md) |
| **04 — Live empty/object system** | Running detection and empty/object classification on camera frames. | [Stage 04 overview](04_Live_Empty_Object_System/README.md) · [Live script](04_Live_Empty_Object_System/CODE/live_empty_object_classification.py) · [Code guide](04_Live_Empty_Object_System/CODE/live_empty_object_classification.py.README.md) |
| **05 — Dynamic pallet detection** | Following pallets through successive frames and applying conveyor-boundary filtering. | [Stage 05 overview](05_Dynamic_Pallet_Detection/README.md) · [Tracking script](05_Dynamic_Pallet_Detection/CODE/dynamic_live_pallet_detection.py) · [Code guide](05_Dynamic_Pallet_Detection/CODE/dynamic_live_pallet_detection.py.README.md) |
| **06 — Integrated live vision** | Earlier combined live detection/empty-object workflow before the final marking/RFID system. | [Stage 06 overview](06_Live_Empty_Object_Detection_Final/README.md) · [Live script](06_Live_Empty_Object_Detection_Final/CODE/live_object_detection_on_pallet.py) · [Code guide](06_Live_Empty_Object_Detection_Final/CODE/live_object_detection_on_pallet.py.README.md) |
| **07 — Final inspection and RFID** | Three-model inspection, temporal voting, top/bottom RFID reads, and physical pallet identification. | [Stage 07 overview](07_Final_Empty_Accepted_Rejected_System/README.md) · [Final Phase 70 runner](07_Final_Empty_Accepted_Rejected_System/CODE/70_conference_dual_rfid_bottom_calibrated.py) |

### Stage 00 — Updated 100-image pallet detector

The conference pipeline uses a revised pallet-localization model developed from **100 labeled conveyor images**: 80 for training, 10 for validation, and 10 for testing. Its role is to locate pallets in the full camera image before any crop-level classification takes place.

- [Stage 00 documentation](00_Pallet_100_YOLO26_Training/README.md)
- [Detector dataset](00_Pallet_100_YOLO26_Training/DATASET/100_IMAGE_DETECTOR)
- [Detector checkpoint directory](00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights)

**Note:** Stage 02 documents the earlier detector development. Stage 00 holds the revised 100-image detector assets used in the later conference workflow; they are not two extra models running at the same time.

### Stages 01–03 — Prepare data and train the first two models

**Stage 01** explains the image annotation process and YOLO bounding-box labels. **Stage 02** contains the detector training and test scripts. **Stage 03** develops the separate `EMPTY` / `OBJECT_PRESENT` image classifier, which receives crops of detected pallets rather than full-conveyor images.

- [Stage 01 code-folder guide](01_Pallet_Annotation/CODE/README_From_DOCS.md)
- [Stage 02 code-folder guide](02_YOLO_Pallet_Detection/CODE/README_From_DOCS.md)
- [Stage 02 training script](02_YOLO_Pallet_Detection/CODE/train_yolo26_pallet_localizer.py) and [evaluation script](02_YOLO_Pallet_Detection/CODE/evaluate_yolo_pallet_localizer_test.py)
- [Stage 03 code-folder guide](03_Empty_Object_Classifier/CODE/README_From_DOCS.md)

### Stages 04–06 — Run and improve the live camera system

**Stage 04** connects detection and empty/object classification to a live camera feed. **Stage 05** adds dynamic tracking and conveyor-boundary handling so pallets can be followed while moving. **Stage 06** documents the combined live workflow that preceded the final `ACCEPTED` / `REJECTED` and RFID integration.

- [Stage 04 code-folder guide](04_Live_Empty_Object_System/CODE/README_From_DOCS.md)
- [Stage 05 code-folder guide](05_Dynamic_Pallet_Detection/CODE/README_From_DOCS.md)
- [Stage 06 code-folder guide](06_Live_Empty_Object_Detection_Final/CODE/README_From_DOCS.md)

### Stage 07 — Final three-model inspection and RFID association

This stage combines pallet detection, tracking, empty/object classification, `X` / `NO_X` classification, and multi-view decision logic. Two RFID readers provide the physical pallet identifiers used to associate decisions with pallets 1–6.

| File | Role |
|---|---|
| [Phase 70 — dual RFID runner](07_Final_Empty_Accepted_Rejected_System/CODE/70_conference_dual_rfid_bottom_calibrated.py) | Main integrated live inspection with top and bottom RFID readers. |
| [Phase 65 — top RFID runner](07_Final_Empty_Accepted_Rejected_System/CODE/65_conference_phase40_persistent_top_rfid.py) | Earlier version with persistent top-reader RFID association. |
| [Phase 65 code guide](07_Final_Empty_Accepted_Rejected_System/CODE/65_conference_phase40_persistent_top_rfid.py.README.md) | Detailed explanation of the Phase 65 implementation. |
| [Dynamic pallet tracker](07_Final_Empty_Accepted_Rejected_System/CODE/dynamic_live_pallet_detection.py) | Tracker-stage dependency loaded by the integrated runner. |
| [Tracker code guide](07_Final_Empty_Accepted_Rejected_System/CODE/dynamic_live_pallet_detection.py.README.md) | Explanation of the tracker component. |
| [Stage 07 code-folder guide](07_Final_Empty_Accepted_Rejected_System/CODE/README_From_DOCS.md) | Additional code-level documentation. |

**Phase 70 is the primary dual-RFID reference runner.** Phase 65 is retained as the earlier top-reader implementation. The numbered development scripts that were removed during repository cleanup are not required to understand these two main entries; earlier experiments remain in Git history.

## 3. Model inputs and outputs

| Component | Input | Output | Why it is needed |
|---|---|---|---|
| Pallet detector (YOLO26) | Full conveyor frame | Pallet bounding boxes | Locates the region to inspect. |
| Empty/object classifier (YOLO26) | Pallet image crop | `EMPTY` or `OBJECT_PRESENT` | Avoids checking markings on empty pallets. |
| Marking classifier (YOLO26) | Focused object crop | `X` or `NO_X` | Supplies evidence for rejection or acceptance. |
| Tracking and voting | Detections across video frames | Stable track and inspection state | Reduces decisions based on one poor view. |
| RFID association | PLC input words and eligible tracks | Physical pallet ID 1–6 | Links a visual result to a physical pallet. |

`CHECKING` means there is not yet enough reliable evidence for an accepted/rejected decision. **ACCEPTED and REJECTED are system-level outcomes**, not the direct output classes of a single three-class neural network.

## 4. RFID identification and PLC interface

The final system uses a shared mapping of the known RFID tags to six physical pallets. Its two read-only input groups are:

| Station | Reader | PLC input words |
|---|---|---|
| Top station | RFID2 | `IO_Link_Master_ST1_to_ST4:I1.Data[128]` through `Data[131]` |
| Bottom station | RFID1 | `IO_Link_Master_ST1_to_ST4:I1.Data[96]` through `Data[99]` |

The application associates a fresh RFID reading only when the visual match is sufficiently clear. Its identity memory can reconnect a pallet after a temporary vision-track loss, but uncertain matches should remain unresolved rather than receiving another pallet's number.

**Safety and scope:** The documented Phase 65 and Phase 70 scripts **read PLC tags**. They do **not** issue PLC motor or piston commands, and they do **not** provide robot actuation control. Robot integration is a separate step.

## 5. Where to find data and trained models

| Asset | Repository location |
|---|---|
| Revised 100-image pallet dataset | [`00/.../DATASET/100_IMAGE_DETECTOR`](00_Pallet_100_YOLO26_Training/DATASET/100_IMAGE_DETECTOR) |
| Revised pallet model checkpoint | [`00/.../pallet_yolo26_100_v2/weights`](00_Pallet_100_YOLO26_Training/pallet_yolo26_100_v2/weights) |
| Empty/object classifier dataset | [`03_Empty_Object_Classifier/DATASET`](03_Empty_Object_Classifier/DATASET) |
| Original X/NO_X dataset | [`07/.../DATASET/CONFERENCE_X_NO_X_ORIGINAL`](07_Final_Empty_Accepted_Rejected_System/DATASET/CONFERENCE_X_NO_X_ORIGINAL) |
| Additional focused X/NO_X training data | [`07/.../DATASET/CREST_MEDIUM_FOCUS_AUGMENTED`](07_Final_Empty_Accepted_Rejected_System/DATASET/CREST_MEDIUM_FOCUS_AUGMENTED) |
| Final classification checkpoints | [`07/.../MODELS`](07_Final_Empty_Accepted_Rejected_System/MODELS) |

These folders document the available assets. **Dataset inclusion alone does not establish the exact training provenance of a checkpoint**; use the saved training configuration and run logs when reproducing a particular experiment.

## 6. Running the final system in the lab

The commands below are for the configured Windows lab environment. They assume the same Python environment, camera, model weights, conveyor boundary configuration, and PLC network access used during development.

**Open Command Prompt and activate the environment:**

```bat
C:\VLM_Finetune\vlm_env\Scripts\activate.bat
cd /d "C:\Do_Not_Delete_PLC\CREST_YOLO26\07_Final_Empty_Accepted_Rejected_System\CODE"
```

**Run offline RFID and integration self-tests (no live camera required):**

```bat
python 70_conference_dual_rfid_bottom_calibrated.py --self-test
```

**Run the live system:**

```bat
python 70_conference_dual_rfid_bottom_calibrated.py
```

These are the original Phase 70 entry points. A successful self-test checks programmed logic; it is **not** proof that the camera, RFID reader alignment, or physical conveyor is behaving correctly. Confirm those on the actual hardware before using live decisions.

The Python files in this repository may include absolute Windows paths. Paths and dependencies must be checked before attempting to run the system from a different computer.

## 7. How to read this repository

1. Read this root README for the overall workflow.
2. Open the [stage list](#2-explore-the-project-stage-by-stage) and select the stage you want to understand.
3. Read that stage's main `README.md` for its purpose, process, and outputs.
4. Open the stage's `CODE` folder. Python files and their `<filename>.README.md` explanations are together; `README_From_DOCS.md` contains additional folder-level guidance.
5. Use the linked datasets, checkpoints, and evaluation records to inspect the supporting artifacts.
6. For the integrated live system, start with [Phase 70](07_Final_Empty_Accepted_Rejected_System/CODE/70_conference_dual_rfid_bottom_calibrated.py) and the [Stage 07 overview](07_Final_Empty_Accepted_Rejected_System/README.md).

For a high-level record of packaged dependencies and any missing items, see [PACKAGE_AUDIT.md](PACKAGE_AUDIT.md), where included in this branch.

---

**Project scope:** Computer-vision-based inspection, temporal classification, and read-only RFID-based pallet identification on a conveyor. Hardware control and robot integration are not part of the documented Phase 70 runtime.
