# Stage 06 — Integrated Live Empty/Object Detection

**CREST | Consolidating the live two-model vision workflow**

[Previous: Stage 05](../05_Dynamic_Pallet_Detection/README.md) · [Project overview](../README.md) · **Next: [Stage 07 — Final Inspection and RFID System](../07_Final_Empty_Accepted_Rejected_System/)**

## 1. What this stage is for

By this stage, CREST had a pallet detector, an EMPTY/OBJECT_PRESENT classifier, and work on dynamic tracking. Stage 06 records the **consolidated live vision workflow** built from those components.

The word **"Final" in this historical folder name refers to that development phase**. It should not be confused with the later full **Stage 07 EMPTY/ACCEPTED/REJECTED plus RFID system**.

## 2. Processing flow

```text
Live conveyor frame
   ↓
Pallet detection
   ↓
Conveyor-area filtering and dynamic tracking
   ↓
Crop each visible pallet
   ↓
EMPTY / OBJECT_PRESENT classification
   ↓
Display the integrated result on the camera feed
```

The exact features of a historical runner depend on the code version. This README summarizes the stage's role rather than claiming that every file in this directory has identical decision logic.

## 3. What we integrated

- **Model reuse:** load the trained pallet detector and the existing EMPTY/OBJECT_PRESENT classifier rather than training them again.
- **Camera processing:** perform detection and classification repeatedly on a moving conveyor feed.
- **Pallet isolation:** use detected boxes to crop each pallet before content classification.
- **Tracking:** maintain temporary visual continuity where supported by the particular runner.
- **Visual inspection:** review labels and boxes in live screenshots or recordings.

This stage is a bridge between early live demos (Stage 04) and the complete inspection and RFID association system (Stage 07).

## 4. How this differs from Stage 07

| Capability | Stage 06 focus | Stage 07 final integration |
|---|---|---|
| Find pallets | Yes | Yes |
| Track moving pallets | Integrated development | Yes, with later persistence logic |
| Determine EMPTY / OBJECT_PRESENT | Yes | Yes |
| Determine X / NO_X | Not the primary scope of this folder | Yes, separate third model |
| Produce ACCEPTED / REJECTED | Not the final three-status pipeline | Yes |
| Link physical pallet numbers using both RFID readers | Not this stage's goal | Yes |

**Important:** Stage 07's `Phase 70` reads RFID/PLC input data; it is not robot, piston, or motor actuation code.

## 5. Files and results

Check the `CODE/` directory for live scripts, the configuration/results directories for camera or test artifacts, and the generated inventory below for tracked files in `conveyor-upload`. Depending on the script, checkpoint files may be referenced from other stages rather than copied into this folder.

To rerun an old stage script, first inspect its imports, model paths, camera index, and environment requirements. **Do not treat the historical Stage 06 runner as the latest Phase 70 program**.

## 6. What this stage delivered

**Output:** an integrated live view that locates pallets and evaluates whether each has an object, helping test the interaction between detector, tracking, camera, and classifier.

## 7. Evaluation and limitations

Successful classification on a few visible frames is a demonstration, not a complete quantitative evaluation. Tracking can still fail around turns, under occlusion, or when pallets touch. A full-system evaluation must separately assess detector localization, content classification, identity tracking, and runtime performance.

## 8. For the professor / reviewer

This stage shows **the last development step before adding the final X/NO_X inspection decision and dual-RFID identity association in Stage 07**.

---

## Files actually included in this Git branch

The list below was generated from Git-tracked files on `conveyor-upload` when this README was installed. It is an inventory, not an instruction to run every script.

**Tracked files in this stage (excluding this README): 6**

### Folder overview
- `CODE/` — 4 tracked file(s)
- `CONFIG/` — 2 tracked file(s)

### Python source files
- `CODE/live_object_detection_on_pallet.py`

### Model checkpoints
- No `.pt` file is tracked inside this stage; a runner may reference weights elsewhere.

### Configuration files
- `CONFIG/conveyor_boundary.json`
