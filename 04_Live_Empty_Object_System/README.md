# Stage 04 — First Live Empty/Object Vision System

**CREST | Connecting models to the live conveyor camera**

[Previous: Stage 03](../03_Empty_Object_Classifier/README.md) · [Project overview](../README.md) · **Next: [Stage 05 — Dynamic Pallet Detection](../05_Dynamic_Pallet_Detection/README.md)**

## 1. What this stage is for

Stages 02 and 03 trained the individual pallet detector and empty/object classifier. In Stage 04, these models were brought into **one live camera workflow** so the conveyor could be inspected while pallets were moving.

This was an integration stage, not the training of a new third model.

## 2. Processing flow

```text
Camera reads a live frame
   ↓
Detect pallet boxes (Stage 02 model)
   ↓
Crop each detected pallet
   ↓
Classify the crop (Stage 03 model)
   ↓
Display a box and EMPTY / OBJECT_PRESENT label
   ↓
Repeat for the next frame
```

The live display shows where pallets were detected and which of the two content classes was predicted for each crop. This is an intermediate vision result, **not a permanent physical pallet/RFID identity**.

## 3. What we developed and checked

1. **Camera acquisition:** read frames from the installed conveyor camera.
2. **Model integration:** load the already trained pallet detector and empty/object classifier.
3. **Per-pallet crops:** pass each detected pallet crop to the second model.
4. **Result visualization:** draw labeled boxes on the video frame.
5. **Live observations:** inspect whether the pipeline keeps working under lighting changes, motion, corners, and partial obstructions.

A model that scores well on saved images can still make mistakes on moving camera frames. This stage provided an early way to see those problems directly.

## 4. Inputs and outputs

| Item | Meaning |
|---|---|
| Input | Live conveyor camera frames |
| Model 1 | Detects visible pallets |
| Model 2 | Labels each crop `EMPTY` or `OBJECT_PRESENT` |
| Visible output | Bounding boxes and labels on live video |
| Typical saved outputs | Screenshots, recordings, or logs, if retained |

**No RFID-based numbering or X/NO_X accepted/rejected decision is established by this stage alone.** Those capabilities belong to later integration work.

## 5. Where the work is stored

Look for camera runner `.py` files in `CODE/`, saved visual evidence in results/image folders, and model configuration references. The actual filenames tracked in the current branch appear in the generated inventory below. Some scripts refer to the trained model checkpoints in earlier stages instead of storing duplicate `.pt` files here.

### Reproducing the live demonstration

The code is designed for the CREST lab's Windows camera/GPU setup. Before executing a historical live script, inspect its `CAMERA_INDEX`, model-checkpoint paths, and dataset/configuration paths. The specific filename and requirements depend on which version was retained; the inventory below identifies the tracked candidates.

## 6. What this stage delivered

**Output:** the first live, two-model inspection flow that could display pallet locations and EMPTY/OBJECT_PRESENT results on video.

## 7. Limitations that motivated Stage 05

Independent frame-by-frame predictions can cause labels or boxes to flicker when pallets turn or overlap. A newly detected pallet may also receive a new temporary number after being obscured. Stage 05 adds dynamic tracking and conveyor-area filtering to improve continuity between frames.

## 8. For the professor / reviewer

This stage demonstrates the move from **offline model testing** to a **working live camera pipeline**. It does not by itself demonstrate the final system's RFID mapping or physical actuation.

---

## Files actually included in this Git branch

The list below was generated from Git-tracked files on `conveyor-upload` when this README was installed. It is an inventory, not an instruction to run every script.

**Tracked files in this stage (excluding this README): 4**

### Folder overview
- `CODE/` — 4 tracked file(s)

### Python source files
- `CODE/live_empty_object_classification.py`

### Model checkpoints
- No `.pt` file is tracked inside this stage; a runner may reference weights elsewhere.

### Configuration files
- No YAML/JSON file is tracked inside this stage.
