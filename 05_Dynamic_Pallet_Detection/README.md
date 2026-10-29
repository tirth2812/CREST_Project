# Stage 05 — Dynamic Pallet Detection and Tracking

**CREST | Maintaining visual continuity while pallets move**

[Previous: Stage 04](../04_Live_Empty_Object_System/README.md) · [Project overview](../README.md) · **Next: [Stage 06 — Integrated Live Empty/Object Detection](../06_Live_Empty_Object_Detection_Final/README.md)**

## 1. What this stage is for

A detector can find a pallet in one camera frame, but **it does not automatically know whether a box in the next frame belongs to the same pallet**. Stage 05 adds a tracking stage to follow moving pallets from frame to frame.

This became important when several pallets were visible at once, especially on the turns or when a pallet briefly disappeared behind conveyor hardware.

## 2. Processing flow

```text
Current camera frame
   ↓
Pallet model predicts possible boxes
   ↓
Keep detections associated with the conveyor region
   ↓
Match boxes to existing temporary tracks
   ↓
Confirm/update tracks and reject obvious duplicates
   ↓
Provide a current pallet position to the live classifier
```

A visual **track ID is temporary**. It is not the RFID number printed on or assigned to a real pallet. Permanent physical pallet numbering was integrated later in Stage 07.

## 3. Key ideas

### Conveyor outer and inner boundaries

The conveyor is represented using two polygons: an **outer boundary** and an **inner boundary**. The useful conveyor region is the ring between them. This helps reject candidate detections that are clearly outside the belt area.

Boundary points are stored in a configuration file such as `conveyor_boundary.json`. If `coordinate_type` is `normalized`, point coordinates are expected between 0 and 1. Pixel values such as `342, 501` must not be treated as normalized values.

### Bounding-box matching

For each new frame, the tracker considers the location and size of current boxes relative to earlier boxes. Similarity measures such as intersection over union (IoU), relative position, and object size can support a match.

### Confirmation and duplicate suppression

A new box may need to appear consistently before it is treated as a stable track. Duplicate filtering aims to avoid two tracks for the same pallet, but it needs careful tuning because **two physically different pallets may be very close or overlapping**.

### Lost tracks and turns

At conveyor corners, a square pallet can look diamond-shaped; its axis-aligned bounding box may become much bigger. Partial obstruction by a pole or wire may also cause the detector to miss a pallet temporarily. Later live versions include short-term track-state persistence and conservative reconnection logic.

## 4. Input, output, and files

| Part | Description |
|---|---|
| Input | Detections from the Stage 02 pallet model |
| Configuration | Camera settings and conveyor boundary polygons |
| Output | Confirmed pallet tracks with temporary track IDs and positions |
| Downstream use | Stable crops and track histories for content classification |

A key supporting script used by the later Phase 70 system is named `dynamic_live_pallet_detection.py`. Its exact location in the final system is Stage 07's `CODE/` folder; the Stage 05 folder records the related development work. See the branch inventory below for files actually stored under Stage 05.

## 5. What this stage delivered

**Output:** the dynamic visual tracking logic needed to follow pallets through successive video frames rather than restarting with unrelated boxes each time.

## 6. Known limitations

Tracking is still sensitive to close pallets, merged detections, abrupt changes in apparent box size, and complete occlusion. Duplicate suppression that is too aggressive can hide a real neighboring pallet; too loose can display multiple boxes for one pallet. These are important live-test cases, especially before integrating a robot that can bring pallets close together.

## 7. For the professor / reviewer

This stage explains the difference between **detecting a pallet** and **continuously following that pallet**. Stage 06 combines the live detection/tracking work with the EMPTY/OBJECT_PRESENT classification workflow.

---

## Files actually included in this Git branch

The list below was generated from Git-tracked files on `conveyor-upload` when this README was installed. It is an inventory, not an instruction to run every script.

**Tracked files in this stage (excluding this README): 6**

### Folder overview
- `CODE/` — 4 tracked file(s)
- `CONFIG/` — 2 tracked file(s)

### Python source files
- `CODE/dynamic_live_pallet_detection.py`

### Model checkpoints
- No `.pt` file is tracked inside this stage; a runner may reference weights elsewhere.

### Configuration files
- `CONFIG/conveyor_boundary.json`
