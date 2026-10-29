# Stage 01 — Pallet Image Collection and Annotation

**CREST | Data preparation for pallet detection**

[Project overview](../README.md) · **Next: [Stage 02 — Pallet Detection](../02_YOLO_Pallet_Detection/README.md)**

## 1. What this stage is for

The first task in CREST was teaching a model **what a pallet looks like and where it appears in a camera image**. We collected images of the conveyor and manually marked each visible pallet with a rectangular bounding box. These marked images are the ground truth used to train and test the pallet detector in Stage 02.

A *bounding box* describes the location of a pallet. It does **not** say whether that pallet is empty, accepted, or rejected; those questions are handled in later stages.

## 2. Work carried out

1. Capture full conveyor images, including pallets on straight sections and around the turns.
2. Draw a box around each pallet in an image. One image can contain several pallets.
3. Assign the detection class `0` (`pallet`) to every box.
4. Save box annotations in YOLO format alongside the corresponding image.
5. Review annotations for missing pallets, incorrect boxes, and inconsistent labeling.
6. Organize the final detection data into separate training, validation, and test portions for Stage 02.

## 3. Dataset used in the final 100-image detector experiment

| Set | Full-frame images | Labeled pallet boxes |
|---|---:|---:|
| Training | 80 | 761 |
| Validation | 10 | 106 |
| Test | 10 | 105 |
| **Total** | **100** | **972** |

The recorded full frames are **1920 × 1080 pixels**. The 100-image collection grew from an earlier 20-image collection, with 80 additional images. **100 images describes the entire dataset, not the number of images used for parameter training**: 80 are in the training split.

### What an annotation looks like

For a single pallet, the annotation file contains a line in this order:

```text
class_id x_center y_center box_width box_height
```

The four location values are normalized between 0 and 1 relative to the image width and height. For example, a center at the middle of an image has approximate normalized coordinates `(0.5, 0.5)`. This is a format illustration, **not** a measured annotation from the dataset.

## 4. How this stage connects to the rest of CREST

```text
Camera images
   ↓
Manually drawn pallet boxes
   ↓
YOLO labels and reviewed dataset splits
   ↓
Stage 02: train and evaluate pallet detection
```

Stage 02 uses these labeled images to learn a pallet's position. Stage 03 uses different, **classification-style** images and labels; it is not the same dataset.

## 5. What to look for in this folder

- **Images / dataset folders:** captured conveyor frames or prepared annotation data, if included in this repository copy.
- **Annotation or preparation scripts (`.py`):** tools for marking, checking, or organizing images, where available.
- **Labels (`.txt`):** one or more normalized YOLO bounding-box records for an image.
- **Configuration (`.yaml`, `.json`):** class names, image locations, or dataset settings, if present.
- **Visual examples/results:** evidence that pallet boxes align with the physical pallets.

The file inventory at the end of this README is generated from the files actually tracked in the `conveyor-upload` branch. Large training datasets may also be retained separately on the lab computer.

## 6. What this stage delivered

**Output:** a labeled, split pallet-detection dataset that could be used for detector training and held-out evaluation. It did not produce accepted/rejected decisions or RFID identity mapping.

## 7. Important limitations

An annotation can be inaccurate if the pallet is partly obscured, viewed at an angle, or cut off by the image boundary. The detector's results depend on annotation consistency. Train/validation/test images also need to remain separated during evaluation.

## 8. For the professor / reviewer

This stage documents the **source of the detector's ground truth**. For the associated learning procedure and detector test results, continue to Stage 02.

---

## Files actually included in this Git branch

The list below was generated from Git-tracked files on `conveyor-upload` when this README was installed. It is an inventory, not an instruction to run every script.

**Tracked files in this stage (excluding this README): 130**

### Folder overview
- `CODE/` — 4 tracked file(s)
- `CONFERENCE_100/` — 103 tracked file(s)
- `DATA/` — 23 tracked file(s)

### Python source files
- `CODE/pallet_annotator.py`

### Model checkpoints
- No `.pt` file is tracked inside this stage; a runner may reference weights elsewhere.

### Configuration files
- `DATA/pallet_annotations.json`
