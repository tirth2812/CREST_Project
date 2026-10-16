# First live EMPTY/OBJECT integration

**Folder:** `04_Live_Empty_Object_System/`

## Goal

Bring camera ingestion, pallet localization and empty/object classification together.

## Process

Camera frame -> pallet detection -> pallet crop -> EMPTY/OBJECT_PRESENT prediction -> displayed result.

## Inputs and artifacts

CODE contains the available live-integration scripts; model assets are documented in their training stages.

## Stage inventory

- 1 code/source files
- 0 image files
- 0 model checkpoint files

## Browse folders

- [CODE](CODE/README.md)

## Code files and documentation

- `CODE/live_empty_object_classification.py` — [source explanation](CODE/DOCS/live_empty_object_classification.py.README.md)

## Checks and limitations

Hardware/camera indexes and model paths are machine-specific; review source before running.
