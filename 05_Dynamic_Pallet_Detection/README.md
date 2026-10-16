# Dynamic detection and tracking

**Folder:** `05_Dynamic_Pallet_Detection/`

## Goal

Follow moving pallets over frames and retain useful track information as the conveyor moves.

## Process

Camera/detector boxes -> geometry and track association -> time-varying pallet identities -> downstream state.

## Inputs and artifacts

CODE stores tracker development scripts; CONFIG stores any available motion or conveyor configuration.

## Stage inventory

- 1 code/source files
- 0 image files
- 0 model checkpoint files

## Browse folders

- [CODE](CODE/README.md)
- [CONFIG](CONFIG/README.md)

## Code files and documentation

- `CODE/dynamic_live_pallet_detection.py` — [source explanation](CODE/DOCS/dynamic_live_pallet_detection.py.README.md)

## Checks and limitations

Tracking IDs are temporary software IDs, not physical RFID pallet numbers.
