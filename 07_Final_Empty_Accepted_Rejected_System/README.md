# Final live inspection and persistent RFID identity

**Folder:** `07_Final_Empty_Accepted_Rejected_System/`

## Goal

Run the complete pallet inspection pipeline and link temporary vision tracks to physical pallet IDs 1–6.

## Process

Camera -> YOLO26 pallet boxes -> dynamic tracking / duplicate filtering -> EMPTY/OBJECT_PRESENT -> X/NO_X symbol evidence -> ACCEPTED/REJECTED -> TOP RFID2 read -> physical identity memory / correction -> display and audit.

## Inputs and artifacts

CODE contains protected Phase 40, final Phase 65, and the dynamic tracker helper. CONFIG contains conveyor boundary and saved RFID zone. MODELS contains classifier checkpoints. The final detector best.pt is in Stage 00 and the original Conference installation.

## Stage inventory

- 5 code/source files
- 444 image files
- 3 model checkpoint files

## Browse folders

- [CODE](CODE/README.md)
- [CONFIG](CONFIG/README.md)
- [DATASET](DATASET/README.md)
- [MODELS](MODELS/README.md)
- [TOOLS](TOOLS/README.md)

## Code files and documentation

- `CODE/40_conference_100_pallet_live.py` — [source explanation](CODE/DOCS/40_conference_100_pallet_live.py.README.md)
- `CODE/65_conference_phase40_persistent_top_rfid.py` — [source explanation](CODE/DOCS/65_conference_phase40_persistent_top_rfid.py.README.md)
- `CODE/dynamic_live_pallet_detection.py` — [source explanation](CODE/DOCS/dynamic_live_pallet_detection.py.README.md)
- `TOOLS/62_crest_verify_rfid1_rfid2_physical_positions.py` — [source explanation](TOOLS/DOCS/62_crest_verify_rfid1_rfid2_physical_positions.py.README.md)
- `TOOLS/64_crest_single_rfid_reader_check.py` — [source explanation](TOOLS/DOCS/64_crest_single_rfid_reader_check.py.README.md)

## Final runnable files

- **Final RFID system:** [`CODE/65_conference_phase40_persistent_top_rfid.py`](CODE/65_conference_phase40_persistent_top_rfid.py)
- **Protected baseline:** [`CODE/40_conference_100_pallet_live.py`](CODE/40_conference_100_pallet_live.py)
- **Required tracker:** [`CODE/dynamic_live_pallet_detection.py`](CODE/dynamic_live_pallet_detection.py)
- **Top RFID mapping:** PLC RFID2 words 128–131; center-entry zone and persistent physical-ID ledger.
- **Original final detector:** from the Conference Stage 00 training run, not the older Stage 02 checkpoint.

See the [Phase 65 code README](CODE/DOCS/65_conference_phase40_persistent_top_rfid.py.README.md) for exact run commands and keypad controls.

## Checks and limitations

X = REJECTED; NO_X = ACCEPTED when classification evidence is sufficient. An RFID number can outlive a tracker box, but a temporarily lost box can still disappear. Validate physical association and camera/PLC wiring on live hardware.
