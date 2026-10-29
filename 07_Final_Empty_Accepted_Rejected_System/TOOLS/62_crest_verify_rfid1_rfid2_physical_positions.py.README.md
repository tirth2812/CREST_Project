# 62_crest_verify_rfid1_rfid2_physical_positions.py

**Stage:** Final live inspection and persistent RFID identity
**Source path in package:** `07_Final_Empty_Accepted_Rejected_System/TOOLS/62_crest_verify_rfid1_rfid2_physical_positions.py`

## Purpose

Read-only test for identifying the physical positions of the two RFID readers.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\62_crest_verify_rfid1_rfid2_physical_positions.py`. This README is generated; the Python file is unchanged.

## Operational context

This module is retained as a development, annotation, training, live-stage or evaluation artifact. Do not assume it is the default live entrypoint. Read its arguments and source before using it with lab hardware.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `load_camera_rois()` | 55 | Handles load camera rois; precise behavior is in the source. |
| `read_words()` | 80 | Read exactly four input words, first batch, then individual fallback. |
| `plc_poll_thread()` | 101 | Handles plc poll thread; precise behavior is in the source. |
| `draw_label()` | 152 | Handles draw label; precise behavior is in the source. |
| `annotate_frame()` | 160 | Handles annotate frame; precise behavior is in the source. |
| `main()` | 188 | Run the program's main processing loop. |

## Imported Python modules

`__future__`, `argparse`, `csv`, `datetime`, `json`, `pathlib`, `queue`, `threading`, `time`

## CLI strings visible in source

`--console-only`, `--self-test`

Other than the documented Phase 65 self-test, these text matches are not verified command-line options.

## Preservation

Only this Markdown document was generated; the program source has not been edited.
