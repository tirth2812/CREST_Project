# 64_crest_single_rfid_reader_check.py

**Stage:** Final live inspection and persistent RFID identity
**Source path in package:** `07_Final_Empty_Accepted_Rejected_System/TOOLS/64_crest_single_rfid_reader_check.py`

## Purpose

Read-only PLC reader test that displays one reader's physical pallet numbers at a time.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\CODE\64_crest_single_rfid_reader_check.py`. This README is generated; the Python file is unchanged.

## Operational context

This module is retained as a development, annotation, training, live-stage or evaluation artifact. Do not assume it is the default live entrypoint. Read its arguments and source before using it with lab hardware.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `read_one_reader()` | 28 | Read four words of exactly one RFID reader (read-only). |
| `stamp()` | 46 | Handles stamp; precise behavior is in the source. |
| `main()` | 50 | Run the program's main processing loop. |

## Imported Python modules

`argparse`, `datetime`, `time`

## CLI strings visible in source

`--reader`

Other than the documented Phase 65 self-test, these text matches are not verified command-line options.

## Preservation

Only this Markdown document was generated; the program source has not been edited.
