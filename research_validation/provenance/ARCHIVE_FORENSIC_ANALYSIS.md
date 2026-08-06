# Archive Forensic Analysis

No ZIP archive is present under `E:\MIMII`; the root contains extracted
directories only. Therefore archive listing, extraction replay, compression
method, ordering, and timestamp comparisons cannot be performed from the
canonical root without introducing a new extraction operation.

Nine official-named ZIPs were found outside the root and were MD5-checked
read-only. Five matched, one mismatched, and three were absent. This establishes
an archive-container discrepancy but does not by itself prove extracted-payload
corruption.
