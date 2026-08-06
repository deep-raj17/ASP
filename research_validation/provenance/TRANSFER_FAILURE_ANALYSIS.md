# Transfer Failure Analysis

The failure class is **Network timeout**. DNS and HTTPS headers succeeded, but
the content stream delivered only 64,652,336 bytes during a 120-second range
resume attempt for `-6_dB_fan.zip`; curl exited 28. Earlier attempts also
stalled. The partial files were quarantined and not treated as archives.

No checksum mismatch was evaluated because no complete archive existed. No
archive was extracted.
