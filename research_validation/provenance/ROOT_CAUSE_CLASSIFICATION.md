# Root Cause Classification

**Final decision: ROOT CAUSE LIKELY IDENTIFIED**

Dominant classification: **Remote server limitation** (medium confidence).
The network path is stable by the bounded ping test, HTTPS negotiation
succeeds, and the server returns explicit retry/rate-limit metadata. The large
archive stream proceeds slowly enough that bounded client timeouts expire.

This is a likelihood assessment, not proof of a specific Zenodo internal
throttle. No evidence supports local storage, DNS, TLS, proxy, firewall, or
antivirus as the dominant cause.
