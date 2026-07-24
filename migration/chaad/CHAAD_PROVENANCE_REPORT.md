# CHAAD Provenance Report

Every accepted evidence record stores a current SHA-256, canonical metadata
SHA-256, original file timestamp, relative source identity, producer, tool
version, repository revision, project reference, and workflow reference.
Verification used deterministic `file-checksum` and, for JSON artifacts,
`structured-schema` verifiers.

The local evidence store necessarily contains absolute paths for live file
verification. It is private runtime state under `.ros/` and is excluded from
public exports and Git. Adapter manifests and migration reports contain no
private dataset root.

The authoritative MIMII Zenodo record (DOI `10.5281/zenodo.3384388`) identifies
release `public 1.0`, four machine IDs 00/02/04/06, four machine types, three
noise levels, and CC BY-SA 4.0. Local structure agrees, but source-archive
identity remains unverified because the official archive MD5 values cannot be
reconstructed from extracted directories.
