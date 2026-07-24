# Registry Architecture

SQLite stores a single ordered append-only event journal partitioned by registry
name. Disposable views are derived by replay.
