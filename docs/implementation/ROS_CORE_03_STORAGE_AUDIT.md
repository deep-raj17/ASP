# ROS-CORE-03 Storage Audit

SQLite was selected for local-first ACID writes, WAL recovery, portability, and
future backend abstraction. UPDATE/DELETE triggers fail closed. Event history is
authoritative; current views replay history.
