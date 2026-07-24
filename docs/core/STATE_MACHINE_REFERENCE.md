# State Machine Reference

Workflow states are `NOT_STARTED`, `READY`, `RUNNING`, `BLOCKED`, `FAILED`,
`COMPLETED`, and `CANCELLED`. Gate states are `UNEVALUATED`, `PENDING`,
`SATISFIED`, `UNSATISFIED`, `BLOCKED`, and `WAIVED`. Missing input blocks;
verified failure is unsatisfied. Terminal workflows reject further transitions.
