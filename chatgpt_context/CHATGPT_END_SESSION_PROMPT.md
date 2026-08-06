# ChatGPT End-of-Session Prompt — CHAAD Project

> Copy and paste this at the end of a ChatGPT working session to ensure proper documentation.

---

I am ending this session. Before concluding, update the project documentation:

## Mandatory Documentation Updates

### 1. Session Log
Add an entry to `docs/SESSION_LOG.md` with:
- Session date
- Objective
- Files changed (exact paths)
- Commands executed (exact command lines)
- Validation results (real output, not assumptions)
- Decisions made
- Errors encountered
- Remaining work
- Next recommended action

### 2. Current State
Update `docs/CURRENT_STATE.md` if the project state changed:
- Change any capability status (VERIFIED, BLOCKED, etc.)
- Update the auto-generated metadata section
- Update any blockers that were resolved or discovered

### 3. Roadmap
Update `docs/ROADMAP.md` if milestones were completed or changed:
- Mark completed items with [x]
- Add new blocked items with [!]
- Update dependencies

### 4. Known Issues
Update `docs/KNOWN_ISSUES.md` if issues were:
- Discovered (new entries)
- Resolved (move to Resolved section)
- Reproduced (add reproduction steps)

### 5. Decision Log
Update `docs/DECISIONS.md` if any significant decision was made:
- Use ADR format with decision ID, context, rationale, evidence

### 6. Experiment Log
Update `docs/EXPERIMENT_LOG.md` if any experiment ran:
- Record experiment ID, date, status, metrics, evidence paths

### 7. Current Handoff
Update `chatgpt_context/CHATGPT_CURRENT_HANDOFF.md`:
- What was done
- What remains
- Exact next commands
- Active blockers

## Exact Output Required

Provide a structured summary:

```markdown
## Session Close Summary — YYYY-MM-DD

### Files Changed
- path/to/file (modified/created)
- ...

### Commands Executed
```bash
command1  # brief note
command2  # output summary: PASS/FAIL/value
```

### Validation Results
| Test | Result | Evidence |
|------|--------|----------|

### Documentation Updated
- docs/SESSION_LOG.md: added session entry
- docs/CURRENT_STATE.md: updated capability X to VERIFIED
- docs/ROADMAP.md: marked milestone Y as [x]

### Active Blockers
- Blocker description (resolution path)

### Next Recommended Action
1. Specific command to run
2. ...

### Suggested Git Commit Message
```

## Do NOT
- Claim tests passed if you didn't run them
- Claim work completed if validation failed
- Mark blocked items as done
- Skip the session log entry
