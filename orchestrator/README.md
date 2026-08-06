# CHAAD Local Orchestrator

This controller connects two separate agents:

1. an OpenAI Responses API planner that selects one bounded task; and
2. the local `codex exec` implementation agent that edits and validates the
   repository.

It does not import browser-chat history. Put durable objectives and evidence
boundaries in `MASTER_GOAL.md`, controller state, or explicit approval records.

## Safety design

- Real runs refuse a dirty worktree unless `--allow-dirty` is supplied.
- Planner and other common credential environment variables are removed from
  the `codex exec` child environment.
- Codex uses its separately saved CLI/ChatGPT login.
- `codex exec` runs with `workspace-write` and `--ephemeral`.
- Planner JSON is type-checked and restricted to known actions/risk flags.
- High-risk tasks stop at a human gate.
- Protected-test access requires both:
  - `"protected_test_authorised": true` in `state.json`; and
  - `reports/submission_recovery/PROTECTED_TEST_APPROVAL.md` containing
    `DECISION: APPROVED` and `SCOPE: PROTECTED_TEST`.
- Scientific experiments also require
  `"research_reentry_authorised": true`.
- Every successful Codex iteration must update the structured Markdown report.
- Raw planner/Codex outputs are archived under the ignored `raw/` directory.

These controls reduce risk; they are not a security boundary against malicious
repository code or a substitute for human review.

## Setup

Create an isolated environment. On PowerShell:

```powershell
python -m venv .orchestrator-venv
.\.orchestrator-venv\Scripts\Activate.ps1
python -m pip install -r orchestrator\requirements.txt
codex login
```

If PowerShell blocks the npm wrapper, use `codex.cmd` or the Codex executable
bundled with the IDE. On WSL:

```bash
python3 -m venv .orchestrator-venv
source .orchestrator-venv/bin/activate
python -m pip install -r orchestrator/requirements.txt
codex login
```

Configure the planner credential in the shell, never in repository files:

```powershell
$env:OPENAI_API_KEY = "..."
$env:PLANNER_MODEL = "gpt-5.1"  # optional explicit override
```

## Safe validation

These commands make no API or nested Codex call:

```powershell
python orchestrator\controller.py --preflight
python orchestrator\controller.py --dry-run
python -m pytest tests\test_orchestrator_controller.py -p no:cacheprovider -q
```

## Controlled execution

The current worktree is heavily dirty, so preserve/review it before enabling
autonomous edits. A one-iteration execution is:

```powershell
python orchestrator\controller.py --max-iterations 1 --timeout 1800 --allow-dirty
```

Then inspect:

```powershell
Get-Content orchestrator\state.json
Get-Content reports\autonomous_loop\LATEST_CODEX_REPORT.md
Get-ChildItem reports\autonomous_loop\raw -Recurse
git status --short
git diff
```

Do not increase the iteration limit until the first run and its diff have been
reviewed. `state.json`, `runtime/`, raw transcripts, `.env`, and the
orchestration virtual environment are intentionally ignored by Git.
