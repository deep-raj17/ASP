# CLI Security

Handlers avoid shell execution, reject output paths outside the workspace,
default import to dry-run in non-interactive mode, require archive approval,
and never expose delete or manual gate-pass commands. Text and JSON rendering
redacts common bearer-token, password, secret, token, and API-key forms.
Deferred mutations fail closed with `COMMAND_NOT_IMPLEMENTED`.
