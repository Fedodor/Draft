# Secrets Management

- Type: Security knowledge page
- Status: Implemented, rotation pending
- Canonical source: [`../../STEP_01_SECRETS_AND_ENV.md`](../../STEP_01_SECRETS_AND_ENV.md)
- Related: [Audit finding C1](../audits/project-audit-2026-10-22.md), [Remediation Step 1](../plans/remediation-plan.md)

## Current state

- `ml/.env` holds `OPENDOTA_API_KEY` and `STRATZ_API_KEY` in plaintext but is now ignored.
- The 19 hardcoded key matches in 12 files under `ml/training/` were removed in Step 1.
- A root `.gitignore` and `ml/.env.example` now exist.
- The directory is not a Git repository, so no history purge is needed today.

## Environment contract

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `STRATZ_API_KEY` | For STRATZ collectors | None | Fail before network access if missing |
| `OPENDOTA_API_KEY` | No | None | Warn once and continue unauthenticated |
| `ML_HTTP_TIMEOUT_SECONDS` | No | `30` | Request timeout |
| `ML_MAX_RETRIES` | No | `3` | Transient retry count |
| `ML_LOG_LEVEL` | No | `INFO` | Collector logging |

Canonical local file: `ml/.env`. Only one env file is loaded. `ml/.env.example` contains placeholders and is versioned once Git is initialized.

Precedence: process environment, then `ml/.env` with `override=False`, then code defaults for non-secret settings. Secrets have no code default.

## Loader rules

- One module owns dotenv loading, recommended `ml/training/settings.py`.
- Resolve paths from the module, not the current working directory.
- `get_opendota_key()` returns `None` when absent.
- `require_stratz_key()` raises `MissingApiKeyError`.
- Empty, whitespace, and placeholder values count as missing.
- Never print or partially print a key.
- Tests use `monkeypatch`, never the real `.env`.

## Repository guard

Before any `git init`:

```text
Create root .gitignore.
Ignore .env, .env.*, except .env.example.
Ignore keys, certificates, venv, caches, generated datasets, node_modules, target, dist.
Keep release models, schemas, manifests, and fixtures tracked.
```

If the repository already exists when this is read, scan history before assuming it is clean.

## Rotation runbook

1. Revoke old OpenDota and STRATZ keys at the providers.
2. Create replacements with minimum scope.
3. Put them only in `ml/.env`.
4. Migrate every collector to the loader.
5. Run a dry-run or one-page collection per provider.
6. Scan for secrets and search source for literals.
7. Confirm logs contain no key material.
8. Record the rotation date here, never the key.

If history exists: rotate first, scan, purge with `git filter-repo`, coordinate a force-push, then re-scan.

## Scanning

```powershell
gitleaks detect --source . --no-git --redact
gitleaks protect --staged --redact
gitleaks detect --source . --redact
```

Add the staged scan as a pre-commit hook and to CI in Step 8.

## Anti-patterns

- A key literal as a fallback.
- A default test key.
- Logging request URLs that contain `api_key`.
- Committing a `.env` file "for convenience".
- Ignoring `ml/datasets/` after it is already tracked.
- Trusting history cleanup instead of rotating.

## Open decisions

1. `ml/.env` only, or also root `.env`? Recommendation: `ml/.env` only.
2. Is OpenDota optional? Recommendation: yes.
3. Rotate now? Recommendation: yes, before the first commit or archive share.
4. Use an OS keyring? Defer until multi-user or CI requirements appear.
5. `gitleaks` or `detect-secrets`? Recommendation: `gitleaks`.

## Status

Step 1 code is implemented: `.gitignore`, `ml/.env.example`, `ml/training/settings.py`, collector migration, and 12 passing tests. The source secret scan is clean. Key rotation and the first Git commit remain pending; no credentials were changed as part of the audit or implementation.
