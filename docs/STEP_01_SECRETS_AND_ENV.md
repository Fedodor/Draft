# Step 1 Design: Secrets and Environment

- Date: 2026-10-22
- Status: Implemented, key rotation pending
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit finding: [`AUDIT.md`](AUDIT.md), C1 and M4

## Implementation status

- [x] Root `.gitignore` added before Git initialization.
- [x] `ml/.env.example` added with placeholders.
- [x] `ml/training/settings.py` added with required/optional key handling.
- [x] All 12 collector and diagnostic scripts migrated off literals.
- [x] `train_pipeline.py` loads the OpenDota key through `settings.py`.
- [x] `data_collector.py` (single-line, key-free) was reconstructed in Step 6 with no credential literals.
- [x] `python-dotenv` and `pytest` added to `ml/requirements.txt`.
- [x] Settings tests and a source secret scan added; 12 tests pass.
- [ ] Rotate the OpenDota and STRATZ keys.
- [ ] Initialize Git only after the rotated working tree is clean.

## Context

The project currently stores API credentials in `ml/.env` and hardcodes them in 12 Python files. The directory is not a Git repository yet, so there is no commit history to purge today. That makes this the safest possible moment to add a repository guard and a single environment contract.

The risk is not only accidental publication. Plaintext keys can leak through logs, screenshots, shell history, backups, generated notebooks, and shared archives. Rotation is still required even without Git history.

## Goals

1. Ensure no credential literal exists in source or documentation.
2. Make the required/optional status of each key explicit.
3. Fail fast and clearly when a required key is missing.
4. Keep local development convenient with one documented `.env` file.
5. Prevent a future `git init` from staging secrets or generated data.
6. Add a repeatable secret scan.

## Non-goals

- Building a production secret manager or cloud key vault.
- Adding key management to the Rust application.
- Changing collector behavior beyond credential loading.
- Purging Git history in a repository that does not exist yet.
- Encrypting the local `.env` file.

## Current-state inventory

| Location | Key | Current state | Target |
| --- | --- | --- | --- |
| `ml/.env` | `OPENDOTA_API_KEY` | Plaintext | Local only, ignored |
| `ml/.env` | `STRATZ_API_KEY` | Plaintext | Local only, ignored |
| `ml/training/collect_opendota.py` | OpenDota | Hardcoded | Loader, optional |
| `ml/training/collect_opendota_pro.py` | OpenDota | Hardcoded | Loader, optional |
| `ml/training/collect_allpick.py` | OpenDota | Hardcoded | Loader, optional |
| `ml/training/collect_100k.py` | OpenDota | Hardcoded | Loader, optional |
| `ml/training/collect_pro_matches.py` | OpenDota | Hardcoded | Loader, optional |
| `ml/training/collect_stratz.py` | STRATZ | Hardcoded | Loader, required |
| `ml/training/collect_stratz_allpick.py` | STRATZ | Hardcoded | Loader, required |
| `ml/training/collect_stratz_tournaments.py` | STRATZ | Hardcoded | Loader, required |
| `ml/training/stratz_player_test.py` | STRATZ | Hardcoded | Loader, required |
| `ml/training/test_stratz_browse.py` | STRATZ | Hardcoded | Loader, required |
| `ml/training/test_pro_players.py` | STRATZ | Hardcoded | Loader, required |
| `ml/training/fetch_versions.py` | STRATZ | Hardcoded | Loader, required |
| Repository root | none | No `.gitignore` | Add before `git init` |
| `.git` | none | Not a Git repository | Initialize only after guard |

Count: 19 credential matches across 12 files.

## Proposed environment contract

### Canonical file

Use `ml/.env` as the only supported local environment file. Do not also load a root `.env`; two locations create ambiguous precedence and make accidental commits more likely.

`ml/.env.example` is versioned once Git is initialized and contains placeholders only.

### Variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `STRATZ_API_KEY` | Yes for STRATZ collectors | None | STRATZ GraphQL authentication |
| `OPENDOTA_API_KEY` | No | None | Higher OpenDota rate limits |
| `ML_HTTP_TIMEOUT_SECONDS` | No | `30` | Collector request timeout |
| `ML_MAX_RETRIES` | No | `3` | Transient HTTP retry count |
| `ML_LOG_LEVEL` | No | `INFO` | Collector logging level |

Do not add model or data paths to `.env` unless they genuinely vary by machine. Paths belong in the settings module with project-relative defaults.

### Precedence

1. Process environment.
2. `ml/.env`, loaded with `override=False`.
3. Code defaults for non-secret settings only.

Secrets never have a code default. CLI flags may override behavior such as output path or request limit, but must not override credentials.

### Required versus optional

- `STRATZ_API_KEY`: required for every script that calls STRATZ. Raise a specific error before any network request.
- `OPENDOTA_API_KEY`: optional. OpenDota permits limited unauthenticated access. When absent, log one warning and continue.
- Empty, whitespace-only, or quoted placeholder values count as missing.

## Proposed module

Recommended path: `ml/training/settings.py`.

Responsibilities:

- Resolve `ML_ROOT` from the module location, not the current working directory.
- Load `ml/.env` once.
- Expose `get_opendota_key() -> str | None`.
- Expose `require_stratz_key() -> str`.
- Expose a redacted configuration summary for startup logs.
- Never log or return the raw key except through the explicit getter used by an HTTP client.

Suggested exceptions:

```python
class ConfigurationError(RuntimeError):
    """Base class for environment configuration failures."""


class MissingApiKeyError(ConfigurationError):
    """Raised when a required API key is absent."""
```

Suggested behavior:

```text
get_opendota_key()
  missing or placeholder -> None, warn once

require_stratz_key()
  missing or placeholder -> MissingApiKeyError with setup instructions
  present -> stripped value
```

Error text should say what to do, not what the value is:

```text
STRATZ_API_KEY is not set. Copy ml/.env.example to ml/.env and set a valid key.
```

### Call-site migration

- OpenDota collectors call `get_opendota_key()` and pass the value only when present.
- STRATZ collectors call `require_stratz_key()` at the start of `main()` or client construction.
- Tests use `monkeypatch.setenv` or `monkeypatch.delenv`; tests must never read the real `.env`.
- No script should call `load_dotenv` directly; only the settings module loads it.

## Repository guard

Create a root `.gitignore` before any of these commands:

```powershell
git init
git add .
```

Minimum patterns:

```gitignore
# Secrets
.env
.env.*
!.env.example
*.key
*.pem
*.p12
*.pfx

# Python
ml/venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ipynb_checkpoints/

# Generated ML data and artifacts that are not release assets
ml/data/
ml/datasets/
ml/cache/
ml/logs/
ml/training/output/

# Node and Tauri
node_modules/
apps/desktop/node_modules/
apps/desktop/dist/
apps/desktop/src-tauri/target/
target/

# Local OS and editor files
.DS_Store
Thumbs.db
.idea/
.vscode/
```

Rules:

- Keep release models under `ml/models/` tracked and intentional.
- Do not ignore `ml/.env.example`.
- Do not ignore feature schemas, model manifests, or deterministic fixtures.
- If a path is already tracked after initialization, run `git rm --cached` and commit the removal.

## Secret scanning

Local working-tree scan, before Git exists:

```powershell
gitleaks detect --source . --no-git --redact
```

After Git is initialized:

```powershell
gitleaks protect --staged --redact
gitleaks detect --source . --redact
```

Recommended follow-up:

- Add a pre-commit hook that runs `gitleaks protect --staged`.
- Add the same scan to CI in Step 8.
- Never use `--no-git` as the only long-term control.

## Key rotation runbook

Because `ml/.env` and source files contain plaintext keys, rotate both keys even though there is no Git history.

1. Revoke the existing OpenDota and STRATZ keys in their provider dashboards.
2. Create replacement keys with the minimum required scope.
3. Put the new values only in `ml/.env`.
4. Update every collector to use the loader.
5. Run a dry-run or one-page collection for each provider.
6. Run the secret scan and a source search.
7. Confirm logs do not contain the keys.
8. Record the rotation date in the wiki security page, never the key value.

If the project is later found to have another Git remote or an exported archive:

1. Rotate first. History cleanup does not make a leaked key safe.
2. Scan the history with `gitleaks detect --source . --redact`.
3. Purge with `git filter-repo --replace-text replacements.txt`.
4. Force-push only after coordinating with all users.
5. Re-scan the rewritten history.

## Failure behavior

| Situation | Behavior | Exit code |
| --- | --- | --- |
| STRATZ key missing in a STRATZ collector | Clear error before network | `2` |
| OpenDota key missing | One warning, continue unauthenticated | `0` |
| Placeholder value detected | Treat as missing | `2` for STRATZ |
| `.env` missing | Same as missing variables | Depends on key |
| Request timeout | Existing retry policy, then error | Existing |
| Key present but rejected by provider | Clear provider error, no key in message | Non-zero |

Never print the key, even partially. A fingerprint such as the last four characters is unnecessary for this project.

## Test plan

| Test | Setup | Expected |
| --- | --- | --- |
| STRATZ key missing | Delete env var, point loader at empty env | `MissingApiKeyError` |
| STRATZ key empty | Set to `""` | `MissingApiKeyError` |
| STRATZ key placeholder | Set to `replace-with-stratz-token` | `MissingApiKeyError` |
| OpenDota key missing | Delete env var | Returns `None`, warns once |
| Process env wins | Set process env and `.env` differently | Process value used |
| `.env` does not override | Set process env, load `.env` | Process value retained |
| No key in error | Trigger missing key | Error text contains variable name, not value |
| Source has no literals | Scan `ml/training` | No key assignment matches |

The source scan should be a focused test or a CI scan, not a fragile regex embedded in application code.

## Implementation sequence

1. Add root `.gitignore`.
2. Add `ml/.env.example`.
3. Add `ml/training/settings.py` and unit tests.
4. Replace OpenDota call sites.
5. Replace STRATZ call sites.
6. Rotate keys and update `ml/.env`.
7. Run collectors in dry-run or one-page mode.
8. Run secret scans and source search.
9. Only then initialize Git and make the first commit.
10. Update the wiki and status board.

## Acceptance criteria

- [x] No credential literal exists in source or documentation.
- [x] `.env` files are ignored and `.env.example` is not.
- [x] STRATZ collectors fail before network access when the key is absent.
- [x] OpenDota collectors continue without a key and warn once.
- [x] Process environment takes precedence over `ml/.env`.
- [x] Tests do not depend on a developer's real `.env`.
- [x] The source secret scan test passes. `gitleaks` remains the recommended full-tree scan before the first commit.
- [ ] Rotation is complete and recorded without the key value.
- [ ] The first Git commit contains no secrets or generated datasets.

## Verification commands

```powershell
Test-Path .git
rg -n "(OPENDOTA_API_KEY|STRATZ_API_KEY)\s*=\s*['\"]" ml
& ml\venv\Scripts\python.exe -m pytest ml/tests -q
& ml\venv\Scripts\python.exe -m compileall -q ml/training
gitleaks detect --source . --no-git --redact
```

The pytest suite includes a source scan for key assignments and JWTs, so it runs without `gitleaks`. Run `gitleaks` too when available, especially before the first commit.

Expected before Git initialization: `Test-Path .git` returns `False`. After the guard is in place and the working tree is clean, `git init` is allowed.

## Rollout and rollback

Rollout is configuration-only for collectors and does not affect the desktop app. If a collector breaks, restore the previous code path temporarily, but do not restore hardcoded keys; fix the loader or add the missing variable.

Rollback must not reintroduce plaintext credentials. If the rotation is incomplete, keep the old key revoked and block the affected collector rather than committing a fallback.

## Decisions

1. **Environment file location:** Applied. `ml/.env` is the only supported file; root `.env` is not loaded.
2. **OpenDota requirement:** Applied. It is optional and warns once when absent.
3. **Rotation timing:** Pending. Rotate before the first Git commit and before sharing the project.
4. **History purge:** Not applicable today because there is no `.git`. Re-check before any remote is added.
5. **OS keyring:** Deferred. A local `.env` is acceptable for a single-developer desktop project; revisit if the project becomes multi-user or CI-driven.
6. **Secret scan tool:** A Python source-scan test is implemented. Add `gitleaks` before the first commit for full-tree and history coverage.

## Estimated effort

- Code and tests: 2 to 3 hours.
- Key rotation and collector smoke tests: 30 to 60 minutes, plus provider propagation time.
- Documentation and verification: 30 minutes.

Total: roughly half a day, excluding provider delays.

## Definition of done

This step closes when the acceptance criteria are checked, the status board in [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md) is updated, and the wiki security page reflects the new environment contract.
