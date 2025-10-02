# Copilot Instructions

## Project snapshot
- Full Auto CI automates cloning repos, running lint/tests, and storing results; entry script is `main.py` → `src/cli.py`.
- Core logic is in `src/service.py`, backed by SQLite and git subprocesses; threads poll repos and queue test runs.

## Key modules & flows
- `CIService` wires `Config`, `GitTracker`, and `ToolRunner`. `start()` spins worker threads plus a monitor loop; `run_tests()` is the synchronous path the CLI uses.
- `GitTracker` (`src/git.py`) persists repo metadata in SQLite and shells out to system `git` (`clone`, `pull`, `log`, `checkout`). Real network access happens unless you patch `subprocess.run`.
- `ToolRunner` runs `Pylint` and `Coverage` (`src/tools.py`) sequentially on the repo path. Coverage changes cwd, runs `coverage run -m pytest`, then `coverage xml` and parses `coverage.xml`.

## Data layout & persistence
- SQLite file defaults to `~/.fullautoci/database.sqlite`; `CIService._setup_database()` creates `repositories`, `commits`, `results`, `users` tables only.
- Several callers expect extra tables/columns (`api_keys`, `test_runs`, `repositories.status/last_check`). Create them in tests/fixtures before using those surfaces.
- Git clones live under `~/.fullautoci/repos/<id>_<name>`; service may delete/recreate directories during `clone()`.

## API & webhooks
- `src/api.py` only activates Flask routes if optional `api` extras are installed. Without Flask, `API.run()` logs a placeholder.
- Webhooks (`src/webhook.py`) require the repo URL in SQLite to match the incoming clone URL exactly. GitHub uses `clone_url`, GitLab `git_http_url`, Bitbucket URLs are normalized to `.git`.
- Webhook handlers return commit dicts fed into `CIService.add_test_task()`. That helper currently enqueues keys `repository_id`/`commit_hash`, while workers expect `repo_id`/`commit`; keep this inconsistency in mind when producing queue items.

## CLI & service operations
- CLI commands (`full-auto-ci …`) are in `src/cli.py`; they mainly forward to `CIService`. The service status flag is `CIService.running` (no PID tracking).
- Starting the service (`service start`) spawns background threads inside the current process; there is no daemonization. Use mocks when testing to avoid hanging threads.

## Testing & local workflows
- Run unit tests with `pytest` from repo root; suites heavily patch threading, subprocess, and filesystem calls. Temporary SQLite files are the norm in fixtures.
- When exercising service/tool code, patch `subprocess.run`, `os.chdir`, and `GitRepo` methods to avoid real git/coverage executions.
- Coverage reports require `coverage` CLI on PATH; ensure `coverage xml` succeeds or adjust tests to stub it.

## Configuration conventions
- Config loads from `~/.fullautoci/config.yml`; copy `config.example.yml` for defaults. `Config.get(section, key)` falls back to the in-memory defaults defined in `src/config.py`.
- Tool configs accept overrides via YAML (`tools.pylint.config_file`, `tools.coverage.run_tests_cmd`) but the concrete runners currently ignore these values.

## Known gaps & cautions
- Service/database schema drift: align DB migrations manually before extending API/webhook functionality.
- Git/webhook paths assume Unix-style home directories; adjust `work_dir` via `GitRepo` constructor in tests if needed.
- Thread loops catch broad exceptions and only log; add instrumentation when debugging silent failures.
