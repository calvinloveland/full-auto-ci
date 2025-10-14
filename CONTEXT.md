# Full Auto CI - Development Context

## Current Status

- Initial project structure created
- Core service implementation started
- CLI interface implemented
- API skeleton in place
- Tool modules for Pylint and Coverage created
- Basic unit tests written

## Next Steps

### Phase 1 (Current Focus)

- [x] Complete Git integration
  - [x] Implement repository cloning
  - [x] Add commit detection and tracking
  - [x] Add webhook support for automatic triggers

- [x] Enhance SQLite database
  - [x] Add proper migrations system *(basic column backfill helpers in service setup)*
  - [x] Implement data access layers *(see `src/db.py`, `CIService` refactor)*
  - [x] Add indexes for performance

- [ ] Complete CLI functionality
  - [x] Add configuration commands *(new `config` subcommands for show/set/path)*
  - [x] Add user management *(new `user` add/list/remove commands backed by service)*
  - [x] Implement better output formatting

- [ ] Improve test coverage
  - [x] Integration tests *(see `tests/test_integration_service_flow.py`)*
  - [x] End-to-end tests *(CLI+service flow exercised in `tests/test_integration_service_flow.py`)*
  - [x] Add dashboard template regression tests *(see `tests/test_dashboard.py`)*
  - [ ] Stand up browser automation suite *(Playwright or Selenium; capture smoke flows for dashboard)*

### Phase 2 (Coming Soon)

- [ ] Implement Flask web dashboard
  - [x] Set up basic Flask application
  - [x] Create dashboard templates with Jinja2
  - [x] Add HTMX for dynamic content
  - [ ] Implement authentication and session management
  - [ ] Wire dashboard smoke tests into CI *(reuse browser automation harness)*

- [ ] Add visualization for test results
  - [x] Historical trends
  - [x] Commit-by-commit comparison

- [ ] External CI system integrations
  - [x] Define provider abstraction layer (shared interface for pollers, webhooks, credential management)
  - [x] Add CLI scaffolding for provider lifecycle (list/add/remove/sync)
  - [ ] Implement shared job/status ingestion workflow (normalize runs → `test_runs` + `results` tables)
  - [ ] Document security posture (secret storage, OAuth/token scopes) and fallback modes
  - [ ] Build provider health dashboard panel + CLI diagnostics (`full-auto-ci providers status`)

  #### GitHub Actions
  - [ ] Support PAT / GitHub App auth flows with fine-grained scopes (repo, actions:read)
  - [ ] Implement webhook handler for `workflow_job` + `check_suite` events → queue CIService tasks
  - [ ] Add poller for scheduled sync of run metadata (GraphQL `workflowRuns` fallback)
  - [ ] Surface re-run hooks (`workflow_dispatch`) from dashboard and CLI

  #### GitLab CI
  - [ ] Add personal-access-token + project access token credential handling
  - [ ] Consume `Pipeline Hook` events; map statuses to internal enum
  - [ ] Implement REST polling for pipelines/jobs when webhooks unavailable (self-hosted installs)
  - [ ] Provide namespace/project selector in config + auto-pagination for large histories

  #### Jenkins
  - [ ] Support basic auth + API token + crumb issuer negotiation
  - [ ] Add job discovery via `/api/json` and selectively subscribe to builds
  - [ ] Stream console output + test reports (JUnit) into `results`
  - [ ] Offer CLI command to trigger parameterized builds and watch completion

  #### Bamboo CI
  - [ ] Handle PAT and basic auth modes; document requirements for remote agent visibility
  - [ ] Poll `/result` endpoints for plan status and artifacts
  - [ ] Parse Bamboo test result API (JUnit/coverage) into dashboard summaries
  - [ ] Add rate-limit aware backoff + alerting when Bamboo queue backlog detected

  #### Additional Providers (stretch)
  - [ ] Azure DevOps pipelines (use Service Principal + OAuth)
  - [ ] CircleCI (personal token, pipeline webhook listener)
  - [ ] Buildkite (graphQL pipeline events, artifact fetcher)
  - [ ] Generic webhook adapter for unsupported providers

  #### Architecture & Sequencing Notes
  - **Unified provider interface**: introduce `BaseProvider` with hooks for `enqueue_from_webhook`, `sync_runs`, `trigger_run`, `validate_credentials`. Providers reside under `src/providers/<name>.py` with shared utilities in `src/providers/base.py`.
  - **Background workers**: extend `CIService` monitor threads to include provider-specific pollers; reuse existing queue for normalized `TestTask` items.
  - **Data model extensions**:
   - Add `external_providers` table (`id`, `type`, `name`, `config`, `created_at`)
   - Add `external_jobs` table to track remote run IDs → internal `test_runs`
   - Add `provider_credentials` encrypted blob storage (integration with OS keyring or user-provided KMS)
  - **Configuration UX**: new CLI commands: `provider add <type>`, `provider list`, `provider status`, `provider sync`. YAML config sections carry defaults per provider; secrets stored separately.
  - **Security**: implement secret masking in logs, rotate tokens via reminders, support read-only mode for audit-first deployments. Document minimal scopes for each provider.
  - **Telemetry/Alerts**: emit provider health metrics (last sync timestamp, error counts) into dashboard status card + optional Prometheus export in future phase.

  #### Milestone Plan
  1. **Foundations (Week 1)**
    - Ship provider abstraction & schema migrations
    - Implement provider registry + CLI scaffolding
    - Add smoke tests for registry and configuration flows
  2. **GitHub Actions (Weeks 2-3)**
    - Webhook ingestion (checks/workflow job)
    - REST/GraphQL polling fallback
    - Dashboard surfacing of GitHub run metadata
  3. **GitLab CI (Week 4)**
    - Pipeline hook listener & manual sync command
    - Support self-hosted namespaces
  4. **Jenkins (Week 5)**
    - API token auth, job discovery, build trigger support
    - Console log + JUnit parser integration
  5. **Bamboo CI (Week 6)**
    - PAT auth + queue monitoring + test artifact fetch
  6. **Stretch Providers (post-Week 6)**
    - Azure DevOps → CircleCI → Buildkite → Generic adapter based on demand

  #### Testing Strategy
  - Add contract tests per provider using recorded fixtures/mocked HTTP responses (VCR.py or responses library)
  - Simulate webhook payload replay in `tests/test_providers_<name>.py`
  - Extend integration suite to validate provider sync populates dashboard + CLI outputs
  - Introduce end-to-end smoke using lightweight dockerized Jenkins/Bamboo when CI resources permit (opt-in job)

  #### Documentation Deliverables
  - Provider setup guides in `docs/providers/<name>.md`
  - Troubleshooting appendix (common HTTP errors, rate limits)
  - Security checklist and sample token rotation playbooks
  - Update `README.md` with quickstart matrix comparing provider capabilities

### UI Testing Strategy

- Current coverage relies on unit-level template tests (`tests/test_dashboard.py`) to validate key contexts render without errors.
- Next milestone is adding a lightweight browser automation flow (Playwright preferred for HTMX support) that loads the dashboard, waits for HTMX swaps, and asserts key panels render.
- CI service should expose a headless-friendly configuration (disable auto-open, bind to 127.0.0.1) so the UI suite can spin up the dashboard via CLI before running checks.
- Capture regression fixtures for empty states (no repositories) and populated repositories (dogfooding repo) to guard against blank-page regressions reported previously.

#### Execution Plan

1. **Tooling bootstrap**
  - [x] Add Playwright (Python) to dev dependencies and record install notes in `README.md`.
  - [x] Configure pytest-playwright fixtures (`ui_tests/conftest.py`) to drive Chromium in headless mode.
  - [x] Extend devcontainer init script to optionally run `playwright install --with-deps chromium` via `FULL_AUTO_CI_INSTALL_PLAYWRIGHT`.
2. **Test harness wiring**
  - [x] Stand up a `werkzeug`-backed dashboard server fixture (`ui_tests/conftest.py`) that boots `create_app` against an isolated database.
  - [x] Implement Playwright fixtures that coordinate startup/teardown and expose the shared `page` fixture.
3. **Scenario coverage**
  - [x] Smoke test: verify dashboard shell renders header, repo list empty-state, and alerts load for HTMX placeholder.
  - [x] Populated repo test: seed SQLite with dogfood repo + mock test runs, ensure insights cards show counts and recent run table populates.
  - [ ] Regression view: confirm navigating to repository detail route loads commit list and tool tabs without console errors.
4. **CI integration**
  - [ ] Add `ui-tests` job to GitHub Actions with xvfb/headless dependencies; gate on optional matrix flag to control runtime cost.
  - [ ] Update `README.md` with instructions for running `npm run ui-test` (or `pytest --ui`) locally and via CI.
  - [ ] Document environment switches (e.g., `FULL_AUTO_CI_OPEN_BROWSER=0`) required for headless execution.

- [ ] Add MCP server
  - [x] Define MCP capabilities surface (list repositories, queue test runs, fetch latest results)
  - [x] Implement `src/mcp/server.py` shim wrapping `CIService` with async-safe adapters
  - [ ] Support secure transport (Unix domain socket in dev, optional TCP with token auth)
  - [x] Provide CLI entrypoint (`full-auto-ci mcp serve`) with graceful shutdown + logging
  - [x] Add contract tests exercising handshake, tool execution, and error propagation
  - [x] Document usage and wiring into assistants (`README` + sample client script)

### Phase 3 (Future)

- [ ] Add support for additional languages
  - [ ] JavaScript/TypeScript
  - [ ] Rust
  - [ ] Go
  - [ ] Java

- [ ] Implement notification system
  - [ ] Email notifications
  - [ ] Slack integration
  - [ ] Custom webhooks

## Technical Debt

- API module needs Flask implementation
- Error handling needs improvement
- Configuration management is basic
- Missing comprehensive logging
- No automated deployment pipeline

## Development Environment

The project now uses a VS Code Dev Container for consistent development environments:

- Docker-based development environment
- Pre-configured with all necessary dependencies
- Automatic initialization of database and config
- Standardized code formatting and linting

To use:

1. Install Docker and VS Code with Remote Containers extension
2. Open project folder in VS Code
3. Click "Reopen in Container" when prompted
4. Environment is automatically initialized

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [HTMX Documentation](https://htmx.org/docs/)
- [GitPython](https://gitpython.readthedocs.io/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/remote/containers)
