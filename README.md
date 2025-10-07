# Full Auto CI

Fully automatic Continuous Integration.
Pulls down the latest code, runs tests, and reports results.
Works in the background on every commit to provide a history of results.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Install from source

```bash
git clone https://github.com/yourusername/full_auto_ci.git
cd full_auto_ci
pip install -e .
```

For API and dashboard features:

```bash
pip install -e ".[api,dashboard]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Configuration

The system uses a YAML configuration file. By default, it looks for a config file at `~/.fullautoci/config.yml`.

You can copy and modify the example configuration:

```bash
mkdir -p ~/.fullautoci
cp config.example.yml ~/.fullautoci/config.yml
```

## Usage

### CLI

Start the service:

```bash
full-auto-ci service start
```

The command launches the service in a background process, prints the dashboard URL (defaults to `http://127.0.0.1:8000` unless overridden via `dashboard.host`/`dashboard.port` in `~/.fullautoci/config.yml`), and—when possible—opens it in your default browser. A PID file is stored under the Full Auto CI data directory (`service.pid`) so you can inspect or stop the process later. Disable the auto-open behavior by setting `dashboard.auto_open: false` in your config or by exporting `FULL_AUTO_CI_OPEN_BROWSER=0`.

Check the status:

```bash
full-auto-ci service status
```

Stop the service:

```bash
full-auto-ci service stop
```

Add a repository:

```bash
full-auto-ci repo add "My Project" https://github.com/username/project.git
```

List repositories:

```bash
full-auto-ci repo list
```

Run tests manually:

```bash
full-auto-ci test run <repo_id> <commit_hash>
```

### REST API

Start the API server (after installing API dependencies):

```bash
python -m src.api
```

The API will be available at `http://localhost:5000`.

See the [design document](design.md) for API endpoints.

## Development

### Using VS Code Dev Container (Recommended)

This project includes a VS Code Dev Container configuration for easy setup:

1. Install [Docker](https://www.docker.com/products/docker-desktop) and [VS Code](https://code.visualstudio.com/)
2. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VS Code extension
3. Open the project folder in VS Code
4. When prompted, click "Reopen in Container" (or use the command palette: "Remote-Containers: Reopen in Container")
5. The container will build and your development environment will be ready to use

### Manual Setup

If you prefer not to use containers:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[api,dashboard,dev]"
```

### Running the dashboard

The Flask-powered dashboard provides a quick status view of repositories and recent test runs:

```bash
python -m src.dashboard
```

By default the dashboard listens on `http://127.0.0.1:8000`. Configure host/port in `~/.fullautoci/config.yml` under the `dashboard` section.

### Dogfooding the project

The dev container initialization script automatically registers this repository so the service can test itself. To customise:

- `FULL_AUTO_CI_DOGFOOD=0` — skip registration
- `FULL_AUTO_CI_REPO_URL` — override repository URL
- `FULL_AUTO_CI_REPO_BRANCH` — override branch (default `main`)
- `FULL_AUTO_CI_REPO_NAME` — change display name

You can also enable always-on dogfooding directly from configuration by setting the `dogfood` section either in `~/.fullautoci/config.yml` or via the dev container config:

```yaml
dogfood:
	enabled: true
	name: "Full Auto CI"
	url: "https://github.com/calvinloveland/full-auto-ci.git"
	branch: "main"
	queue_on_start: true  # queue the latest commit immediately
```

When enabled, the service will ensure the repository is registered on startup and automatically queue the latest commit for testing unless `queue_on_start` is set to `false` (or `FULL_AUTO_CI_DOGFOOD_QUEUE=0`).

Set these variables before starting the container or rerun `.devcontainer/init_dev_env.sh` after adjusting them.

### Run tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src tests/
```

### Code style

This project follows PEP 8 style guide. You can check your code with:

```bash
pylint src/ tests/
```

## License

MIT License
