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

Check the status:

```bash
full-auto-ci service status
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
