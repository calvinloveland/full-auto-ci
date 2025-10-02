# Full Auto CI

Fully automatic Continuous Integration.
Pulls down the latest code, runs tests, and reports results.
Works in the background on every commit to provide a history of results.

## Architecture

Full Auto CI consists of two main components:

1. A service that runs as a background process or daemon, monitoring repositories for new commits
2. A CLI tool that allows users to interact with the service, view results, and configure settings

### Service Features

- Monitors Git repositories for new commits
- Pulls down the latest code automatically
- Runs configured tests and tools in isolated environments
- Stores test results and history in a database
- Provides a REST API for accessing results and configuration
- Scales via multiprocessing to handle multiple commits and tools concurrently

### CLI Features

- Configure repositories to monitor
- Manually trigger test runs
- View test results and history
- Manage authentication and settings

## Version Control Support

- Git (primary)
- Future expansion possible for other VCS

## Supported Environments

- Docker
- Nix

## Supported Languages

- Python (initial)
- Future planned languages:
  - JavaScript/TypeScript
  - Rust
  - Go
  - Java

## Supported Tools

- Pylint
- Coverage
- Future tools:
  - Testing frameworks (pytest, unittest)
  - Type checkers (mypy)
  - Formatters (black, isort)

## Result Reporting

### Dashboard

- Web-based dashboard built with:
  - Flask backend
  - HTMX for dynamic content
  - Jinja templates for rendering
  - CSS for styling
- Features:
  - Overview of all repositories
  - Detailed test results per commit
  - Historical trend analysis
  - Filtering and searching capabilities

### Notifications (Future)

- Email alerts
- Slack/Discord integration
- Webhooks for custom integrations

## Configuration

- Repository-based configuration via special file (e.g., `.fullautoci.yml`)
- Reasonable defaults for common project types
- Service-level configuration for global settings
- Project-specific overrides

## Database

- SQLite database for storing:
  - Repository information
  - Commit history
  - Test results
  - User authentication
  - Configuration settings

## REST API

### Authentication

- API key authentication
- Rate limiting for security

### Endpoints

- `GET /api/v1/repos` - List all monitored repositories
- `GET /api/v1/repos/{repo_id}` - Get repository details
- `POST /api/v1/repos` - Register a new repository
- `DELETE /api/v1/repos/{repo_id}` - Remove a repository

- `GET /api/v1/repos/{repo_id}/commits` - List commits for a repository
- `GET /api/v1/repos/{repo_id}/commits/{commit_id}` - Get details for a specific commit
- `GET /api/v1/repos/{repo_id}/commits/{commit_id}/results` - Get test results for a specific commit

- `POST /api/v1/trigger/{repo_id}` - Manually trigger a test run
- `GET /api/v1/status` - Get service status

- `GET /api/v1/users` - List users (admin only)
- `POST /api/v1/users` - Create a new user
- `GET /api/v1/users/{user_id}` - Get user details
- `PUT /api/v1/users/{user_id}` - Update user information
- `DELETE /api/v1/users/{user_id}` - Remove a user

## Security

- API keys stored hashed in the database
- Authentication required for all API endpoints
- Role-based access control:
  - Admin: Full access to all endpoints
  - User: Access to assigned repositories
  - Read-only: Can only view results

## Implementation Plan

### Phase 1: Core Infrastructure

- Service setup with Git integration
- Basic Python tool support (Pylint, Coverage)
- SQLite database setup
- Basic CLI for configuration

### Phase 2: Web Dashboard

- Flask web application
- Basic results viewing
- Authentication system

### Phase 3: Advanced Features

- Additional language support
- Notification system
- More sophisticated reporting
- Performance optimizations

## Deployment Options

- Self-hosted service
- Docker container
- Systemd service on Linux

## Future Considerations

- Distributed testing across multiple machines
- Cloud integration (AWS, GCP, Azure)
- Plugin system for custom tool integration
- Support for monorepo structures
