#!/bin/bash

# Initialize the Full Auto CI development environment
# This script sets up the necessary directories and configuration

# Ensure the Full Auto CI directory exists
mkdir -p ~/.fullautoci

# Copy the config file if it doesn't exist
if [ ! -f ~/.fullautoci/config.yml ]; then
    cp /workspaces/full_auto_ci/.devcontainer/config.yml ~/.fullautoci/config.yml
    echo "✅ Configuration file created at ~/.fullautoci/config.yml"
else
    echo "ℹ️ Configuration file already exists at ~/.fullautoci/config.yml"
fi

# Create necessary subdirectories
mkdir -p ~/.fullautoci/repositories
mkdir -p ~/.fullautoci/backups

# Initialize the database
python -c "
import os
import sys
sys.path.append('/workspaces/full_auto_ci')
from src.service import CIService

service = CIService()
db_path = os.path.expanduser('~/.fullautoci/database.sqlite')

print('✅ Database initialized at', db_path)

dogfood_enabled = os.getenv('FULL_AUTO_CI_DOGFOOD', '1').lower() not in {'0', 'false', 'no'}
if dogfood_enabled:
    repo_url = os.getenv('FULL_AUTO_CI_REPO_URL', 'https://github.com/calvinloveland/full-auto-ci.git')
    repo_name = os.getenv('FULL_AUTO_CI_REPO_NAME', 'Full Auto CI')
    repo_branch = os.getenv('FULL_AUTO_CI_REPO_BRANCH', 'main')

    existing = [repo for repo in service.list_repositories() if repo['url'] == repo_url]
    if existing:
        repo_id = existing[0]['id']
        print(f'ℹ️ Dogfooding repository already registered (ID {repo_id})')
    else:
        repo_id = service.add_repository(repo_name, repo_url, repo_branch)
        if repo_id:
            print(f'✅ Dogfooding repository registered (ID {repo_id})')
        else:
            print('⚠️ Failed to register dogfooding repository. Check logs for details.')
else:
    print('ℹ️ Skipping dogfooding repository registration (FULL_AUTO_CI_DOGFOOD disabled)')
"

INSTALL_PLAYWRIGHT=${FULL_AUTO_CI_INSTALL_PLAYWRIGHT:-0}
if [ "$INSTALL_PLAYWRIGHT" != "0" ]; then
    if command -v playwright >/dev/null 2>&1; then
        echo "\n🎭 Installing Playwright Chromium runtime (FULL_AUTO_CI_INSTALL_PLAYWRIGHT=${INSTALL_PLAYWRIGHT})"
        playwright install --with-deps chromium || echo "⚠️ Playwright browser installation failed"
    else
        echo "⚠️ Playwright CLI not found; skip browser installation"
    fi
else
    echo "ℹ️ Skipping Playwright browser installation (FULL_AUTO_CI_INSTALL_PLAYWRIGHT=${INSTALL_PLAYWRIGHT})"
fi

echo ""
echo "🚀 Full Auto CI development environment initialized!"
echo "To run the service:"
echo "  - Development mode: python -m src.service"
echo "  - API server: python -m src.api"
echo "  - CLI: full-auto-ci --help"
