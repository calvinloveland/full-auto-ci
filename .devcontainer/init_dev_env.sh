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
import sqlite3
import os
import sys
sys.path.append('/workspaces/full_auto_ci')
from src.service import CIService
service = CIService()
print('✅ Database initialized at', os.path.expanduser('~/.fullautoci/database.sqlite'))
"

echo ""
echo "🚀 Full Auto CI development environment initialized!"
echo "To run the service:"
echo "  - Development mode: python -m src.service"
echo "  - API server: python -m src.api"
echo "  - CLI: full-auto-ci --help"
