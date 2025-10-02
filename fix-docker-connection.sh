#!/bin/sh

# This script helps fix Docker connection issues for VS Code Dev Containers

echo "Checking Docker configuration for VS Code Dev Containers..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running! Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Check which socket Docker is using
if [ -S /var/run/docker.sock ]; then
    DOCKER_SOCKET="/var/run/docker.sock"
    echo "✅ Found Docker socket at $DOCKER_SOCKET"
elif [ -S /run/docker.sock ]; then
    DOCKER_SOCKET="/run/docker.sock"
    echo "✅ Found Docker socket at $DOCKER_SOCKET"
else
    echo "❌ Could not find Docker socket at standard locations"
    exit 1
fi

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Create or update settings.json
cat > .vscode/settings.json << EOF
{
    "remote.containers.dockerPath": "docker",
    "docker.host": "unix://$DOCKER_SOCKET",
    "dev.containers.dockerComposePath": "docker compose"
}
EOF

echo "✅ Created VS Code settings.json with Docker socket configuration"

# Temporarily unset DOCKER_HOST for this session
echo "For this terminal session, unset the DOCKER_HOST variable:"
echo "export DOCKER_HOST="
echo ""
echo "Add this to your ~/.bashrc or ~/.zshrc to make it permanent:"
echo "# Use system Docker socket for VS Code"
echo "export DOCKER_HOST=unix://$DOCKER_SOCKET"
echo ""
echo "Then restart VS Code and try opening the folder in a container again."

export DOCKER_HOST=unix://$DOCKER_SOCKET

echo "✅ DOCKER_HOST temporarily set to unix://$DOCKER_SOCKET for this session"
echo "🚀 You should now be able to use Dev Containers in VS Code!"
