ARG VARIANT="3.10-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install additional OS packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    sqlite3 \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user with sudo access
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user home directory and .fullautoci directory
RUN mkdir -p /home/$USERNAME/.fullautoci \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME/.fullautoci

# Install pip requirements
COPY requirements.txt /tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Setup Python tools and linters
RUN pip --disable-pip-version-check --no-cache-dir install \
    black \
    flake8 \
    isort \
    mypy \
    pylint \
    pytest \
    pytest-cov

# Create the workspace directory
WORKDIR /workspaces/full_auto_ci
