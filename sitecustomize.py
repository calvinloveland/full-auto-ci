"""Project-wide Python startup customizations."""
from __future__ import annotations

import os

# Prevent globally installed pytest plugins (like pytest-cov) from auto-loading
# when running this project's test suite. This avoids compatibility issues with
# plugins outside of our dependency control. Users can override by exporting
# PYTEST_DISABLE_PLUGIN_AUTOLOAD=0 before invoking pytest.
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
