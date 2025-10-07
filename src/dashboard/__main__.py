"""Run the Full Auto CI dashboard application."""
from __future__ import annotations

import logging
from typing import Any

from . import create_app

logger = logging.getLogger(__name__)


def main() -> None:
    app = create_app()
    service = app.config["CI_SERVICE"]
    dashboard_config: dict[str, Any] = service.config.get("dashboard") or {}

    host = dashboard_config.get("host", "127.0.0.1")
    port = int(dashboard_config.get("port", 8000) or 8000)
    debug = bool(dashboard_config.get("debug", False))

    logger.info("Starting Full Auto CI dashboard on %s:%s", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
