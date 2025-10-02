#!/usr/bin/env python3
"""Main entry point for Full Auto CI."""
import sys
import os
import logging

from src.cli import CLI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    cli = CLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
