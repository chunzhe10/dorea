"""Shared fixtures for dorea pipeline tests."""

import sys
from pathlib import Path

# Add scripts/ to sys.path so tests can import pipeline modules
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
