"""Make `bible_content` importable when scripts are run directly."""

import sys
from pathlib import Path

CONTENT_DIR = Path(__file__).resolve().parent.parent
if str(CONTENT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTENT_DIR))
