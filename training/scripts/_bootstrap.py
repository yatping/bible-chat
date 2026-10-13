"""Make ``bible_ft`` importable when running ``python training/scripts/<name>.py``."""

import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
