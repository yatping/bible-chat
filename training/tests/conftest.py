import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
for p in (TRAINING_DIR, TRAINING_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
