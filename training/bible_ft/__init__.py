"""Pure-Python helpers for the Bible AI fine-tuning pipeline.

Nothing in this package imports torch/transformers; heavy dependencies live inside the
scripts' ``main`` functions so the package and its tests run with only ``pytest`` installed.
"""

from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_DIR.parent
SHARED_DIR = REPO_ROOT / "shared"
