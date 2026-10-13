import subprocess
import sys

import pytest

from bible_ft import TRAINING_DIR

SCRIPTS = sorted(p for p in (TRAINING_DIR / "scripts").glob("*.py") if not p.name.startswith("_"))


def test_expected_scripts_exist():
    names = {p.name for p in SCRIPTS}
    assert {
        "build_dataset.py", "synthesize_with_teacher.py", "split_dataset.py", "train_qlora.py", "evaluate.py",
        "compare_reports.py", "merge_adapter.py", "dump_chat_template.py",
    } <= names


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_help_runs_without_heavy_deps(script):
    out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout


def test_convert_gguf_help():
    script = TRAINING_DIR / "scripts" / "convert_gguf.sh"
    out = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "Q4_K_M" in out.stdout
    assert "set -euo pipefail" in script.read_text()
