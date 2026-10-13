#!/usr/bin/env python3
"""Run ``mlx_lm.lora`` with Gemma 4 thinking mode OFF, matching the app and the eval.

mlx-lm's ``TokenizerWrapper.apply_chat_template`` defaults ``enable_thinking`` to True for models that support it
(Hugging Face's template default is false), and ``mlx_lm.lora``'s chat dataset passes no kwargs. Trained that way,
every prompt carries ``<|think|>`` in the system turn, which the device never sends; the adapter then behaves
differently at inference time. This wrapper changes the default to False, checks the rendering of the first
training conversation, and hands all arguments to ``mlx_lm.lora``.

Example:
  python training/scripts/train_mlx.py -c training/configs/mlx_qlora.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THINK_TOKEN = "<|think|>"


def patch_thinking_default() -> None:
    from mlx_lm import tokenizer_utils

    original = tokenizer_utils.TokenizerWrapper.apply_chat_template
    if getattr(original, "_thinking_off", False):
        return

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(self, *args, **kwargs)

    apply_chat_template._thinking_off = True  # type: ignore[attr-defined]
    tokenizer_utils.TokenizerWrapper.apply_chat_template = apply_chat_template


def check_rendering(model: str, data_dir: Path) -> None:
    """Fail fast if the first training conversation would still be rendered in thinking mode."""
    from mlx_lm.utils import _download, load_tokenizer

    tokenizer = load_tokenizer(_download(model))
    first = json.loads((data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    text = tokenizer.apply_chat_template(first["messages"], tokenize=False)
    if THINK_TOKEN in text:
        raise SystemExit(f"training prompt still contains {THINK_TOKEN}; refusing to train")
    print(f"chat template check: no {THINK_TOKEN} in training prompts")


def config_value(argv: list[str], key: str) -> str | None:
    flag = "--" + key.replace("_", "-")
    if flag in argv:
        return argv[argv.index(flag) + 1]
    for config_flag in ("-c", "--config"):
        if config_flag in argv:
            import yaml

            config = yaml.safe_load(Path(argv[argv.index(config_flag) + 1]).read_text(encoding="utf-8"))
            return config.get(key)
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print("usage: train_mlx.py [mlx_lm.lora arguments...]\n")
        print(__doc__)
        return 0
    patch_thinking_default()
    model, data = config_value(argv, "model"), config_value(argv, "data")
    if model and data:
        check_rendering(model, Path(data))
    from mlx_lm import lora

    sys.argv = ["mlx_lm.lora", *argv]
    lora.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
