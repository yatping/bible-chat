#!/usr/bin/env python3
"""Merge an mlx-lm LoRA adapter into the original Hugging Face checkpoint.

QLoRA trains the adapter against a 4-bit base, but the deployable model should be the *full-precision* base plus
the adapter delta (quantising once, at GGUF time). ``mlx_lm.fuse`` would instead write an MLX-layout, text-only
checkpoint that llama.cpp's converter does not expect, so this script edits the original safetensors directly:

    W_hf  +=  scale * (lora_b.T @ lora_a.T)            (identical to mlx_lm LoRALinear.fuse)

Every other tensor (vision/audio towers, embeddings, norms) and every config/tokenizer file is copied unchanged,
so the output directory is a drop-in replacement for ``google/gemma-4-E2B-it`` and feeds straight into
``scripts/convert_gguf.sh``.

Example:
  python training/scripts/merge_mlx_adapter.py \\
      --base google/gemma-4-E2B-it --adapter training/outputs/mlx/adapter \\
      --out training/outputs/bible-gemma-merged
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ADAPTER_FILE = "adapters.safetensors"


def hf_weight_key(mlx_module_path: str) -> str:
    """Map an mlx-lm Gemma 4 module path to the original HF tensor name.

    mlx-lm's gemma4 ``sanitize`` strips the leading ``model.`` and rewrites ``language_model.`` to
    ``language_model.model.``; this inverts that.
    """
    path = mlx_module_path
    if path.startswith("language_model.model."):
        path = "language_model." + path.removeprefix("language_model.model.")
    return f"model.{path}.weight"


def lora_pairs(adapter: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Group adapter tensors into {module_path: (lora_a, lora_b)}."""
    pairs: dict[str, dict[str, Any]] = {}
    for key, value in adapter.items():
        module, _, leaf = key.rpartition(".")
        if leaf in ("lora_a", "lora_b"):
            pairs.setdefault(module, {})[leaf] = value
    missing = [m for m, p in pairs.items() if set(p) != {"lora_a", "lora_b"}]
    if missing:
        raise ValueError(f"incomplete LoRA pairs for: {missing[:3]}")
    return {m: (p["lora_a"], p["lora_b"]) for m, p in pairs.items()}


def merged_weight(weight: Any, lora_a: Any, lora_b: Any, scale: float) -> Any:
    """Return ``weight + scale * (lora_b.T @ lora_a.T)`` in the weight's dtype (computed in float32)."""
    import numpy as np

    w = weight.astype(np.float32)
    delta = scale * (lora_b.astype(np.float32).T @ lora_a.astype(np.float32).T)
    if delta.shape != w.shape:
        raise ValueError(f"LoRA delta {delta.shape} does not match weight {w.shape}")
    return (w + delta).astype(weight.dtype)


def resolve_base(base: str) -> Path:
    path = Path(base)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base))


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path, shard_bytes: int = 2 << 30) -> dict[str, Any]:
    """Stream base tensors one at a time (keeps peak memory near one shard on a 16 GB Mac) into 2 GB shards."""
    import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = float(config["lora_parameters"]["scale"])
    pairs = {hf_weight_key(m): ab for m, ab in lora_pairs(load_file(str(adapter_dir / ADAPTER_FILE))).items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in base_dir.iterdir():
        if src.is_file() and not src.name.startswith("model") and src.suffix != ".safetensors":
            shutil.copy2(src, out_dir / src.name)

    # Each shard is written as soon as it is full, then renamed once the shard count is known.
    written: list[tuple[Path, list[str]]] = []
    shard: dict[str, Any] = {}
    size = total = 0
    merged: set[str] = set()

    def flush() -> None:
        nonlocal shard, size
        if shard:
            tmp = out_dir / f"shard-{len(written):05d}.tmp"
            save_file(shard, str(tmp), metadata={"format": "pt"})
            written.append((tmp, list(shard)))
            shard, size = {}, 0

    for src in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(src), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in pairs:
                    tensor = merged_weight(tensor, *pairs[key], scale)
                    merged.add(key)
                if size and size + tensor.nbytes > shard_bytes:
                    flush()
                shard[key] = tensor
                size += tensor.nbytes
                total += tensor.nbytes
    flush()

    weight_map: dict[str, str] = {}
    for i, (tmp, keys) in enumerate(written, 1):
        name = f"model-{i:05d}-of-{len(written):05d}.safetensors"
        tmp.rename(out_dir / name)
        weight_map.update(dict.fromkeys(keys, name))
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    unmatched = sorted(pairs.keys() - merged)
    if unmatched:
        raise SystemExit(f"{len(unmatched)} adapter modules had no matching base weight, e.g. {unmatched[:3]}")
    return {"merged_modules": len(merged), "scale": scale, "shards": len(written), "out": str(out_dir)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="google/gemma-4-E2B-it", help="HF model id or local HF checkpoint dir")
    p.add_argument("--adapter", type=Path, required=True, help="mlx-lm adapter dir (adapters.safetensors)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(merge(resolve_base(args.base), args.adapter, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
