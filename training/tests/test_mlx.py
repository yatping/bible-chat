"""Apple Silicon (mlx-lm) path: data conversion and adapter merge math."""

import json

import merge_mlx_adapter as merge
import prepare_mlx_data
import pytest

from bible_ft import TRAINING_DIR
from bible_ft.jsonl import read_jsonl, write_jsonl
from bible_ft.schema import to_messages


@pytest.fixture
def seed_records():
    return list(read_jsonl(TRAINING_DIR / "data" / "seed" / "direct_scripture.jsonl"))[:2]


def test_hf_weight_key_inverts_mlx_gemma4_sanitize():
    assert (
        merge.hf_weight_key("language_model.model.layers.3.self_attn.q_proj")
        == "model.language_model.layers.3.self_attn.q_proj.weight"
    )
    assert merge.hf_weight_key("language_model.model.layers.0.mlp.down_proj") == (
        "model.language_model.layers.0.mlp.down_proj.weight"
    )


def test_lora_pairs_groups_and_rejects_incomplete():
    pairs = merge.lora_pairs({"a.q.lora_a": 1, "a.q.lora_b": 2, "a.q.linear.weight": 3})
    assert pairs == {"a.q": (1, 2)}
    with pytest.raises(ValueError):
        merge.lora_pairs({"a.q.lora_a": 1})


def test_merged_weight_matches_lora_forward():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 4)).astype(np.float32)  # (out, in)
    a = rng.standard_normal((4, 2)).astype(np.float32)  # (in, r)
    b = rng.standard_normal((2, 6)).astype(np.float32)  # (r, out)
    x = rng.standard_normal((3, 4)).astype(np.float32)
    lora_out = x @ w.T + 0.5 * (x @ a) @ b  # mlx_lm LoRALinear.__call__
    merged = merge.merged_weight(w, a, b, 0.5)
    np.testing.assert_allclose(x @ merged.T, lora_out, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        merge.merged_weight(w, b.T, a.T, 1.0)


def test_prepare_mlx_data_writes_runtime_messages(tmp_path, seed_records):
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "val", "test"):
        write_jsonl(splits / f"{name}.jsonl", seed_records)
    counts = prepare_mlx_data.convert(splits, tmp_path / "mlx")
    assert counts == {"train": 2, "valid": 2, "test": 2}
    row = json.loads((tmp_path / "mlx" / "train.jsonl").read_text().splitlines()[0])
    assert row == {"messages": to_messages(seed_records[0])}
    assert row["messages"][-1]["role"] == "assistant"


def test_strip_thinking_mirrors_device_filter():
    from bible_ft.thinking import strip_thinking

    assert strip_thinking("<|channel>thought\nplan<channel|>Answer [S1].") == ("Answer [S1].", True)
    assert strip_thinking("Plain answer.") == ("Plain answer.", False)
    assert strip_thinking("Hi <|channel>thought\nnever closed") == ("Hi", True)
