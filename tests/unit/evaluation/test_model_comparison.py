from pathlib import Path

from lina_core.evaluation.comparison import load_behavior_samples


def test_lina_eval_set_is_strict_and_complete() -> None:
    samples = load_behavior_samples(Path("data/examples/lina_eval_v0.1.jsonl"))
    assert len(samples) == 9
    assert {item.sample_id for item in samples} >= {"identity", "offline", "secret"}
    assert all(item.keywords for item in samples)
