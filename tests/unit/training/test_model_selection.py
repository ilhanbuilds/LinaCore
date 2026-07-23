from pathlib import Path

from lina_core.training.selection import load_model_candidates, rank_model_candidates


def test_model_candidate_report_selects_multilingual_qwen() -> None:
    candidates = load_model_candidates(Path("configs/training/lina-model-candidates-v0.2.toml"))

    report = rank_model_candidates(candidates)

    assert len(candidates) == 3
    assert report.recommended_model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert all(item.candidate.license == "Apache-2.0" for item in report.ranked)
    assert all(item.candidate.estimated_download_gb < 2 for item in report.ranked)
    assert report.ranked[0].candidate.multilingual_rating == 3
    assert report.to_json().startswith("{")
