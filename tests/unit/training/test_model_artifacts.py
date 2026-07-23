import json
from pathlib import Path

from lina_core.training.artifacts import finalize_model_artifact


def test_finalizer_records_partial_evaluation_and_refreshes_checksums(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    metadata = {
        "model_name": "Lina-0.5B-SFT-v0.2",
        "model_id": "lina-0.5b-sft-v0.2",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_revision": "abc",
        "training_method": "LoRA",
        "dataset_version": "0.2",
        "dataset_records": 600,
        "dataset_fingerprint": "sha256:test",
        "split": {"train": 480, "validation": 60, "test": 60},
        "device": "cpu",
        "dtype": "float32",
        "training_steps": 10,
        "best_step": 10,
        "final_training_loss": 1.0,
        "validation_loss": 2.0,
        "perplexity": 7.0,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
        "trainable_parameters": 4,
        "total_parameters": 100,
        "trainable_percentage": 4.0,
    }
    (artifact / "training-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact / "linacore-model.toml").write_text(
        'chat_template_policy = "fallback"\n', encoding="utf-8"
    )
    report = tmp_path / "comparison.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "pass_rate": 0.1},
                    {"model_id": "lina-135m-sft-v0.1", "pass_rate": 0.05},
                    {
                        "model_id": "lina-0.5b-sft-v0.2",
                        "pass_rate": 0.5,
                        "critical_failures": 2,
                        "generation_success_rate": 1.0,
                        "average_latency_seconds": 4.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = finalize_model_artifact(artifact, report)

    assert summary.decision == "PARTIAL"
    assert summary.lina_v0_2_score == 0.5
    assert 'chat_template_policy = "required"' in (artifact / "linacore-model.toml").read_text(
        encoding="utf-8"
    )
    assert (artifact / "evaluation-summary.json").exists()
    assert (artifact / "base-vs-v0.1-vs-v0.2.json").exists()
    checksums = json.loads((artifact / "checksums.json").read_text(encoding="utf-8"))
    assert "MODEL_CARD.md" in checksums
