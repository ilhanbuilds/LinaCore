from dataclasses import replace
from pathlib import Path

import pytest

from lina_core.core.exceptions import TrainingError
from lina_core.datasets.sft import SFTRecord
from lina_core.inference.request import ChatMessage, ChatRole
from lina_core.training.config import apply_training_profile, load_training_config
from lina_core.training.runner import TrainingRunner

CONFIG = Path("configs/training/lina-135m-sft-v0.1.toml")
V0_2_CONFIG = Path("configs/training/lina-0.5b-sft-v0.2.toml")


class _Linear:
    pass


class _TorchDouble:
    class nn:
        Linear = _Linear


class _ModelDouble:
    def named_modules(self) -> list[tuple[str, object]]:
        return [
            ("model.layers.0.self_attn.q_proj", _Linear()),
            ("model.layers.0.self_attn.v_proj", _Linear()),
            ("lm_head", _Linear()),
        ]


class _CanonicalTokenizerDouble:
    chat_template = "canonical"
    eos_token = "~"
    eos_token_id = ord("~")

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>~"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _multi_turn_record() -> SFTRecord:
    return SFTRecord(
        "multi-turn",
        (
            ChatMessage(ChatRole.USER, "first question"),
            ChatMessage(ChatRole.ASSISTANT, "first answer"),
            ChatMessage(ChatRole.USER, "second question"),
            ChatMessage(ChatRole.ASSISTANT, "second answer"),
        ),
        "diagnostic",
        "en",
        "synthetic-test",
        "CC0-1.0",
        "approved",
        {},
    )


def _copy_training_dataset(workspace: Path, dataset_path: str) -> None:
    target = workspace / dataset_path
    target.parent.mkdir(parents=True)
    target.write_text(Path(dataset_path).read_text(encoding="utf-8"), encoding="utf-8")


def test_training_config_and_profiles_are_typed() -> None:
    config = load_training_config(CONFIG)
    assert config.local_files_only and not config.allow_download
    assert not config.trust_remote_code and config.report_to == "none"
    assert apply_training_profile(config, "smoke").max_steps == 1
    quick = apply_training_profile(config, "quick")
    assert quick.max_steps == 0 and quick.num_train_epochs == 3
    assert config.fingerprint.startswith("sha256:")


def test_training_plan_does_not_start_training(tmp_path: Path) -> None:
    config = load_training_config(CONFIG)
    _copy_training_dataset(tmp_path, config.dataset_path)
    (tmp_path / "base-model").mkdir()
    config = replace(
        config,
        base_model="base-model",
        output_dir="models/unit-test-training-plan",
    )
    plan = TrainingRunner(tmp_path).plan(config)
    assert plan.record_count == 24
    assert not plan.download_required
    assert plan.method == "full-supervised-fine-tuning"


def test_training_plan_rejects_missing_local_model(tmp_path: Path) -> None:
    config = load_training_config(CONFIG)
    _copy_training_dataset(tmp_path, config.dataset_path)
    config = replace(
        config,
        base_model="missing-base-model",
        output_dir="models/unit-test-training-plan",
    )

    with pytest.raises(TrainingError, match="Base model local değil"):
        TrainingRunner(tmp_path).plan(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("trust_remote_code", True),
        ("report_to", "wandb"),
        ("dtype", "float16"),
        ("merge_adapter", True),
        ("train_split", -1.0),
    ],
)
def test_unsafe_training_config_is_rejected(field: str, value: object) -> None:
    config = load_training_config(CONFIG)
    with pytest.raises(TrainingError):
        replace(config, **{field: value})


def test_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    config = load_training_config(CONFIG)
    dataset = tmp_path / config.dataset_path
    dataset.parent.mkdir(parents=True)
    dataset.write_text(Path(config.dataset_path).read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "model"
    output.mkdir()
    (output / "existing").write_text("keep", encoding="utf-8")
    config = replace(config, output_dir=str(output.relative_to(tmp_path)))
    with pytest.raises(TrainingError, match="Output dizini dolu"):
        TrainingRunner(tmp_path).validate_output(config)


def test_lora_config_is_typed_and_requires_valid_hyperparameters() -> None:
    config = replace(
        load_training_config(CONFIG),
        use_lora=True,
        merge_adapter=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=("q_proj", "v_proj"),
    )

    assert config.use_lora and config.merge_adapter
    assert config.lora_target_modules == ("q_proj", "v_proj")
    assert config.fingerprint.startswith("sha256:")
    with pytest.raises(TrainingError, match="LoRA"):
        replace(config, lora_r=0)


def test_v0_2_quality_config_uses_grouped_lora_training() -> None:
    config = load_training_config(V0_2_CONFIG)
    quality = apply_training_profile(config, "quality")

    assert quality.use_lora and quality.merge_adapter
    assert quality.split_strategy == "grouped"
    assert quality.max_steps == 720
    assert quality.max_sequence_length == 256
    assert quality.base_model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert quality.model_context_length == 32768


def test_lora_targets_are_discovered_from_model_architecture() -> None:
    targets = TrainingRunner._discover_lora_targets(_ModelDouble(), _TorchDouble())

    assert targets == ("q_proj", "v_proj")


def test_training_tokenization_labels_every_assistant_turn_and_one_eos() -> None:
    tokenizer = _CanonicalTokenizerDouble()
    result = TrainingRunner._training_tokenization(
        tokenizer,
        _multi_turn_record(),
        512,
        fallback_enabled=False,
    )

    assert len(result.assistant_spans) == 2
    assert not result.terminal_eos_added
    assert not result.truncated
    labeled = [
        "".join(chr(token) for token in result.token_ids[start:end])
        for start, end in result.assistant_spans
    ]
    assert labeled == ["first answer</assistant>~", "second answer</assistant>~"]
    assert result.token_ids[-2:] != (tokenizer.eos_token_id, tokenizer.eos_token_id)


def test_training_tokenization_masks_assistant_prefix_and_reports_truncation() -> None:
    tokenizer = _CanonicalTokenizerDouble()
    complete = TrainingRunner._training_tokenization(
        tokenizer,
        _multi_turn_record(),
        512,
        fallback_enabled=False,
    )
    first_start, _ = complete.assistant_spans[0]

    assert complete.token_ids[first_start] == ord("f")

    truncated = TrainingRunner._training_tokenization(
        tokenizer,
        _multi_turn_record(),
        first_start + 3,
        fallback_enabled=False,
    )
    assert truncated.truncated
    assert truncated.assistant_spans == ((first_start, first_start + 3),)
