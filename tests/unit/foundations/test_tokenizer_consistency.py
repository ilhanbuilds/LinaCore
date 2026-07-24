from typing import ClassVar

from lina_core.tokenizer.consistency import compare_tokenizers, tokenizer_signature


class QwenTokenizerDouble:
    chat_template = "{{ messages }}"
    vocab_size = 100
    bos_token_id = None
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = None
    special_tokens_map: ClassVar[dict[str, str]] = {
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
    model_max_length = 4096
    padding_side = "right"
    truncation_side = "right"

    def __len__(self) -> int:
        return 102

    def get_added_vocab(self) -> dict[str, int]:
        return {"<eos>": 2, "<pad>": 0}


def test_tokenizer_signature_covers_template_tokens_and_padding_policy() -> None:
    signature = tokenizer_signature(QwenTokenizerDouble())

    assert signature.length == 102
    assert signature.vocab_size == 100
    assert signature.eos_token_id == 2
    assert signature.padding_side == "right"
    assert len(signature.chat_template_sha256) == 64


def test_tokenizer_parity_reports_artifact_drift() -> None:
    source = QwenTokenizerDouble()
    artifact = QwenTokenizerDouble()
    artifact.chat_template = "different"
    artifact.padding_side = "left"

    report = compare_tokenizers(source, artifact)

    assert not report.passed
    assert report.differences == ("chat_template_sha256", "padding_side")
