from dataclasses import replace

from lina_core.training.merge_validation import GenerationTrace, compare_generation_traces


def _trace() -> GenerationTrace:
    return GenerationTrace(
        (0.1, 0.5, -0.2),
        (1, 3, 4),
        (1, 3, 4, 2),
        "Doğal ve açık bir cevap.",
        "torch.float32",
        "a" * 64,
    )


def test_merge_equivalence_accepts_numerically_close_identical_generation() -> None:
    adapter = _trace()
    merged = replace(adapter, first_token_logits=(0.100001, 0.499999, -0.2))

    report = compare_generation_traces(adapter, merged, tolerance=1e-5)

    assert report.passed
    assert report.max_first_logit_delta < report.tolerance


def test_merge_equivalence_rejects_token_or_tokenizer_drift() -> None:
    adapter = _trace()
    merged = replace(adapter, final_tokens=(1, 9, 2), tokenizer_sha256="b" * 64)

    report = compare_generation_traces(adapter, merged)

    assert not report.passed
    assert not report.final_tokens_equal
    assert not report.tokenizer_equal
