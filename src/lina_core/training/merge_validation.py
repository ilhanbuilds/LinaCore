from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    first_token_logits: tuple[float, ...]
    argmax_tokens: tuple[int, ...]
    final_tokens: tuple[int, ...]
    decoded_output: str
    parameter_dtype: str
    tokenizer_sha256: str


@dataclass(frozen=True, slots=True)
class MergeEquivalenceReport:
    max_first_logit_delta: float
    argmax_tokens_equal: bool
    final_tokens_equal: bool
    decoded_output_equal: bool
    dtype_equal: bool
    tokenizer_equal: bool
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.max_first_logit_delta <= self.tolerance
            and self.argmax_tokens_equal
            and self.final_tokens_equal
            and self.decoded_output_equal
            and self.dtype_equal
            and self.tokenizer_equal
        )


def compare_generation_traces(
    adapter: GenerationTrace,
    merged: GenerationTrace,
    *,
    tolerance: float = 1e-4,
) -> MergeEquivalenceReport:
    if len(adapter.first_token_logits) != len(merged.first_token_logits):
        max_delta = float("inf")
    else:
        max_delta = max(
            (
                abs(left - right)
                for left, right in zip(
                    adapter.first_token_logits,
                    merged.first_token_logits,
                    strict=True,
                )
            ),
            default=0.0,
        )
    return MergeEquivalenceReport(
        max_delta,
        adapter.argmax_tokens == merged.argmax_tokens,
        adapter.final_tokens == merged.final_tokens,
        adapter.decoded_output == merged.decoded_output,
        adapter.parameter_dtype == merged.parameter_dtype,
        adapter.tokenizer_sha256 == merged.tokenizer_sha256,
        tolerance,
    )
