from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_prompt_characters: int = 32_768
    max_messages: int = 128
    max_output_tokens: int = 4_096
    max_dataset_record_size: int = 1_000_000
    max_config_size: int = 1_000_000
    max_report_size: int = 5_000_000
    max_streamed_events: int = 4_098

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) <= 0 for item in fields(self)):
            raise ValueError("Resource limits must be positive")
