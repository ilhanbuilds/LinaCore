from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StopMatch:
    text: str
    matched: str | None
    stopped: bool


def apply_stop_sequences(text: str, stops: tuple[str, ...]) -> StopMatch:
    matches = [(text.find(stop), stop) for stop in stops if stop and stop in text]
    if not matches:
        return StopMatch(text, None, False)
    index, matched = min(matches, key=lambda item: item[0])
    return StopMatch(text[:index], matched, True)
