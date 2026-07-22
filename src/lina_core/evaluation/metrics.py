import re

from lina_core.evaluation.models import MetricResult


def exact_match(actual: str, expected: str) -> MetricResult:
    return MetricResult("exact_match", float(actual == expected), passed=actual == expected)


def normalized_exact_match(actual: str, expected: str) -> MetricResult:
    def norm(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    passed = norm(actual) == norm(expected)
    return MetricResult("normalized_exact_match", float(passed), passed=passed)


def contains_expected(actual: str, expected: str) -> MetricResult:
    passed = expected.casefold() in actual.casefold()
    return MetricResult("contains_expected", float(passed), passed=passed)


def response_length(actual: str) -> MetricResult:
    return MetricResult("response_length", float(len(actual)), {"characters": len(actual)})
