import pytest

from training.schedules.implementation import warmup_cosine


def test_schedule_boundaries() -> None:
    assert warmup_cosine(0, 2, 8) == 0.5
    assert warmup_cosine(1, 2, 8) == warmup_cosine(2, 2, 8) == 1
    assert warmup_cosine(7, 2, 8, 0.1) == pytest.approx(0.1)
    assert warmup_cosine(100, 2, 8, 0.1) == pytest.approx(0.1)
    assert warmup_cosine(0, 0, 8) == 1
