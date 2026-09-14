"""Counterexamples for overlapping GridSample fragments; no CUDA required."""
import numpy as np
import pytest
from src.training.inference_contract import FragmentLogitAccumulator


def test_overlapping_fragments_average_logits_not_last_prediction():
    fragments = [([0, 1], [[20, 0], [0, 8]]), ([0, 2], [[0, 3], [4, 0]])]
    for order in (fragments, fragments[::-1]):
        accumulator = FragmentLogitAccumulator(3)
        for index, logits in order:
            accumulator.add(np.array(index), np.array(logits, dtype=np.float32))
        assert accumulator.finish().tolist() == [0, 1, 0]
        assert accumulator.coverage.tolist() == [2, 1, 1]
        assert accumulator.summary()["fragment_count"] == 2


def test_incomplete_cloud_cannot_be_published_as_background():
    accumulator = FragmentLogitAccumulator(3)
    accumulator.add(np.array([0, 1]), np.ones((2, 2)))
    with pytest.raises(ValueError, match="omitted 1"):
        accumulator.finish()


@pytest.mark.parametrize("index,logits", [
    ([0, 0], [[1, 0], [0, 1]]), ([0, 3], [[1, 0], [0, 1]]),
    ([0.0, 1.0], [[1, 0], [0, 1]]), ([0, 1], [[1, 0]]),
    ([0], [[np.nan, 1]]), ([], []),
])
def test_malformed_fragments_rejected(index, logits):
    with pytest.raises(ValueError):
        FragmentLogitAccumulator(3).add(np.array(index), np.array(logits))


def test_class_width_cannot_change_between_fragments():
    accumulator = FragmentLogitAccumulator(2)
    accumulator.add(np.array([0]), np.ones((1, 2)))
    with pytest.raises(ValueError, match="class count changed"):
        accumulator.add(np.array([1]), np.ones((1, 3)))
