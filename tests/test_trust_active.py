from __future__ import annotations

import numpy as np

from surroforge.active import select_batch
from surroforge.trust import ConformalIntervals, distance_to_training_score, trust_level


def test_diversity_selects_requested_batch_size():
    candidates = np.array([[0.0], [0.5], [1.0], [2.0]])
    training = np.array([[0.0]])
    selected = select_batch(candidates, batch_size=2, training_x=training, acquisition="diversity")
    assert selected == [3, 2]


def test_distance_score_and_trust_level():
    score = distance_to_training_score(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert score == 0.0
    assert trust_level(warnings=[], distance_score=score, uncertainty=0.1) == "high"
    assert trust_level(warnings=["x is near boundary of training domain"]) == "medium"


def test_conformal_intervals_cover_residual_quantile():
    conformal = ConformalIntervals(alpha=0.2).fit(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[1.1], [1.9], [2.5]]),
    )
    lower, upper = conformal.interval(np.array([[2.0]]))
    assert lower.shape == upper.shape == (1, 1)
    assert lower[0, 0] <= 2.0 <= upper[0, 0]
