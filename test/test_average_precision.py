import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import numpy as np

from ..average_precision import get_true_pred_topk


def test_get_true_pred_topk():
    k = 3
    labels = ["a", "b", "c", "d"]
    y_pred_prob = [
        [0.1, 0.2, 0.4, 0.15],
        [0.1, 0.5, 0.4, 0.15],
        [0.5, 0.2, 0.2, 0.15],
        [0.1, 0.2, 0.4, 0.5],
    ]
    yval = [[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1]]

    pred_expected = [
        ["c", "b", "d"],
        ["b", "c", "d"],
        ["a", "b", "c"],
        ["d", "c", "b"],
    ]

    true_expected = [
        ["c", "b", "d"],
        ["b", "c"],
        ["c", "d", "b"],
        ["a", "b", "c", "d"],
    ]

    true, pred = get_true_pred_topk(
        labels, np.asarray(y_pred_prob), np.asarray(yval), k
    )

    assert pred_expected == pred
    for i in range(len(true)):
        assert sorted(true_expected[i]) == sorted(true[i])
