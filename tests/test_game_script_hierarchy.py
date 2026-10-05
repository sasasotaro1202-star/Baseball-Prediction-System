from __future__ import annotations

import numpy as np

from research.game_script_common import TransitionKernel


def test_sparse_primary_state_uses_parent_support():
    k = TransitionKernel(min_support=4)
    key = (1, "T", 0, 0, 0, 0, 0)
    primary = ("B", 1, 0, 0, 0, 0, "N")
    parent = ("B", 0, 1, 0, 0, 1, "A")
    k.views["rich"][key][primary] = 1
    k.views["no_count"][(1, "T", 0, 0, 0)][parent] = 5
    rng = np.random.default_rng(123)
    draws = [k.sample(key, rng, {"H": 1.0, "A": 1.0})[0] for _ in range(500)]
    assert primary in draws
    assert parent in draws
