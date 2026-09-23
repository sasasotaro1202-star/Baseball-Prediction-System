import numpy as np
import pandas as pd

from research.hierarchical_result_model import HierarchicalNPBClassifier


def test_hierarchical_npb_result_outputs_coherent_three_way_probabilities():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(180, 8)))
    y = np.tile(np.array([0, 0, 2, 2, 1, 0, 2, 0, 2], dtype=int), 20)
    model = HierarchicalNPBClassifier(hgb_max_iter=30)
    model.fit(X, y)
    p = model.predict_proba(X.iloc[:12])
    assert p.shape == (12, 3)
    assert np.isfinite(p).all()
    assert np.all(p > 0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.array_equal(model.classes_, np.array([0, 1, 2]))


def test_hierarchical_npb_uses_only_non_draw_rows_for_stage_two():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(120, 5)))
    y = np.array(([0, 1, 2, 0, 2] * 24), dtype=int)
    model = HierarchicalNPBClassifier(hgb_max_iter=20)
    model.fit(X, y)
    assert set(model.non_draw_model.classes_) == {0, 2}
