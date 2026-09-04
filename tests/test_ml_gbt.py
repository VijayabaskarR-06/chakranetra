"""The learner underneath every model in roadlens.ml.

A gradient-boosting implementation is easy to write in a way that looks right
— the loss goes down, the predictions look plausible — and is quietly wrong in
a way that only shows up as mediocre accuracy nobody can explain. These tests
pin the parts that fail silently: that a split actually separates the rows the
leaf was fit on, that serialisation is bit-exact rather than approximately
right, and that `explain()` adds up to the number it claims to explain.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roadlens.ml.gbt import GradientBoostedTrees, RegressionTree, _sigmoid


def _regression_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 3 * X[:, 0] + 2 * np.sin(2 * X[:, 1]) + X[:, 2] * X[:, 3] + rng.normal(0, 0.2, n)
    return X, y


def _binary_data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    z = 1.5 * X[:, 0] - 2 * X[:, 1] + X[:, 2] ** 2 - 1
    return X, (rng.random(n) < _sigmoid(z)).astype(float)


def test_learns_a_nonlinear_function():
    X, y = _regression_data()
    m = GradientBoostedTrees(n_estimators=300, learning_rate=0.06,
                             subsample=0.8, random_state=1).fit(X[:450], y[:450])
    pred = m.predict(X[450:])
    ss_res = float(((y[450:] - pred) ** 2).sum())
    ss_tot = float(((y[450:] - y[450:].mean()) ** 2).sum())
    assert 1 - ss_res / ss_tot > 0.9


def test_training_loss_decreases():
    X, y = _regression_data()
    m = GradientBoostedTrees(n_estimators=60, learning_rate=0.1, random_state=1).fit(X, y)
    assert m.train_loss_[-1] < m.train_loss_[0]
    # Newton boosting with a small step should not oscillate upward on the
    # training set; a rise means the leaf weights or the hessians are wrong.
    assert all(b <= a + 1e-12 for a, b in zip(m.train_loss_, m.train_loss_[1:]))


def test_classifier_ranks_and_stays_in_range():
    X, y = _binary_data()
    m = GradientBoostedTrees(loss="logistic", n_estimators=200, learning_rate=0.08,
                             random_state=2).fit(X[:450], y[:450])
    p = m.predict(X[450:])
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[y[450:] == 1].mean() > p[y[450:] == 0].mean()


def test_logistic_rejects_non_binary_labels():
    X, y = _regression_data(n=100)
    with pytest.raises(ValueError, match="labels in"):
        GradientBoostedTrees(loss="logistic", n_estimators=5).fit(X, y)


def test_split_threshold_separates_the_rows_it_was_fit_on():
    """A midpoint computed as (a+b)/2 can round to exactly `a` in float, which
    puts a training row on the wrong side of its own split — the leaf is then
    fit on rows that will never reach it. This checks the actual routing."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    g = X[:, 0] * 2.0
    h = np.ones(200)
    tree = RegressionTree(max_depth=1, min_samples_leaf=5).fit(X, g, h)

    root = tree.nodes[0]
    assert root.feature is not None, "expected a split on separable data"
    left = X[X[:, root.feature] < root.threshold]
    right = X[X[:, root.feature] >= root.threshold]
    assert len(left) >= 5 and len(right) >= 5
    assert left[:, root.feature].max() < root.threshold <= right[:, root.feature].min()


def test_never_splits_inside_a_run_of_equal_values():
    """`x < t` cannot separate two rows holding the same value, so a split
    proposed between them would silently send both the same way and leave one
    child empty."""
    X = np.array([[1.0], [1.0], [1.0], [1.0], [2.0], [2.0], [2.0], [2.0]])
    g = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
    h = np.ones(8)
    tree = RegressionTree(max_depth=2, min_samples_leaf=1).fit(X, g, h)
    for node in tree.nodes:
        if node.feature is not None:
            assert 1.0 < node.threshold <= 2.0


def test_constant_target_produces_no_splits():
    X = np.random.default_rng(4).normal(size=(100, 3))
    m = GradientBoostedTrees(n_estimators=50, random_state=4).fit(X, np.full(100, 7.0))
    assert np.allclose(m.predict(X), 7.0)


def test_json_round_trip_is_exact():
    X, y = _regression_data()
    m = GradientBoostedTrees(n_estimators=80, random_state=5).fit(X, y)
    before = m.predict(X)
    restored = GradientBoostedTrees.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.array_equal(before, restored.predict(X))


def test_stale_format_version_is_refused():
    X, y = _regression_data(n=100)
    data = GradientBoostedTrees(n_estimators=5).fit(X, y).to_dict()
    data["format_version"] = 999
    with pytest.raises(ValueError, match="format v999"):
        GradientBoostedTrees.from_dict(data)


def test_per_row_base_score_is_the_cold_start_guarantee():
    """Zero trees must reproduce the base score exactly — that is what lets
    CostModel fall back to the rules engine with no special case."""
    X, y = _regression_data(n=200)
    base = np.linspace(1.0, 5.0, 200)
    m = GradientBoostedTrees(n_estimators=0).fit(X, y, base_score=base)
    assert np.array_equal(m.predict(X, base_score=base), base)


def test_base_score_array_must_match_row_count():
    X, y = _regression_data(n=50)
    with pytest.raises(ValueError, match="one entry per row"):
        GradientBoostedTrees(n_estimators=2).fit(X, y, base_score=np.ones(49))


def test_predict_rejects_wrong_feature_count():
    X, y = _regression_data(n=100)
    m = GradientBoostedTrees(n_estimators=5).fit(X, y)
    with pytest.raises(ValueError, match="expects 5 features"):
        m.predict(np.zeros((3, 4)))


def test_explanation_sums_to_the_prediction():
    """An explanation whose parts do not add up to the prediction is worse
    than no explanation, because it will be believed."""
    X, y = _regression_data(n=400)
    m = GradientBoostedTrees(n_estimators=120, max_depth=3, random_state=6).fit(X, y)
    for i in range(25):
        ex = m.explain(X[i], base_score=m.base_score)
        assert ex["raw_prediction"] == pytest.approx(float(m.predict(X[i:i + 1])[0]),
                                                     abs=1e-9)


def test_feature_importance_is_a_distribution_over_used_features():
    X, y = _regression_data()
    m = GradientBoostedTrees(n_estimators=100, random_state=7).fit(X, y)
    imp = m.feature_importance()
    assert imp.shape == (5,)
    assert imp.sum() == pytest.approx(1.0)
    # y ignores column 4 entirely; it should be the least-used feature.
    assert imp[4] <= imp[:4].min() + 0.05


def test_sigmoid_does_not_overflow():
    z = np.array([-1e4, -50.0, 0.0, 50.0, 1e4])
    p = _sigmoid(z)
    assert np.all(np.isfinite(p))
    assert p[0] == pytest.approx(0.0) and p[-1] == pytest.approx(1.0)
    assert p[2] == pytest.approx(0.5)


def test_empty_and_malformed_input_is_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        GradientBoostedTrees().fit(np.zeros((0, 3)), np.zeros(0))
    with pytest.raises(ValueError, match="2-dimensional"):
        GradientBoostedTrees().fit(np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError, match="10 rows but y has 9"):
        GradientBoostedTrees().fit(np.zeros((10, 2)), np.zeros(9))
    with pytest.raises(ValueError, match="Unknown loss"):
        GradientBoostedTrees(loss="hinge")


def test_deterministic_across_runs():
    X, y = _regression_data()
    a = GradientBoostedTrees(n_estimators=40, subsample=0.7, random_state=11).fit(X, y)
    b = GradientBoostedTrees(n_estimators=40, subsample=0.7, random_state=11).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
