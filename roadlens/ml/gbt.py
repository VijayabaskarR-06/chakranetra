"""
Chakranetra — Gradient-Boosted Trees
====================================
The learner behind every model in `roadlens.ml`. Written out in NumPy rather
than pulled from scikit-learn for three reasons that are specific to this
project, not general preference:

  1. The dashboard runs in the browser with no server (see dashboard/scan.js).
     A model the city cannot execute client-side breaks that promise. A tree
     ensemble serialises to a few kilobytes of JSON and evaluates in twenty
     lines of JavaScript, so `tools/generate_ml_js.py` can emit the identical
     model and `tests/test_ml_js_parity.py` can prove it agrees to 1e-9.

  2. sklearn's ensembles pull in a large binary dependency for what is, at
     this data scale (thousands of tickets, not millions), a few hundred
     lines of array arithmetic.

  3. Public spending needs an audit trail. Every split here is inspectable,
     and `explain()` decomposes any prediction into per-feature rupee
     contributions a municipal officer can read.

The formulation is Newton boosting, the same one XGBoost uses. Each round
fits a regression tree to the gradient and hessian of the loss at the current
prediction, and the optimal weight of a leaf holding rows `I` is

    w = -sum(g_i) / (sum(h_i) + lambda)

with the gain of a candidate split measured as the drop in that objective:

    gain = 0.5 * [ GL^2/(HL+l) + GR^2/(HR+l) - G^2/(H+l) ] - gamma

Two losses are supported, and they differ only in how (g, h) are computed:

    squared    g = pred - y            h = 1
    logistic   g = sigmoid(pred) - y   h = p(1-p)

so regression (cost, degradation) and classification (repair failure) share
one code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Serialisation format version. Bumped when the on-disk shape of a model
# changes, so a stale models/*.json is rejected rather than silently
# misinterpreted by a newer predictor.
FORMAT_VERSION = 1

# Hessian floor for the logistic loss. p(1-p) collapses to zero for confident
# predictions; without a floor the leaf weight -G/(H+l) explodes on a pure
# leaf and one round of boosting undoes the whole ensemble.
_MIN_HESSIAN = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic. np.exp(1000) warns and returns inf; the
    piecewise form keeps both tails exact."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class _Node:
    """One node of a regression tree.

    Internal nodes use `feature`/`threshold`/`left`/`right`; leaves use
    `value`. The two are distinguished by `feature is None`, and the flat
    `nodes` list is what gets serialised — index-addressed so the JavaScript
    port can walk it with no object graph.
    """
    feature: int | None = None
    threshold: float = 0.0
    left: int = -1
    right: int = -1
    value: float = 0.0


class RegressionTree:
    """A single CART regression tree fit to gradients and hessians.

    Splits are found by exact greedy search: for each feature the rows are
    sorted once and prefix sums of g and h give every candidate split's gain
    in one pass. Exact rather than histogram-binned because a municipal
    ticket table is thousands of rows, where exactness is free and removes a
    source of Python/JavaScript disagreement.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5,
                 lambda_: float = 1.0, gamma: float = 0.0,
                 min_child_hessian: float = 1e-3):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.min_child_hessian = float(min_child_hessian)
        self.nodes: list[_Node] = []

    # -- fitting ------------------------------------------------------------

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> "RegressionTree":
        self.nodes = []
        idx = np.arange(X.shape[0])
        self._build(X, g, h, idx, depth=0)
        return self

    def _leaf_value(self, g: np.ndarray, h: np.ndarray, idx: np.ndarray) -> float:
        return float(-g[idx].sum() / (h[idx].sum() + self.lambda_))

    def _build(self, X, g, h, idx, depth: int) -> int:
        node_id = len(self.nodes)
        self.nodes.append(_Node())

        if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf:
            self.nodes[node_id] = _Node(value=self._leaf_value(g, h, idx))
            return node_id

        feature, threshold, gain = self._best_split(X, g, h, idx)
        if feature is None or gain <= 0.0:
            self.nodes[node_id] = _Node(value=self._leaf_value(g, h, idx))
            return node_id

        # The same comparison the predictor uses, so the rows a leaf was
        # fit on are exactly the rows that will route to it.
        mask = X[idx, feature] < threshold
        left_idx, right_idx = idx[mask], idx[~mask]

        left = self._build(X, g, h, left_idx, depth + 1)
        right = self._build(X, g, h, right_idx, depth + 1)
        self.nodes[node_id] = _Node(feature=feature, threshold=threshold,
                                    left=left, right=right)
        return node_id

    def _best_split(self, X, g, h, idx):
        """Return (feature, threshold, gain) for the best split, or (None, 0, 0).

        Gain is measured against the objective of keeping `idx` as one leaf,
        so a positive gain means splitting genuinely reduces the loss.
        """
        G, H = g[idx].sum(), h[idx].sum()
        parent = (G * G) / (H + self.lambda_)

        best_feature, best_threshold, best_gain = None, 0.0, 0.0
        n_features = X.shape[1]

        for f in range(n_features):
            values = X[idx, f]
            order = np.argsort(values, kind="mergesort")   # stable: ties keep row order
            v = values[order]
            gs = np.cumsum(g[idx][order])
            hs = np.cumsum(h[idx][order])

            # A split is only legal between two *different* values; splitting
            # inside a run of equal values is not expressible as `x < t`.
            distinct = v[:-1] < v[1:]
            # Honour min_samples_leaf on both sides.
            n = v.size
            counts = np.arange(1, n)
            legal = distinct & (counts >= self.min_samples_leaf) & \
                    ((n - counts) >= self.min_samples_leaf)
            if not legal.any():
                continue

            GL, HL = gs[:-1], hs[:-1]
            GR, HR = G - GL, H - HL
            legal &= (HL >= self.min_child_hessian) & (HR >= self.min_child_hessian)
            if not legal.any():
                continue

            gains = 0.5 * ((GL * GL) / (HL + self.lambda_)
                           + (GR * GR) / (HR + self.lambda_)
                           - parent) - self.gamma
            gains = np.where(legal, gains, -np.inf)

            k = int(np.argmax(gains))
            if gains[k] > best_gain:
                # Midpoint of the two bracketing values. Computed as
                # a + (b-a)/2 rather than (a+b)/2 so it cannot overflow or,
                # more importantly here, round to exactly `a` or `b` and put
                # a training row on the wrong side of its own threshold.
                lo, hi = float(v[k]), float(v[k + 1])
                threshold = lo + (hi - lo) / 2.0
                if threshold <= lo or threshold > hi:
                    threshold = hi
                best_feature, best_threshold, best_gain = f, threshold, float(gains[k])

        return best_feature, best_threshold, best_gain

    # -- prediction ---------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            node = 0
            while self.nodes[node].feature is not None:
                nd = self.nodes[node]
                node = nd.left if X[i, nd.feature] < nd.threshold else nd.right
            out[i] = self.nodes[node].value
        return out

    def decision_path(self, x: np.ndarray) -> list[int]:
        """Node ids visited by one row, root first. Used by `explain()`."""
        path, node = [0], 0
        while self.nodes[node].feature is not None:
            nd = self.nodes[node]
            node = nd.left if x[nd.feature] < nd.threshold else nd.right
            path.append(node)
        return path

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": [
                ([-1, 0.0, -1, -1, n.value] if n.feature is None
                 else [n.feature, n.threshold, n.left, n.right, 0.0])
                for n in self.nodes
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionTree":
        tree = cls()
        tree.nodes = [
            _Node(value=float(row[4])) if int(row[0]) < 0
            else _Node(feature=int(row[0]), threshold=float(row[1]),
                       left=int(row[2]), right=int(row[3]))
            for row in data["nodes"]
        ]
        return tree


class GradientBoostedTrees:
    """An additive ensemble of `RegressionTree`s fit by Newton boosting.

    `base_score` is the raw-space prediction before any tree runs, and it is
    what makes cold start honest elsewhere in this package: the cost model
    passes the rules-based estimate as a per-row base score, so an ensemble
    with zero trees reproduces `roadlens.severity.assess` exactly and every
    tree after that is a *correction* to it.
    """

    def __init__(self, loss: str = "squared", n_estimators: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 3,
                 min_samples_leaf: int = 5, lambda_: float = 1.0,
                 gamma: float = 0.0, subsample: float = 1.0,
                 random_state: int = 0):
        if loss not in ("squared", "logistic"):
            raise ValueError(f"Unknown loss {loss!r}; use 'squared' or 'logistic'")
        self.loss = loss
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.subsample = float(subsample)
        self.random_state = int(random_state)

        self.trees: list[RegressionTree] = []
        self.base_score: float = 0.0
        self.n_features_: int = 0
        self.train_loss_: list[float] = []

    # -- loss ---------------------------------------------------------------

    def _grad_hess(self, y: np.ndarray, pred: np.ndarray):
        if self.loss == "squared":
            return pred - y, np.ones_like(y)
        p = _sigmoid(pred)
        return p - y, np.maximum(p * (1.0 - p), _MIN_HESSIAN)

    def _loss_value(self, y: np.ndarray, pred: np.ndarray) -> float:
        if self.loss == "squared":
            return float(0.5 * np.mean((y - pred) ** 2))
        p = np.clip(_sigmoid(pred), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _initial_base_score(self, y: np.ndarray) -> float:
        if self.loss == "squared":
            return float(np.mean(y))
        # Log-odds of the base rate, clipped so an all-one/all-zero label
        # column does not start the ensemble at infinity.
        p = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        return float(math.log(p / (1.0 - p)))

    # -- fitting ------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            base_score: np.ndarray | float | None = None) -> "GradientBoostedTrees":
        """Fit the ensemble.

        `base_score` may be a per-row array (an existing estimate the model
        should correct) or a scalar; when omitted it is derived from `y`.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if self.loss == "logistic" and not np.all((y == 0) | (y == 1)):
            raise ValueError("logistic loss requires labels in {0, 1}")

        self.n_features_ = X.shape[1]
        self.trees = []
        self.train_loss_ = []

        if base_score is None:
            self.base_score = self._initial_base_score(y)
            pred = np.full(y.shape, self.base_score, dtype=np.float64)
        elif np.isscalar(base_score):
            self.base_score = float(base_score)
            pred = np.full(y.shape, self.base_score, dtype=np.float64)
        else:
            # Per-row offset: the ensemble stores no constant of its own, and
            # callers must supply the same offset at predict time.
            self.base_score = 0.0
            pred = np.asarray(base_score, dtype=np.float64).ravel().copy()
            if pred.shape != y.shape:
                raise ValueError("base_score array must have one entry per row")

        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        for _ in range(self.n_estimators):
            g, h = self._grad_hess(y, pred)

            if self.subsample < 1.0:
                take = rng.random(n) < self.subsample
                if take.sum() < 2 * self.min_samples_leaf:
                    take = np.ones(n, dtype=bool)
                # Zeroing the hessian and gradient of held-out rows keeps the
                # tree's index arithmetic on the full matrix while making the
                # excluded rows contribute nothing to any split or leaf.
                gs, hs = np.where(take, g, 0.0), np.where(take, h, 0.0)
            else:
                gs, hs = g, h

            tree = RegressionTree(max_depth=self.max_depth,
                                  min_samples_leaf=self.min_samples_leaf,
                                  lambda_=self.lambda_, gamma=self.gamma)
            tree.fit(X, gs, hs)

            step = self.learning_rate * tree.predict(X)
            # A tree that found no split contributes a constant; keep it only
            # if it actually moves the loss, otherwise the ensemble grows
            # with dead rounds.
            if np.allclose(step, 0.0):
                break

            pred = pred + step
            self.trees.append(tree)
            self.train_loss_.append(self._loss_value(y, pred))

        return self

    # -- prediction ---------------------------------------------------------

    def decision_function(self, X: np.ndarray,
                          base_score: np.ndarray | float | None = None) -> np.ndarray:
        """Raw additive score, before any link function."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.n_features_ and X.shape[1] != self.n_features_:
            raise ValueError(
                f"Model expects {self.n_features_} features, got {X.shape[1]}"
            )

        if base_score is None:
            out = np.full(X.shape[0], self.base_score, dtype=np.float64)
        elif np.isscalar(base_score):
            out = np.full(X.shape[0], float(base_score), dtype=np.float64)
        else:
            out = np.asarray(base_score, dtype=np.float64).ravel().copy()
            if out.shape[0] != X.shape[0]:
                raise ValueError("base_score array must have one entry per row")

        for tree in self.trees:
            out += self.learning_rate * tree.predict(X)
        return out

    def predict(self, X, base_score=None) -> np.ndarray:
        """Regression value, or probability under the logistic loss."""
        raw = self.decision_function(X, base_score=base_score)
        return _sigmoid(raw) if self.loss == "logistic" else raw

    # -- interpretation -----------------------------------------------------

    def feature_importance(self) -> np.ndarray:
        """How often each feature is split on, weighted by tree depth-1 usage.

        Deliberately the simple split-count measure: it needs no held-out
        data and cannot be misread as a causal claim.
        """
        counts = np.zeros(self.n_features_, dtype=np.float64)
        for tree in self.trees:
            for node in tree.nodes:
                if node.feature is not None:
                    counts[node.feature] += 1.0
        total = counts.sum()
        return counts / total if total else counts

    def explain(self, x: np.ndarray, base_score: float = 0.0) -> dict:
        """Attribute one prediction to features, in raw score units.

        Each tree's contribution is walked down its decision path and charged
        to the feature that was split on at each step, apportioned by how much
        the prediction moved at that step. The parts sum to the prediction
        minus the base score, exactly — that identity is asserted in the tests,
        because an explanation that does not add up is worse than none.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        contributions = np.zeros(self.n_features_, dtype=np.float64)

        for tree in self.trees:
            path = tree.decision_path(x)
            # Value of a node = weighted mean of the leaves under it. Walking
            # the path, the change in that value from parent to child is the
            # credit due to the parent's split feature.
            values = [self._subtree_value(tree, node) for node in path]
            for depth in range(len(path) - 1):
                feature = tree.nodes[path[depth]].feature
                contributions[feature] += self.learning_rate * (
                    values[depth + 1] - values[depth]
                )

        # Whatever the root values contribute is a constant of the ensemble,
        # not attributable to any feature; fold it into the intercept.
        root_total = sum(
            self.learning_rate * self._subtree_value(tree, 0) for tree in self.trees
        )
        return {
            "base_score": float(base_score),
            "intercept": float(root_total),
            "contributions": contributions,
            "raw_prediction": float(base_score + root_total + contributions.sum()),
        }

    def _subtree_value(self, tree: RegressionTree, node: int) -> float:
        """Mean leaf value under `node`, unweighted.

        Unweighted because the ensemble does not retain training counts once
        serialised, and the identity `explain()` guarantees (parts sum to the
        prediction) holds for any node-value definition, since the terms
        telescope along the path.
        """
        nd = tree.nodes[node]
        if nd.feature is None:
            return nd.value
        return 0.5 * (self._subtree_value(tree, nd.left)
                      + self._subtree_value(tree, nd.right))

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "base_score": self.base_score,
            "n_features": self.n_features_,
            "trees": [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GradientBoostedTrees":
        version = int(data.get("format_version", 0))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Model format v{version} cannot be read by this build "
                f"(expected v{FORMAT_VERSION}); retrain with `make train`"
            )
        model = cls(loss=data["loss"], learning_rate=float(data["learning_rate"]))
        model.base_score = float(data["base_score"])
        model.n_features_ = int(data["n_features"])
        model.trees = [RegressionTree.from_dict(t) for t in data["trees"]]
        return model
