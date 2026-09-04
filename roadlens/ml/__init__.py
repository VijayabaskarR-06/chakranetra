"""
Chakranetra — Learned Models
============================
Chakranetra's detection stack uses a pretrained YOLOv8-seg network. This
package is the part the project trains itself: four models built on a
gradient-boosted-tree learner written out in NumPy, which predict repair
cost, defect growth, repair failure and departmental budget.

They correct the rules engine rather than replacing it. `roadlens.severity`
is right that the arithmetic deciding public spending must be explainable, so
every model here starts from the rule's answer, learns only the correction,
refuses to install itself if it cannot beat the rule on held-out data, and
reports where its training data came from.

    from roadlens.ml import get_registry
    reg = get_registry(store, predictive)
    reg.cost.predict(ticket)          # rupees, with a conformal interval
    reg.degradation.forecast(ticket)  # days until the next severity band
    reg.failure.predict(ticket)       # P(this repair comes back)
    reg.budget.forecast(tickets)      # 30/60/90-day spend, with a band
"""

from .gbt import GradientBoostedTrees, RegressionTree
from .features import COST_SPEC, DEGRADATION_SPEC, FAILURE_SPEC, FeatureSpec
from .models import (
    BudgetForecast,
    CostModel,
    DegradationModel,
    RepairFailureModel,
)
from .registry import ModelRegistry, get_registry, reset_registry

__all__ = [
    "GradientBoostedTrees", "RegressionTree",
    "FeatureSpec", "COST_SPEC", "DEGRADATION_SPEC", "FAILURE_SPEC",
    "CostModel", "DegradationModel", "RepairFailureModel", "BudgetForecast",
    "ModelRegistry", "get_registry", "reset_registry",
]
