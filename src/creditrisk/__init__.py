"""A WoE/IV credit scorecard with reason codes, calibration and a fairness audit."""

from .binning import BinnedFeature, bin_categorical, bin_numeric, is_missing, quantile_edges
from .datasets import synthetic
from .fairness import audit, group_metrics, threshold_for_parity
from .scorecard import (
    Scorecard,
    brier_score,
    calibration_table,
    fit_logistic,
    fit_scorecard,
    gini,
)

__version__ = "0.2.0"

__all__ = [
    "BinnedFeature",
    "Scorecard",
    "audit",
    "bin_categorical",
    "bin_numeric",
    "brier_score",
    "calibration_table",
    "fit_logistic",
    "fit_scorecard",
    "gini",
    "group_metrics",
    "is_missing",
    "quantile_edges",
    "synthetic",
    "threshold_for_parity",
]
