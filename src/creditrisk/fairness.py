"""Fairness audit.

Four measures, because they are **mathematically incompatible** and a report that
quotes one of them is hiding that fact. Except in degenerate cases, a model cannot
satisfy demographic parity and equalised odds simultaneously when base rates differ
between groups — this is a theorem, not a tuning problem.

So the honest output is a table showing where the model stands on each, and the honest
posture is to state which one the lender has chosen to optimise and why.

    demographic parity   equal approval rates across groups
                         ignores whether the groups differ in actual risk
    equal opportunity    equal true-positive rates
                         among people who would repay, equal chance of approval
    equalised odds       equal TPR *and* FPR
                         the strictest, and rarely achievable
    disparate impact     ratio of approval rates; the US "four-fifths rule" treats
                         below 0.8 as prima facie evidence of adverse impact

Note on Pakistan: the PDPA does not yet define algorithmic fairness thresholds, so the
four-fifths rule is used here as the strictest widely recognised standard rather than
as a local legal requirement. Applying a stricter standard than the law demands is the
defensible direction to err in.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .binning import check_binary_target, is_missing


@dataclass
class GroupMetrics:
    group: str
    n: int
    approved: int
    approval_rate: float
    # Among those who would actually have repaid.
    true_positive_rate: float
    false_positive_rate: float
    bad_rate: float


def _rates(approved: Sequence[int], outcomes: Sequence[int]) -> tuple[float, float, float]:
    """(TPR, FPR, bad rate). `outcomes` is 1 for bad, so a good is 0."""
    goods = [(a, y) for a, y in zip(approved, outcomes, strict=True) if y == 0]
    bads = [(a, y) for a, y in zip(approved, outcomes, strict=True) if y == 1]

    tpr = sum(a for a, _ in goods) / len(goods) if goods else 0.0
    fpr = sum(a for a, _ in bads) / len(bads) if bads else 0.0
    bad_rate = len(bads) / len(outcomes) if outcomes else 0.0
    return tpr, fpr, bad_rate


def _group_labels(groups: Sequence[Any]) -> list[str]:
    """Group labels as strings, so 1 and "1" are one group and sorting never mixes types."""
    labels = []
    for i, g in enumerate(groups):
        if is_missing(g):
            raise ValueError(
                f"group at row {i} is missing; label it explicitly (e.g. 'unknown') "
                "so it is audited rather than dropped"
            )
        labels.append(str(g))
    return labels


def group_metrics(
    groups: Sequence[Any], approved: Sequence[Any], outcomes: Sequence[Any]
) -> dict[str, GroupMetrics]:
    """Per-group approval rate, TPR, FPR and bad rate.

    `approved` is 1 for approved, 0 for declined; `outcomes` is 1 for bad (default), 0
    for good. Group labels are compared as strings.
    """
    if not (len(groups) == len(approved) == len(outcomes)):
        raise ValueError(
            "groups, approved and outcomes must be the same length "
            f"({len(groups)}, {len(approved)}, {len(outcomes)})"
        )
    labels = _group_labels(groups)
    approved = check_binary_target(approved, "approved")
    outcomes = check_binary_target(outcomes, "outcomes")

    out: dict[str, GroupMetrics] = {}
    for name in sorted(set(labels)):
        idx = [i for i, g in enumerate(labels) if g == name]
        group_approved = [approved[i] for i in idx]
        group_outcomes = [outcomes[i] for i in idx]
        tpr, fpr, bad_rate = _rates(group_approved, group_outcomes)
        out[name] = GroupMetrics(
            group=name,
            n=len(idx),
            approved=sum(group_approved),
            approval_rate=round(sum(group_approved) / len(idx), 6) if idx else 0.0,
            true_positive_rate=round(tpr, 6),
            false_positive_rate=round(fpr, 6),
            bad_rate=round(bad_rate, 6),
        )
    return out


def audit(
    groups: Sequence[Any],
    approved: Sequence[Any],
    outcomes: Sequence[Any],
    *,
    reference: str | None = None,
) -> dict:
    """Full fairness report.

    The reference group defaults to the **largest** group rather than to an
    alphabetically first one, because the comparison people actually care about is
    "relative to the majority", and picking it by name is an accident waiting to
    change when a label changes.
    """
    metrics = group_metrics(groups, approved, outcomes)
    if len(metrics) < 2:
        return {
            "groups": {k: v.__dict__ for k, v in metrics.items()},
            "note": "fewer than two groups; nothing to compare",
        }

    if reference is None:
        reference = max(metrics.values(), key=lambda m: m.n).group
    elif str(reference) not in metrics:
        raise ValueError(f"reference group {reference!r} not found; groups are {sorted(metrics)}")
    reference = str(reference)
    ref = metrics[reference]

    comparisons = {}
    for name, m in metrics.items():
        if name == reference:
            continue
        comparisons[name] = {
            "approval_rate_difference": round(m.approval_rate - ref.approval_rate, 6),
            "disparate_impact": (
                round(m.approval_rate / ref.approval_rate, 6) if ref.approval_rate else None
            ),
            "equal_opportunity_gap": round(m.true_positive_rate - ref.true_positive_rate, 6),
            "equalised_odds_gap": round(
                max(
                    abs(m.true_positive_rate - ref.true_positive_rate),
                    abs(m.false_positive_rate - ref.false_positive_rate),
                ),
                6,
            ),
            # Reported because it is the usual explanation offered for a gap, and the
            # reader is entitled to judge it rather than be told it.
            "bad_rate_difference": round(m.bad_rate - ref.bad_rate, 6),
        }

    # `is not None`, not truthiness: a group with zero approvals has a ratio of 0.0,
    # the most extreme adverse impact there is, and must not be dropped.
    ratios = [
        c["disparate_impact"] for c in comparisons.values() if c["disparate_impact"] is not None
    ]
    worst = min(ratios) if ratios else None

    return {
        "reference_group": reference,
        "groups": {k: v.__dict__ for k, v in metrics.items()},
        "comparisons": comparisons,
        "worst_disparate_impact": worst,
        # A flag, not a verdict. Four-fifths is a screening threshold that triggers
        # investigation; it is not a finding of discrimination on its own.
        "four_fifths_rule_flag": (worst is not None and worst < 0.8),
        "note": (
            "Demographic parity and equalised odds cannot both hold when base rates "
            "differ between groups. State which standard applies and why."
        ),
    }


def threshold_for_parity(
    scores: Sequence[float], groups: Sequence[Any], *, target_rate: float
) -> dict[str, float]:
    """Per-group cut-offs that equalise approval rates.

    Included so the trade-off is concrete rather than theoretical: group-specific
    thresholds achieve demographic parity exactly, and in most jurisdictions using
    them is itself unlawful, because the protected attribute enters the decision.

    A tool that computes this should say so, which is why it is said here.

    Returns, per group, the lowest score to approve (approve when ``score >= cutoff``).
    `target_rate` is the fraction to approve, in [0, 1]; 0 gives a cut-off of +inf.
    Tied scores at the cut-off are all approved, so a group's realised rate can exceed
    the target when many applicants share one score.
    """
    if isinstance(target_rate, bool) or not 0.0 <= target_rate <= 1.0:
        raise ValueError(f"target_rate must be a fraction in [0, 1], got {target_rate!r}")
    if len(scores) != len(groups):
        raise ValueError(
            f"scores and groups must be the same length ({len(scores)} vs {len(groups)})"
        )
    labels = _group_labels(groups)
    out: dict[str, float] = {}
    for name in sorted(set(labels)):
        group_scores = sorted(
            (float(s) for s, g in zip(scores, labels, strict=True) if g == name), reverse=True
        )
        # Approve the top ceil(target_rate * n) applicants.
        n_approve = math.ceil(target_rate * len(group_scores) - 1e-9)
        out[name] = math.inf if n_approve == 0 else group_scores[n_approve - 1]
    return out
