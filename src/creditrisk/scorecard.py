"""Scorecard construction, scoring, and reason codes.

A scorecard turns a logistic model into integer points, which is what actually gets
deployed in lending. The conversion is fixed by two numbers a business chooses, not by
anything the model knows:

    PDO           Points to Double the Odds. "Every 20 points halves the risk."
    base_score    the score at base_odds

    factor = PDO / ln(2)
    offset = base_score − factor × ln(base_odds)
    score  = offset + factor × ln(odds)

The reason this survives in regulated lending is that the score decomposes exactly:
each feature contributes a whole number of points, and those points sum to the score.
So "declined because: 6 months at address (−42), 3 recent enquiries (−31)" is not an
approximation of the decision — it *is* the decision, restated.

Adverse-action notices are a legal requirement in most jurisdictions. A model that
cannot produce them cannot be deployed, whatever its AUC.
"""

from __future__ import annotations

import difflib
import json
import math
import numbers
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .binning import (
    BinnedFeature,
    bin_categorical,
    bin_numeric,
    check_binary_target,
    is_missing,
)

FORMAT_VERSION = 1

# Label used when a value fits no bin: a blank field for a feature that had no missing
# values in training, or a category never seen in training with no missing bin to fall
# back on. Such a value gets WoE 0, i.e. the population-average points.
NO_BIN = "no matching bin (neutral)"


@dataclass
class ScorecardPoints:
    feature: str
    bin_label: str
    points: int


@dataclass
class Scorecard:
    """Integer points per feature bin, plus the intercept.

    `on_unknown_field`: an application key that is not a scorecard feature (often a typo
    such as ``"Income"``) raises ValueError by default; ``"ignore"`` skips it, which
    suits whole database rows with id columns.

    `on_unmatched`: a value that fits no bin - a blank field where training had no
    missing values, or an unseen category with no missing bin - gets the neutral
    points (WoE 0, the population average) by default; ``"raise"`` refuses it.
    """

    features: dict[str, BinnedFeature]
    coefficients: dict[str, float]
    intercept: float
    pdo: float = 20.0
    base_score: int = 600
    base_odds: float = 50.0
    points: dict[str, dict[str, int]] = field(default_factory=dict)
    neutral_points: dict[str, int] = field(default_factory=dict)
    on_unknown_field: str = "raise"
    on_unmatched: str = "neutral"

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def build(self) -> Scorecard:
        """Allocate points to every bin.

        **Sign.** The fitted model predicts the probability of *bad*, while a credit
        score is by universal convention the log-odds of *good* — higher is safer. So
        the logit is negated on the way into points:

            ln(odds_good) = −(intercept + Σ βᵢ·WoEᵢ)

        Getting this backwards produces a scorecard that runs the right way round
        statistically and the wrong way round commercially: the best applicants score
        lowest, and nobody notices until the portfolio does.

        The intercept is spread evenly across features so the points of any complete
        application sum to the score with no leftover term. A scorecard with a hidden
        constant cannot be explained line by line, which defeats the purpose.
        """
        if not self.features:
            raise ValueError("a scorecard needs at least one feature")
        if set(self.coefficients) != set(self.features):
            missing = sorted(set(self.features) - set(self.coefficients))
            extra = sorted(set(self.coefficients) - set(self.features))
            raise ValueError(
                "coefficients must have exactly one entry per feature; "
                f"missing {missing}, unexpected {extra}"
            )
        if not self.pdo > 0:
            raise ValueError("pdo must be positive")
        if not self.base_odds > 0:
            raise ValueError("base_odds must be positive")
        if self.on_unknown_field not in ("raise", "ignore"):
            raise ValueError("on_unknown_field must be 'raise' or 'ignore'")
        if self.on_unmatched not in ("neutral", "raise"):
            raise ValueError("on_unmatched must be 'neutral' or 'raise'")
        for name, feature in self.features.items():
            if feature.name != name:
                raise ValueError(f"feature key {name!r} holds a feature named {feature.name!r}")

        shared = (self.offset - self.factor * self.intercept) / len(self.features)

        self.points = {}
        self.neutral_points = {}
        for name, feature in self.features.items():
            beta = self.coefficients[name]
            self.points[name] = {
                label: int(round(shared - self.factor * beta * woe))
                for label, woe in feature.woe.items()
            }
            self.neutral_points[name] = int(round(shared))
        return self

    # ---- scoring -------------------------------------------------------------

    def _check(self, application: Mapping[str, Any]) -> None:
        if not self.points:
            raise RuntimeError("scorecard has no points yet; call .build() first")
        if not isinstance(application, Mapping):
            raise TypeError(
                f"an application must be a dict of feature -> value, got {type(application)}"
            )
        if self.on_unknown_field == "raise":
            unknown = [k for k in application if k not in self.features]
            if unknown:
                hints = []
                for k in unknown:
                    close = difflib.get_close_matches(str(k), list(self.features), n=1)
                    hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
                raise ValueError(
                    f"unknown field(s) {', '.join(hints)}; scorecard features are "
                    f"{sorted(self.features)}. To skip extra fields set on_unknown_field='ignore' "
                    "(CLI: --ignore-unknown)."
                )

    def contributions(self, application: Mapping[str, Any]) -> list[ScorecardPoints]:
        """Points per feature for one application. Always one entry per feature."""
        self._check(application)
        out: list[ScorecardPoints] = []
        for name, feature in self.features.items():
            value = application.get(name)
            b = feature.find_bin(value)
            if b is None:
                if self.on_unmatched == "raise":
                    what = "missing" if is_missing(value) else f"unseen value {value!r}"
                    raise ValueError(f"feature {name!r}: {what} and no bin to put it in")
                out.append(ScorecardPoints(name, NO_BIN, self.neutral_points[name]))
            else:
                out.append(ScorecardPoints(name, b.label, self.points[name][b.label]))
        return out

    def score(self, application: Mapping[str, Any]) -> int:
        return sum(c.points for c in self.contributions(application))

    def probability_from_score(self, score: float) -> float:
        """Probability of *bad* at a given score."""
        odds = math.exp((score - self.offset) / self.factor)
        return round(1.0 / (1.0 + odds), 6)

    def probability(self, application: Mapping[str, Any]) -> float:
        """Probability of *bad*, recovered from the score.

        The inverse of the points transformation, so score and probability can never
        disagree — they are the same number in two units.
        """
        return self.probability_from_score(self.score(application))

    # ---- explanation ---------------------------------------------------------

    def reason_codes(self, application: Mapping[str, Any], *, limit: int = 4) -> list[dict]:
        """Why this application scored what it did, worst first.

        Measured against each feature's **best attainable** points, not against zero
        or against the population mean. "You lost 42 points relative to the best
        possible answer on this question" is actionable; "this feature contributed
        180 points" is not.
        """
        reasons = []
        for c in self.contributions(application):
            best = max(max(self.points[c.feature].values()), c.points)
            reasons.append(
                {
                    "feature": c.feature,
                    "bin": c.bin_label,
                    "points": c.points,
                    "points_lost": best - c.points,
                }
            )
        reasons.sort(key=lambda r: -r["points_lost"])
        # A reason that cost nothing is not a reason for the decline.
        return [r for r in reasons if r["points_lost"] > 0][:limit]

    def explain(self, application: Mapping[str, Any], *, cutoff: int | None = None) -> dict:
        """Score, probability, per-feature points and reason codes as one JSON-ready dict."""
        contributions = self.contributions(application)
        score = sum(c.points for c in contributions)
        out: dict[str, Any] = {
            "score": score,
            "probability_of_default": self.probability_from_score(score),
            "points": [
                {"feature": c.feature, "bin": c.bin_label, "points": c.points}
                for c in contributions
            ],
            "reason_codes": self.reason_codes(application),
        }
        if cutoff is not None:
            out["cutoff"] = cutoff
            out["decision"] = "APPROVE" if score >= cutoff else "DECLINE"
        return out

    # ---- export --------------------------------------------------------------

    def points_table(self) -> list[dict]:
        """Every bin of every feature with its WoE and points: the card a committee signs."""
        if not self.points:
            raise RuntimeError("scorecard has no points yet; call .build() first")
        rows = []
        for name, feature in self.features.items():
            for b in feature.bins:
                rows.append(
                    {
                        "feature": name,
                        "bin": b.label,
                        "n": b.total,
                        "bad_rate": round(b.bad_rate, 6),
                        "woe": feature.woe[b.label],
                        "points": self.points[name][b.label],
                    }
                )
            if not any(b.is_missing for b in feature.bins):
                rows.append(
                    {
                        "feature": name,
                        "bin": NO_BIN,
                        "n": 0,
                        "bad_rate": None,
                        "woe": 0.0,
                        "points": self.neutral_points[name],
                    }
                )
        return rows

    def to_dict(self) -> dict:
        """Everything needed to score, as plain JSON types. Round-trips with from_dict."""
        if not self.points:
            raise RuntimeError("scorecard has no points yet; call .build() first")
        return {
            "format": "credit-risk-engine/scorecard",
            "version": FORMAT_VERSION,
            "pdo": self.pdo,
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "intercept": self.intercept,
            "coefficients": dict(self.coefficients),
            "on_unknown_field": self.on_unknown_field,
            "on_unmatched": self.on_unmatched,
            "features": [f.to_dict() for f in self.features.values()],
            "points": self.points,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Scorecard:
        if d.get("format") != "credit-risk-engine/scorecard":
            raise ValueError("not a credit-risk-engine scorecard (missing 'format' marker)")
        if d.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported scorecard version {d.get('version')!r}")
        features = {f["name"]: BinnedFeature.from_dict(f) for f in d["features"]}
        card = cls(
            features=features,
            coefficients={k: float(v) for k, v in d["coefficients"].items()},
            intercept=float(d["intercept"]),
            pdo=float(d["pdo"]),
            base_score=int(d["base_score"]),
            base_odds=float(d["base_odds"]),
            on_unknown_field=d.get("on_unknown_field", "raise"),
            on_unmatched=d.get("on_unmatched", "neutral"),
        ).build()
        if d.get("points") and d["points"] != card.points:
            raise ValueError("stored points do not match the points rebuilt from the model")
        return card

    def save(self, path: str | os.PathLike[str]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Scorecard:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def logistic(x: float) -> float:
    # Guarded against overflow at the tails, which a scorecard reaches routinely.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. The system is (features+1) square."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ValueError(
                "the fit is singular: two features are collinear or one is constant; "
                "drop it or raise l2"
            )
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            if f:
                for c in range(col, n + 1):
                    a[r][c] -= f * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def fit_logistic(
    rows: Sequence[Sequence[float]],
    target: Sequence[Any],
    *,
    l2: float = 0.001,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> tuple[list[float], float]:
    """L2-regularised logistic regression by Newton's method (IRLS).

    Minimises mean log-loss + (l2 / 2)·Σw² (the bias is not penalised). A scorecard has
    a handful of WoE features, so each Newton step is a (k+1)×(k+1) solve and the fit
    converges to the exact optimum in well under 20 steps - no learning rate to tune,
    and no under-converged coefficients. L2 keeps a bin that separates perfectly from
    taking an unbounded coefficient, which is the usual failure on small samples.

    Returns (weights, bias). Raises ValueError on ragged rows, non-finite values, a
    target that is not 0/1, or a target with only one class.
    """
    n = len(rows)
    if n != len(target):
        raise ValueError(f"rows and target must be the same length ({n} vs {len(target)})")
    if n == 0:
        raise ValueError("cannot fit on zero rows")
    y = check_binary_target(target)
    if len(set(y)) < 2:
        raise ValueError("target has only one class; a default model needs goods and bads")
    k = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != k:
            raise ValueError(f"row {i} has {len(row)} values, expected {k}")
        if not all(
            (type(v) is float or isinstance(v, numbers.Real)) and math.isfinite(v) for v in row
        ):
            raise ValueError(f"row {i} has a missing or non-finite value: {list(row)!r}")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    # WoE rows repeat heavily (a 5-bin, 3-feature card has at most ~200 distinct rows),
    # so collapse identical (row, outcome) pairs into weighted patterns first.
    counts: dict[tuple[tuple[float, ...], int], int] = {}
    for row, yi in zip(rows, y, strict=True):
        key = (tuple(float(v) for v in row) + (1.0,), yi)  # bias column last
        counts[key] = counts.get(key, 0) + 1
    patterns = [(xi, yi, c) for (xi, yi), c in counts.items()]
    beta = [0.0] * (k + 1)
    penalty = [l2] * k + [0.0]

    def objective(b: list[float]) -> float:
        total = 0.0
        for xi, yi, c in patterns:
            z = sum(w * v for w, v in zip(b, xi, strict=True))
            # log(1 + e^z) - y z, computed stably.
            total += c * ((z if z > 0 else 0.0) + math.log1p(math.exp(-abs(z))) - yi * z)
        return total / n + 0.5 * sum(p * w * w for p, w in zip(penalty, b, strict=True))

    current = objective(beta)
    for _ in range(max_iter):
        grad = [0.0] * (k + 1)
        hess = [[0.0] * (k + 1) for _ in range(k + 1)]
        for xi, yi, c in patterns:
            p = logistic(sum(w * v for w, v in zip(beta, xi, strict=True)))
            err = c * (p - yi)
            weight = c * p * (1.0 - p)
            for a in range(k + 1):
                grad[a] += err * xi[a]
                wa = weight * xi[a]
                row = hess[a]
                for col in range(a, k + 1):
                    row[col] += wa * xi[col]
        for a in range(k + 1):
            grad[a] = grad[a] / n + penalty[a] * beta[a]
            for col in range(a, k + 1):
                hess[a][col] /= n
                hess[col][a] = hess[a][col]
            hess[a][a] += penalty[a]
        step = _solve(hess, grad)
        # Damped Newton: halve the step until the objective does not get worse.
        t = 1.0
        while True:
            candidate = [b - t * s for b, s in zip(beta, step, strict=True)]
            value = objective(candidate)
            if value <= current + 1e-15 or t < 1e-6:
                break
            t /= 2
        beta, current = candidate, value
        if max(abs(t * s) for s in step) < tol:
            break

    return beta[:k], beta[k]


# --- one-call construction --------------------------------------------------------


def _looks_categorical(values: Sequence[Any]) -> bool:
    for v in values:
        if is_missing(v) or isinstance(v, bool):
            continue
        if isinstance(v, str):
            try:
                float(v)
            except ValueError:
                return True
        elif not isinstance(v, numbers.Real):
            return True
    return False


def fit_scorecard(
    columns: Mapping[str, Sequence[Any]],
    target: Sequence[Any],
    *,
    categorical: Sequence[str] | None = None,
    n_bins: int = 5,
    min_bin_fraction: float = 0.05,
    l2: float = 0.001,
    pdo: float = 20.0,
    base_score: int = 600,
    base_odds: float = 50.0,
) -> Scorecard:
    """Bin every column, fit the logistic model on WoE, and build the points.

    `columns` maps feature name to its values (a dict of lists, or ``{c: df[c] for c in
    cols}`` from pandas). Columns named in `categorical`, and any column holding text
    that is not a number, are binned by category; the rest by equal-frequency quantile.
    None/NaN/blank are missing everywhere.
    """
    if not columns:
        raise ValueError("no feature columns given")
    y = check_binary_target(list(target))
    categorical = set(categorical or ())
    unknown = categorical - set(columns)
    if unknown:
        raise ValueError(f"categorical names columns that are not features: {sorted(unknown)}")

    features: dict[str, BinnedFeature] = {}
    for name, values in columns.items():
        values = list(values)
        if len(values) != len(y):
            raise ValueError(f"column {name!r} has {len(values)} values, target has {len(y)}")
        if name in categorical or _looks_categorical(values):
            features[name] = bin_categorical(name, values, y)
        else:
            features[name] = bin_numeric(
                name, values, y, n_bins=n_bins, min_bin_fraction=min_bin_fraction
            )

    names = list(features)
    column_values = {name: list(columns[name]) for name in names}
    rows = [
        [features[name].transform(column_values[name][i]) for name in names] for i in range(len(y))
    ]
    weights, bias = fit_logistic(rows, y, l2=l2)
    return Scorecard(
        features,
        dict(zip(names, weights, strict=True)),
        bias,
        pdo=pdo,
        base_score=base_score,
        base_odds=base_odds,
    ).build()


# --- calibration ----------------------------------------------------------------


def _check_scored(probabilities: Sequence[float], outcomes: Sequence[Any]) -> list[int]:
    if len(probabilities) != len(outcomes):
        raise ValueError(
            "probabilities and outcomes must be the same length "
            f"({len(probabilities)} vs {len(outcomes)})"
        )
    for i, p in enumerate(probabilities):
        if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(p):
            raise ValueError(f"probability at row {i} is {p!r}; expected a finite number")
    return check_binary_target(outcomes, "outcomes")


def brier_score(probabilities: Sequence[float], outcomes: Sequence[Any]) -> float:
    """Mean squared error of a probability forecast. Lower is better.

    Reported alongside discrimination because they answer different questions. A model
    can rank perfectly and still be wrong about the level - and a lender prices from
    the level, not the ranking.
    """
    y = _check_scored(probabilities, outcomes)
    if not y:
        return 0.0
    if any(not 0.0 <= p <= 1.0 for p in probabilities):
        raise ValueError("brier_score needs probabilities in [0, 1]")
    return round(sum((p - t) ** 2 for p, t in zip(probabilities, y, strict=True)) / len(y), 6)


def calibration_table(
    probabilities: Sequence[float], outcomes: Sequence[Any], *, n_bins: int = 10
) -> list[dict]:
    """Predicted versus observed bad rate, by quantile of predicted risk.

    The rows are split as evenly as possible (sizes differ by at most one), so there is
    no runt group at the end.
    """
    y = _check_scored(probabilities, outcomes)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if not y:
        return []
    # Sort on the probability alone. Sorting the (probability, outcome) pairs would
    # order tied scores goods-first, so a group boundary that falls inside a tie would
    # hand the goods to the lower group and the bads to the next - a miscalibration
    # manufactured by the sort. A scorecard has few distinct scores, so ties are the
    # norm, not an edge case.
    paired = sorted(zip(probabilities, y, strict=True), key=lambda pair: pair[0])
    groups = min(n_bins, len(paired))
    size, extra = divmod(len(paired), groups)
    table = []
    start = 0
    for g in range(groups):
        end = start + size + (1 if g < extra else 0)
        chunk = paired[start:end]
        start = end
        table.append(
            {
                "n": len(chunk),
                "predicted": round(sum(p for p, _ in chunk) / len(chunk), 6),
                "observed": round(sum(t for _, t in chunk) / len(chunk), 6),
            }
        )
    return table


def gini(probabilities: Sequence[float], outcomes: Sequence[Any]) -> float:
    """Gini coefficient, = 2·AUC − 1. The discrimination measure lenders quote.

    Exact AUC from ranks (Mann-Whitney U) with tied scores given their mid-rank, so a
    tie counts as half a concordant pair. O(n log n): a 100k-row book takes well under
    a second, and the answer equals counting every bad/good pair directly.
    Returns 0.0 when only one class is present.
    """
    y = _check_scored(probabilities, outcomes)
    n_bad = sum(y)
    n_good = len(y) - n_bad
    if not n_bad or not n_good:
        return 0.0
    order = sorted(range(len(y)), key=lambda i: probabilities[i])
    rank_sum_bad = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probabilities[order[j + 1]] == probabilities[order[i]]:
            j += 1
        mid_rank = (i + j) / 2 + 1  # ranks are 1-based
        rank_sum_bad += mid_rank * sum(y[order[m]] for m in range(i, j + 1))
        i = j + 1
    auc = (rank_sum_bad - n_bad * (n_bad + 1) / 2) / (n_bad * n_good)
    return round(2 * auc - 1, 6)


__all__ = [
    "NO_BIN",
    "Scorecard",
    "ScorecardPoints",
    "brier_score",
    "calibration_table",
    "fit_logistic",
    "fit_scorecard",
    "gini",
    "logistic",
]
