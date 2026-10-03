"""Weight of Evidence binning and Information Value.

The transformation credit scoring has used for forty years, and the reason scorecards
survive in regulated lending while more accurate models do not: **every step is
explainable to a regulator, a customer and a court.**

Weight of Evidence replaces a raw feature value with the log-odds contributed by its
bin:

    WoE = ln( P(good | bin) / P(bad | bin) )

Three properties fall out of that, and together they are why the technique persists:

  monotonic    after binning, a higher WoE always means a lower risk, so the direction
               of every feature can be stated in one sentence
  linear       WoE is already log-odds, which is exactly the scale logistic regression
               works on, so the model stays a sum of terms
  robust       outliers land in the edge bin and stop mattering; missing values get
               their own bin rather than an imputed lie

Information Value scores how much a feature separates good from bad:

    IV = Σ (P(good|bin) − P(bad|bin)) × WoE

    < 0.02   useless
    0.02-0.1 weak
    0.1-0.3  medium
    0.3-0.5  strong
    > 0.5    suspicious — usually leakage, not a great feature

That last band matters more than the others. An IV above 0.5 almost always means the
feature encodes the outcome: a "collections flag" that is only ever set after default.

Missing values: ``None``, ``float('nan')`` (what pandas and numpy use) and ``pandas.NA``
are all treated as missing, both when binning and when scoring.
"""

from __future__ import annotations

import bisect
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

# Prevents a bin with zero goods or zero bads from sending the logarithm to infinity.
# 0.5 is the conventional continuity correction in credit scoring, not an arbitrary
# epsilon - it is the same adjustment used for zero cells in a contingency table.
_SMOOTHING = 0.5

MISSING_LABEL = "missing"


def is_missing(value: Any) -> bool:
    """True for None, NaN (float or numpy), pandas.NA and a blank string.

    NaN is the value pandas hands you for a blank cell. Treating it as a number sorts
    it into arbitrary places and corrupts every quantile edge, so it is normalised to
    missing everywhere a value enters this package.
    """
    kind = type(value)
    if kind is float:  # fast path: this runs once per cell
        return value != value
    if kind is int or kind is bool:
        return False
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()  # a blank CSV/JSON field
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return math.isnan(value)
    # pandas.NA / pandas.NaT, recognised without importing pandas.
    return type(value).__name__ in ("NAType", "NaTType")


def check_binary_target(target: Sequence[Any], name: str = "target") -> list[int]:
    """Return `target` as a list of 0/1 ints, or raise ValueError naming the first bad value.

    Strings such as "0" and "1" are refused on purpose: "0" is truthy in Python, so a
    CSV column read as text would otherwise silently count every row as a default.
    """
    out: list[int] = []
    for i, y in enumerate(target):
        if type(y) is int and (y == 0 or y == 1):  # fast path
            out.append(y)
            continue
        if isinstance(y, bool):
            out.append(int(y))
            continue
        if isinstance(y, numbers.Real) and not is_missing(y) and y in (0, 1):
            out.append(int(y))
            continue
        raise ValueError(
            f"{name} must contain only 0 (good) and 1 (bad); row {i} is {y!r}"
            + (" - convert text columns with int()" if isinstance(y, str) else "")
        )
    return out


def _as_number(feature: str, value: Any) -> float:
    """A non-missing value as a finite float, or a TypeError/ValueError naming the feature."""
    number: float | None = None
    if type(value) is float:
        number = value
    elif isinstance(value, numbers.Real):  # int, bool, numpy scalars
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            number = None
    if number is None:
        raise TypeError(
            f"feature {feature!r} is numeric but got {value!r}; "
            "pass a number, None/NaN for missing, or bin it with bin_categorical()"
        )
    if not math.isfinite(number):
        raise ValueError(f"feature {feature!r}: {value!r} is not a finite number")
    return number


@dataclass
class Bin:
    """One bin. Numeric bins cover ``[lower, upper)``; categorical bins hold one category."""

    label: str
    lower: float = -math.inf
    upper: float = math.inf
    goods: int = 0
    bads: int = 0
    is_missing: bool = False
    category: str | None = None

    @property
    def total(self) -> int:
        return self.goods + self.bads

    @property
    def bad_rate(self) -> float:
        return self.bads / self.total if self.total else 0.0

    def contains(self, value: Any) -> bool:
        if is_missing(value):
            return self.is_missing
        if self.is_missing:
            return False
        if self.category is not None:
            return str(value) == self.category
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return False
        return self.lower <= value < self.upper

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"label": self.label, "goods": self.goods, "bads": self.bads}
        if self.is_missing:
            d["is_missing"] = True
        elif self.category is not None:
            d["category"] = self.category
        else:
            # JSON has no infinity; None stands for an open end.
            d["lower"] = None if math.isinf(self.lower) else self.lower
            d["upper"] = None if math.isinf(self.upper) else self.upper
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Bin:
        lower = d.get("lower")
        upper = d.get("upper")
        return cls(
            label=d["label"],
            lower=-math.inf if lower is None else float(lower),
            upper=math.inf if upper is None else float(upper),
            goods=int(d.get("goods", 0)),
            bads=int(d.get("bads", 0)),
            is_missing=bool(d.get("is_missing", False)),
            category=d.get("category"),
        )


@dataclass
class BinnedFeature:
    name: str
    bins: list[Bin] = field(default_factory=list)
    woe: dict[str, float] = field(default_factory=dict)
    iv: float = 0.0
    kind: str = "numeric"  # or "categorical"
    # Smallest and largest value seen in training (numeric only; None on old cards).
    observed_min: float | None = None
    observed_max: float | None = None

    def range_warning(self, value: Any) -> str | None:
        """Why `value` is outside what this feature was trained on, or None.

        The bins are open-ended, so age 200 or income -5 still land in an end bin and
        get a score. That score is an extrapolation nobody validated; this says so.
        """
        if is_missing(value):
            return None
        if self.kind == "categorical":
            if any(b.category == str(value) for b in self.bins):
                return None
            return f"{self.name}={value!r} was never seen in training; scored as missing/neutral"
        if self.observed_min is None or self.observed_max is None:
            return None
        number = _as_number(self.name, value)
        if self.observed_min <= number <= self.observed_max:
            return None
        return (
            f"{self.name}={number:g} is outside the training range "
            f"[{self.observed_min:g}, {self.observed_max:g}]; the score extrapolates"
        )

    def find_bin(self, value: Any) -> Bin | None:
        """The bin a value falls in, or None if no bin fits.

        No bin fits when the value is missing and training had no missing values, or
        when a categorical value was never seen in training.
        """
        if self.kind == "numeric" and not is_missing(value):
            value = _as_number(self.name, value)
            ranges = [b for b in self.bins if not b.is_missing]
            # Numeric bins are contiguous and sorted, so bisect on the upper bounds.
            i = bisect.bisect_right([b.upper for b in ranges], value)
            if i < len(ranges) and ranges[i].contains(value):
                return ranges[i]
            return None
        for b in self.bins:
            if b.contains(value):
                return b
        if not is_missing(value):
            # An unseen category is scored like a missing value if training had any.
            return next((b for b in self.bins if b.is_missing), None)
        return None

    def transform(self, value: Any) -> float:
        """WoE for one value. A value no bin fits gets 0.0, the neutral WoE."""
        b = self.find_bin(value)
        return self.woe[b.label] if b is not None else 0.0

    def is_monotonic(self) -> bool:
        """Does WoE move in one direction across the numeric bins?

        Checked rather than assumed. A non-monotonic feature cannot be described in
        one sentence to a credit committee, and a scorecard that cannot be described
        will not be approved. Categories have no order, so a categorical feature has
        nothing to violate and returns True.
        """
        if self.kind == "categorical":
            return True
        values = [self.woe[b.label] for b in self.bins if not b.is_missing]
        if len(values) < 2:
            return True
        increasing = all(b >= a for a, b in pairwise(values))
        decreasing = all(b <= a for a, b in pairwise(values))
        return increasing or decreasing

    def strength(self) -> str:
        for limit, label in ((0.02, "useless"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong")):
            if self.iv < limit:
                return label
        return "suspicious (check for leakage)"

    def table(self) -> list[dict]:
        """One row per bin: counts, bad rate and WoE. What a credit committee reads."""
        return [
            {
                "bin": b.label,
                "n": b.total,
                "goods": b.goods,
                "bads": b.bads,
                "bad_rate": round(b.bad_rate, 6),
                "woe": self.woe[b.label],
            }
            for b in self.bins
        ]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "iv": self.iv,
            **(
                {"range": [self.observed_min, self.observed_max]}
                if self.observed_min is not None
                else {}
            ),
            "bins": [dict(b.to_dict(), woe=self.woe[b.label]) for b in self.bins],
        }

    @classmethod
    def from_dict(cls, d: dict) -> BinnedFeature:
        bins = [Bin.from_dict(b) for b in d["bins"]]
        woe = {b["label"]: float(b["woe"]) for b in d["bins"]}
        lo, hi = d.get("range") or (None, None)
        return cls(
            name=d["name"],
            bins=bins,
            woe=woe,
            iv=float(d["iv"]),
            kind=d["kind"],
            observed_min=None if lo is None else float(lo),
            observed_max=None if hi is None else float(hi),
        )


def _woe_and_iv(bins: list[Bin]) -> tuple[dict[str, float], float]:
    total_goods = sum(b.goods for b in bins)
    total_bads = sum(b.bads for b in bins)
    if total_goods == 0 or total_bads == 0:
        # One class absent entirely: there is no evidence to weigh.
        return {b.label: 0.0 for b in bins}, 0.0

    woe: dict[str, float] = {}
    iv = 0.0
    for b in bins:
        good_share = (b.goods + _SMOOTHING) / (total_goods + _SMOOTHING * len(bins))
        bad_share = (b.bads + _SMOOTHING) / (total_bads + _SMOOTHING * len(bins))
        value = math.log(good_share / bad_share)
        woe[b.label] = round(value, 6)
        iv += (good_share - bad_share) * value
    return woe, round(iv, 6)


def quantile_edges(values: Sequence[Any], n_bins: int) -> list[float]:
    """Cut points at equal-frequency quantiles. Missing values (None/NaN) are ignored.

    Equal frequency, not equal width: equal-width bins on a skewed feature - income,
    balance, almost everything in lending - put 95% of the population in one bin and
    learn nothing.
    """
    clean = sorted(float(v) for v in values if not is_missing(v))
    if not clean or n_bins < 2:
        return []
    edges = [clean[min(len(clean) - 1, int(len(clean) * i / n_bins))] for i in range(1, n_bins)]
    # An edge at the minimum would create an empty first bin.
    return sorted({e for e in edges if e > clean[0]})


def _label(lower: float, upper: float) -> str:
    return f"[{lower:g}, {upper:g})"


def bin_numeric(
    name: str,
    values: Sequence[Any],
    target: Sequence[Any],
    *,
    n_bins: int = 5,
    edges: Sequence[float] | None = None,
    min_bin_fraction: float = 0.05,
) -> BinnedFeature:
    """Bin a numeric feature. `target` is 1 for bad (default), 0 for good.

    None and NaN go to a separate ``missing`` bin. Bins smaller than
    `min_bin_fraction` of the population are merged into a neighbour: a bin of nine
    accounts produces a WoE that will not survive contact with next quarter's data.
    """
    if len(values) != len(target):
        raise ValueError(
            f"values and target must be the same length ({len(values)} vs {len(target)})"
        )
    target = check_binary_target(target)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if not 0 <= min_bin_fraction < 1:
        raise ValueError("min_bin_fraction must be in [0, 1)")

    numbers_ = [None if is_missing(v) else _as_number(name, v) for v in values]
    if edges is not None:
        cuts = sorted({float(e) for e in edges if not is_missing(e)})
    else:
        cuts = quantile_edges([v for v in numbers_ if v is not None], n_bins)

    bounds = [-math.inf, *cuts, math.inf]
    bins = [Bin(label=_label(lo, hi), lower=lo, upper=hi) for lo, hi in pairwise(bounds)]
    missing_bin = Bin(label=MISSING_LABEL, is_missing=True)

    for value, y in zip(numbers_, target, strict=True):
        # bisect_right on the cuts gives the index of the [lower, upper) bin.
        target_bin = missing_bin if value is None else bins[bisect.bisect_right(cuts, value)]
        if y:
            target_bin.bads += 1
        else:
            target_bin.goods += 1

    # Merge undersized bins. Each small bin joins the next one along; a small bin at the
    # end joins the previous one. Repeat until every bin clears the floor or one is left.
    floor = len(values) * min_bin_fraction
    merged = list(bins)
    while len(merged) > 1:
        small = next((i for i, b in enumerate(merged) if b.total < floor), None)
        if small is None:
            break
        j = small + 1 if small + 1 < len(merged) else small - 1
        a, b = sorted((small, j))
        left, right = merged[a], merged[b]
        left.upper = right.upper
        left.goods += right.goods
        left.bads += right.bads
        left.label = _label(left.lower, left.upper)
        del merged[b]

    if missing_bin.total:
        merged.append(missing_bin)

    woe, iv = _woe_and_iv(merged)
    seen = [v for v in numbers_ if v is not None]
    return BinnedFeature(
        name=name,
        bins=merged,
        woe=woe,
        iv=iv,
        kind="numeric",
        observed_min=min(seen) if seen else None,
        observed_max=max(seen) if seen else None,
    )


def bin_categorical(name: str, values: Sequence[Any], target: Sequence[Any]) -> BinnedFeature:
    """One bin per category, plus a bin for missing (None/NaN).

    At scoring time a category never seen in training goes to the missing bin if there
    is one; otherwise it gets the neutral WoE of 0.
    """
    if len(values) != len(target):
        raise ValueError(
            f"values and target must be the same length ({len(values)} vs {len(target)})"
        )
    target = check_binary_target(target)

    by_label: dict[str, Bin] = {}
    for value, y in zip(values, target, strict=True):
        if is_missing(value):
            b = by_label.setdefault(MISSING_LABEL, Bin(label=MISSING_LABEL, is_missing=True))
        else:
            category = str(value)
            if category == MISSING_LABEL:
                raise ValueError(f"feature {name!r}: the category name 'missing' is reserved")
            b = by_label.setdefault(category, Bin(label=category, category=category))
        if y:
            b.bads += 1
        else:
            b.goods += 1

    bins = sorted(by_label.values(), key=lambda b: (b.is_missing, b.label))
    woe, iv = _woe_and_iv(bins)
    return BinnedFeature(name=name, bins=bins, woe=woe, iv=iv, kind="categorical")
