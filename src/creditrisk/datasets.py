"""A synthetic loan book where the true relationships are known.

Used by the README quickstart, demo.py and the tests, so every number the README
prints can be reproduced from the package alone.

The generating model:

    income       higher income means lower risk (the real signal); 5% left blank
    age          pure noise - should earn roughly no points
    employment   "salaried" < "self-employed" < "unemployed" in risk (a weaker signal)

    P(default) = logistic(0.2 − 0.014·income + employment effect)
"""

from __future__ import annotations

import math
import random

EMPLOYMENT_EFFECT = {"salaried": 0.0, "self-employed": 0.3, "unemployed": 0.8}


def synthetic(n: int = 2000, seed: int = 12) -> dict[str, list]:
    """Return columns ``income``, ``age``, ``employment``, ``default`` (1 = bad).

    ``income`` is in thousands and is None for about 5% of rows, as a blank field would
    be in a real application file.
    """
    rng = random.Random(seed)
    columns: dict[str, list] = {"income": [], "age": [], "employment": [], "default": []}
    for _ in range(n):
        income = rng.uniform(20, 200)
        age = rng.uniform(21, 70)
        employment = rng.choices(list(EMPLOYMENT_EFFECT), [0.6, 0.3, 0.1])[0]
        z = 0.2 - 0.014 * income + EMPLOYMENT_EFFECT[employment]
        default = 1 if rng.random() < 1 / (1 + math.exp(-z)) else 0
        if rng.random() < 0.05:
            income = None
        columns["income"].append(None if income is None else round(income, 1))
        columns["age"].append(round(age))
        columns["employment"].append(employment)
        columns["default"].append(default)
    return columns
