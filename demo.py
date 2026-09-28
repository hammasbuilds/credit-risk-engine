"""One applicant in, a score and its adverse-action reasons out.

    python demo.py

Fits a scorecard on the package's synthetic loan book, where the true relationships
are known (higher income means lower risk, employment type matters a little, age is
noise), then scores one applicant and explains the decline in points.
No network, no dependencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from creditrisk import calibration_table, fit_scorecard, gini, synthetic  # noqa: E402

# 527 is the score at which P(default) = 20% on this card's scale (600 at 50:1, PDO 20).
CUTOFF = 527
FEATURES = ("income", "age", "employment")

data = synthetic()
columns = {name: data[name] for name in FEATURES}
card = fit_scorecard(columns, data["default"])

applicant = {"income": 34.0, "age": 29, "employment": "self-employed"}

print("INPUT")
print(f"   applicant          {applicant}")
print(f"   cutoff             {CUTOFF}   (P(default) = 20%)")
print(
    f"   scorecard          fitted on {len(data['default'])} synthetic accounts, "
    f"PDO={card.pdo:.0f}, {card.base_score} at {card.base_odds:.0f}:1"
)
print()
print("   features")
for name, f in card.features.items():
    print(f"      {name:12} IV {f.iv:.3f}  {f.strength():10} monotonic={f.is_monotonic()}")
print()

result = card.explain(applicant, cutoff=CUTOFF)

print("OUTPUT")
print(f"   score              {result['score']}")
print(f"   P(default)         {result['probability_of_default']:.1%}")
print(f"   decision           {result['decision']}   (cutoff {CUTOFF})")
print()
print("   points breakdown")
for p in result["points"]:
    print(f"      {p['feature']:12} {p['bin']:>16}   {p['points']:+4d} pts")
print()
print("   adverse-action reason codes, worst first")
for r in result["reason_codes"]:
    print(f"      {r}")
print()
probs = [card.probability({n: data[n][i] for n in FEATURES}) for i in range(len(data["default"]))]
table = calibration_table(probs, data["default"], n_bins=5)
print(f"   model gini         {gini(probs, data['default']):.3f}   (in-sample)")
print("   calibration        predicted vs observed bad rate, by quintile of risk")
for row in table:
    print(f"      n={row['n']}   predicted {row['predicted']:.3f}   observed {row['observed']:.3f}")
