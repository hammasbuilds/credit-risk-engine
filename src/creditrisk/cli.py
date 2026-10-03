"""Command line: fit a scorecard from a CSV, score applications, audit decisions.

    creditrisk sample-data loans.csv
    creditrisk fit loans.csv --target default --out card.json
    creditrisk points card.json
    creditrisk score card.json applicant.json --cutoff 600
    creditrisk audit decisions.csv --group gender --approved approved --outcome default

Every command takes --json for machine-readable output. Errors in the input are
reported as one line on stderr with exit status 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .datasets import synthetic
from .fairness import audit
from .scorecard import Scorecard, brier_score, fit_scorecard, gini


class InputError(Exception):
    pass


def _read_csv(path: str) -> list[dict[str, str]]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        raise InputError(f"file not found: {path}") from None
    except UnicodeDecodeError:
        raise InputError(f"{path} is not UTF-8 text; re-save it as UTF-8 CSV") from None
    if not rows:
        raise InputError(f"{path} has no data rows")
    return rows


def _column(rows: list[dict[str, str]], name: str, path: str) -> list[Any]:
    if name not in rows[0]:
        raise InputError(f"column {name!r} not in {path}; columns are {list(rows[0])}")
    return [None if (r[name] is None or not r[name].strip()) else r[name] for r in rows]


def _binary(values: list[Any], name: str) -> list[int]:
    out = []
    for i, v in enumerate(values, start=2):  # row 1 is the header
        try:
            f = float(v) if v is not None else math.nan
        except ValueError:
            f = math.nan
        if f not in (0.0, 1.0):
            raise InputError(f"column {name!r} must be 0 or 1; line {i} is {v!r}")
        out.append(int(f))
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and math.isinf(o):
        return None
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _emit(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=_json_default, allow_nan=False))


# ---- commands ---------------------------------------------------------------------


def cmd_sample_data(args: argparse.Namespace) -> int:
    data = synthetic(n=args.rows, seed=args.seed)
    names = list(data)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in zip(*(data[n] for n in names), strict=True):
            writer.writerow(["" if v is None else v for v in row])
    print(f"wrote {args.rows} rows to {args.out} (columns: {', '.join(names)})")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    rows = _read_csv(args.data)
    target = _binary(_column(rows, args.target, args.data), args.target)
    drop = set(_split(args.drop)) | {args.target}
    wanted = _split(args.features) or [c for c in rows[0] if c not in drop]
    if not wanted:
        raise InputError("no feature columns left after --drop")
    columns = {name: _column(rows, name, args.data) for name in wanted}
    if not 0 <= args.holdout < 1:
        raise InputError(f"--holdout must be in [0, 1), got {args.holdout}")
    # Stratified split: the same bad rate in both parts, so a small book does not end
    # up with a holdout that has no defaults to rank.
    rng = random.Random(args.seed)
    test_idx: set[int] = set()
    for label in (0, 1):
        idx = [i for i, t in enumerate(target) if t == label]
        rng.shuffle(idx)
        test_idx.update(idx[: round(len(idx) * args.holdout)])
    train_idx = [i for i in range(len(target)) if i not in test_idx]
    held = sorted(test_idx)
    if args.holdout and (not held or len(set(target[i] for i in held)) < 2):
        raise InputError("holdout has too few rows to contain both classes; lower --holdout")
    card = fit_scorecard(
        {n: [columns[n][i] for i in train_idx] for n in wanted},
        [target[i] for i in train_idx],
        categorical=_split(args.categorical),
        n_bins=args.n_bins,
        min_bin_fraction=args.min_bin_fraction,
        pdo=args.pdo,
        base_score=args.base_score,
        base_odds=args.base_odds,
    )

    def scored(idx: list[int]) -> tuple[list[float], list[int]]:
        return [card.probability({n: columns[n][i] for n in wanted}) for i in idx], [
            target[i] for i in idx
        ]

    train_p, train_y = scored(train_idx)
    summary = {
        "rows": len(target),
        "train_rows": len(train_idx),
        "holdout_rows": len(held),
        "bad_rate": round(sum(target) / len(target), 6),
        "training_gini": gini(train_p, train_y),
        "features": [
            {
                "feature": name,
                "kind": f.kind,
                "iv": f.iv,
                "strength": f.strength(),
                "monotonic": f.is_monotonic(),
                "coefficient": round(card.coefficients[name], 6),
                "bins": len(f.bins),
            }
            for name, f in card.features.items()
        ],
    }
    if held:
        test_p, test_y = scored(held)
        summary["holdout_gini"] = gini(test_p, test_y)
        summary["holdout_brier"] = brier_score(test_p, test_y)
        summary["holdout_mean_predicted"] = round(sum(test_p) / len(test_p), 6)
        summary["holdout_bad_rate"] = round(sum(test_y) / len(test_y), 6)
    if args.out:
        card.save(args.out)
        summary["saved_to"] = str(args.out)
    if args.json:
        _emit(summary)
        return 0
    print(
        f"fitted on {summary['train_rows']} of {summary['rows']} rows "
        f"({summary['holdout_rows']} held out), bad rate {summary['bad_rate']:.1%}"
    )
    print(f"{'feature':<16} {'kind':<12} {'IV':>7}  {'strength':<30} {'beta':>8}")
    for f in summary["features"]:
        print(
            f"{f['feature']:<16} {f['kind']:<12} {f['iv']:>7.3f}  "
            f"{f['strength']:<30} {f['coefficient']:>8.3f}"
        )
    print(f"training gini {summary['training_gini']:.3f} (in-sample)")
    if held:
        print(
            f"holdout gini  {summary['holdout_gini']:.3f}, brier {summary['holdout_brier']:.4f}, "
            f"mean P(default) {summary['holdout_mean_predicted']:.1%} vs observed "
            f"{summary['holdout_bad_rate']:.1%} ({summary['holdout_rows']} rows never seen)"
        )
    else:
        print("no holdout (--holdout 0): the gini above flatters the card")
    if args.out:
        print(f"saved scorecard to {args.out}")
    else:
        print("(not saved: pass --out card.json to keep it)")
    return 0


def cmd_points(args: argparse.Namespace) -> int:
    card = _load_card(args.card)
    table = card.points_table()
    if args.json:
        _emit(table)
        return 0
    print(f"{'feature':<16} {'bin':<28} {'n':>6} {'bad rate':>9} {'WoE':>9} {'points':>7}")
    for r in table:
        rate = "" if r["bad_rate"] is None else f"{r['bad_rate']:.1%}"
        print(
            f"{r['feature']:<16} {r['bin']:<28} {r['n']:>6} {rate:>9} "
            f"{r['woe']:>9.4f} {r['points']:>7}"
        )
    return 0


def _load_card(path: str) -> Scorecard:
    try:
        return Scorecard.load(path)
    except FileNotFoundError:
        raise InputError(f"file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise InputError(f"{path} is not valid JSON: {e}") from None


def _applications(path: str) -> list[dict]:
    if path.lower().endswith(".csv"):
        return [{k: v for k, v in r.items()} for r in _read_csv(path)]
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise InputError(f"file not found: {path}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InputError(f"{path} is not valid JSON: {e}") from None
    apps = data if isinstance(data, list) else [data]
    if not all(isinstance(a, dict) for a in apps):
        raise InputError("expected a JSON object or a list of objects (feature -> value)")
    return apps


def cmd_score(args: argparse.Namespace) -> int:
    card = _load_card(args.card)
    if args.ignore_unknown:
        card.on_unknown_field = "ignore"
    results = []
    for i, app in enumerate(_applications(args.applications)):
        try:
            r = card.explain(app, cutoff=args.cutoff)
        except (ValueError, TypeError) as e:
            raise InputError(f"application {i + 1}: {e}") from None
        if r["warnings"] and args.strict:
            raise InputError(f"application {i + 1}: {r['warnings'][0]} (--strict)")
        for w in r["warnings"]:
            print(f"warning: application {i + 1}: {w}", file=sys.stderr)
        results.append(r)
    if args.json:
        _emit(results if len(results) != 1 else results[0])
        return 0
    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"--- application {i + 1}")
        line = f"score {r['score']}   P(default) {r['probability_of_default']:.1%}"
        if "decision" in r:
            line += f"   {r['decision']} (cutoff {r['cutoff']})"
        print(line)
        for p in r["points"]:
            print(f"   {p['feature']:<16} {p['bin']:<28} {p['points']:>5} pts")
        if r["reason_codes"]:
            print("   reasons, worst first:")
            for rc in r["reason_codes"]:
                print(f"      {rc['feature']:<16} {rc['bin']:<28} -{rc['points_lost']} pts")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    rows = _read_csv(args.data)
    groups = _column(rows, args.group, args.data)
    approved = _binary(_column(rows, args.approved, args.data), args.approved)
    outcomes = _binary(_column(rows, args.outcome, args.data), args.outcome)
    groups = ["(blank)" if g is None else g for g in groups]
    report = audit(groups, approved, outcomes, reference=args.reference)
    if args.json:
        _emit(report)
        return 0
    if "comparisons" not in report:
        print(report["note"])
        return 0
    print(f"reference group: {report['reference_group']}")
    print(f"{'group':<16} {'n':>6} {'approval':>9} {'TPR':>7} {'FPR':>7} {'bad rate':>9}")
    for g in report["groups"].values():
        print(
            f"{g['group']:<16} {g['n']:>6} {g['approval_rate']:>9.1%} "
            f"{g['true_positive_rate']:>7.1%} {g['false_positive_rate']:>7.1%} "
            f"{g['bad_rate']:>9.1%}"
        )
    print()
    for name, c in report["comparisons"].items():
        di = c["disparate_impact"]
        print(
            f"{name} vs {report['reference_group']}: disparate impact "
            f"{'n/a' if di is None else f'{di:.3f}'}, "
            f"equal-opportunity gap {c['equal_opportunity_gap']:+.3f}, "
            f"equalised-odds gap {c['equalised_odds_gap']:.3f}, "
            f"bad-rate difference {c['bad_rate_difference']:+.3f}"
        )
    flag = report["four_fifths_rule_flag"]
    print()
    print(
        "four-fifths rule: "
        + (
            "FLAGGED (worst ratio {:.3f} < 0.8)".format(report["worst_disparate_impact"])
            if flag
            else "not flagged"
        )
    )
    print(report["note"])
    return 0


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="creditrisk",
        description="WoE/IV credit scorecard: fit, score with reason codes, fairness audit.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample-data", help="write the synthetic loan book to a CSV")
    s.add_argument("out")
    s.add_argument("--rows", type=int, default=2000)
    s.add_argument("--seed", type=int, default=12)
    s.set_defaults(func=cmd_sample_data)

    f = sub.add_parser("fit", help="fit a scorecard from a CSV with a 0/1 target column")
    f.add_argument("data", help="CSV with a header row; blank cells are missing")
    f.add_argument("--target", required=True, help="column that is 1 for default, 0 for good")
    f.add_argument("--features", help="comma-separated feature columns (default: all others)")
    f.add_argument("--drop", help="comma-separated columns to ignore, e.g. an id column")
    f.add_argument("--categorical", help="comma-separated columns to bin by category")
    f.add_argument("--n-bins", type=int, default=5)
    f.add_argument("--min-bin-fraction", type=float, default=0.05)
    f.add_argument("--pdo", type=float, default=20.0)
    f.add_argument("--base-score", type=int, default=600)
    f.add_argument("--base-odds", type=float, default=50.0)
    f.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="fraction held out (stratified) for an out-of-sample gini; 0 to fit on all",
    )
    f.add_argument("--seed", type=int, default=0, help="seed for the holdout split")
    f.add_argument("--out", help="write the scorecard JSON here")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_fit)

    t = sub.add_parser("points", help="print every bin's WoE and points")
    t.add_argument("card")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=cmd_points)

    c = sub.add_parser("score", help="score applications from JSON (object or list) or CSV")
    c.add_argument("card")
    c.add_argument("applications", help="a .json file, a .csv file, or - for JSON on stdin")
    c.add_argument("--cutoff", type=int, help="approve at or above this score")
    c.add_argument(
        "--ignore-unknown", action="store_true", help="skip fields the scorecard does not use"
    )
    c.add_argument(
        "--strict",
        action="store_true",
        help="refuse values outside the training range or unseen categories (default: warn)",
    )
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_score)

    a = sub.add_parser("audit", help="fairness audit of approval decisions from a CSV")
    a.add_argument("data")
    a.add_argument("--group", required=True, help="protected-attribute column")
    a.add_argument("--approved", required=True, help="column that is 1 if approved")
    a.add_argument("--outcome", required=True, help="column that is 1 if the loan defaulted")
    a.add_argument("--reference", help="reference group (default: the largest)")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_audit)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InputError, ValueError, TypeError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
