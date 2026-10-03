"""The command line, end to end, in a temp directory."""

from __future__ import annotations

import csv
import json

import pytest

from creditrisk.cli import main


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("cli")


@pytest.fixture(scope="module")
def loans(workdir):
    path = workdir / "loans.csv"
    assert main(["sample-data", str(path)]) == 0
    return path


@pytest.fixture(scope="module")
def card_path(workdir, loans):
    """Fitted once for the module; the tests only read it."""
    path = workdir / "card.json"
    assert (
        main(["fit", str(loans), "--target", "default", "--out", str(path), "--holdout", "0"]) == 0
    )
    return path


def test_sample_data_has_blank_cells_for_missing_income(loans):
    with open(loans, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2000
    assert any(r["income"] == "" for r in rows)


def test_fit_reports_iv_per_feature_as_json(loans, tmp_path, capsys):
    out = tmp_path / "c.json"
    assert main(["fit", str(loans), "--target", "default", "--out", str(out), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    by_name = {f["feature"]: f for f in summary["features"]}
    assert by_name["income"]["strength"] == "strong"
    assert by_name["employment"]["kind"] == "categorical"
    assert out.exists()


def test_the_csv_route_matches_the_python_route(card_path):
    """A CSV round trip (text cells, blanks) must give the same card as in memory."""
    from creditrisk import Scorecard, fit_scorecard, synthetic

    d = synthetic()
    in_memory = fit_scorecard({k: d[k] for k in ("income", "age", "employment")}, d["default"])
    assert Scorecard.load(card_path).points == in_memory.points


def test_score_json_file(card_path, tmp_path, capsys):
    app = tmp_path / "app.json"
    app.write_text(json.dumps({"income": 34, "age": 29, "employment": "self-employed"}))
    assert main(["score", str(card_path), str(app), "--cutoff", "527", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == 487
    assert result["decision"] == "DECLINE"
    assert result["reason_codes"][0]["feature"] == "income"


def test_score_a_csv_of_applications(card_path, loans, capsys):
    assert main(["score", str(card_path), str(loans), "--ignore-unknown", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 2000


def test_score_text_output(card_path, tmp_path, capsys):
    app = tmp_path / "app.json"
    app.write_text('{"income": 34, "employment": "self-employed"}')
    assert main(["score", str(card_path), str(app), "--cutoff", "527"]) == 0
    out = capsys.readouterr().out
    assert "DECLINE" in out and "reasons, worst first" in out
    assert "no matching bin (neutral)" in out  # the blank age, shown honestly


def test_a_typo_is_one_clear_line_and_exit_2(card_path, tmp_path, capsys):
    app = tmp_path / "app.json"
    app.write_text('{"Income": 34}')
    assert main(["score", str(card_path), str(app)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: application 1: unknown field(s) 'Income'")
    assert "Traceback" not in err


def test_points_table(card_path, capsys):
    assert main(["points", str(card_path), "--json"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert {r["feature"] for r in table} == {"income", "age", "employment"}


def test_audit_flags_a_zero_approval_group(tmp_path, capsys):
    path = tmp_path / "decisions.csv"
    lines = ["group,approved,default"]
    lines += ["maj,1,0"] * 80 + ["maj,0,0"] * 20 + ["min,0,0"] * 50
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = ["audit", str(path), "--group", "group", "--approved", "approved"]
    assert main([*args, "--outcome", "default", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["four_fifths_rule_flag"] is True
    assert main([*args, "--outcome", "default"]) == 0
    assert "FLAGGED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["fit", "nope.csv", "--target", "default"], "file not found"),
        (["fit", "{loans}", "--target", "nope"], "column 'nope' not in"),
        (["fit", "{loans}", "--target", "employment"], "must be 0 or 1"),
        (["score", "{loans}", "{loans}"], "not valid JSON"),
    ],
)
def test_bad_input_is_reported_not_raised(argv, message, loans, capsys):
    argv = [a.replace("{loans}", str(loans)) for a in argv]
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_impossible_values_warn_and_strict_refuses(card_path, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"income": -5, "age": 200, "employment": "astronaut"}', encoding="utf-8")
    assert main(["score", str(card_path), str(bad), "--json"]) == 0
    captured = capsys.readouterr()
    warnings = json.loads(captured.out)["warnings"]
    assert len(warnings) == 3
    assert "income=-5 is outside the training range" in captured.err
    assert "age=200" in captured.err and "'astronaut' was never seen" in captured.err
    assert main(["score", str(card_path), str(bad), "--strict"]) == 2
    assert "--strict" in capsys.readouterr().err


def test_in_range_applicant_has_no_warnings(card_path, tmp_path, capsys):
    ok = tmp_path / "ok.json"
    ok.write_text('{"income": 80, "age": 40, "employment": "salaried"}', encoding="utf-8")
    assert main(["score", str(card_path), str(ok), "--json", "--strict"]) == 0
    assert json.loads(capsys.readouterr().out)["warnings"] == []


def test_fit_reports_a_held_out_gini(loans, tmp_path, capsys):
    args = ["fit", str(loans), "--target", "default", "--json"]
    assert main(args) == 0
    s = json.loads(capsys.readouterr().out)
    assert (s["train_rows"], s["holdout_rows"]) == (1600, 400)
    assert 0.2 < s["holdout_gini"] < 0.6 and 0 < s["holdout_brier"] < 0.25
    assert main([*args, "--holdout", "0"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["holdout_rows"] == 0 and "holdout_gini" not in s
    assert main([*args, "--holdout", "1.5"]) == 2
    assert "--holdout" in capsys.readouterr().err


def test_card_without_a_stored_range_still_loads(card_path):
    from creditrisk import Scorecard

    d = json.loads(card_path.read_text(encoding="utf-8"))
    for f in d["features"].values() if isinstance(d["features"], dict) else d["features"]:
        f.pop("range", None)
    card = Scorecard.from_dict(d)
    assert card.warnings({"income": -5, "age": 200, "employment": "salaried"}) == []
    assert Scorecard.load(card_path).features["age"].observed_max == 70
