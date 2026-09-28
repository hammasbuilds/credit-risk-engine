"""Credit risk engine tests.

Scorecard behaviour is almost entirely exact — WoE is a logarithm, points are a linear
transform, the four-fifths rule is a ratio — so nearly everything here is asserted
rather than compared against a tolerance.
"""

from __future__ import annotations

import json
import math
import random

import pytest

import creditrisk
from creditrisk import synthetic
from creditrisk.binning import bin_categorical, bin_numeric, is_missing, quantile_edges
from creditrisk.fairness import audit, group_metrics, threshold_for_parity
from creditrisk.scorecard import (
    NO_BIN,
    Scorecard,
    brier_score,
    calibration_table,
    fit_logistic,
    fit_scorecard,
    gini,
    logistic,
)

FEATURES = ("income", "age", "employment")


@pytest.fixture(scope="module")
def data():
    return synthetic()


@pytest.fixture(scope="module")
def card(data):
    """Built once: the fit is deterministic, and the tests only read it."""
    return fit_scorecard({k: data[k] for k in FEATURES}, data["default"])


def applications(data):
    return [{k: data[k][i] for k in FEATURES} for i in range(len(data["default"]))]


def two_feature_card():
    """The manual route: bin, transform, fit, build."""
    d = synthetic()
    f_income = bin_numeric("income", d["income"], d["default"])
    f_age = bin_numeric("age", d["age"], d["default"])
    pairs = zip(d["income"], d["age"], strict=True)
    rows = [[f_income.transform(i), f_age.transform(a)] for i, a in pairs]
    weights, bias = fit_logistic(rows, d["default"])
    return Scorecard(
        {"income": f_income, "age": f_age}, {"income": weights[0], "age": weights[1]}, bias
    ).build()


class TestBinning:
    def test_quantile_edges_are_equal_frequency(self):
        """Equal-width bins on a skewed feature put 95% of the population in one bin."""
        edges = quantile_edges(list(range(100)), 4)
        assert edges == [25, 50, 75]

    def test_woe_is_negative_where_risk_is_high(self, data):
        """WoE is ln(P(good)/P(bad)), so a high-risk bin must be negative. Getting
        this sign wrong inverts the entire scorecard."""
        feature = bin_numeric("income", data["income"], data["default"])
        worst = max(feature.bins, key=lambda b: b.bad_rate)
        assert feature.woe[worst.label] < 0

    def test_woe_is_monotonic_for_a_monotonic_feature(self, data):
        assert bin_numeric("income", data["income"], data["default"]).is_monotonic()

    def test_a_noise_feature_scores_useless(self, data):
        assert bin_numeric("age", data["age"], data["default"]).iv < 0.02

    def test_the_demo_signal_is_strong_not_flagged_as_leakage(self, data):
        """The flagship feature must land in 'strong'. An example whose real signal
        trips the leakage warning teaches the reader to ignore the warning."""
        feature = bin_numeric("income", data["income"], data["default"])
        assert 0.3 < feature.iv < 0.5
        assert feature.strength() == "strong"

    def test_leakage_is_flagged_as_suspicious(self):
        """An IV above 0.5 almost always means the feature encodes the outcome."""
        target = [i % 2 for i in range(400)]
        leak = [float(y) for y in target]  # the label itself
        assert "suspicious" in bin_numeric("leak", leak, target, n_bins=2).strength()

    def test_a_bin_with_no_bads_does_not_produce_infinity(self):
        """Continuity correction, not an arbitrary epsilon."""
        values = [1.0] * 50 + [100.0] * 50
        target = [0] * 50 + [1] * 50
        feature = bin_numeric("x", values, target, n_bins=2)
        assert all(math.isfinite(w) for w in feature.woe.values())

    def test_missing_values_get_their_own_bin(self):
        """Rather than an imputed lie. Missingness is frequently predictive in
        lending — a blank employer field is information."""
        values = [1.0, 2.0, 3.0, None, None, 4.0]
        feature = bin_numeric("x", values, [0, 0, 1, 1, 1, 0], n_bins=2)
        assert any(b.is_missing and b.total == 2 for b in feature.bins)

    def test_nan_is_missing_not_a_number(self):
        """pandas hands you NaN for a blank cell. NaN inside sorted() used to produce
        arbitrary quantile edges; the bins must be identical to the None run."""
        rng = random.Random(3)
        values = [rng.uniform(0, 100) for _ in range(1500)] + [None] * 150
        rng.shuffle(values)
        target = [rng.randint(0, 1) for _ in values]
        with_none = bin_numeric("x", values, target)
        with_nan = bin_numeric("x", [math.nan if v is None else v for v in values], target)
        assert [b.total for b in with_nan.bins] == [b.total for b in with_none.bins]
        assert with_nan.woe == with_none.woe
        assert with_nan.bins[-1].is_missing and with_nan.bins[-1].total == 150
        assert with_nan.transform(math.nan) == with_nan.woe["missing"]

    def test_is_missing_recognises_the_usual_blanks(self):
        class NAType:  # stands in for pandas.NA without importing pandas
            pass

        for blank in (None, math.nan, float("nan"), "", "  ", NAType()):
            assert is_missing(blank)
        for value in (0, 0.0, False, "0", "A", -1.5):
            assert not is_missing(value)

    def test_undersized_bins_are_merged(self):
        """A bin of nine accounts produces a WoE that will not survive next quarter."""
        values = [float(i) for i in range(100)]
        target = [i % 2 for i in range(100)]
        feature = bin_numeric("x", values, target, n_bins=20, min_bin_fraction=0.15)
        numeric = [b for b in feature.bins if not b.is_missing]
        assert all(b.total >= 15 for b in numeric)

    def test_an_undersized_first_bin_is_merged_forward(self):
        """The merge used to only look backwards, so a tiny first bin survived."""
        values = [0.0] * 3 + [float(v) for v in range(1, 198)]
        target = [i % 2 for i in range(200)]
        feature = bin_numeric("x", values, target, edges=[0.5, 50, 100, 150])
        assert all(b.total >= 10 for b in feature.bins)
        assert feature.bins[0].lower == -math.inf

    def test_user_edges_are_sorted(self):
        """Unsorted edges used to label a bin [-inf, 20) that held values up to 49."""
        values = [float(v) for v in range(100)]
        target = [i % 2 for i in range(100)]
        feature = bin_numeric("x", values, target, edges=[50, 20])
        assert [b.label for b in feature.bins] == ["[-inf, 20)", "[20, 50)", "[50, inf)"]
        assert [b.total for b in feature.bins] == [20, 30, 50]

    def test_categorical_binning(self):
        feature = bin_categorical("grade", ["A", "A", "B", "B", "C", None], [0, 0, 1, 1, 1, 0])
        assert {b.label for b in feature.bins} == {"A", "B", "C", "missing"}

    def test_categorical_features_transform(self):
        """Used to raise TypeError: '<=' not supported between float and str."""
        feature = bin_categorical("grade", ["A", "B"] * 10, [0, 1] * 10)
        assert feature.transform("A") > 0 > feature.transform("B")
        assert feature.is_monotonic()  # no order to violate

    def test_an_unseen_category_goes_to_the_missing_bin(self):
        feature = bin_categorical("grade", ["A", "B", None] * 10, [0, 1, 1] * 10)
        assert feature.transform("Z") == feature.woe["missing"]

    def test_an_unseen_category_without_a_missing_bin_is_neutral(self):
        feature = bin_categorical("grade", ["A", "B"] * 10, [0, 1] * 10)
        assert feature.transform("Z") == 0.0

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            bin_numeric("x", [1.0, 2.0], [0])

    @pytest.mark.parametrize("bad_target", [["0", "1"], [-1, 1], [0, 2], [0, None]])
    def test_a_target_that_is_not_zero_one_is_refused(self, bad_target):
        """'0' is truthy, so a text column used to count every row as a default."""
        with pytest.raises(ValueError, match="0 .good. and 1 .bad."):
            bin_numeric("x", [1.0, 2.0], bad_target)

    def test_booleans_and_floats_are_accepted_targets(self):
        feature = bin_numeric("x", [1.0, 2.0, 3.0, 4.0], [False, True, 0.0, 1.0], n_bins=2)
        assert sum(b.bads for b in feature.bins) == 2

    def test_text_in_a_numeric_feature_is_refused(self):
        with pytest.raises(TypeError, match="bin_categorical"):
            bin_numeric("x", [1.0, "salaried"], [0, 1])

    def test_a_single_class_yields_no_evidence(self):
        feature = bin_numeric("x", [1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], n_bins=2)
        assert feature.iv == 0.0


class TestScorecard:
    def test_higher_quality_applicants_score_higher(self, card):
        """The bug this test exists for: the model predicts P(bad) while a score is
        log-odds of *good*, so the sign flips on the way into points. Get it wrong and
        the scorecard is statistically right and commercially backwards."""
        good = {"income": 180, "age": 45, "employment": "salaried"}
        poor = {"income": 25, "age": 22, "employment": "unemployed"}
        assert card.score(good) > card.score(poor)

    def test_score_and_probability_never_disagree(self, card):
        """They are the same number in two units, so this is exact."""
        app = {"income": 120, "age": 40, "employment": "salaried"}
        odds = math.exp((card.score(app) - card.offset) / card.factor)
        assert card.probability(app) == pytest.approx(1 / (1 + odds), abs=1e-6)

    def test_points_sum_to_the_score(self, card):
        """No hidden constant, or the decision cannot be explained line by line."""
        app = {"income": 120, "age": 40, "employment": "salaried"}
        assert sum(c.points for c in card.contributions(app)) == card.score(app)

    def test_pdo_doubles_the_odds(self, card):
        """The definition of the points scale: +PDO points halves the risk."""

        def odds(score):
            return math.exp((score - card.offset) / card.factor)

        assert odds(600 + card.pdo) == pytest.approx(2 * odds(600), rel=1e-9)

    def test_base_score_sits_at_base_odds(self, card):
        expected = card.offset + card.factor * math.log(card.base_odds)
        assert expected == pytest.approx(card.base_score, abs=1e-6)

    def test_the_manual_route_builds_the_same_kind_of_card(self):
        manual = two_feature_card()
        assert manual.score({"income": 180, "age": 45}) > manual.score({"income": 25, "age": 45})

    def test_a_categorical_feature_scores(self, card):
        """Categorical features used to crash at scoring."""
        base = {"income": 120, "age": 40}
        salaried = card.score({**base, "employment": "salaried"})
        unemployed = card.score({**base, "employment": "unemployed"})
        assert salaried > unemployed

    def test_a_missing_field_gets_neutral_points_not_zero(self, card):
        """A missing field with no training-time missing bin used to drop that
        feature's whole share of the points (529 -> 265), guaranteeing a decline."""
        full = {"income": 120, "age": 40, "employment": "salaried"}
        partial = {"income": 120, "employment": "salaried"}
        assert abs(card.score(full) - card.score(partial)) <= 5
        age = next(c for c in card.contributions(partial) if c.feature == "age")
        assert age.bin_label == NO_BIN
        assert age.points == card.neutral_points["age"]

    @pytest.mark.parametrize("blank", [None, math.nan, ""])
    def test_blank_values_score_like_an_absent_field(self, card, blank):
        absent = card.score({"income": 120, "employment": "salaried"})
        assert card.score({"income": 120, "age": blank, "employment": "salaried"}) == absent

    def test_a_blank_income_uses_the_trained_missing_bin(self, card):
        c = card.contributions({"income": None, "age": 40, "employment": "salaried"})[0]
        assert c.bin_label == "missing"

    def test_an_empty_application_scores_near_the_population_average(self, card):
        """Used to score 0."""
        assert abs(card.score({}) - sum(card.neutral_points.values())) <= 5

    def test_a_typo_in_a_field_name_is_refused(self, card):
        """'Income' used to be silently ignored, costing the applicant ~half the score."""
        with pytest.raises(ValueError, match="did you mean 'income'"):
            card.score({"Income": 120, "age": 40, "employment": "salaried"})

    def test_extra_fields_can_be_ignored_explicitly(self, data):
        lenient = fit_scorecard({"income": data["income"]}, data["default"])
        lenient.on_unknown_field = "ignore"
        assert lenient.score({"income": 120, "id": 7}) == lenient.score({"income": 120})

    def test_unmatched_values_can_be_refused_explicitly(self, data):
        strict = fit_scorecard({"age": data["age"]}, data["default"])
        strict.on_unmatched = "raise"
        with pytest.raises(ValueError, match="no bin"):
            strict.score({"age": None})

    def test_a_numeric_string_is_scored_as_a_number(self, card):
        a = {"income": "120", "age": "40", "employment": "salaried"}
        b = {"income": 120, "age": 40, "employment": "salaried"}
        assert card.score(a) == card.score(b)

    def test_text_in_a_numeric_field_is_refused(self, card):
        with pytest.raises(TypeError, match="income"):
            card.score({"income": "lots", "age": 40, "employment": "salaried"})

    def test_a_coefficient_typo_is_refused(self, data):
        """Used to fall back to beta = 0, giving every bin the same points."""
        f = bin_numeric("income", data["income"], data["default"])
        with pytest.raises(ValueError, match="missing \\['income'\\]"):
            Scorecard({"income": f}, {"Income": -1.0}, 0.0).build()

    def test_scoring_before_build_is_refused(self, data):
        f = bin_numeric("income", data["income"], data["default"])
        with pytest.raises(RuntimeError, match="build"):
            Scorecard({"income": f}, {"income": -1.0}, 0.0).score({"income": 1})

    def test_reason_codes_name_the_worst_factor_first(self, card):
        reasons = card.reason_codes({"income": 25, "age": 45, "employment": "salaried"})
        assert reasons and reasons[0]["feature"] == "income"
        assert reasons[0]["points_lost"] > 0

    def test_a_factor_that_cost_nothing_is_not_a_reason(self, card):
        """Adverse-action notices must list reasons for the decline, not a feature
        list."""
        best = {"income": 195, "age": 35, "employment": "salaried"}
        assert card.reason_codes(best) == []

    def test_reason_codes_are_measured_against_the_best_attainable(self, card):
        reasons = card.reason_codes({"income": 25, "age": 22, "employment": "salaried"})
        income_best = max(card.points["income"].values())
        income_reason = next(r for r in reasons if r["feature"] == "income")
        assert income_reason["points_lost"] == income_best - income_reason["points"]

    def test_explain_is_json_ready(self, card):
        out = card.explain({"income": 34, "age": 29, "employment": "self-employed"}, cutoff=527)
        assert json.loads(json.dumps(out)) == out
        assert out["decision"] == "DECLINE"
        assert sum(p["points"] for p in out["points"]) == out["score"]


class TestPersistence:
    def test_round_trip_through_json_scores_identically(self, card, data, tmp_path):
        path = tmp_path / "card.json"
        card.save(path)
        loaded = Scorecard.load(path)
        for app in applications(data)[:300]:
            assert loaded.score(app) == card.score(app)
            assert loaded.reason_codes(app) == card.reason_codes(app)
        assert loaded.score({"age": 40}) == card.score({"age": 40})

    def test_a_tampered_file_is_refused(self, card):
        d = json.loads(json.dumps(card.to_dict()))
        d["points"]["income"][next(iter(d["points"]["income"]))] += 50
        with pytest.raises(ValueError, match="do not match"):
            Scorecard.from_dict(d)

    def test_a_foreign_json_is_refused(self):
        with pytest.raises(ValueError, match="not a credit-risk-engine scorecard"):
            Scorecard.from_dict({"hello": 1})

    def test_points_table_lists_every_bin(self, card):
        table = card.points_table()
        income_rows = [r for r in table if r["feature"] == "income"]
        assert [r["bin"] for r in income_rows] == [b.label for b in card.features["income"].bins]
        age_neutral = [r for r in table if r["feature"] == "age" and r["bin"] == NO_BIN]
        assert age_neutral and age_neutral[0]["points"] == card.neutral_points["age"]


class TestFit:
    def test_newton_matches_the_closed_form_mle(self):
        """With one binary feature and no penalty the MLE is known exactly:
        bias = logit(bad rate at x=0), weight = logit(rate at x=1) - bias."""
        rows = [[0.0]] * 100 + [[1.0]] * 100
        target = [1] * 20 + [0] * 80 + [1] * 60 + [0] * 40
        w, b = fit_logistic(rows, target, l2=0.0)

        def logit(p):
            return math.log(p / (1 - p))

        assert b == pytest.approx(logit(0.2), abs=1e-9)
        assert w[0] == pytest.approx(logit(0.6) - logit(0.2), abs=1e-9)

    def test_the_fit_reaches_the_optimum(self, data):
        """Gradient descent used to stop far short (age beta -0.06 vs MLE -1.15)."""
        f = bin_numeric("income", data["income"], data["default"])
        rows = [[f.transform(v)] for v in data["income"]]
        w, b = fit_logistic(rows, data["default"], l2=0.001)
        n = len(rows)
        errors = [
            (logistic(w[0] * r[0] + b) - y, r[0])
            for r, y in zip(rows, data["default"], strict=True)
        ]
        grad_w = sum(e * x for e, x in errors)
        grad_b = sum(e for e, _ in errors)
        assert abs(grad_w / n + 0.001 * w[0]) < 1e-8
        assert abs(grad_b / n) < 1e-8

    def test_perfect_separation_stays_finite(self):
        w, b = fit_logistic([[0.0]] * 10 + [[1.0]] * 10, [0] * 10 + [1] * 10)
        assert math.isfinite(w[0]) and math.isfinite(b)

    def test_one_class_is_refused(self):
        with pytest.raises(ValueError, match="one class"):
            fit_logistic([[0.1], [0.2]], [0, 0])

    def test_ragged_or_missing_rows_are_refused(self):
        with pytest.raises(ValueError, match="row 1"):
            fit_logistic([[0.1, 0.2], [0.3]], [0, 1])
        with pytest.raises(ValueError, match="non-finite"):
            fit_logistic([[0.1], [math.nan]], [0, 1])

    def test_collinear_features_are_explained_without_a_penalty(self):
        rows = [[float(i % 3), float(i % 3)] for i in range(30)]
        with pytest.raises(ValueError, match="collinear"):
            fit_logistic(rows, [i % 2 for i in range(30)], l2=0.0)

    def test_fit_scorecard_detects_text_columns_as_categorical(self, card):
        assert card.features["employment"].kind == "categorical"
        assert card.features["income"].kind == "numeric"

    def test_fit_scorecard_rejects_unknown_categorical_names(self, data):
        with pytest.raises(ValueError, match="not features"):
            fit_scorecard({"income": data["income"]}, data["default"], categorical=["job"])


class TestMetrics:
    def test_gini_is_one_for_perfect_separation(self):
        assert gini([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0

    def test_gini_is_zero_for_a_coin_flip(self):
        assert gini([0.5] * 4, [0, 1, 0, 1]) == 0.0

    def test_gini_is_negative_when_the_model_is_inverted(self):
        """Which is what an inverted scorecard would look like in production."""
        assert gini([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == -1.0

    def test_gini_needs_both_classes(self):
        assert gini([0.1, 0.9], [0, 0]) == 0.0

    def test_rank_gini_equals_counting_every_pair(self):
        """The O(n log n) version must agree with the definition, ties included."""
        rng = random.Random(5)
        probs = [round(rng.random(), 1) for _ in range(400)]  # heavy ties
        outcomes = [1 if rng.random() < p else 0 for p in probs]
        bads = [p for p, y in zip(probs, outcomes, strict=True) if y]
        goods = [p for p, y in zip(probs, outcomes, strict=True) if not y]
        pairs = sum(1.0 if b > g else 0.5 if b == g else 0.0 for b in bads for g in goods)
        assert gini(probs, outcomes) == pytest.approx(2 * pairs / (len(bads) * len(goods)) - 1)

    def test_mismatched_lengths_are_refused(self):
        """gini([.1,.2,.3], [0,1]) used to return 1.0."""
        with pytest.raises(ValueError, match="same length"):
            gini([0.1, 0.2, 0.3], [0, 1])
        with pytest.raises(ValueError, match="same length"):
            brier_score([0.1, 0.2, 0.3], [0, 1])

    def test_the_model_discriminates(self, card, data):
        probs = [card.probability(a) for a in applications(data)]
        assert gini(probs, data["default"]) > 0.3

    def test_brier_rewards_calibration_not_just_ranking(self):
        """A model can rank perfectly and still be wrong about the level — and a
        lender prices from the level."""
        confident = brier_score([0.01, 0.99], [0, 1])
        timid = brier_score([0.45, 0.55], [0, 1])
        assert confident < timid

    def test_calibration_table_compares_predicted_with_observed(self, card, data):
        probs = [card.probability(a) for a in applications(data)]
        table = calibration_table(probs, data["default"], n_bins=5)
        assert len(table) == 5
        assert table[0]["predicted"] < table[-1]["predicted"]
        assert table[0]["observed"] < table[-1]["observed"]
        for row in table:
            assert abs(row["predicted"] - row["observed"]) < 0.06

    def test_calibration_groups_have_no_runt(self):
        table = calibration_table([i / 23 for i in range(23)], [i % 2 for i in range(23)])
        assert sorted(r["n"] for r in table) == [2] * 7 + [3] * 3

    def test_tied_scores_are_not_sorted_goods_first(self):
        """Sorting (probability, outcome) pairs put every good before every bad inside
        a tie, so a group boundary through a tie manufactured miscalibration. Scorecards
        have few distinct scores, so this hit real output."""
        table = calibration_table([0.5] * 10, [1, 0] * 5, n_bins=2)
        assert [r["observed"] for r in table] == [0.6, 0.4]

    def test_empty_inputs_do_not_raise(self):
        assert brier_score([], []) == 0.0
        assert calibration_table([], []) == []


class TestFairness:
    def test_group_metrics(self):
        metrics = group_metrics(["a", "a", "b", "b"], [1, 0, 1, 1], [0, 1, 0, 1])
        assert metrics["a"].n == 2
        assert metrics["b"].approval_rate == 1.0

    def test_disparate_impact_flags_the_four_fifths_rule(self):
        groups = ["majority"] * 100 + ["minority"] * 100
        approved = [1] * 90 + [0] * 10 + [1] * 50 + [0] * 50
        outcomes = [0] * 200
        report = audit(groups, approved, outcomes)
        assert report["worst_disparate_impact"] == pytest.approx(50 / 90, abs=1e-4)
        assert report["four_fifths_rule_flag"] is True

    def test_a_group_with_zero_approvals_is_flagged(self):
        """A ratio of 0.0 is falsy; it used to be dropped, so the most extreme adverse
        impact there is came back as 'not flagged'."""
        groups = ["maj"] * 100 + ["min"] * 50
        approved = [1] * 80 + [0] * 20 + [0] * 50
        report = audit(groups, approved, [0] * 150)
        assert report["comparisons"]["min"]["disparate_impact"] == 0.0
        assert report["worst_disparate_impact"] == 0.0
        assert report["four_fifths_rule_flag"] is True

    def test_an_even_handed_model_is_not_flagged(self):
        groups = ["majority"] * 100 + ["minority"] * 100
        approved = [1] * 85 + [0] * 15 + [1] * 80 + [0] * 20
        report = audit(groups, approved, [0] * 200)
        assert report["four_fifths_rule_flag"] is False

    def test_the_reference_group_is_the_largest(self):
        """Picking it alphabetically is an accident waiting for a label to change."""
        groups = ["zzz"] * 100 + ["aaa"] * 10
        report = audit(groups, [1] * 110, [0] * 110)
        assert report["reference_group"] == "zzz"

    def test_an_unknown_reference_group_is_a_clear_error(self):
        with pytest.raises(ValueError, match="reference group 'xyz' not found"):
            audit(["a", "b"], [1, 0], [0, 0], reference="xyz")

    def test_mixed_label_types_do_not_crash(self):
        report = audit([1, "1", 2, 2], [1, 1, 0, 1], [0, 0, 0, 0])
        assert set(report["groups"]) == {"1", "2"}

    def test_approved_must_be_zero_or_one(self):
        with pytest.raises(ValueError, match="approved"):
            audit(["a", "b"], ["yes", "no"], [0, 0])

    def test_equal_opportunity_gap_looks_only_at_those_who_would_repay(self):
        groups = ["a"] * 4 + ["b"] * 4
        outcomes = [0, 0, 1, 1, 0, 0, 1, 1]
        approved = [1, 1, 0, 0, 1, 0, 0, 0]  # group b rejects a good applicant
        report = audit(groups, approved, outcomes, reference="a")
        assert report["comparisons"]["b"]["equal_opportunity_gap"] == pytest.approx(-0.5)

    def test_equalised_odds_takes_the_worse_of_the_two_gaps(self):
        groups = ["a"] * 4 + ["b"] * 4
        outcomes = [0, 0, 1, 1, 0, 0, 1, 1]
        approved = [1, 1, 0, 0, 1, 1, 1, 1]  # b approves everyone: TPR equal, FPR worse
        report = audit(groups, approved, outcomes, reference="a")
        comparison = report["comparisons"]["b"]
        assert comparison["equal_opportunity_gap"] == 0.0
        assert comparison["equalised_odds_gap"] == pytest.approx(1.0)

    def test_base_rate_difference_is_reported_alongside(self):
        """It is the usual explanation offered for a gap, and the reader is entitled
        to judge it rather than be told it."""
        groups = ["a"] * 4 + ["b"] * 4
        report = audit(groups, [1] * 8, [0, 0, 0, 0, 1, 1, 1, 1], reference="a")
        assert report["comparisons"]["b"]["bad_rate_difference"] == pytest.approx(1.0)

    def test_the_incompatibility_is_stated_not_hidden(self):
        """Demographic parity and equalised odds cannot both hold when base rates
        differ. A report quoting one measure is concealing that."""
        report = audit(["a", "a", "b", "b"], [1, 0, 1, 0], [0, 1, 0, 1])
        assert "equalised odds" in report["note"]

    def test_the_report_is_json_serialisable(self):
        report = audit(["a", "a", "b", "b"], [1, 0, 1, 0], [0, 1, 0, 1])
        assert json.loads(json.dumps(report)) == report

    def test_a_single_group_has_nothing_to_compare(self):
        assert "nothing to compare" in audit(["a", "a"], [1, 0], [0, 1])["note"]

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            group_metrics(["a"], [1, 0], [0])

    def test_parity_thresholds_are_exported_and_group_specific(self):
        """Computed so the trade-off is concrete: this achieves parity exactly, and
        using it is unlawful in most jurisdictions because the protected attribute
        enters the decision."""
        assert creditrisk.threshold_for_parity is threshold_for_parity
        scores = [float(i) for i in range(100)]
        groups = ["a"] * 50 + ["b"] * 50
        cutoffs = threshold_for_parity(scores, groups, target_rate=0.5)
        assert cutoffs == {"a": 25.0, "b": 75.0}
        for name, cut in cutoffs.items():
            group = [s for s, g in zip(scores, groups, strict=True) if g == name]
            assert sum(s >= cut for s in group) / len(group) == 0.5

    def test_parity_target_rate_edges(self):
        scores, groups = [1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"]
        assert threshold_for_parity(scores, groups, target_rate=0.0) == {
            "a": math.inf,
            "b": math.inf,
        }
        assert threshold_for_parity(scores, groups, target_rate=1.0) == {"a": 1.0, "b": 3.0}
        for bad in (1.5, -0.1):
            with pytest.raises(ValueError, match="target_rate"):
                threshold_for_parity(scores, groups, target_rate=bad)
