"""Tests for the practitioner guidance module."""

import pytest

from diff_diff import (
    BaconDecomposition,
    CallawaySantAnna,
    DifferenceInDifferences,
    MultiPeriodDiD,
    generate_did_data,
    generate_staggered_data,
)
from diff_diff.continuous_did_results import ContinuousDiDResults
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.imputation_results import ImputationDiDResults
from diff_diff.practitioner import STEPS, practitioner_next_steps
from diff_diff.results import DiDResults, SyntheticDiDResults
from diff_diff.stacked_did_results import StackedDiDResults
from diff_diff.sun_abraham import SunAbrahamResults
from diff_diff.triple_diff import TripleDifferenceResults
from diff_diff.trop_results import TROPResults
from diff_diff.two_stage_results import TwoStageDiDResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def did_data():
    return generate_did_data(n_units=50, treatment_effect=3.0, seed=42)


@pytest.fixture(scope="session")
def staggered_data():
    return generate_staggered_data(
        n_units=60, n_periods=8, treatment_effect=2.0, seed=42
    )


@pytest.fixture(scope="session")
def did_results(did_data):
    did = DifferenceInDifferences()
    return did.fit(did_data, outcome="outcome", treatment="treated", time="post")


@pytest.fixture(scope="session")
def multi_period_results(did_data):
    es = MultiPeriodDiD()
    return es.fit(
        did_data, outcome="outcome", unit="unit", time="period", treatment="treated"
    )


@pytest.fixture(scope="session")
def cs_results(staggered_data):
    cs = CallawaySantAnna()
    return cs.fit(
        staggered_data,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
    )


@pytest.fixture(scope="session")
def bacon_results(staggered_data):
    bacon = BaconDecomposition()
    return bacon.fit(
        staggered_data,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
    )


# ---------------------------------------------------------------------------
# Mock result fixtures for expensive estimators
# ---------------------------------------------------------------------------
def _mock_result(cls, **overrides):
    """Create a minimal mock of a results dataclass."""
    # Provide default fields that most result types share
    defaults = dict(
        att=0.5,
        se=0.1,
        t_stat=5.0,
        p_value=0.001,
        conf_int=(0.3, 0.7),
        n_obs=100,
        n_treated=50,
        n_control=50,
    )
    defaults.update(overrides)
    try:
        return cls(**defaults)
    except TypeError:
        # Some result classes have different required fields
        return cls.__new__(cls)


@pytest.fixture
def mock_synth_results():
    r = SyntheticDiDResults.__new__(SyntheticDiDResults)
    r.att = 1.0
    r.se = 0.3
    return r


@pytest.fixture
def mock_trop_results():
    r = TROPResults.__new__(TROPResults)
    r.att = 0.8
    r.se = 0.2
    return r


@pytest.fixture
def mock_efficient_results():
    r = EfficientDiDResults.__new__(EfficientDiDResults)
    r.att = 0.6
    r.se = 0.15
    r.hausman_pretest = None
    return r


@pytest.fixture
def mock_continuous_results():
    r = ContinuousDiDResults.__new__(ContinuousDiDResults)
    r.att = 0.4
    r.se = 0.1
    return r


@pytest.fixture
def mock_triple_results():
    r = TripleDifferenceResults.__new__(TripleDifferenceResults)
    r.att = 0.7
    r.se = 0.2
    return r


@pytest.fixture
def mock_sa_results():
    r = SunAbrahamResults.__new__(SunAbrahamResults)
    r.att = 0.5
    r.se = 0.1
    return r


@pytest.fixture
def mock_imputation_results():
    r = ImputationDiDResults.__new__(ImputationDiDResults)
    r.att = 0.5
    r.se = 0.1
    return r


@pytest.fixture
def mock_two_stage_results():
    r = TwoStageDiDResults.__new__(TwoStageDiDResults)
    r.att = 0.5
    r.se = 0.1
    return r


@pytest.fixture
def mock_stacked_results():
    r = StackedDiDResults.__new__(StackedDiDResults)
    r.att = 0.5
    r.se = 0.1
    return r


# ---------------------------------------------------------------------------
# Tests: return schema
# ---------------------------------------------------------------------------
class TestReturnSchema:
    def test_has_expected_keys(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert "estimator" in output
        assert "completed" in output
        assert "next_steps" in output
        assert "warnings" in output

    def test_estimator_name(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert output["estimator"] == "DifferenceInDifferences"

    def test_estimation_always_completed(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert "estimation" in output["completed"]

    def test_next_steps_are_dicts(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        for step in output["next_steps"]:
            assert "baker_step" in step
            assert "label" in step
            assert "why" in step
            assert "code" in step
            assert "priority" in step

    def test_warnings_are_strings(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        for w in output["warnings"]:
            assert isinstance(w, str)


# ---------------------------------------------------------------------------
# Tests: each result type produces guidance
# ---------------------------------------------------------------------------
class TestResultTypeDispatch:
    def test_did_results(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_multi_period_results(self, multi_period_results):
        output = practitioner_next_steps(multi_period_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "MultiPeriodDiD (Event Study)"

    def test_cs_results(self, cs_results):
        output = practitioner_next_steps(cs_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "CallawaySantAnna"

    def test_bacon_results(self, bacon_results):
        output = practitioner_next_steps(bacon_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "BaconDecomposition"
        # Bacon should suggest switching to a robust estimator
        labels = [s["label"] for s in output["next_steps"]]
        assert any("heterogeneity-robust" in lbl for lbl in labels)

    def test_sa_results(self, mock_sa_results):
        output = practitioner_next_steps(mock_sa_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "SunAbraham"

    def test_imputation_results(self, mock_imputation_results):
        output = practitioner_next_steps(mock_imputation_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_two_stage_results(self, mock_two_stage_results):
        output = practitioner_next_steps(mock_two_stage_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_stacked_results(self, mock_stacked_results):
        output = practitioner_next_steps(mock_stacked_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_synth_results(self, mock_synth_results):
        output = practitioner_next_steps(mock_synth_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "SyntheticDiD"

    def test_trop_results(self, mock_trop_results):
        output = practitioner_next_steps(mock_trop_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_efficient_results(self, mock_efficient_results):
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_continuous_results(self, mock_continuous_results):
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_triple_results(self, mock_triple_results):
        output = practitioner_next_steps(mock_triple_results, verbose=False)
        assert len(output["next_steps"]) > 0


# ---------------------------------------------------------------------------
# Tests: completed_steps filtering
# ---------------------------------------------------------------------------
class TestCompletedSteps:
    def test_filter_parallel_trends(self, cs_results):
        full = practitioner_next_steps(cs_results, verbose=False)
        filtered = practitioner_next_steps(
            cs_results, completed_steps=["parallel_trends"], verbose=False
        )
        assert len(filtered["next_steps"]) < len(full["next_steps"])
        # No step should have baker_step 2 about parallel trends
        for s in filtered["next_steps"]:
            if s["baker_step"] == 2:
                assert "parallel trends" not in s["label"].lower()

    def test_filter_sensitivity(self, cs_results):
        full = practitioner_next_steps(cs_results, verbose=False)
        filtered = practitioner_next_steps(
            cs_results, completed_steps=["sensitivity"], verbose=False
        )
        assert len(filtered["next_steps"]) < len(full["next_steps"])

    def test_filter_all_steps(self, cs_results):
        output = practitioner_next_steps(
            cs_results, completed_steps=list(STEPS), verbose=False
        )
        assert len(output["next_steps"]) == 0

    def test_invalid_step_name_raises(self, did_results):
        with pytest.raises(ValueError, match="Unknown step names"):
            practitioner_next_steps(
                did_results, completed_steps=["invalid_step"], verbose=False
            )


# ---------------------------------------------------------------------------
# Tests: verbose output
# ---------------------------------------------------------------------------
class TestVerboseOutput:
    def test_verbose_prints(self, did_results, capsys):
        practitioner_next_steps(did_results, verbose=True)
        captured = capsys.readouterr()
        assert "Practitioner Guidance" in captured.out
        assert "Baker et al." in captured.out
        assert "DifferenceInDifferences" in captured.out

    def test_no_print_when_silent(self, did_results, capsys):
        practitioner_next_steps(did_results, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------
class TestNaNHandling:
    def test_nan_att_produces_warning(self):
        r = DiDResults(
            att=float("nan"),
            se=float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            conf_int=(float("nan"), float("nan")),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        output = practitioner_next_steps(r, verbose=False)
        assert len(output["warnings"]) > 0
        assert any("NaN" in w for w in output["warnings"])


# ---------------------------------------------------------------------------
# Tests: unknown result type fallback
# ---------------------------------------------------------------------------
class TestFallback:
    def test_unknown_type(self):
        class FakeResults:
            att = 1.0
            se = 0.5

        output = practitioner_next_steps(FakeResults(), verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "FakeResults"
