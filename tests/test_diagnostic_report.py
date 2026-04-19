"""Tests for ``diff_diff.diagnostic_report.DiagnosticReport``.

Covers:
- Schema contract: every top-level key always present, stable enum values.
- Applicability matrix: per-estimator ``applicable_checks`` property.
- JSON round-trip.
- ``precomputed=`` passthrough (sensitivity).
- Pre-trends verdict thresholds (three bins).
- Power-aware tier thresholds (three bins + fallback).
- DEFF reads from ``survey_metadata`` when present.
- EfficientDiD ``hausman_pretest`` pathway.
- SDiD / TROP native diagnostics.
- Error-doesn't-break-report (diagnostic raises -> section records error).
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import patch

import numpy as np
import pytest

import diff_diff as dd
from diff_diff import (
    CallawaySantAnna,
    DiagnosticReport,
    DiagnosticReportResults,
    DifferenceInDifferences,
    EfficientDiD,
    MultiPeriodDiD,
    SyntheticDiD,
    generate_did_data,
    generate_factor_data,
    generate_staggered_data,
)
from diff_diff.diagnostic_report import (
    DIAGNOSTIC_REPORT_SCHEMA_VERSION,
    _power_tier,
    _pt_verdict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_TOP_LEVEL_KEYS = {
    "schema_version",
    "estimator",
    "headline_metric",
    "parallel_trends",
    "pretrends_power",
    "sensitivity",
    "placebo",
    "bacon",
    "design_effect",
    "heterogeneity",
    "epv",
    "estimator_native_diagnostics",
    "skipped",
    "warnings",
    "overall_interpretation",
    "next_steps",
}

_STATUS_ENUM = {
    "ran",
    "skipped",
    "error",
    "not_applicable",
    "not_run",
    "computed",
}


@pytest.fixture(scope="module")
def did_fit():
    warnings.filterwarnings("ignore")
    df = generate_did_data(n_units=80, n_periods=4, treatment_effect=1.5, seed=7)
    did = DifferenceInDifferences().fit(df, outcome="outcome", treatment="treated", time="post")
    return did, df


@pytest.fixture(scope="module")
def multi_period_fit():
    warnings.filterwarnings("ignore")
    df = generate_did_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
    es = MultiPeriodDiD().fit(
        df,
        outcome="outcome",
        treatment="treated",
        time="period",
        unit="unit",
        reference_period=3,
    )
    return es, df


@pytest.fixture(scope="module")
def cs_fit():
    warnings.filterwarnings("ignore")
    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    # Use base_period='universal' so HonestDiD sensitivity can run on this
    # fixture. CS's default is 'varying', which DR now skips with a
    # methodology-critical reason (Rambachan-Roth bounds are not valid for
    # interpretation on consecutive-comparison pre-periods). See the
    # round-5 CI review on PR #318.
    cs = CallawaySantAnna(base_period="universal").fit(
        sdf,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        aggregate="event_study",
    )
    return cs, sdf


@pytest.fixture(scope="module")
def edid_fit():
    warnings.filterwarnings("ignore")
    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    edid = EfficientDiD().fit(
        sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
    )
    return edid, sdf


@pytest.fixture(scope="module")
def sdid_fit():
    warnings.filterwarnings("ignore")
    fdf = generate_factor_data(n_units=25, n_pre=8, n_post=4, n_treated=4, seed=11)
    sdid = SyntheticDiD().fit(fdf, outcome="outcome", unit="unit", time="period", treatment="treat")
    return sdid, fdf


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
class TestSchemaContract:
    """The AI-legible schema is the public promise. These tests lock it down."""

    def test_every_top_level_key_present_did(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        schema = dr.to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_multiperiod(self, multi_period_fit):
        fit, _ = multi_period_fit
        schema = DiagnosticReport(fit).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_cs(self, cs_fit):
        fit, sdf = cs_fit
        schema = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        schema = DiagnosticReport(fit).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_schema_version_constant(self, multi_period_fit):
        fit, _ = multi_period_fit
        schema = DiagnosticReport(fit).to_dict()
        assert schema["schema_version"] == DIAGNOSTIC_REPORT_SCHEMA_VERSION
        assert DIAGNOSTIC_REPORT_SCHEMA_VERSION == "1.0"

    def test_all_statuses_use_closed_enum(self, cs_fit):
        fit, sdf = cs_fit
        schema = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).to_dict()
        for key in [
            "parallel_trends",
            "pretrends_power",
            "sensitivity",
            "placebo",
            "bacon",
            "design_effect",
            "heterogeneity",
            "epv",
            "estimator_native_diagnostics",
        ]:
            section = schema.get(key)
            assert isinstance(section, dict), f"{key} missing"
            assert (
                section.get("status") in _STATUS_ENUM
            ), f"{key}.status = {section.get('status')!r} not in {_STATUS_ENUM}"

    def test_json_round_trip_multiperiod(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        dumped = json.dumps(dr.to_dict())
        assert len(dumped) > 0
        round = json.loads(dumped)
        assert round["schema_version"] == DIAGNOSTIC_REPORT_SCHEMA_VERSION

    def test_json_round_trip_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        dumped = json.dumps(DiagnosticReport(fit).to_dict())
        assert len(dumped) > 0


# ---------------------------------------------------------------------------
# Applicability matrix
# ---------------------------------------------------------------------------
class TestApplicabilityMatrix:
    """Per-estimator applicability set filtered by instance state + options."""

    def test_did_without_data_skips_pt(self, did_fit):
        fit, _ = did_fit
        dr = DiagnosticReport(fit)  # no data
        assert "parallel_trends" not in dr.applicable_checks
        assert "parallel_trends" in dr.skipped_checks
        reason = dr.skipped_checks["parallel_trends"]
        assert "data" in reason.lower()

    def test_did_with_data_runs_pt(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        assert "parallel_trends" in dr.applicable_checks

    def test_did_with_data_but_no_column_kwargs_skips_pt(self, did_fit):
        """Round-11 regression: ``applicable_checks`` must match the
        runner's full argument contract. 2x2 PT needs ``data`` AND
        ``outcome`` / ``time`` / ``treatment`` — not just ``data``."""
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df)  # missing column kwargs
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks["parallel_trends"]
        assert "outcome" in reason
        assert "time" in reason
        assert "treatment" in reason

    def test_bacon_applicability_requires_all_column_kwargs(self, cs_fit):
        """Round-11 regression: Bacon needs the full ``outcome`` / ``time``
        / ``unit`` / ``first_treat`` contract from ``bacon_decompose``."""
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            first_treat="first_treat",
            # intentionally omit outcome / time / unit
        )
        assert "bacon" not in dr.applicable_checks
        reason = dr.skipped_checks["bacon"]
        assert "outcome" in reason or "time" in reason or "unit" in reason

    def test_multiperiod_runs_pt_and_power_and_sensitivity(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        applicable = set(dr.applicable_checks)
        assert "parallel_trends" in applicable
        assert "pretrends_power" in applicable
        assert "sensitivity" in applicable

    def test_cs_runs_heterogeneity(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        applicable = set(dr.applicable_checks)
        assert "heterogeneity" in applicable
        assert "bacon" in applicable
        assert "parallel_trends" in applicable

    def test_sdid_has_estimator_native(self, sdid_fit):
        fit, _ = sdid_fit
        dr = DiagnosticReport(fit)
        assert "estimator_native" in dr.applicable_checks

    def test_run_opt_outs_move_checks_to_skipped(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit, run_sensitivity=False)
        assert "sensitivity" not in dr.applicable_checks
        assert dr.skipped_checks["sensitivity"].startswith("run_sensitivity=False")

    def test_placebo_is_reserved_and_skipped(self, did_fit):
        """Placebo is always in _CHECK_NAMES, always skipped in MVP."""
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        placebo_section = dr.to_dict()["placebo"]
        assert placebo_section["status"] in {"skipped", "not_applicable"}


# ---------------------------------------------------------------------------
# Precomputed passthrough
# ---------------------------------------------------------------------------
class TestPrecomputed:
    def test_precomputed_sensitivity_is_used_verbatim(self, multi_period_fit):
        fit, _ = multi_period_fit

        # Construct a minimal SensitivityResults-shaped object the formatter recognizes.
        class _FakeSens:
            M_values = np.array([0.5, 1.0])
            bounds = [(0.1, 2.0), (-0.2, 2.5)]
            robust_cis = [(0.05, 2.1), (-0.3, 2.6)]
            breakdown_M = 0.75
            method = "relative_magnitude"
            original_estimate = 1.0
            original_se = 0.2
            alpha = 0.05

        fake = _FakeSens()
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            dr = DiagnosticReport(fit, precomputed={"sensitivity": fake})
            dr.to_dict()
            mock.assert_not_called()
        schema = dr.to_dict()
        assert schema["sensitivity"]["status"] == "ran"
        assert schema["sensitivity"]["breakdown_M"] == 0.75

    def test_precomputed_pretrends_power_parity_with_default_path(self, cs_fit):
        """Round-20 P1 regression: ``precomputed={"pretrends_power": ...}``
        must apply the same covariance-source annotation and conservative
        diagonal-fallback downgrade as ``_check_pretrends_power``. Otherwise
        the same fit can be labeled ``well_powered`` through the precomputed
        path and ``moderately_powered`` through the default path.
        """
        from diff_diff.pretrends import compute_pretrends_power

        fit, data = cs_fit

        # Precompute the power result from the same fit. The compute function
        # populates ``original_results`` on the output so DR's precomputed
        # adapter can inspect the source fit's event_study_vcov.
        pp = compute_pretrends_power(fit, alpha=0.05, target_power=0.80, violation_type="linear")
        assert getattr(pp, "original_results", None) is fit

        dr_default = DiagnosticReport(fit, data=data).to_dict()
        dr_precomputed = DiagnosticReport(
            fit, data=data, precomputed={"pretrends_power": pp}
        ).to_dict()

        default_block = dr_default["pretrends_power"]
        precomp_block = dr_precomputed["pretrends_power"]

        # Both paths are "ran"; the precomputed path flags itself with
        # ``precomputed=True`` while the default path sets ``method=
        # compute_pretrends_power``.
        assert default_block["status"] == "ran"
        assert precomp_block["status"] == "ran"
        assert precomp_block.get("precomputed") is True

        # Tier and covariance_source must agree across paths so downstream
        # BR prose does not diverge based on which path produced the block.
        assert default_block["tier"] == precomp_block["tier"]
        assert default_block["covariance_source"] == precomp_block["covariance_source"]

    def test_precomputed_pretrends_power_downgrades_when_full_vcov_unused(self):
        """Stub-based regression: when the source fit has both
        ``event_study_vcov`` and ``event_study_vcov_index`` populated but
        the diagonal fallback was used, the precomputed adapter must emit
        ``covariance_source='diag_fallback_available_full_vcov_unused'`` and
        downgrade a ``well_powered`` tier to ``moderately_powered`` — just
        like the default compute path. Complements the live-fit parity test
        by exercising the tier-bumping edge explicitly.
        """

        # Minimal CS-shaped stub with full vcov flagged.
        class _CSStub:
            overall_att = 1.0
            overall_se = 0.25
            overall_t_stat = 4.0
            overall_p_value = 0.001
            overall_conf_int = (0.5, 1.5)
            alpha = 0.05
            n_obs = 400
            n_treated = 80
            n_control = 320
            survey_metadata = None
            event_study_effects = None
            event_study_vcov = np.eye(3)
            event_study_vcov_index = {-2: 0, -1: 1, 0: 2}

        stub = _CSStub()
        stub.__class__.__name__ = "CallawaySantAnnaResults"

        class _PPStub:
            mdv = 0.1  # |ATT| = 1.0 -> ratio = 0.1 -> well_powered before downgrade
            violation_type = "linear"
            alpha = 0.05
            target_power = 0.80
            violation_magnitude = 0.1
            power = 0.80
            n_pre_periods = 2
            original_results = stub

        dr = DiagnosticReport(stub, precomputed={"pretrends_power": _PPStub()})
        block = dr.to_dict()["pretrends_power"]
        assert block["status"] == "ran"
        assert block["covariance_source"] == "diag_fallback_available_full_vcov_unused"
        # Downgrade must apply: pre-tier is well_powered, post-tier is moderately_powered.
        assert block["tier"] == "moderately_powered"

    def test_precomputed_parallel_trends_bypasses_applicability_gate(self, cs_fit):
        """Round-22 P1 regression: ``precomputed["parallel_trends"]`` was
        documented as supported but ``_instance_skip_reason`` skipped the
        PT check on applicability grounds (missing raw panel / columns
        for the event-study replay, non-replayable EfficientDiD fits,
        etc.) BEFORE the precomputed runner could fire. The fix
        short-circuits the gate when the precomputed key is present so
        advertised passthroughs actually land on the runner.
        """
        fit, _ = cs_fit
        precomputed_pt = {
            "status": "ran",
            "method": "event_study",
            "joint_p_value": 0.42,
            "n_pre_periods": 3,
            "verdict": "no_detected_violation",
        }

        # Without passing ``data`` + column kwargs, the applicability
        # gate would previously have marked PT as skipped. With the
        # precomputed override, it must land on the formatter instead.
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": precomputed_pt})
        pt_block = dr.to_dict()["parallel_trends"]
        assert pt_block["status"] == "ran", (
            f"precomputed parallel_trends must bypass the applicability gate. "
            f"Got status={pt_block.get('status')}, reason={pt_block.get('reason')}"
        )

    def test_precomputed_parallel_trends_preserves_schema_shaped_joint_p(self, cs_fit):
        """Round-23 P1 regression: schema-shaped PT dicts with
        ``joint_p_value`` (the key emitted by the default DR path and
        the shape users are most likely to replay from one DR to
        another) must land on ``joint_p_value`` in the output, not
        silently fall through to ``None``. Prior formatter read only
        ``p_value``, so a dict with ``joint_p_value=0.42`` was
        degraded to ``joint_p_value=None`` / ``verdict="inconclusive"``.
        """
        fit, _ = cs_fit
        dr = DiagnosticReport(
            fit,
            precomputed={
                "parallel_trends": {
                    "joint_p_value": 0.42,
                    "test_statistic": 5.6,
                    "df": 3,
                    "method": "hausman",
                }
            },
        )
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "hausman"
        assert pt["joint_p_value"] == 0.42, (
            f"joint_p_value must survive formatting; got {pt.get('joint_p_value')}"
        )
        assert pt["test_statistic"] == 5.6
        assert pt["df"] == 3
        # Verdict must be derived from the surviving p-value, not None.
        assert pt["verdict"] != "inconclusive"

    def test_precomputed_parallel_trends_accepts_native_hausman_result(self, cs_fit):
        """Round-23 P1 regression: ``_pt_hausman`` tells users with
        non-replayable EfficientDiD fits to pass a precomputed pretest
        result, but the formatter previously rejected non-dict inputs
        outright. The ``HausmanPretestResult`` dataclass — the exact
        object ``EfficientDiD.hausman_pretest(...)`` returns — must
        now pass through with ``statistic`` / ``p_value`` / ``df``
        preserved on the schema.
        """
        from types import SimpleNamespace

        fit, _ = cs_fit
        # Mirror HausmanPretestResult: the key fields are ``statistic``,
        # ``p_value``, ``df``. Uses SimpleNamespace so the test does
        # not need EfficientDiD's construction path.
        hausman = SimpleNamespace(
            statistic=7.2,
            p_value=0.065,
            df=3,
            reject=False,
            alpha=0.05,
            att_all=1.0,
            att_post=1.05,
            recommendation="pt_all",
        )
        dr = DiagnosticReport(fit, precomputed={"parallel_trends": hausman})
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran", (
            f"Native HausmanPretestResult must be accepted; got "
            f"status={pt.get('status')}, reason={pt.get('reason')}"
        )
        assert pt["joint_p_value"] == 0.065
        # ``statistic`` on the source object maps to ``test_statistic``
        # in the emitted schema (matches the default ``_pt_hausman``
        # path that also exposes it as ``test_statistic``).
        assert pt["test_statistic"] == 7.2
        assert pt["df"] == 3

    def test_precomputed_parallel_trends_rejects_input_without_p_value(self, cs_fit):
        """Inputs without any recognized p-value field (neither
        ``joint_p_value`` nor ``p_value``) must surface a clear error,
        not silently land on ``joint_p_value=None``. Keeps the formatter
        permissive about absent ``test_statistic`` / ``df`` (2x2 PT has
        neither) while catching obviously-wrong inputs.
        """
        fit, _ = cs_fit
        dr = DiagnosticReport(
            fit, precomputed={"parallel_trends": {"method": "event_study"}}
        )
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "error"
        assert "joint_p_value" in pt["reason"] or "p_value" in pt["reason"]

    def test_precomputed_bacon_bypasses_applicability_gate(self, cs_fit):
        """Round-22 P1 regression: ``precomputed["bacon"]`` was
        documented as supported but ``_instance_skip_reason`` skipped
        Bacon on applicability grounds (``data`` / column kwargs missing)
        before the runner could fire. Users with an already-computed
        ``BaconDecompositionResults`` must be able to pass it through
        without re-supplying the raw panel.
        """
        from types import SimpleNamespace

        fit, _ = cs_fit
        precomputed_bacon = SimpleNamespace(
            weights=None,
            att=1.2,
            comparison_types={},
            total_weight_later_vs_earlier=0.02,
        )

        dr = DiagnosticReport(fit, precomputed={"bacon": precomputed_bacon})
        bacon_block = dr.to_dict()["bacon"]
        assert bacon_block["status"] == "ran", (
            f"precomputed bacon must bypass the applicability gate. "
            f"Got status={bacon_block.get('status')}, reason={bacon_block.get('reason')}"
        )

    def test_precomputed_single_m_sensitivity_exposes_original_estimate_and_se(self, cs_fit):
        """Pre-emptive audit regression: ``_format_precomputed_sensitivity``
        used to drop ``original_estimate`` and ``original_se`` on the
        single-M ``HonestDiDResults`` branch, even though both
        ``SensitivityResults`` and ``HonestDiDResults`` carry those fields.
        The grid branch surfaces them via ``_format_sensitivity_results``,
        so dropping them on the single-M branch made the schema shape
        dependent on which object type the user passed. Parity fix: the
        single-M branch now carries the same fields.
        """
        from types import SimpleNamespace

        fit, _ = cs_fit
        single_m = SimpleNamespace(
            lb=0.3,
            ub=1.8,
            ci_lb=0.15,
            ci_ub=1.95,
            M=1.0,
            method="relative_magnitude",
            original_estimate=1.05,
            original_se=0.22,
            alpha=0.05,
        )

        block = DiagnosticReport(
            fit, precomputed={"sensitivity": single_m}
        ).to_dict()["sensitivity"]
        assert block["status"] == "ran"
        assert block["conclusion"] == "single_M_precomputed"
        # Parity with the grid branch: these fields must be present and
        # reflect the passed object's values.
        assert block["original_estimate"] == 1.05
        assert block["original_se"] == 0.22


# ---------------------------------------------------------------------------
# Verdict / tier helpers
# ---------------------------------------------------------------------------
class TestJointWaldAlignment:
    """Cover the event-study PT joint-Wald vs Bonferroni fallback paths.

    These tests address the correctness-sensitive codepath in
    ``_pt_event_study`` where pre-period coefficient keys must align with
    ``interaction_indices`` before the joint Wald statistic can be indexed
    into the right vcov rows/columns. When alignment fails, the code must
    fall back to Bonferroni rather than compute a Wald statistic on the
    wrong rows.
    """

    @staticmethod
    def _stub_result(pre_effects, interaction_indices, vcov, **extra):
        """Build a minimal MultiPeriodDiDResults-shaped stub for PT tests.

        ``pre_effects`` is an iterable of ``(period_key, effect, se, p_value)``
        tuples. Returns an object whose class name is ``MultiPeriodDiDResults``
        so DR's name-keyed dispatch routes it to the event-study PT path.
        """
        from types import SimpleNamespace

        pre_map = {
            k: SimpleNamespace(effect=eff, se=se, p_value=p) for (k, eff, se, p) in pre_effects
        }

        class MultiPeriodDiDResults:  # noqa: D401 — test stub that mimics the real class name
            pass

        obj = MultiPeriodDiDResults()
        obj.pre_period_effects = pre_map
        obj.interaction_indices = interaction_indices
        obj.vcov = np.asarray(vcov, dtype=float) if vcov is not None else None
        obj.avg_att = 1.0
        obj.avg_se = 0.1
        obj.avg_p_value = 0.001
        obj.avg_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 50
        obj.n_control = 50
        obj.survey_metadata = None
        for k, v in extra.items():
            setattr(obj, k, v)
        return obj

    def test_joint_wald_runs_when_keys_align(self):
        """With aligned pre_effects + interaction_indices + vcov, Wald runs
        and the computed chi-squared statistic matches the closed form."""
        pre = [(-3, 0.0, 0.5, 0.99), (-2, 0.0, 0.5, 0.99), (-1, 0.0, 0.5, 0.99)]
        interaction_indices = {-3: 0, -2: 1, -1: 2, 0: 3}  # maps period -> vcov row
        vcov = np.diag([0.25, 0.25, 0.25, 0.25])  # SE = 0.5 for each pre-period
        stub = self._stub_result(pre, interaction_indices, vcov)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert (
            pt["method"] == "joint_wald"
        ), f"Expected joint_wald with aligned keys; got {pt.get('method')}"
        # beta=0 across all periods -> test_statistic = 0 -> p = 1.0
        assert pt["test_statistic"] == pytest.approx(0.0)
        assert pt["joint_p_value"] == pytest.approx(1.0)
        assert pt["df"] == 3

    def test_joint_wald_computes_expected_statistic(self):
        """Verify the Wald statistic matches a known closed-form value."""
        # beta = [1.0, -0.5, 0.2]; vcov diagonal with variances [0.25, 0.25, 0.16]
        # -> test_statistic = 1.0^2/0.25 + 0.5^2/0.25 + 0.2^2/0.16
        #                   = 4.0 + 1.0 + 0.25 = 5.25
        pre = [(-3, 1.0, 0.5, 0.04), (-2, -0.5, 0.5, 0.30), (-1, 0.2, 0.4, 0.61)]
        interaction_indices = {-3: 0, -2: 1, -1: 2}
        vcov = np.diag([0.25, 0.25, 0.16])
        stub = self._stub_result(pre, interaction_indices, vcov)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["method"] == "joint_wald"
        assert pt["test_statistic"] == pytest.approx(5.25, rel=1e-6)

    def test_falls_back_to_bonferroni_without_interaction_indices(self):
        pre = [(-2, 1.0, 0.5, 0.04), (-1, 0.2, 0.5, 0.69)]
        stub = self._stub_result(pre, interaction_indices=None, vcov=np.diag([0.25, 0.25]))
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "bonferroni", (
            "Missing interaction_indices must force Bonferroni fallback, "
            "never attempt a Wald statistic on misaligned rows."
        )
        # Bonferroni: min(per-period p) * n = 0.04 * 2 = 0.08 (< 1)
        assert pt["joint_p_value"] == pytest.approx(0.08, rel=1e-6)

    def test_falls_back_to_bonferroni_when_keys_misaligned(self):
        """pre_effects has keys [-2, -1] but interaction_indices uses [2019, 2020]."""
        pre = [(-2, 1.0, 0.5, 0.04), (-1, 0.2, 0.5, 0.69)]
        interaction_indices = {2019: 0, 2020: 1}  # deliberately different namespace
        vcov = np.diag([0.25, 0.25])
        stub = self._stub_result(pre, interaction_indices, vcov)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "bonferroni", (
            "Misaligned interaction_indices must force Bonferroni fallback — "
            "the len(keys_in_vcov) == df guard should prevent the Wald path."
        )

    def test_falls_back_to_bonferroni_when_vcov_missing(self):
        pre = [(-2, 1.0, 0.5, 0.04), (-1, 0.2, 0.5, 0.69)]
        interaction_indices = {-2: 0, -1: 1}
        stub = self._stub_result(pre, interaction_indices, vcov=None)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["method"] == "bonferroni"


class TestNarrowedApplicabilityAndPlaceboSchema:
    """Regressions for the round-3 CI-review findings.

    * ``pretrends_power`` and ``sensitivity`` are now restricted to the
      result families that their backing helpers actually support, so
      default reports no longer land in ``error`` for SA / Imputation /
      Stacked / EfficientDiD / StaggeredTripleDiff / Wooldridge.
    * ``placebo`` is always ``status="skipped"`` in MVP regardless of
      estimator, matching the ``REPORTING.md`` contract.
    """

    def test_placebo_is_always_skipped_not_not_applicable(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        placebo = dr.to_dict()["placebo"]
        assert placebo["status"] == "skipped", (
            f"placebo must always be status='skipped' per REPORTING.md; "
            f"got {placebo['status']!r}"
        )

    def test_placebo_skipped_for_multiperiod_fit(self, multi_period_fit):
        fit, _ = multi_period_fit
        placebo = DiagnosticReport(fit).to_dict()["placebo"]
        assert placebo["status"] == "skipped"

    def test_placebo_skipped_for_sdid_fit(self, sdid_fit):
        fit, _ = sdid_fit
        placebo = DiagnosticReport(fit).to_dict()["placebo"]
        assert placebo["status"] == "skipped"

    def test_sun_abraham_sensitivity_not_applicable(self):
        """SA is not in HonestDiD's adapter list; DR must not try to run it."""
        import warnings

        from diff_diff import SunAbraham, generate_staggered_data

        warnings.filterwarnings("ignore")
        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        fit = SunAbraham().fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        dr = DiagnosticReport(fit)
        applicable = set(dr.applicable_checks)
        sensitivity = dr.to_dict()["sensitivity"]
        assert "sensitivity" not in applicable, (
            "SunAbrahamResults has no HonestDiD adapter; sensitivity must not "
            "be marked applicable"
        )
        assert sensitivity["status"] == "not_applicable"

    def test_n_obs_zero_reference_marker_filtered(self):
        """Stacked / TwoStage / Imputation reference markers use n_obs=0
        (not n_groups=0). ``_collect_pre_period_coefs`` must filter both."""
        import numpy as np

        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        class StackedDiDResults:
            pass

        obj = StackedDiDResults()
        obj.event_study_effects = {
            -2: {"effect": 0.1, "se": 0.3, "p_value": 0.74, "n_obs": 50},
            -1: {
                "effect": 0.0,
                "se": np.nan,
                "p_value": np.nan,
                "n_obs": 0,  # synthetic reference marker
            },
            0: {"effect": 1.5, "se": 0.2, "p_value": 0.0001, "n_obs": 50},
        }
        coefs = _collect_pre_period_coefs(obj)
        keys = [k for (k, _, _, _) in coefs]
        assert -1 not in keys, "n_obs==0 row must be filtered out"
        assert -2 in keys


class TestReferenceMarkerAndNaNFiltering:
    """Regression for the P0 finding that reference markers + NaN pre-periods
    were being swept into Bonferroni / Wald PT as real evidence.

    Universal-base CS / SA / ImputationDiD / Stacked event-study output
    injects a synthetic reference-period row (``effect=0``, ``se=NaN``,
    ``p_value=NaN``, ``n_groups=0``). Treating that row as valid
    pre-period evidence would inflate the Bonferroni denominator and
    collapse all-NaN fallbacks to a false-clean verdict.
    """

    @staticmethod
    def _cs_stub_with_reference_marker():
        import numpy as np

        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        # Two real pre-period rows + one universal-base reference marker (n_groups=0).
        obj.event_study_effects = {
            -3: {"effect": 0.1, "se": 0.3, "p_value": 0.74, "n_groups": 5},
            -2: {"effect": -0.2, "se": 0.3, "p_value": 0.51, "n_groups": 5},
            -1: {
                "effect": 0.0,
                "se": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "n_groups": 0,
            },
            0: {"effect": 1.5, "se": 0.2, "p_value": 0.0001, "n_groups": 5},
        }
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None
        return obj

    def test_reference_marker_excluded_from_pt_collection(self):
        from diff_diff.diagnostic_report import _collect_pre_period_coefs

        obj = self._cs_stub_with_reference_marker()
        coefs = _collect_pre_period_coefs(obj)
        keys = [k for (k, _, _, _) in coefs]
        assert -1 not in keys, (
            "Universal-base reference marker (n_groups=0) must not appear "
            "as a valid pre-period coefficient"
        )
        assert -3 in keys and -2 in keys
        # Every returned SE must be finite.
        for _k, _eff, se, _p in coefs:
            assert np.isfinite(se), f"Non-finite SE leaked through: {se}"

    def test_all_nan_pre_periods_do_not_produce_clean_verdict(self):
        """If *every* pre-period row is a reference marker / NaN, the PT
        check must return inconclusive / skipped — never a clean p_value=1.0.
        """
        import numpy as np

        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        obj.event_study_effects = {
            -1: {
                "effect": 0.0,
                "se": np.nan,
                "p_value": np.nan,
                "n_groups": 0,
            },
            0: {"effect": 1.5, "se": 0.2, "p_value": 0.0001, "n_groups": 5},
        }
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None
        dr = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        # All pre-period rows were reference markers → no valid data → skipped.
        assert pt["status"] == "skipped"
        # Verdict must not falsely say "no detected violation" when the only
        # "data" was a reference marker.
        assert pt.get("verdict") != "no_detected_violation"

    def test_bonferroni_excludes_nan_p_values(self):
        """If a pre-period row has a finite effect/SE but NaN p-value (edge
        case on some exotic fits), Bonferroni must skip it, not feed it in."""
        import numpy as np

        class MultiPeriodDiDResults:
            pass

        from types import SimpleNamespace

        obj = MultiPeriodDiDResults()
        obj.pre_period_effects = {
            -2: SimpleNamespace(effect=1.0, se=0.5, p_value=0.04),
            -1: SimpleNamespace(effect=0.5, se=0.5, p_value=np.nan),
        }
        obj.vcov = None
        obj.interaction_indices = None
        obj.event_study_vcov = None
        obj.event_study_vcov_index = None
        obj.avg_att = 1.0
        obj.avg_se = 0.1
        obj.avg_p_value = 0.001
        obj.avg_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 50
        obj.n_control = 50
        obj.survey_metadata = None

        dr = DiagnosticReport(obj, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        # With only one valid p-value (0.04), Bonferroni should be min(1.0, 0.04*1) = 0.04.
        # If the NaN were naively included the test would either error or coerce to 1.0.
        assert pt["method"] == "bonferroni"
        assert pt["joint_p_value"] == pytest.approx(0.04, abs=1e-9)


class TestPrecomputedValidation:
    """Regression for the P1 finding that ``precomputed=`` silently accepted
    keys that were never implemented. Unsupported keys now raise."""

    def test_unsupported_precomputed_key_raises(self, multi_period_fit):
        fit, _ = multi_period_fit
        with pytest.raises(ValueError, match="not implemented"):
            DiagnosticReport(fit, precomputed={"design_effect": object()})

    def test_supported_precomputed_keys_accepted(self, multi_period_fit):
        fit, _ = multi_period_fit
        # The four implemented keys should not raise at construction.
        DiagnosticReport(fit, precomputed={"parallel_trends": {"p_value": 0.5}})

    def test_mixed_supported_and_unsupported_raises(self, multi_period_fit):
        fit, _ = multi_period_fit
        with pytest.raises(ValueError, match="epv"):
            DiagnosticReport(fit, precomputed={"sensitivity": None, "epv": object()})


class TestSingleMSensitivityPrecomputed:
    """Single-M HonestDiDResults must NOT be narrated as full-grid robustness.

    Regression for the P0 CI-review finding that ``conclusion='single_M_precomputed'``
    was being swallowed because both renderers checked ``breakdown_M is None`` and
    fell through to the "robust across the full grid" phrasing.
    """

    def _fake_single_m(self, M=1.5, ci_lb=1.0, ci_ub=3.0):
        from types import SimpleNamespace

        return SimpleNamespace(
            M=M,
            lb=ci_lb,
            ub=ci_ub,
            ci_lb=ci_lb,
            ci_ub=ci_ub,
            method="relative_magnitude",
            alpha=0.05,
        )

    def test_dr_schema_preserves_single_m_marker(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit, precomputed={"sensitivity": self._fake_single_m()})
        sens = dr.to_dict()["sensitivity"]
        assert sens["status"] == "ran"
        assert sens["conclusion"] == "single_M_precomputed"
        assert sens["breakdown_M"] is None
        assert len(sens["grid"]) == 1

    def test_dr_summary_does_not_claim_full_grid_robustness(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit, precomputed={"sensitivity": self._fake_single_m()})
        summary = dr.summary()
        assert "across the entire HonestDiD grid" not in summary
        assert "robust across the grid" not in summary
        # It should narrate the single-M check honestly.
        assert "single point checked" in summary
        assert "not a breakdown" in summary or "not a grid" in summary

    def test_br_summary_does_not_claim_full_grid_robustness(self, multi_period_fit):
        """BR via honest_did_results= passthrough must not oversell a point check."""
        from diff_diff import BusinessReport

        fit, _ = multi_period_fit
        br = BusinessReport(fit, honest_did_results=self._fake_single_m())
        summary = br.summary()
        assert "full grid" not in summary
        assert "single point checked" in summary


class TestEPVDictBacked:
    """EPV diagnostics on fits that use the dict-of-dicts convention.

    Regression for the P0 CI-review finding that ``_check_epv`` assumed
    ``low_epv_cells`` / ``min_epv`` attributes but the library stores
    ``epv_diagnostics`` as ``{(g, t): {"is_low": ..., "epv": ...}}``.
    """

    def _make_cs_stub(self, epv_diag, threshold=10.0):
        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.epv_diagnostics = epv_diag
        obj.epv_threshold = threshold
        return obj

    def test_low_epv_cells_counted_from_is_low_flag(self):
        epv = {
            (2020, 1): {"is_low": True, "epv": 4.5},
            (2020, 2): {"is_low": False, "epv": 18.0},
            (2021, 1): {"is_low": True, "epv": 2.0},
            (2021, 2): {"is_low": False, "epv": 22.0},
        }
        stub = self._make_cs_stub(epv, threshold=10.0)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        section = dr.to_dict()["epv"]
        assert section["status"] == "ran"
        assert section["n_cells_low"] == 2
        assert section["n_cells_total"] == 4
        assert section["min_epv"] == pytest.approx(2.0)
        assert section["threshold"] == pytest.approx(10.0)

    def test_no_low_cells_reports_clean(self):
        epv = {(2020, 1): {"is_low": False, "epv": 15.0}}
        stub = self._make_cs_stub(epv, threshold=10.0)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        section = dr.to_dict()["epv"]
        assert section["n_cells_low"] == 0
        assert section["min_epv"] == pytest.approx(15.0)

    def test_threshold_read_from_results_not_hardcoded(self):
        """Pass a non-default epv_threshold and confirm DR echoes it."""
        epv = {(2020, 1): {"is_low": True, "epv": 7.0}}
        stub = self._make_cs_stub(epv, threshold=8.5)
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        assert dr.to_dict()["epv"]["threshold"] == pytest.approx(8.5)


class TestCSEventStudyVCovSupport:
    """CS sensitivity + pretrends_power must not be skipped for absence of results.vcov.

    Regression for the P1 CI-review finding that the applicability gate required
    ``results.vcov`` but CS exposes ``event_study_vcov`` / ``event_study_vcov_index``.
    """

    def test_cs_sensitivity_runs_on_aggregated_fit(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert (
            "sensitivity" in dr.applicable_checks
        ), "CS fit with event_study aggregation must not skip sensitivity"
        sens = dr.to_dict()["sensitivity"]
        # It may run successfully or emit an error depending on data shape,
        # but it must NOT be skipped for "results.vcov not available".
        assert sens["status"] in {"ran", "error"}, sens

    def test_cs_pretrends_power_runs_on_aggregated_fit(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert (
            "pretrends_power" in dr.applicable_checks
        ), "CS fit with event_study aggregation must not skip pretrends_power"


class TestCSJointWaldViaEventStudyVCov:
    """CS PT should use joint_wald via event_study_vcov when interaction_indices is absent.

    Regression for the P1 CI-review finding that CS always fell back to Bonferroni
    even though ``event_study_vcov`` + ``event_study_vcov_index`` were available.
    """

    def _make_cs_stub_with_es_vcov(self):
        class CallawaySantAnnaResults:
            pass

        obj = CallawaySantAnnaResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 200
        obj.n_treated = 40
        obj.n_control = 160
        obj.survey_metadata = None
        # Pre-period event-study entries with known coefficients + vcov.
        obj.event_study_effects = {
            -3: {"effect": 0.5, "se": 0.5, "p_value": 0.32},
            -2: {"effect": -0.5, "se": 0.5, "p_value": 0.32},
            -1: {"effect": 0.2, "se": 0.4, "p_value": 0.62},
            0: {"effect": 2.0, "se": 0.3, "p_value": 0.0001},
            1: {"effect": 2.5, "se": 0.3, "p_value": 0.0001},
        }
        obj.event_study_vcov = np.diag([0.25, 0.25, 0.16, 0.09, 0.09])
        obj.event_study_vcov_index = [-3, -2, -1, 0, 1]
        obj.vcov = None  # CS convention
        obj.interaction_indices = None
        return obj

    def test_cs_pt_uses_event_study_vcov_wald(self):
        stub = self._make_cs_stub_with_es_vcov()
        dr = DiagnosticReport(stub, run_sensitivity=False, run_bacon=False)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert (
            pt["method"] == "joint_wald_event_study"
        ), f"Expected event-study-backed Wald; got method={pt.get('method')!r}"
        # Closed-form: 0.5^2/0.25 + (-0.5)^2/0.25 + 0.2^2/0.16 = 1 + 1 + 0.25 = 2.25
        assert pt["test_statistic"] == pytest.approx(2.25, rel=1e-6)
        assert pt["df"] == 3


class TestContinuousDiDHeadline:
    """ContinuousDiDResults exposes overall_att_se/p_value/conf_int, not overall_se/…

    Regression for the P1 CI-review finding that both report classes missed
    ContinuousDiDResults inference fields.
    """

    def test_extract_scalar_headline_resolves_continuous_did_aliases(self):
        from diff_diff.diagnostic_report import _extract_scalar_headline

        class ContinuousDiDResults:
            pass

        obj = ContinuousDiDResults()
        obj.overall_att = 2.5
        obj.overall_att_se = 0.4
        obj.overall_att_p_value = 0.00001
        obj.overall_att_conf_int = (1.7, 3.3)
        obj.alpha = 0.05

        result = _extract_scalar_headline(obj)
        assert result is not None
        name, value, se, p, ci, alpha = result
        assert name == "overall_att"
        assert value == pytest.approx(2.5)
        assert se == pytest.approx(0.4)
        assert p == pytest.approx(0.00001)
        assert ci == [pytest.approx(1.7), pytest.approx(3.3)]
        assert alpha == pytest.approx(0.05)


class TestVerdictsAndTiers:
    def test_pt_verdict_three_bins(self):
        assert _pt_verdict(0.001) == "clear_violation"
        assert _pt_verdict(0.049) == "clear_violation"
        assert _pt_verdict(0.10) == "some_evidence_against"
        assert _pt_verdict(0.29) == "some_evidence_against"
        assert _pt_verdict(0.30) == "no_detected_violation"
        assert _pt_verdict(0.99) == "no_detected_violation"
        assert _pt_verdict(None) == "inconclusive"
        assert _pt_verdict(float("nan")) == "inconclusive"

    def test_power_tier_three_bins_plus_unknown(self):
        assert _power_tier(0.1) == "well_powered"
        assert _power_tier(0.24) == "well_powered"
        assert _power_tier(0.25) == "moderately_powered"
        assert _power_tier(0.99) == "moderately_powered"
        assert _power_tier(1.0) == "underpowered"
        assert _power_tier(5.0) == "underpowered"
        assert _power_tier(None) == "unknown"
        assert _power_tier(float("nan")) == "unknown"


# ---------------------------------------------------------------------------
# EfficientDiD hausman pathway
# ---------------------------------------------------------------------------
class TestEfficientDiDHausman:
    def test_hausman_pretest_runs_with_data_kwargs(self, edid_fit):
        fit, sdf = edid_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "hausman"

    def test_hausman_skipped_without_data_kwargs(self, edid_fit):
        """Without the raw panel kwargs, PT is now gated at the
        applicability level (round-10 CI review) — no method field on
        the skip section, but ``applicable_checks`` excludes
        ``parallel_trends`` and ``skipped_checks`` names it with the
        missing-kwargs reason."""
        fit, _ = edid_fit
        dr = DiagnosticReport(fit)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "skipped"
        assert "parallel_trends" not in dr.applicable_checks
        assert "parallel_trends" in dr.skipped_checks
        assert "hausman_pretest" in dr.skipped_checks["parallel_trends"]


# ---------------------------------------------------------------------------
# SDiD native
# ---------------------------------------------------------------------------
class TestSDiDNative:
    def test_sdid_pt_uses_synthetic_fit_method(self, sdid_fit):
        fit, _ = sdid_fit
        pt = DiagnosticReport(fit).to_dict()["parallel_trends"]
        assert pt["method"] == "synthetic_fit"
        assert pt["verdict"] == "design_enforced_pt"
        assert isinstance(pt.get("pre_treatment_fit_rmse"), float)

    def test_sdid_native_section_populated(self, sdid_fit):
        fit, _ = sdid_fit
        native = DiagnosticReport(fit).to_dict()["estimator_native_diagnostics"]
        assert native["status"] == "ran"
        assert native["estimator"] == "SyntheticDiD"
        assert "weight_concentration" in native
        assert "in_time_placebo" in native
        assert "zeta_sensitivity" in native

    def test_sdid_does_not_call_honest_did(self, sdid_fit):
        """HonestDiD sensitivity should NOT run on SDiD (native path used instead)."""
        fit, _ = sdid_fit
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            DiagnosticReport(fit).to_dict()
            mock.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_sensitivity_error_does_not_break_report(self, multi_period_fit):
        """A failing diagnostic records its error in the section; the report still renders."""
        fit, _ = multi_period_fit

        def _raise(*args, **kwargs):
            raise RuntimeError("synthetic test failure")

        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis", side_effect=_raise):
            dr = DiagnosticReport(fit)
            schema = dr.to_dict()
            sens = schema["sensitivity"]
            assert sens["status"] == "error"
            assert "synthetic test failure" in sens["reason"]
            # Other sections still ran.
            assert schema["parallel_trends"]["status"] == "ran"


# ---------------------------------------------------------------------------
# Overall prose
# ---------------------------------------------------------------------------
class TestOverallInterpretation:
    def test_overall_interpretation_nonempty_for_fit(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        prose = dr.summary()
        assert isinstance(prose, str)
        assert len(prose) > 50  # a real paragraph

    def test_full_report_has_headers(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        md = dr.full_report()
        assert "# Diagnostic Report" in md
        assert "## Overall Interpretation" in md
        assert "## Parallel trends" in md
        assert "## HonestDiD sensitivity" in md


# ---------------------------------------------------------------------------
# Public result class
# ---------------------------------------------------------------------------
class TestDiagnosticReportResults:
    def test_run_all_returns_dataclass(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        results = dr.run_all()
        assert isinstance(results, DiagnosticReportResults)
        assert isinstance(results.applicable_checks, tuple)
        assert isinstance(results.schema, dict)

    def test_run_all_is_idempotent(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        a = dr.run_all()
        b = dr.run_all()
        assert a is b  # cached


class TestDCDHParallelTrendsViaPlaceboEventStudy:
    """Regression for the round-6 P1 finding that dCDH was advertised as
    PT-applicable but ``_collect_pre_period_coefs`` never read
    ``placebo_event_study``, so the PT check was silently skipped even
    on fits with valid placebo horizons.
    """

    def _stub(self, with_placebo: bool):
        class ChaisemartinDHaultfoeuilleResults:
            pass

        stub = ChaisemartinDHaultfoeuilleResults()
        stub.att = 1.0
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.event_study_effects = None
        if with_placebo:
            stub.placebo_event_study = {
                -3: {
                    "effect": 0.05,
                    "se": 0.1,
                    "p_value": 0.62,
                    "conf_int": (-0.15, 0.25),
                    "n_obs": 40,
                },
                -2: {
                    "effect": -0.08,
                    "se": 0.09,
                    "p_value": 0.38,
                    "conf_int": (-0.26, 0.10),
                    "n_obs": 45,
                },
                -1: {
                    "effect": 0.04,
                    "se": 0.10,
                    "p_value": 0.69,
                    "conf_int": (-0.16, 0.24),
                    "n_obs": 50,
                },
            }
        else:
            stub.placebo_event_study = None
        return stub

    def test_pt_check_reads_placebo_event_study(self):
        stub = self._stub(with_placebo=True)
        dr = DiagnosticReport(stub).run_all()
        pt = dr.schema["parallel_trends"]
        assert (
            pt["status"] == "ran"
        ), f"dCDH PT check must run on a fit with placebo_event_study; got {pt}"
        # Per-period rows should come from the placebo keys (negative horizons).
        per_period = pt.get("per_period") or pt.get("periods") or []
        assert per_period, "PT output must include per-period rows"
        periods = [row.get("period") for row in per_period]
        assert all(
            isinstance(p, int) and p < 0 for p in periods
        ), f"dCDH PT must use negative placebo horizons; got {periods}"

    def test_pt_check_skips_when_no_placebo_event_study(self):
        stub = self._stub(with_placebo=False)
        dr = DiagnosticReport(stub).run_all()
        pt = dr.schema["parallel_trends"]
        assert (
            pt["status"] == "skipped"
        ), f"dCDH PT must skip when placebo_event_study is missing; got {pt}"


class TestHeterogeneityPostTreatmentOnly:
    """Regression for the round-6 P1 finding that ``_check_heterogeneity``
    was mixing pre- and post-treatment coefficients into the CV / range /
    sign-consistency summary.
    """

    def test_collector_prefers_post_period_effects_over_period_effects(self):
        """On a MultiPeriod-shaped stub, ``_collect_effect_scalars`` must read
        ``post_period_effects`` (post-treatment only), not ``period_effects``
        (which mixes pre- and post-treatment coefficients). If the pre-period
        value leaked in, sign_consistency would flip and the range would span
        a much larger interval."""
        from diff_diff.diagnostic_report import DiagnosticReport

        class MultiPeriodDiDResults:
            pass

        stub = MultiPeriodDiDResults()
        pe_pre = type("PeriodEffect", (), {"effect": -1.0, "se": 0.2})()
        pe_post_1 = type("PeriodEffect", (), {"effect": 1.0, "se": 0.2})()
        pe_post_2 = type("PeriodEffect", (), {"effect": 3.0, "se": 0.2})()
        stub.period_effects = {-1: pe_pre, 0: pe_post_1, 1: pe_post_2}
        stub.post_period_effects = {0: pe_post_1, 1: pe_post_2}
        stub.pre_period_effects = {-1: pe_pre}
        stub.avg_att = 2.0
        stub.avg_se = 0.1
        stub.avg_p_value = 0.001
        stub.avg_conf_int = (1.8, 2.2)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None

        # Bypass the applicability-matrix gate by constructing the report
        # object and calling the extractor directly: the fix is in the
        # extractor, and MultiPeriod's applicability matrix may or may
        # not include heterogeneity at any given release.
        dr = DiagnosticReport(stub)
        effects = sorted(dr._collect_effect_scalars())
        assert effects == [1.0, 3.0], (
            f"Extractor must return only post-treatment effects "
            f"(no pre-period -1.0); got {effects}"
        )
        assert dr._heterogeneity_source() == "post_period_effects"

    def test_event_study_filters_pre_period_and_reference_markers(self):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        # Event study: pre horizons (rel<0), reference marker (n_groups=0),
        # non-finite row, and two valid post rows.
        stub.event_study_effects = {
            -2: {"effect": -3.0, "se": 0.2, "n_groups": 15},
            -1: {"effect": 0.0, "se": float("nan"), "n_groups": 0},  # reference marker
            0: {"effect": 1.0, "se": 0.2, "n_groups": 15},
            1: {"effect": 2.0, "se": 0.2, "n_groups": 12},
            2: {"effect": float("nan"), "se": 0.2, "n_groups": 5},  # non-finite
        }
        stub.overall_att = 1.5
        stub.overall_se = 0.1
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (1.3, 1.7)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.base_period = "universal"

        dr = DiagnosticReport(
            stub,
            run_parallel_trends=False,
            run_sensitivity=False,
            run_bacon=False,
        ).run_all()
        het = dr.schema["heterogeneity"]
        assert het["status"] == "ran"
        assert het["source"] == "event_study_effects_post"
        # Only rel>=0, finite, non-reference rows: {1.0, 2.0}.
        assert het["n_effects"] == 2
        assert het["min"] == pytest.approx(1.0)
        assert het["max"] == pytest.approx(2.0)
        assert het["sign_consistent"] is True


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------
def test_public_api_exports():
    for name in ("DiagnosticReport", "DiagnosticReportResults", "DIAGNOSTIC_REPORT_SCHEMA_VERSION"):
        assert hasattr(dd, name), f"diff_diff must export {name}"
