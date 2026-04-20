"""Tests for the BR/DR ``target_parameter`` block (gap #6).

Covers:

- Per-estimator dispatch in ``describe_target_parameter`` emits the
  expected ``aggregation`` tag and non-empty prose for each of the 16
  result classes in ``_APPLICABILITY``.
- Fit-time config reads are honored:
  - ``EfficientDiDResults.pt_assumption`` branches the tag between
    ``pt_all_combined`` and ``pt_post_single_baseline``.
  - ``StackedDiDResults.clean_control`` varies the ``definition``
    clause (never_treated vs strict vs not_yet_treated).
  - ``ChaisemartinDHaultfoeuilleResults.L_max`` + ``covariate_residuals``
    + ``linear_trends_effects`` branches the dCDH estimand tag.
- BR and DR emit identical ``target_parameter`` blocks (cross-surface
  parity).
- Exhaustiveness: every result-class name in ``_APPLICABILITY`` gets
  a non-default, non-empty target-parameter block.
- Prose renders the target-parameter sentence in BR's summary + full
  report and DR's overall_interpretation + full report.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from diff_diff._reporting_helpers import describe_target_parameter
from diff_diff.diagnostic_report import _APPLICABILITY


def _minimal_result(class_name: str, **attrs):
    """Build a minimal stub result object with the given class name."""
    cls = type(class_name, (), {})
    obj = cls()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class TestTargetParameterPerEstimator:
    """Dispatch table lookup for each of the 16 result classes."""

    def test_did_results(self):
        tp = describe_target_parameter(_minimal_result("DiDResults"))
        assert tp["aggregation"] == "2x2"
        assert tp["headline_attribute"] == "att"
        assert "ATT" in tp["name"]

    def test_multi_period_did_results(self):
        tp = describe_target_parameter(_minimal_result("MultiPeriodDiDResults"))
        assert tp["aggregation"] == "event_study"
        assert tp["headline_attribute"] == "avg_att"
        assert "event-study" in tp["name"].lower()

    def test_did_results_mentions_twfe(self):
        """``TwoWayFixedEffects.fit()`` returns ``DiDResults`` (verified in
        PR #347 R1 review), so the DiDResults branch must cover both
        the 2x2 DiD and TWFE interpretations. There is no separate
        ``TwoWayFixedEffectsResults`` class.
        """
        tp = describe_target_parameter(_minimal_result("DiDResults"))
        assert tp["aggregation"] == "2x2"
        assert "TWFE" in tp["name"] or "TWFE" in tp["definition"]

    def test_callaway_santanna(self):
        tp = describe_target_parameter(_minimal_result("CallawaySantAnnaResults"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        assert "ATT(g,t)" in tp["name"] or "ATT(g, t)" in tp["definition"]

    def test_sun_abraham(self):
        tp = describe_target_parameter(_minimal_result("SunAbrahamResults"))
        assert tp["aggregation"] == "iw"
        assert "interaction-weighted" in tp["name"].lower() or "IW" in tp["name"]

    def test_imputation(self):
        tp = describe_target_parameter(_minimal_result("ImputationDiDResults"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        assert (
            "imputation" in tp["name"].lower()
            or "BJS" in tp["reference"]
            or "Borusyak" in tp["reference"]
        )

    def test_two_stage(self):
        tp = describe_target_parameter(_minimal_result("TwoStageDiDResults"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        assert "Gardner" in tp["reference"] or "two-stage" in tp["name"].lower()

    def test_stacked(self):
        tp = describe_target_parameter(
            _minimal_result("StackedDiDResults", clean_control="not_yet_treated")
        )
        assert tp["aggregation"] == "stacked"
        assert tp["headline_attribute"] == "overall_att"
        assert "sub-experiment" in tp["definition"].lower()

    def test_wooldridge(self):
        tp = describe_target_parameter(_minimal_result("WooldridgeDiDResults"))
        assert tp["aggregation"] == "simple"
        assert tp["headline_attribute"] == "overall_att"
        assert "ETWFE" in tp["name"] or "ETWFE" in tp["definition"] or "ASF" in tp["name"]

    def test_efficient_did_pt_all(self):
        tp = describe_target_parameter(_minimal_result("EfficientDiDResults", pt_assumption="all"))
        assert tp["aggregation"] == "pt_all_combined"
        assert "PT-All" in tp["name"]

    def test_efficient_did_pt_post(self):
        tp = describe_target_parameter(_minimal_result("EfficientDiDResults", pt_assumption="post"))
        assert tp["aggregation"] == "pt_post_single_baseline"
        assert "PT-Post" in tp["name"]

    def test_continuous_did(self):
        tp = describe_target_parameter(_minimal_result("ContinuousDiDResults"))
        assert tp["aggregation"] == "dose_overall"
        # Definition must name both PT and SPT readings per plan-review CRITICAL #2.
        defn = tp["definition"]
        assert "PT" in defn and "SPT" in defn

    def test_triple_difference(self):
        tp = describe_target_parameter(_minimal_result("TripleDifferenceResults"))
        assert tp["aggregation"] == "ddd"
        assert tp["headline_attribute"] == "att"

    def test_staggered_triple_difference(self):
        tp = describe_target_parameter(_minimal_result("StaggeredTripleDiffResults"))
        assert tp["aggregation"] == "staggered_ddd"
        assert tp["headline_attribute"] == "overall_att"

    def test_dcdh_m(self):
        tp = describe_target_parameter(
            _minimal_result("ChaisemartinDHaultfoeuilleResults", L_max=None)
        )
        assert tp["aggregation"] == "M"
        assert "DID_M" in tp["name"]
        # R1 PR #347 review: headline lives in ``overall_att``, not
        # ``att``. Machine-readable field must match the raw attribute
        # downstream consumers should read.
        assert tp["headline_attribute"] == "overall_att"

    def test_dcdh_l(self):
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=None,
                linear_trends_effects=None,
            )
        )
        assert tp["aggregation"] == "l"
        assert "DID_l" in tp["name"]
        assert tp["headline_attribute"] == "overall_att"

    def test_dcdh_l_with_controls(self):
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=SimpleNamespace(),
                linear_trends_effects=None,
            )
        )
        assert tp["aggregation"] == "l_x"
        assert "DID^X_l" in tp["name"]

    def test_dcdh_l_with_trends(self):
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=2,
                covariate_residuals=None,
                linear_trends_effects={"foo": "bar"},
            )
        )
        assert tp["aggregation"] == "l_fd"
        assert "DID^{fd}_l" in tp["name"]

    def test_sdid(self):
        tp = describe_target_parameter(_minimal_result("SyntheticDiDResults"))
        assert tp["aggregation"] == "synthetic"
        assert "synthetic" in tp["name"].lower()

    def test_trop(self):
        tp = describe_target_parameter(_minimal_result("TROPResults"))
        assert tp["aggregation"] == "factor_model"
        assert "factor" in tp["name"].lower()


class TestTargetParameterFitConfigReads:
    """Parameterized fit-config-branching tests."""

    @pytest.mark.parametrize(
        "clean_control, expected_clause",
        [
            ("never_treated", "never-treated"),
            ("strict", "strictly untreated"),
            ("not_yet_treated", "not yet treated"),
        ],
    )
    def test_stacked_clean_control_branches_definition(self, clean_control, expected_clause):
        tp = describe_target_parameter(
            _minimal_result("StackedDiDResults", clean_control=clean_control)
        )
        assert tp["aggregation"] == "stacked"
        assert expected_clause in tp["definition"]

    @pytest.mark.parametrize(
        "pt_assumption, expected_tag",
        [("all", "pt_all_combined"), ("post", "pt_post_single_baseline")],
    )
    def test_efficient_did_pt_assumption_branches_tag(self, pt_assumption, expected_tag):
        tp = describe_target_parameter(
            _minimal_result("EfficientDiDResults", pt_assumption=pt_assumption)
        )
        assert tp["aggregation"] == expected_tag

    @pytest.mark.parametrize(
        "L_max, controls, trends, expected_tag",
        [
            (None, None, None, "M"),
            (2, None, None, "l"),
            (2, SimpleNamespace(), None, "l_x"),
            (2, None, {"foo": "bar"}, "l_fd"),
            (2, SimpleNamespace(), {"foo": "bar"}, "l_x_fd"),
        ],
    )
    def test_dcdh_config_branches_tag(self, L_max, controls, trends, expected_tag):
        tp = describe_target_parameter(
            _minimal_result(
                "ChaisemartinDHaultfoeuilleResults",
                L_max=L_max,
                covariate_residuals=controls,
                linear_trends_effects=trends,
            )
        )
        assert tp["aggregation"] == expected_tag


class TestTargetParameterCoversEveryResultClass:
    """Exhaustiveness guard (per plan-review MEDIUM #5): every result-
    class name in DR's ``_APPLICABILITY`` dict must get a non-default,
    non-empty target-parameter block."""

    def test_every_applicability_class_has_explicit_branch(self):
        missing = []
        for class_name in _APPLICABILITY:
            tp = describe_target_parameter(_minimal_result(class_name))
            if tp["aggregation"] == "unknown":
                missing.append(class_name)
        assert not missing, (
            "Every result class in _APPLICABILITY must have an explicit "
            f"describe_target_parameter branch. Missing: {missing}"
        )

    def test_every_applicability_class_has_nonempty_fields(self):
        for class_name in _APPLICABILITY:
            tp = describe_target_parameter(_minimal_result(class_name))
            assert tp["name"], f"{class_name}: name is empty"
            assert tp["definition"], f"{class_name}: definition is empty"
            assert tp["aggregation"], f"{class_name}: aggregation is empty"
            assert tp["headline_attribute"], f"{class_name}: headline_attribute is empty"
            assert tp["reference"], f"{class_name}: reference is empty"


class TestTargetParameterCrossSurfaceParity:
    """BR and DR emit identical target_parameter blocks (the shared
    helper is the single source of truth)."""

    def test_br_and_dr_emit_identical_target_parameter(self):
        from diff_diff import BusinessReport, DiagnosticReport

        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 500
        stub.n_treated = 100
        stub.n_control_units = 400
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.base_period = "universal"
        stub.inference_method = "analytical"

        br_tp = BusinessReport(stub, auto_diagnostics=False).to_dict()["target_parameter"]
        dr_tp = DiagnosticReport(stub).to_dict()["target_parameter"]
        assert br_tp == dr_tp


class TestTargetParameterProseRendering:
    """The short ``name`` must render in BR summary + DR overall_interpretation;
    the ``definition`` must render in BR full_report and DR full report."""

    def _cs_stub(self):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 500
        stub.n_treated = 100
        stub.n_control_units = 400
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.base_period = "universal"
        stub.inference_method = "analytical"
        return stub

    def test_br_summary_emits_target_parameter_name(self):
        from diff_diff import BusinessReport

        summary = BusinessReport(self._cs_stub(), auto_diagnostics=False).summary()
        assert "Target parameter:" in summary
        assert "overall ATT" in summary
        # But summary must not embed the full verbose definition —
        # stakeholder paragraph stays within the 6-10 sentence target.
        assert "event_study_effects" not in summary

    def test_br_full_report_emits_target_parameter_section(self):
        from diff_diff import BusinessReport

        md = BusinessReport(self._cs_stub(), auto_diagnostics=False).full_report()
        assert "## Target Parameter" in md
        assert "cohort-size-weighted" in md
        # Full report DOES carry the definition (structural context for
        # readers who scan the markdown).
        assert "event_study_effects" in md or "event-study" in md.lower()

    def test_dr_overall_interpretation_emits_target_parameter_sentence(self):
        from diff_diff import DiagnosticReport

        dr = DiagnosticReport(self._cs_stub()).run_all()
        prose = dr.interpretation
        assert "Target parameter:" in prose
        assert "overall ATT" in prose

    def test_dr_full_report_emits_target_parameter_section(self):
        from diff_diff import DiagnosticReport

        md = DiagnosticReport(self._cs_stub()).full_report()
        assert "## Target Parameter" in md
        assert "Aggregation tag:" in md
        assert "Headline attribute:" in md


class TestTargetParameterRealFitIntegration:
    """PR #347 R1 review P2: stub-based dispatch tests missed (a) the
    TWFE-returns-DiDResults mismatch, (b) the dCDH headline_attribute
    bug. Exercise real fits so these contracts are enforced end-to-end.
    """

    def test_twfe_fit_returns_did_results_branch(self):
        """``TwoWayFixedEffects.fit()`` returns ``DiDResults``, so the
        target-parameter block must be the DiD/TWFE-covering branch.
        Guards against reintroducing a dead-code
        ``TwoWayFixedEffectsResults`` branch.
        """
        import warnings

        from diff_diff import TwoWayFixedEffects, generate_did_data

        warnings.filterwarnings("ignore")
        df = generate_did_data(n_units=40, n_periods=4, seed=7)
        fit = TwoWayFixedEffects().fit(
            df, outcome="outcome", treatment="treated", time="post", unit="unit"
        )
        # Real TWFE fit returns DiDResults (no separate TWFE result class).
        assert type(fit).__name__ == "DiDResults"
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "2x2"
        # The DiDResults branch must name TWFE explicitly so the
        # description is source-faithful for both DiD and TWFE fits.
        assert "TWFE" in tp["name"] or "TWFE" in tp["definition"]

    def test_stacked_did_fit_headline_attribute_matches_real_estimand(self):
        """``StackedDiDResults.overall_att`` is the average of
        post-treatment event-study coefficients ``delta_h`` with
        delta-method SE (``stacked_did.py`` around line 541). Real-fit
        regression against the reviewer's R1 P1 wording catch.
        """
        import warnings

        from diff_diff import StackedDiD, generate_staggered_data

        warnings.filterwarnings("ignore")
        df = generate_staggered_data(n_units=60, n_periods=6, seed=13)
        fit = StackedDiD(kappa_pre=1, kappa_post=1).fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert type(fit).__name__ == "StackedDiDResults"
        tp = describe_target_parameter(fit)
        assert tp["headline_attribute"] == "overall_att"
        # R1 P1 fix: the definition must describe the actual estimand —
        # the average of post-treatment delta_h event-study coefficients.
        assert (
            "event-study" in tp["definition"].lower()
            or "delta_h" in tp["definition"]
            or "post-treatment" in tp["definition"].lower()
        )

    def _dcdh_reversible_panel(self, seed):
        """Build a minimal reversible-treatment panel that dCDH
        accepts (group / time / treatment columns, at least one
        switcher)."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        units = list(range(20))
        periods = list(range(6))
        rows = []
        for u in units:
            # Half of units switch from 0 -> 1 at period 3.
            for t in periods:
                d = 1 if (u < 10 and t >= 3) else 0
                y = d * 2.0 + rng.normal(0.0, 0.5)
                rows.append({"unit": u, "period": t, "treated": d, "outcome": y})
        return pd.DataFrame(rows)

    def test_dcdh_did_m_fit_headline_attribute_is_overall_att(self):
        """``ChaisemartinDHaultfoeuilleResults.overall_att`` holds the
        DID_M headline scalar (``chaisemartin_dhaultfoeuille_results.py``
        line ~357). R1 P1: previously ``headline_attribute="att"``
        pointed at a non-existent attribute.
        """
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=11)
        # DID_M regime: L_max not supplied.
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            group="unit",
            time="period",
            treatment="treated",
        )
        assert type(fit).__name__ == "ChaisemartinDHaultfoeuilleResults"
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "M"
        assert tp["headline_attribute"] == "overall_att"
        # Sanity: the attribute BR/DR points at actually exists on the
        # real fit object.
        assert hasattr(fit, tp["headline_attribute"]), (
            f"headline_attribute={tp['headline_attribute']!r} must name "
            "an attribute that actually exists on the real result object."
        )

    def test_dcdh_did_l_fit_headline_attribute_is_overall_att(self):
        """Same guard for the DID_l dynamic-horizon regime
        (``L_max >= 1``). Real-fit regression.
        """
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille

        warnings.filterwarnings("ignore")
        df = self._dcdh_reversible_panel(seed=12)
        fit = ChaisemartinDHaultfoeuille().fit(
            df,
            outcome="outcome",
            group="unit",
            time="period",
            treatment="treated",
            L_max=2,
        )
        tp = describe_target_parameter(fit)
        assert tp["aggregation"] == "l"
        assert tp["headline_attribute"] == "overall_att"
        assert hasattr(fit, tp["headline_attribute"])
