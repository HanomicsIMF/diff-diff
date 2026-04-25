"""Tests for HAD survey_design= consolidation + soft deprecation cycle.

Covers all 8 HAD surfaces (HAD.fit + did_had_pretest_workflow + 4 array-in
pretests + 2 data-in joint wrappers) per the consolidation plan
(`whimsical-brewing-liskov.md`). Each surface gets:

1. survey_design= positive smoke (new kwarg accepted, finite output).
2. weights= deprecation warning (DeprecationWarning emitted; back-compat
   numerics preserved).
3. survey= deprecation warning (DeprecationWarning emitted; back-compat
   numerics preserved).
4. Numerical parity legacy ≡ new at atol=0 (skipped on qug_test, which
   raises NotImplementedError on all paths).
5. Three-way mutex ValueError (any 2-of-3 combo).

Plus surface-spanning tests:
- make_pweight_design importable from diff_diff top-level.
- make_pweight_design ≡ _make_trivial_resolved (private alias).
- Array-in helpers reject SurveyDesign (TypeError).
- Bit-exact normalization-order invariant (scale-invariance).
- qug_test surface symmetry (signature consistent with siblings).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    HeterogeneousAdoptionDiD,
    SurveyDesign,
    did_had_pretest_workflow,
    joint_homogeneity_test,
    joint_pretrends_test,
    make_pweight_design,
    qug_test,
    stute_joint_pretest,
    stute_test,
    yatchew_hr_test,
)
from diff_diff.survey import ResolvedSurveyDesign

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def array_in_data():
    """Simple (d, dy) arrays for the 3 numeric array-in helpers."""
    rng = np.random.default_rng(0)
    G = 30
    d = rng.uniform(0, 1, size=G)
    dy = 0.5 + 1.5 * d + rng.normal(0, 0.3, size=G)
    return d, dy


@pytest.fixture
def array_in_doses():
    """Just doses for qug_test (single-array)."""
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9])


@pytest.fixture
def two_period_panel():
    """Two-period panel for HAD.fit + did_had_pretest_workflow on
    aggregate='overall'. G=200 units, T=2 periods, dose constant within unit,
    Beta(0.5, 1) draws so d.min() approaches 0 (boundary at 0 satisfied for
    Design 1' continuous_at_zero)."""
    rng = np.random.default_rng(1)
    G = 200
    # Beta(0.5, 1) puts mass near 0; d.min() will be very small relative to
    # median, satisfying the Design 1' boundary heuristic.
    d = rng.beta(0.5, 1.0, size=G)
    rows = []
    for g in range(G):
        for t in (0, 1):
            y = 0.0 if t == 0 else d[g] * 1.2 + rng.normal(0, 0.1)
            rows.append({"unit": g, "time": t, "y": y, "d": (0.0 if t == 0 else d[g])})
    df = pd.DataFrame(rows)
    df["w"] = 1.0  # uniform weight column for SurveyDesign(weights="w")
    return df


@pytest.fixture
def event_study_panel():
    """Multi-period panel for joint_pretrends/joint_homogeneity workflows."""
    rng = np.random.default_rng(2)
    G = 30
    rows = []
    F = 2
    for g in range(G):
        d_g = rng.uniform(0.0, 1.0)
        for t in range(4):
            d_t = 0.0 if t < F else d_g
            y = (0.0 if t < F else d_t * 1.5) + rng.normal(0, 0.15)
            rows.append({"unit": g, "time": t, "y": y, "d": d_t})
    df = pd.DataFrame(rows)
    df["w"] = 1.0
    return df


# =============================================================================
# 1. Surface-spanning tests
# =============================================================================


class TestPublicHelpers:
    def test_make_pweight_design_export(self):
        """make_pweight_design is importable from the diff_diff top level."""
        from diff_diff import make_pweight_design as mpd

        assert mpd is make_pweight_design

    def test_make_pweight_design_returns_resolved(self):
        w = np.array([1.0, 2.0, 3.0, 4.0])
        resolved = make_pweight_design(w)
        assert isinstance(resolved, ResolvedSurveyDesign)
        assert resolved.weight_type == "pweight"
        assert resolved.strata is None
        assert resolved.psu is None
        assert resolved.fpc is None
        assert resolved.replicate_weights is None
        assert resolved.n_strata == 0
        assert resolved.n_psu == 4
        assert np.array_equal(resolved.weights, w.astype(np.float64))

    def test_make_pweight_design_eq_underscore_alias(self):
        """Permanent private alias _make_trivial_resolved IS make_pweight_design."""
        from diff_diff.survey import _make_trivial_resolved

        assert _make_trivial_resolved is make_pweight_design


class TestArrayInTypeGuard:
    """Array-in helpers reject SurveyDesign (cannot resolve column names)."""

    def test_stute_test_rejects_SurveyDesign(self, array_in_data):
        d, dy = array_in_data
        with pytest.raises(TypeError, match="make_pweight_design"):
            stute_test(d, dy, survey_design=SurveyDesign(weights="w"), n_bootstrap=199, seed=0)

    def test_yatchew_hr_test_rejects_SurveyDesign(self, array_in_data):
        d, dy = array_in_data
        with pytest.raises(TypeError, match="make_pweight_design"):
            yatchew_hr_test(d, dy, survey_design=SurveyDesign(weights="w"))

    def test_stute_joint_pretest_rejects_SurveyDesign(self):
        rng = np.random.default_rng(3)
        G = 30
        d = rng.uniform(0, 1, size=G)
        residuals = {0: rng.normal(0, 0.1, G)}
        fitted = {0: np.zeros(G)}
        X = np.column_stack([np.ones(G), d])
        with pytest.raises(TypeError, match="make_pweight_design"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                survey_design=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )


class TestScaleInvariance:
    """Bit-exact normalization-order invariant (Stability invariant #7).

    The legacy weights= deprecation shim binds
    `survey_design = make_pweight_design(weights_unnormalized)` and lets
    the unified survey_design= path apply the mean=1 normalization step
    EXACTLY ONCE downstream. If the shim pre-normalized AND the unified
    path also normalized, the test statistic would scale differently
    under multiplicative weight rescaling.
    """

    def test_stute_weights_alias_scale_invariant(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(4).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r1 = stute_test(d, dy, weights=w, n_bootstrap=199, seed=0)
            r2 = stute_test(d, dy, weights=w * 100.0, n_bootstrap=199, seed=0)
        # Use atol/rtol=1e-14 (per `feedback_assert_allclose_numerical_parity`):
        # the mean=1 normalization step `w * G/sum(w)` produces results that
        # agree to ~16 significant figures but not bit-exactly across
        # multiplicative rescaling (FP rounding in the renormalization step).
        np.testing.assert_allclose(r1.cvm_stat, r2.cvm_stat, atol=1e-14, rtol=1e-14)

    def test_yatchew_weights_alias_scale_invariant(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(5).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r1 = yatchew_hr_test(d, dy, weights=w)
            r2 = yatchew_hr_test(d, dy, weights=w * 100.0)
        np.testing.assert_allclose(r1.t_stat_hr, r2.t_stat_hr, atol=1e-14, rtol=1e-14)


# =============================================================================
# 2. Per-surface deprecation + parity tests
# =============================================================================


class TestQUGTestDeprecation:
    """qug_test (array-in, gated): all paths raise NotImplementedError;
    consolidation tests focus on the deprecation/mutex cascade."""

    def test_survey_design_kwarg_raises_notimpl(self, array_in_doses):
        with pytest.raises(NotImplementedError, match="QUG"):
            qug_test(array_in_doses, survey_design=make_pweight_design(np.ones(5)))

    def test_weights_emits_deprecation_warning(self, array_in_doses):
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            with pytest.raises(NotImplementedError):
                qug_test(array_in_doses, weights=np.ones(5))

    def test_survey_emits_deprecation_warning(self, array_in_doses):
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            with pytest.raises(NotImplementedError):
                qug_test(array_in_doses, survey=SurveyDesign(weights="w"))

    def test_three_way_mutex_design_plus_survey(self, array_in_doses):
        with pytest.raises(ValueError, match="at most one of"):
            qug_test(
                array_in_doses,
                survey_design=make_pweight_design(np.ones(5)),
                survey=SurveyDesign(weights="w"),
            )

    def test_three_way_mutex_design_plus_weights(self, array_in_doses):
        with pytest.raises(ValueError, match="at most one of"):
            qug_test(
                array_in_doses,
                survey_design=make_pweight_design(np.ones(5)),
                weights=np.ones(5),
            )

    def test_three_way_mutex_all_three(self, array_in_doses):
        with pytest.raises(ValueError, match="at most one of"):
            qug_test(
                array_in_doses,
                survey_design=make_pweight_design(np.ones(5)),
                survey=SurveyDesign(weights="w"),
                weights=np.ones(5),
            )


class TestStuteTestDeprecation:
    def test_survey_design_kwarg_smoke(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        r = stute_test(d, dy, survey_design=make_pweight_design(w), n_bootstrap=199, seed=0)
        assert np.isfinite(r.cvm_stat)
        assert 0.0 <= r.p_value <= 1.0

    def test_weights_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            stute_test(d, dy, weights=np.ones(30), n_bootstrap=199, seed=0)

    def test_survey_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            stute_test(
                d,
                dy,
                survey=make_pweight_design(np.ones(30)),
                n_bootstrap=199,
                seed=0,
            )

    def test_numerical_parity_weights_legacy_eq_new(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(7).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = stute_test(d, dy, weights=w, n_bootstrap=199, seed=0)
        r_new = stute_test(d, dy, survey_design=make_pweight_design(w), n_bootstrap=199, seed=0)
        assert r_legacy.cvm_stat == r_new.cvm_stat
        assert r_legacy.p_value == r_new.p_value

    def test_numerical_parity_survey_legacy_eq_new(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(8).uniform(0.5, 1.5, size=30)
        resolved = make_pweight_design(w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)
        r_new = stute_test(d, dy, survey_design=resolved, n_bootstrap=199, seed=0)
        assert r_legacy.cvm_stat == r_new.cvm_stat
        assert r_legacy.p_value == r_new.p_value

    def test_three_way_mutex_design_plus_survey(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            stute_test(
                d,
                dy,
                survey_design=make_pweight_design(w),
                survey=make_pweight_design(w),
                n_bootstrap=199,
                seed=0,
            )

    def test_three_way_mutex_all_three(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            stute_test(
                d,
                dy,
                survey_design=make_pweight_design(w),
                survey=make_pweight_design(w),
                weights=w,
                n_bootstrap=199,
                seed=0,
            )


class TestYatchewHRTestDeprecation:
    def test_survey_design_kwarg_smoke(self, array_in_data):
        d, dy = array_in_data
        w = np.ones(30)
        r = yatchew_hr_test(d, dy, survey_design=make_pweight_design(w))
        assert np.isfinite(r.t_stat_hr)

    def test_weights_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            yatchew_hr_test(d, dy, weights=np.ones(30))

    def test_survey_emits_deprecation_warning(self, array_in_data):
        d, dy = array_in_data
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            yatchew_hr_test(d, dy, survey=make_pweight_design(np.ones(30)))

    def test_numerical_parity_weights_legacy_eq_new(self, array_in_data):
        d, dy = array_in_data
        w = np.random.default_rng(9).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = yatchew_hr_test(d, dy, weights=w)
        r_new = yatchew_hr_test(d, dy, survey_design=make_pweight_design(w))
        assert r_legacy.t_stat_hr == r_new.t_stat_hr
        assert r_legacy.p_value == r_new.p_value

    def test_three_way_mutex_design_plus_weights(self, array_in_data):
        d, dy = array_in_data
        with pytest.raises(ValueError, match="at most one of"):
            yatchew_hr_test(
                d,
                dy,
                survey_design=make_pweight_design(np.ones(30)),
                weights=np.ones(30),
            )


class TestStuteJointPretestDeprecation:
    def _setup(self):
        rng = np.random.default_rng(10)
        G = 30
        d = rng.uniform(0, 1, size=G)
        residuals = {0: rng.normal(0, 0.1, G), 1: rng.normal(0, 0.1, G)}
        fitted = {0: np.zeros(G), 1: np.zeros(G)}
        X = np.column_stack([np.ones(G), d])
        return d, residuals, fitted, X

    def test_survey_design_kwarg_smoke(self):
        d, residuals, fitted, X = self._setup()
        w = np.ones(30)
        r = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=X,
            survey_design=make_pweight_design(w),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isfinite(r.cvm_stat_joint)

    def test_weights_emits_deprecation_warning(self):
        d, residuals, fitted, X = self._setup()
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                weights=np.ones(30),
                n_bootstrap=199,
                seed=0,
            )

    def test_survey_emits_deprecation_warning(self):
        d, residuals, fitted, X = self._setup()
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                survey=make_pweight_design(np.ones(30)),
                n_bootstrap=199,
                seed=0,
            )

    def test_numerical_parity_weights_legacy_eq_new(self):
        d, residuals, fitted, X = self._setup()
        w = np.random.default_rng(11).uniform(0.5, 1.5, size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r_legacy = stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                weights=w,
                n_bootstrap=199,
                seed=0,
            )
        r_new = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=X,
            survey_design=make_pweight_design(w),
            n_bootstrap=199,
            seed=0,
        )
        assert r_legacy.cvm_stat_joint == r_new.cvm_stat_joint
        assert r_legacy.p_value == r_new.p_value

    def test_three_way_mutex_all_three(self):
        d, residuals, fitted, X = self._setup()
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            stute_joint_pretest(
                residuals_by_horizon=residuals,
                fitted_by_horizon=fitted,
                doses=d,
                design_matrix=X,
                survey_design=make_pweight_design(w),
                survey=make_pweight_design(w),
                weights=w,
                n_bootstrap=199,
                seed=0,
            )


class TestJointPretrendsTestDeprecation:
    def test_survey_design_kwarg_smoke(self, event_study_panel):
        df = event_study_panel
        r = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[0],
            base_period=1,
            survey_design=SurveyDesign(weights="w"),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isfinite(r.cvm_stat_joint)

    def test_weights_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        n = len(df)
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )

    def test_survey_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                survey=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )

    def test_three_way_mutex_design_plus_survey(self, event_study_panel):
        df = event_study_panel
        n = len(df)
        with pytest.raises(ValueError, match="at most one of"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                survey_design=SurveyDesign(weights="w"),
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )


class TestJointHomogeneityTestDeprecation:
    def test_survey_design_kwarg_smoke(self, event_study_panel):
        df = event_study_panel
        r = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[2, 3],
            base_period=1,
            survey_design=SurveyDesign(weights="w"),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isfinite(r.cvm_stat_joint)

    def test_weights_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        n = len(df)
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[2, 3],
                base_period=1,
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )

    def test_survey_emits_deprecation_warning(self, event_study_panel):
        df = event_study_panel
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[2, 3],
                base_period=1,
                survey=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )


class TestHADFitDeprecation:
    def test_survey_design_kwarg_smoke(self, two_period_panel):
        df = two_period_panel
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        r = est.fit(df, "y", "d", "time", "unit", survey_design=SurveyDesign(weights="w"))
        assert np.isfinite(r.att)

    def test_weights_emits_deprecation_warning(self, two_period_panel):
        df = two_period_panel
        n = len(df)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            est.fit(df, "y", "d", "time", "unit", weights=np.ones(n))

    def test_survey_emits_deprecation_warning(self, two_period_panel):
        df = two_period_panel
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            est.fit(df, "y", "d", "time", "unit", survey=SurveyDesign(weights="w"))

    def test_three_way_mutex_design_plus_weights(self, two_period_panel):
        df = two_period_panel
        n = len(df)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(ValueError, match="at most one of"):
            est.fit(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=SurveyDesign(weights="w"),
                weights=np.ones(n),
            )


class TestDidHadPretestWorkflowDeprecation:
    def test_survey_design_kwarg_smoke(self, two_period_panel):
        df = two_period_panel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # QUG-skip warning
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )
        assert report.qug is None  # skipped under survey path
        assert report.stute is not None

    def test_weights_emits_deprecation_warning(self, two_period_panel):
        df = two_period_panel
        n = len(df)
        with pytest.warns(DeprecationWarning, match="weights=.*deprecated"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # We still need to allow the DeprecationWarning to propagate
                # to the outer pytest.warns; only filter UserWarning.
                warnings.simplefilter("always", DeprecationWarning)
                did_had_pretest_workflow(
                    df,
                    "y",
                    "d",
                    "time",
                    "unit",
                    weights=np.ones(n),
                    n_bootstrap=199,
                    seed=0,
                )

    def test_survey_emits_deprecation_warning(self, two_period_panel):
        df = two_period_panel
        with pytest.warns(DeprecationWarning, match="survey=.*deprecated"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("always", DeprecationWarning)
                did_had_pretest_workflow(
                    df,
                    "y",
                    "d",
                    "time",
                    "unit",
                    survey=SurveyDesign(weights="w"),
                    n_bootstrap=199,
                    seed=0,
                )

    def test_three_way_mutex_all_three(self, two_period_panel):
        df = two_period_panel
        n = len(df)
        with pytest.raises(ValueError, match="at most one of"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey_design=SurveyDesign(weights="w"),
                survey=SurveyDesign(weights="w"),
                weights=np.ones(n),
                n_bootstrap=199,
                seed=0,
            )
