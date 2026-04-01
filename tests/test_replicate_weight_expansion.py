"""Tests for replicate weight support expansion to 7 additional estimators."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DifferenceInDifferences,
    ImputationDiD,
    MultiPeriodDiD,
    StackedDiD,
    SunAbraham,
    TwoStageDiD,
    TwoWayFixedEffects,
)
from diff_diff.survey import SurveyDesign


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_panel():
    """2-period panel for DiD/MultiPeriodDiD (treatment/post binary columns)."""
    np.random.seed(123)
    n_units = 40
    rows = []
    for i in range(n_units):
        treated = 1 if i < 20 else 0
        wt = 1.0 + 0.2 * (i % 5)
        for t in [0, 1]:
            y = 5.0 + 0.5 * treated + 1.0 * t
            if treated and t == 1:
                y += 2.0  # ATT = 2
            y += np.random.normal(0, 0.3)
            rows.append({
                "unit": i, "time": t, "treated": treated, "post": t,
                "outcome": y, "weight": wt,
            })
    data = pd.DataFrame(rows)
    return data


def _make_staggered_panel():
    """Multi-period staggered panel for TWFE/SA/Stacked/Imputation/TwoStage."""
    np.random.seed(456)
    n_units, n_periods = 50, 8
    rows = []
    for i in range(n_units):
        if i < 15:
            ft = 4  # cohort 1
        elif i < 30:
            ft = 6  # cohort 2
        else:
            ft = 0  # never-treated
        wt = 1.0 + 0.3 * (i % 5)
        for t in range(1, n_periods + 1):
            y = 10.0 + i * 0.03 + t * 0.2
            if ft > 0 and t >= ft:
                y += 2.0
            y += np.random.normal(0, 0.4)
            rows.append({
                "unit": i, "time": t, "first_treat": ft,
                "outcome": y, "weight": wt,
                "treated": 1 if ft > 0 else 0,
                "post": 1 if ft > 0 and t >= ft else 0,
            })
    data = pd.DataFrame(rows)
    return data


def _add_jk1_replicates(data, n_rep=15, unit_col="unit"):
    """Add JK1 (delete-cluster jackknife) replicate weight columns."""
    units = sorted(data[unit_col].unique())
    cluster_size = max(1, len(units) // n_rep)
    rep_cols = []
    for r in range(n_rep):
        start = r * cluster_size
        end = min((r + 1) * cluster_size, len(units))
        deleted_units = set(units[start:end])
        w_r = data["weight"].values.copy()
        mask = data[unit_col].isin(deleted_units).values
        w_r[mask] = 0.0
        w_r[~mask] *= n_rep / (n_rep - 1)
        col = f"rep_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return rep_cols


def _add_brr_replicates(data, n_rep=16, unit_col="unit"):
    """Add BRR replicate weight columns (random sign perturbation)."""
    rng = np.random.RandomState(789)
    units = sorted(data[unit_col].unique())
    rep_cols = []
    for r in range(n_rep):
        signs = rng.choice([-1, 1], size=len(units))
        sign_map = dict(zip(units, signs))
        perturbation = data[unit_col].map(sign_map).values.astype(float)
        # BRR: w_r = w * (1 + epsilon) / 2, where epsilon in {-1, 1}
        # Simplified: w_r = w * (1 + perturbation) (combined_weights=True style)
        w_r = data["weight"].values * (1.0 + 0.5 * perturbation)
        w_r = np.maximum(w_r, 0.0)
        col = f"brr_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return rep_cols


# ---------------------------------------------------------------------------
# Smoke tests — each estimator × {JK1, BRR}
# ---------------------------------------------------------------------------

class TestDiDReplicate:
    """DifferenceInDifferences with replicate weights."""

    def test_did_jk1(self):
        data = _make_simple_panel()
        rep_cols = _add_jk1_replicates(data, n_rep=10)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = DifferenceInDifferences().fit(
            data, "outcome", "treated", "post", survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_did_brr(self):
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = DifferenceInDifferences().fit(
            data, "outcome", "treated", "post", survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0

    def test_did_wild_bootstrap_rejected(self):
        """Wild bootstrap + survey is rejected before replicate check."""
        data = _make_simple_panel()
        rep_cols = _add_jk1_replicates(data, n_rep=10)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        with pytest.raises((ValueError, NotImplementedError)):
            DifferenceInDifferences(inference="wild_bootstrap", cluster="unit").fit(
                data, "outcome", "treated", "post", survey_design=sd,
            )


class TestMultiPeriodDiDReplicate:
    """MultiPeriodDiD with replicate weights."""

    def test_multiperiod_jk1(self):
        data = _make_simple_panel()
        rep_cols = _add_jk1_replicates(data, n_rep=10)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = MultiPeriodDiD().fit(
            data, "outcome", "treated", "time", post_periods=[1], survey_design=sd,
        )
        assert np.isfinite(result.avg_att)
        assert np.isfinite(result.avg_se) and result.avg_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_multiperiod_brr(self):
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = MultiPeriodDiD().fit(
            data, "outcome", "treated", "time", post_periods=[1], survey_design=sd,
        )
        assert np.isfinite(result.avg_att)
        assert np.isfinite(result.avg_se) and result.avg_se > 0


class TestTWFEReplicate:
    """TwoWayFixedEffects with replicate weights."""

    @staticmethod
    def _make_twfe_panel():
        """Balanced 2-period panel with variation in treatment timing."""
        np.random.seed(321)
        n_units = 40
        rows = []
        for i in range(n_units):
            treated = 1 if i < 20 else 0
            wt = 1.0 + 0.2 * (i % 5)
            for t in [0, 1]:
                y = 5.0 + i * 0.05 + t * 1.0
                if treated and t == 1:
                    y += 2.0
                y += np.random.normal(0, 0.3)
                rows.append({
                    "unit": i, "time": t, "treated": treated,
                    "post": t, "outcome": y, "weight": wt,
                })
        return pd.DataFrame(rows)

    def test_twfe_brr(self):
        """BRR works well with TWFE: perturbation doesn't zero out units."""
        data = self._make_twfe_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = TwoWayFixedEffects().fit(
            data, "outcome", "treated", "post", "unit", survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_twfe_brr_larger(self):
        """Second BRR test with different seed."""
        data = self._make_twfe_panel()
        rep_cols = _add_brr_replicates(data, n_rep=20)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = TwoWayFixedEffects().fit(
            data, "outcome", "treated", "post", "unit", survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0


class TestSunAbrahamReplicate:
    """SunAbraham with replicate weights."""

    def test_sun_abraham_brr(self):
        """BRR replicates are less aggressive than JK1 for SunAbraham."""
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = SunAbraham(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_sun_abraham_bootstrap_rejected(self):
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        with pytest.raises(ValueError, match="n_bootstrap"):
            SunAbraham(n_bootstrap=100).fit(
                data, "outcome", "unit", "time", "first_treat", survey_design=sd,
            )


class TestStackedDiDReplicate:
    """StackedDiD with replicate weights."""

    def test_stacked_jk1(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = StackedDiD().fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_stacked_brr(self):
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = StackedDiD().fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0


class TestImputationDiDReplicate:
    """ImputationDiD with replicate weights."""

    def test_imputation_jk1(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = ImputationDiD(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_imputation_bootstrap_rejected(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        with pytest.raises(ValueError, match="n_bootstrap"):
            ImputationDiD(n_bootstrap=100).fit(
                data, "outcome", "unit", "time", "first_treat", survey_design=sd,
            )


class TestTwoStageDiDReplicate:
    """TwoStageDiD with replicate weights."""

    def test_two_stage_jk1(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = TwoStageDiD(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_two_stage_bootstrap_rejected(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        with pytest.raises(ValueError, match="n_bootstrap"):
            TwoStageDiD(n_bootstrap=100).fit(
                data, "outcome", "unit", "time", "first_treat", survey_design=sd,
            )
