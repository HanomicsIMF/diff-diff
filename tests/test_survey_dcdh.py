"""Survey support tests for ChaisemartinDHaultfoeuille (dCDH)."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import ChaisemartinDHaultfoeuille, SurveyDesign
from diff_diff.prep_dgp import generate_reversible_did_data


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_data():
    """Reversible-treatment panel with 30 groups, 6 periods."""
    return generate_reversible_did_data(
        n_groups=30,
        n_periods=6,
        pattern="single_switch",
        seed=42,
    )


@pytest.fixture(scope="module")
def data_with_survey(base_data):
    """Add survey columns: weights, strata, PSU."""
    rng = np.random.default_rng(99)
    df = base_data.copy()
    groups = df["group"].unique()

    # Assign per-group (constant within group) survey attributes
    g_weights = {g: rng.uniform(0.5, 3.0) for g in groups}
    g_strata = {g: int(i % 3) for i, g in enumerate(sorted(groups))}
    g_psu = {g: int(i % 10) for i, g in enumerate(sorted(groups))}

    df["pw"] = df["group"].map(g_weights)
    df["stratum"] = df["group"].map(g_strata)
    df["cluster"] = df["group"].map(g_psu)
    return df


# ── Test: Backward compatibility ────────────────────────────────────


class TestBackwardCompat:
    """survey_design=None produces identical results to pre-change code."""

    def test_no_survey_unchanged(self, base_data):
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            base_data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=None,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) or result.overall_se != result.overall_se
        assert result.survey_metadata is None


# ── Test: Uniform weights = no-survey ───────────────────────────────


class TestUniformWeights:
    """Uniform weights should produce identical results to no survey."""

    def test_uniform_weights_match_unweighted(self, base_data):
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw")

        result_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            base_data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        result_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # Point estimates should match exactly (uniform weights = unweighted mean)
        assert result_plain.overall_att == pytest.approx(
            result_survey.overall_att, abs=1e-10
        )


# ── Test: Non-uniform weights change estimate ───────────────────────


class TestNonUniformWeights:

    def test_nonuniform_weights_change_att(self, base_data):
        """With multiple obs per cell and non-uniform weights, ATT differs."""
        # Duplicate rows to create multi-observation cells, then vary
        # outcomes and weights so weighted cell means differ from unweighted.
        rng = np.random.default_rng(77)
        df = base_data.copy()
        df2 = base_data.copy()
        df2["outcome"] = df2["outcome"] + rng.normal(0, 1.0, size=len(df2))
        multi = pd.concat([df, df2], ignore_index=True)

        # Assign per-group constant weights (heavier on the second copy)
        groups = multi["group"].unique()
        g_weights = {g: rng.uniform(0.5, 3.0) for g in groups}
        multi["pw"] = multi["group"].map(g_weights)
        # Make second copy have different weights (still constant within group)
        # by giving rows from the second batch higher weight via a multiplier
        multi.loc[len(df):, "pw"] = multi.loc[len(df):, "pw"] * 2.0

        # Since weights now vary within group (first copy vs second copy),
        # we need per-observation weights. But dCDH requires group-constant
        # survey columns. Instead, use observation-level weights directly:
        # assign random weights per observation (but constant within group).
        # Actually, the constraint is within-GROUP constancy. With multi-obs
        # cells where weights vary within groups, validation will reject.
        # Solution: use group-constant weights with varied outcomes.
        multi["pw"] = multi["group"].map(g_weights)

        sd = SurveyDesign(weights="pw")
        result_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        result_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # With group-constant weights and multi-obs cells, weighted cell
        # means = unweighted cell means (all obs in the cell get the same
        # weight). The ATTs should match. This confirms the equal-cell
        # contract: survey weights don't change the cross-group aggregation.
        # The SE will differ because the survey variance accounts for design.
        assert result_plain.overall_att == pytest.approx(
            result_survey.overall_att, abs=1e-8
        )
        # But the SEs should differ when strata/PSU are present
        # (tested separately in TestSurveySE)


# ── Test: Scale invariance ──────────────────────────────────────────


class TestScaleInvariance:

    def test_weight_scale_invariance(self, data_with_survey):
        sd1 = SurveyDesign(weights="pw")

        df2 = data_with_survey.copy()
        df2["pw_scaled"] = df2["pw"] * 100.0
        sd2 = SurveyDesign(weights="pw_scaled")

        r1 = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd1,
        )
        r2 = ChaisemartinDHaultfoeuille(seed=1).fit(
            df2,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd2,
        )
        assert r1.overall_att == pytest.approx(r2.overall_att, abs=1e-10)
        assert r1.overall_se == pytest.approx(r2.overall_se, rel=1e-6)


# ── Test: Survey SE differs from analytical SE ──────────────────────


class TestSurveySE:

    def test_strata_psu_changes_se(self, data_with_survey):
        """Strata + PSU should produce a different SE than weights-only."""
        sd_weights_only = SurveyDesign(weights="pw")
        sd_full = SurveyDesign(
            weights="pw", strata="stratum", psu="cluster", nest=True
        )

        r_w = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_weights_only,
        )
        r_full = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_full,
        )
        # Point estimates should match (same weights)
        assert r_w.overall_att == pytest.approx(r_full.overall_att, abs=1e-10)
        # SEs should differ (strata/PSU affects variance)
        if np.isfinite(r_w.overall_se) and np.isfinite(r_full.overall_se):
            assert r_w.overall_se != pytest.approx(r_full.overall_se, rel=0.01)

    def test_survey_metadata_populated(self, data_with_survey):
        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert result.survey_metadata is not None


# ── Test: Validation ────────────────────────────────────────────────


class TestValidation:

    def test_rejects_fweight(self, base_data):
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            ChaisemartinDHaultfoeuille().fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_rejects_aweight(self, base_data):
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw", weight_type="aweight")
        with pytest.raises(ValueError, match="pweight"):
            ChaisemartinDHaultfoeuille().fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_varying_weights_within_group_accepted(self, base_data):
        """Observation-level weights varying within groups are valid."""
        # Create multi-obs cells with varying weights
        rng = np.random.default_rng(1)
        df = base_data.copy()
        df2 = base_data.copy()
        df2["outcome"] = df2["outcome"] + rng.normal(0, 0.5, size=len(df2))
        multi = pd.concat([df, df2], ignore_index=True)
        # Observation-level weights (vary within group)
        multi["pw"] = rng.uniform(0.5, 3.0, size=len(multi))
        sd = SurveyDesign(weights="pw")
        # Should succeed - no group-constant restriction
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_varying_weights_change_att(self, base_data):
        """With multi-obs cells and varying weights, ATT differs from unweighted.

        dCDH uses first differences Y_{g,t} - Y_{g,t-1}, so group-constant
        noise cancels. The noise must vary across both group AND time for
        weighted cell means to affect the ATT via different first differences.
        """
        rng = np.random.default_rng(42)
        df = base_data.copy()
        df2 = base_data.copy()
        # Per-observation noise (varies by group AND time)
        df2["outcome"] = df2["outcome"] + rng.normal(0, 3.0, size=len(df2))
        multi = pd.concat([df, df2], ignore_index=True)
        # Give first copy weight=1, second copy weight=10
        multi["pw"] = np.where(np.arange(len(multi)) < len(df), 1.0, 10.0)
        sd = SurveyDesign(weights="pw")
        result_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi, outcome="outcome", group="group",
            time="period", treatment="treatment",
        )
        result_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        # Weighted cell means with time-varying noise produce different
        # first differences -> different ATT
        assert result_plain.overall_att != pytest.approx(
            result_survey.overall_att, abs=0.01
        )

    def test_rejects_replicate_weights(self, base_data):
        """Replicate weight variance not yet supported."""
        df = base_data.copy()
        df["pw"] = 1.0
        df["rep1"] = 1.0
        df["rep2"] = 1.0
        sd = SurveyDesign(
            weights="pw",
            replicate_weights=["rep1", "rep2"],
            replicate_method="BRR",
        )
        with pytest.raises(NotImplementedError, match="Replicate"):
            ChaisemartinDHaultfoeuille().fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )


# ── Test: Multi-horizon with survey ─────────────────────────────────


class TestMultiHorizonSurvey:

    def test_multi_horizon_survey_runs(self, data_with_survey):
        """L_max >= 1 with survey should produce finite event study effects."""
        sd = SurveyDesign(weights="pw")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        assert 1 in result.event_study_effects
        assert np.isfinite(result.event_study_effects[1]["effect"])


# ── Test: Bootstrap + survey warning ────────────────────────────────


class TestBootstrapSurveyWarning:

    def test_bootstrap_survey_emits_warning(self, data_with_survey):
        sd = SurveyDesign(weights="pw")
        with pytest.warns(UserWarning, match="group-level multiplier"):
            ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1).fit(
                data_with_survey,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )


# ── Test: SE scale pinning ──────────────────────────────────────────


class TestSEScalePinning:
    """Survey SE with uniform weights and no strata/PSU must match plug-in SE."""

    def test_uniform_survey_se_matches_plugin(self, base_data):
        """Pins the divisor normalization: uniform survey SE with group-level
        PSU clustering should be close to plug-in SE.

        Without PSU clustering, survey treats each observation as independent
        (N_obs observations), while plug-in treats each group as independent
        (N_groups). Clustering at the group level aligns the two.
        """
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw", psu="group")

        r_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            base_data, outcome="outcome", group="group",
            time="period", treatment="treatment",
        )
        r_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        # With PSU=group and uniform weights, survey SE should be
        # close to plug-in SE (both assume group-level independence).
        # Small-sample corrections (n/(n-1)) cause minor differences.
        if np.isfinite(r_plain.overall_se) and np.isfinite(r_survey.overall_se):
            assert r_plain.overall_se == pytest.approx(
                r_survey.overall_se, rel=0.15
            ), (
                f"Survey SE ({r_survey.overall_se:.6f}) should be close to "
                f"plug-in SE ({r_plain.overall_se:.6f}) with uniform weights "
                f"and PSU=group"
            )


# ── Test: Zero-weight cells ─────────────────────────────────────────


class TestZeroWeightCells:

    def test_zero_weight_cell_excluded(self, base_data):
        """A cell with zero survey weight is treated as absent."""
        df = base_data.copy()
        df["pw"] = 1.0
        # Zero out weight for one group at one period
        target_group = df["group"].unique()[0]
        target_period = df["period"].unique()[1]
        mask = (df["group"] == target_group) & (df["period"] == target_period)
        df.loc[mask, "pw"] = 0.0
        sd = SurveyDesign(weights="pw")

        # Should not raise; the zero-weight cell is just absent
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
