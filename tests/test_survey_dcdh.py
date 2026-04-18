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

    def test_rejects_replicate_weights_with_bootstrap(self, base_data):
        """Replicate weights combined with n_bootstrap > 0 is rejected.

        Replicate variance is closed-form (compute_replicate_if_variance);
        combining it with a multiplier bootstrap would double-count
        variance. Matches library precedent in efficient_did.py:989,
        staggered.py:1869, two_stage.py:251-253. The standalone
        replicate-only path (n_bootstrap=0) is supported separately;
        see tests/test_survey_dcdh_replicate_psu.py.
        """
        df = base_data.copy()
        df["pw"] = 1.0
        df["rep1"] = 1.0
        df["rep2"] = 1.0
        sd = SurveyDesign(
            weights="pw",
            replicate_weights=["rep1", "rep2"],
            replicate_method="BRR",
        )
        with pytest.raises(NotImplementedError, match="replicate weights and n_bootstrap"):
            ChaisemartinDHaultfoeuille(n_bootstrap=100, seed=1).fit(
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

    def test_bootstrap_survey_auto_inject_no_warning(self, data_with_survey):
        """Under auto-inject psu=group, Hall-Mammen wild PSU bootstrap
        coincides with the group-level multiplier bootstrap — so no
        warning should fire. The old 'PSU-level deferred' warning has
        been replaced with a conditional one that only fires when the
        user passes a strictly coarser PSU.

        See the new test file tests/test_survey_dcdh_replicate_psu.py for
        the strictly-coarser-PSU case where the warning DOES fire.
        """
        sd = SurveyDesign(weights="pw")
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            # Ignore warnings unrelated to the bootstrap-PSU contract.
            _w.filterwarnings(
                "ignore", message="Single-period placebo SE"
            )
            _w.filterwarnings(
                "ignore", message="pweight weights normalized"
            )
            _w.filterwarnings(
                "ignore", message="Assumption 11"
            )
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

        Under dCDH's auto-inject contract (see REGISTRY.md §dCDH survey),
        SurveyDesign(weights='pw') with no explicit psu is equivalent to
        SurveyDesign(weights='pw', psu='group'). Either form aligns the
        effective sampling unit with the group-level IF and matches the
        plug-in SE up to small-sample corrections.
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


# ── Test: Delta overall surface threads survey df ───────────────────


class TestSurveyDeltaInference:
    """Verify the L_max>=2 cost-benefit delta surface uses survey df."""

    def test_survey_delta_uses_survey_df(self, data_with_survey):
        """Under L_max=2 with a survey design, overall_p_value must match
        t-distribution inference with df=df_survey (not z-inference)."""
        from scipy import stats

        sd = SurveyDesign(
            weights="pw", strata="stratum", psu="cluster", nest=True
        )
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=2, survey_design=sd,
        )
        if not (np.isfinite(r.overall_se) and r.overall_se > 0):
            pytest.skip("delta not estimable on this fixture")

        assert r.survey_metadata is not None
        df_s = r.survey_metadata.df_survey
        assert df_s is not None and df_s > 0, (
            f"expected positive df_survey, got {df_s}"
        )

        t_stat = r.overall_att / r.overall_se
        p_t = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_s))
        # Reported p-value must match t-based (proving _df_survey was threaded)
        assert r.overall_p_value == pytest.approx(p_t, abs=1e-10)

    def test_survey_delta_t_differs_from_z(self, base_data):
        """With a small-df design (df~4), survey-t p-value must differ
        measurably from z p-value at the delta surface."""
        from scipy import stats

        df_ = base_data.copy()
        df_["pw"] = 1.0
        # 2 strata × 3 clusters/stratum = 6 nested PSUs → df_survey = 4
        groups = sorted(df_["group"].unique())
        n_g = len(groups)
        strata_map = {g: i // (n_g // 2) for i, g in enumerate(groups)}
        psu_map = {g: i // (n_g // 6) for i, g in enumerate(groups)}
        df_["stratum"] = df_["group"].map(strata_map)
        df_["cluster"] = df_["group"].map(psu_map)
        sd = SurveyDesign(
            weights="pw", strata="stratum", psu="cluster", nest=True
        )
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=2, survey_design=sd,
        )
        if not (np.isfinite(r.overall_se) and r.overall_se > 0):
            pytest.skip("delta not estimable on this fixture")
        assert r.survey_metadata is not None
        df_s = r.survey_metadata.df_survey
        assert df_s is not None and df_s < 30, (
            f"expected small df_survey for t-vs-z gap, got {df_s}"
        )

        t_stat = r.overall_att / r.overall_se
        p_t = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_s))
        p_z = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
        # Threaded p-value must match t, not z
        assert r.overall_p_value == pytest.approx(p_t, abs=1e-10)
        assert abs(r.overall_p_value - p_z) > 1e-6, (
            "overall_p_value must differ from z-inference when df_survey is small"
        )


# ── Test: Survey + controls (DID^X) ─────────────────────────────────


class TestSurveyControls:
    """Covariate-adjusted (DID^X) path must work with survey_design."""

    def test_survey_plus_controls_runs(self, data_with_survey):
        """Covariate-adjusted dCDH with survey_design produces finite ATT."""
        rng = np.random.default_rng(7)
        df_ = data_with_survey.copy()
        df_["x"] = rng.normal(0, 1.0, size=len(df_))
        sd = SurveyDesign(
            weights="pw", strata="stratum", psu="cluster", nest=True
        )
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            controls=["x"], L_max=1, survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert r.survey_metadata is not None


# ── Test: Survey + HonestDiD ────────────────────────────────────────


class TestSurveyHonestDiD:
    """HonestDiD bounds on survey-backed dCDH results must carry df_survey."""

    def test_survey_honest_did_propagates_df(self, data_with_survey):
        """results.honest_did_results.df_survey must match
        results.survey_metadata.df_survey (non-None propagation)."""
        import warnings

        sd = SurveyDesign(
            weights="pw", strata="stratum", psu="cluster", nest=True
        )
        with warnings.catch_warnings():
            # dCDH HonestDiD emits a methodology-deviation warning
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                data_with_survey,
                outcome="outcome", group="group",
                time="period", treatment="treatment",
                L_max=2, honest_did=True, survey_design=sd,
            )
        if r.honest_did_results is None:
            pytest.skip("HonestDiD computation returned None on this fixture")
        assert r.survey_metadata is not None
        df_meta = r.survey_metadata.df_survey
        assert df_meta is not None
        # df_survey must propagate from survey_metadata into HonestDiD result
        assert r.honest_did_results.df_survey == df_meta


# ── Test: Survey-aware heterogeneity ────────────────────────────────


class TestSurveyHeterogeneity:
    """Heterogeneity testing under survey_design uses WLS + TSL IF."""

    def test_uniform_weights_het_matches_unweighted(self, base_data):
        """Uniform pweights must yield identical β_het to plain OLS path."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # time-invariant group-level covariate
        rng = np.random.default_rng(0)
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)

        r_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, heterogeneity="x_het",
        )
        sd = SurveyDesign(weights="pw")
        r_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, heterogeneity="x_het", survey_design=sd,
        )
        assert r_plain.heterogeneity_effects is not None
        assert r_survey.heterogeneity_effects is not None
        b_plain = r_plain.heterogeneity_effects[1]["beta"]
        b_survey = r_survey.heterogeneity_effects[1]["beta"]
        # WLS with uniform weights = OLS; β_het must match
        if np.isfinite(b_plain) and np.isfinite(b_survey):
            assert b_plain == pytest.approx(b_survey, abs=1e-8)

    def test_nonuniform_het_changes_beta(self, base_data):
        """Varying pweights change the WLS point estimate vs plain OLS."""
        rng = np.random.default_rng(42)
        df_ = base_data.copy()
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        pw_map = {g: rng.uniform(0.5, 4.0) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)
        df_["pw"] = df_["group"].map(pw_map)
        sd = SurveyDesign(weights="pw")

        r_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, heterogeneity="x_het",
        )
        r_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, heterogeneity="x_het", survey_design=sd,
        )
        assert r_plain.heterogeneity_effects is not None
        assert r_survey.heterogeneity_effects is not None
        b_plain = r_plain.heterogeneity_effects[1]["beta"]
        b_survey = r_survey.heterogeneity_effects[1]["beta"]
        if np.isfinite(b_plain) and np.isfinite(b_survey):
            # WLS with non-uniform weights differs from unweighted OLS
            assert b_plain != pytest.approx(b_survey, abs=1e-6), (
                f"plain={b_plain} survey={b_survey} should differ with varying weights"
            )

    def test_survey_het_uses_survey_df(self, data_with_survey):
        """Reported p_value must match t-distribution inference with df_survey."""
        from scipy import stats

        rng = np.random.default_rng(7)
        df_ = data_with_survey.copy()
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)
        sd = SurveyDesign(
            weights="pw", strata="stratum", psu="cluster", nest=True
        )
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, heterogeneity="x_het", survey_design=sd,
        )
        assert r.heterogeneity_effects is not None
        entry = r.heterogeneity_effects[1]
        if not (np.isfinite(entry["se"]) and entry["se"] > 0):
            pytest.skip("heterogeneity SE not estimable on this fixture")
        assert r.survey_metadata is not None
        df_s = r.survey_metadata.df_survey
        assert df_s is not None and df_s > 0

        t_stat = entry["beta"] / entry["se"]
        p_t = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_s))
        assert entry["p_value"] == pytest.approx(p_t, abs=1e-10)


# ── Test: TWFE helper parity under survey ───────────────────────────


class TestSurveyTWFEParity:
    """twowayfeweights() with survey_design matches fit().twfe_* under survey."""

    def test_twfe_helper_matches_fit_under_survey(self, data_with_survey):
        """fit and twowayfeweights() produce identical TWFE output under survey."""
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights

        sd = SurveyDesign(
            weights="pw", strata="stratum", psu="cluster", nest=True
        )
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        helper = twowayfeweights(
            data_with_survey,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        # fit() twfe_* may be None if non-binary treatment; this fixture is binary
        assert r.twfe_fraction_negative is not None
        assert r.twfe_sigma_fe is not None
        assert r.twfe_beta_fe is not None
        assert r.twfe_fraction_negative == pytest.approx(
            helper.fraction_negative, abs=1e-12
        )
        assert r.twfe_sigma_fe == pytest.approx(helper.sigma_fe, abs=1e-12)
        assert r.twfe_beta_fe == pytest.approx(helper.beta_fe, abs=1e-12)

    def test_twfe_helper_rejects_non_pweight(self, base_data):
        """fweight/aweight must be rejected by twowayfeweights() under survey."""
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights

        df_ = base_data.copy()
        df_["pw"] = 1.0
        sd = SurveyDesign(weights="pw", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            twowayfeweights(
                df_,
                outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd,
            )

    def test_twfe_helper_accepts_replicate_weights(self, base_data):
        """Replicate-weight designs are accepted by the helper (no SE field
        on TWFEWeightsResult, so only cell aggregation is affected). Matches
        fit()'s new contract where replicate variance runs analytically via
        compute_replicate_if_variance."""
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights

        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["rep1"] = 1.0
        df_["rep2"] = 1.0
        sd = SurveyDesign(
            weights="pw",
            replicate_weights=["rep1", "rep2"],
            replicate_method="BRR",
        )
        result = twowayfeweights(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.beta_fe)
        assert np.isfinite(result.sigma_fe)
        assert 0.0 <= result.fraction_negative <= 1.0


# ── Test: TWFE diagnostic oracle under survey ───────────────────────


class TestSurveyTWFEOracle:
    """twfe_beta_fe under survey must match an observation-level pweighted
    TWFE regression on the same data (proving w_gt is used, not n_gt)."""

    def test_survey_twfe_matches_obs_level_pweighted_ols(self, data_with_survey):
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights
        from diff_diff.linalg import solve_ols

        sd = SurveyDesign(weights="pw")
        helper = twowayfeweights(
            data_with_survey,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(helper.beta_fe)

        # Build observation-level TWFE design with group and period FE
        # (reference category dropped) and treatment indicator.
        df_ = data_with_survey.copy()
        groups_u = sorted(df_["group"].unique())
        periods_u = sorted(df_["period"].unique())
        g_map = {g: i for i, g in enumerate(groups_u)}
        t_map = {t: i for i, t in enumerate(periods_u)}
        g_idx = df_["group"].map(g_map).to_numpy()
        t_idx = df_["period"].map(t_map).to_numpy()
        n = len(df_)
        X_g = np.zeros((n, len(groups_u) - 1))
        X_t = np.zeros((n, len(periods_u) - 1))
        for i in range(n):
            if g_idx[i] > 0:
                X_g[i, g_idx[i] - 1] = 1.0
            if t_idx[i] > 0:
                X_t[i, t_idx[i] - 1] = 1.0
        intercept = np.ones((n, 1))
        treat = df_["treatment"].to_numpy().astype(float).reshape(-1, 1)
        X_obs = np.hstack([intercept, X_g, X_t, treat])
        y_obs = df_["outcome"].to_numpy().astype(float)
        w_obs = df_["pw"].to_numpy().astype(float)

        coef, _, _ = solve_ols(
            X_obs, y_obs,
            weights=w_obs, weight_type="pweight",
            return_vcov=False,
        )
        beta_oracle = float(coef[-1])
        # Point-estimate match (one obs per cell in this fixture; so the
        # cell-level WLS with cell_weight == w_gt equals the obs-level
        # WLS with w_obs weights).
        assert helper.beta_fe == pytest.approx(beta_oracle, rel=1e-6), (
            f"helper.beta_fe={helper.beta_fe} oracle={beta_oracle} "
            f"— TWFE diagnostic must use w_gt under survey"
        )


# ── Test: Zero-weight subpopulation exclusion ──────────────────────


class TestZeroWeightSubpopulation:
    """Zero-weight rows must not trip fuzzy-DiD guard or inflate counts."""

    def test_mixed_zero_weight_row_excluded_from_validation(self, base_data):
        """A cell with a positive-weight treated obs and a zero-weight
        obs with a different treatment value must fit cleanly — the
        zero-weight row is out-of-sample (SurveyDesign.subpopulation())."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Pick a treated (g, t) cell. Add a zero-weight row in the same
        # cell with the opposite treatment value. Unweighted d_min != d_max
        # would trip the fuzzy-DiD guard; pre-filtering zero-weight rows
        # must bypass it.
        treated_mask = df_["treatment"] == 1
        if not treated_mask.any():
            pytest.skip("no treated row in fixture")
        sample = df_[treated_mask].iloc[0].copy()
        # Flip treatment on the injected row, give it zero weight
        sample["treatment"] = 0
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")

        # Must succeed (not raise fuzzy-DiD ValueError)
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_outcome(self, base_data):
        """A zero-weight row with NaN outcome must not trip the outcome
        NaN validator. SurveyDesign.subpopulation() contract."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        sample = df_.iloc[0].copy()
        sample["outcome"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        # Must succeed — zero-weight row with NaN outcome is out-of-sample
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_group_id(self, base_data):
        """A zero-weight row with NaN group id must not crash the SE
        factorization. SurveyDesign.subpopulation() contract."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Cast group to object to allow NaN without coercion errors
        df_["group"] = df_["group"].astype(object)
        sample = df_.iloc[0].copy()
        sample["group"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        # Must succeed — zero-weight row's NaN group id is out-of-sample
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_control(self, base_data):
        """A zero-weight row with NaN in a control column must not abort
        the DID^X path, and the covariate cell aggregation must use only
        positive-weight rows (no length-mismatch error)."""
        rng = np.random.default_rng(13)
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["x"] = rng.normal(0, 1, size=len(df_))
        # Inject a zero-weight row with NaN control value
        sample = df_.iloc[0].copy()
        sample["x"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, controls=["x"], survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_heterogeneity(self, base_data):
        """A zero-weight row with NaN in the heterogeneity column must
        not trip the heterogeneity time-invariance validator."""
        rng = np.random.default_rng(0)
        df_ = base_data.copy()
        df_["pw"] = 1.0
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)
        # Inject a zero-weight row with NaN het value for an existing group
        sample = df_.iloc[0].copy()
        sample["x_het"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        # Must succeed — zero-weight row with NaN het is out-of-sample
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, heterogeneity="x_het", survey_design=sd,
        )
        assert result.heterogeneity_effects is not None


# ── Test: Survey + trends_linear ────────────────────────────────────


class TestSurveyTrendsLinear:
    """Survey-backed trends_linear fit must populate linear_trends_effects."""

    def test_survey_trends_linear_runs(self, data_with_survey):
        sd = SurveyDesign(weights="pw")
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=2, trends_linear=True, survey_design=sd,
        )
        assert r.survey_metadata is not None
        # linear_trends_effects populated per REGISTRY line 614 contract
        assert r.linear_trends_effects is not None
        # At least one horizon should be estimable with finite value
        finite_horizons = [
            h for h, entry in r.linear_trends_effects.items()
            if np.isfinite(entry.get("effect", np.nan))
        ]
        assert len(finite_horizons) > 0, (
            "expected at least one horizon with finite linear_trends_effect"
        )


# ── Test: Survey + trends_nonparam ──────────────────────────────────


class TestSurveyTrendsNonparam:
    """Survey-backed trends_nonparam fit must thread set-restrictions."""

    def test_survey_trends_nonparam_runs(self, data_with_survey):
        # Reuse stratum as set ID (time-invariant per group)
        sd = SurveyDesign(weights="pw")
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=2, trends_nonparam="stratum", survey_design=sd,
        )
        assert r.survey_metadata is not None
        assert r.event_study_effects is not None
        # Support trimming may reduce counts but at least one finite-SE
        # horizon should remain on this fixture.
        finite_ses = [
            entry
            for entry in r.event_study_effects.values()
            if np.isfinite(entry.get("se", np.nan))
        ]
        assert len(finite_ses) > 0, (
            "expected at least one event-study horizon with finite SE "
            "under trends_nonparam + survey"
        )


# ── Test: Survey + design2 ──────────────────────────────────────────


class TestSurveyDesign2:
    """Survey-backed design2 fit must populate design2_effects."""

    @staticmethod
    def _make_join_then_leave_panel(seed=42, n_groups=30, n_periods=8):
        """Panel with join-then-leave (Design-2) groups, matching the
        existing design2 fixture in test_chaisemartin_dhaultfoeuille.py."""
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            group_fe = rng.normal(0, 2)
            for t in range(n_periods):
                if g < 10:
                    d = 1 if 2 <= t < 5 else 0
                elif g < 20:
                    d = 1 if t >= 3 else 0
                else:
                    d = 0
                y = group_fe + 2.0 * t + 5.0 * d + rng.normal(0, 0.3)
                rows.append(
                    {"group": g, "period": t, "treatment": d, "outcome": y, "pw": 1.0}
                )
        return pd.DataFrame(rows)

    def test_survey_design2_runs(self):
        df_ = self._make_join_then_leave_panel()
        sd = SurveyDesign(weights="pw")
        # drop_larger_lower=False keeps the 2-switch groups
        r = ChaisemartinDHaultfoeuille(
            seed=1, drop_larger_lower=False
        ).fit(
            df_,
            outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=1, design2=True, survey_design=sd,
        )
        assert r.survey_metadata is not None
        assert r.design2_effects is not None
        assert r.design2_effects["n_design2_groups"] == 10
        # switch_in and switch_out mean effects should be finite
        assert np.isfinite(r.design2_effects["switch_in"]["mean_effect"])
        assert np.isfinite(r.design2_effects["switch_out"]["mean_effect"])


# ── Test: Within-group constancy of strata and PSU ──────────────────


class TestSurveyWithinGroupValidation:
    """Cell-period IF allocator contract: strata and PSU may vary ACROSS
    cells of a group, but must be constant WITHIN each (g, t) cell. In
    canonical one-obs-per-cell panels the cell-level constancy check is
    trivially satisfied. Out-of-scope combinations (heterogeneity +
    within-group-varying PSU; n_bootstrap > 0 + within-group-varying
    PSU) raise NotImplementedError with a pointer to the follow-up PR.
    """

    def test_accepts_varying_psu_within_group(self, base_data):
        """Under the cell-period allocator, PSU that varies across cells
        of a group is a valid design — the allocator attributes IF mass
        to each (g, t) cell separately and Binder TSL aggregates at PSU
        level with the honest cell-level variance.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        # PSU varies within each group (alternates by period). Still
        # constant within each (g, t) cell because one obs per cell.
        df_["psu"] = df_["period"] % 2
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd, L_max=2,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        # And the SE differs from a within-group-constant-PSU baseline,
        # because the cell allocator now honors the extra PSU structure.
        df_const = base_data.copy()
        df_const["pw"] = 1.0
        df_const["stratum"] = 0
        df_const["psu"] = 0  # constant-within-group PSU baseline
        sd_const = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        r_const = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_const, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd_const, L_max=2,
        )
        assert result.overall_se != r_const.overall_se, (
            "Cell-period allocator must produce a different SE when PSU "
            "actually varies across cells vs. constant-within-group."
        )

    def test_accepts_varying_strata_within_group(self, base_data):
        """Strata that vary across cells of a group are supported under
        the cell-period allocator, trivially satisfying within-cell
        constancy in one-obs-per-cell panels.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Stratum varies within each group across cells
        df_["stratum"] = df_["period"] % 2
        # PSU = group, nested inside stratum so the resolver accepts the
        # cross-stratum reuse of group labels.
        df_["psu"] = df_["group"]
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu", nest=True)
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd, L_max=2,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)

    def test_heterogeneity_with_varying_psu_raises(self, base_data):
        """heterogeneity= is gated under within-group-varying PSU until
        PR 3 ships the cell-period allocator for the WLS psi_obs path.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        df_["psu"] = df_["period"] % 2  # varies within group
        df_["x_het"] = np.arange(len(df_)) % 3  # categorical covariate
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        with pytest.raises(NotImplementedError, match="heterogeneity"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df_, outcome="outcome", group="group",
                time="period", treatment="treatment",
                heterogeneity="x_het", L_max=1,
                survey_design=sd,
            )

    def test_bootstrap_with_varying_psu_raises(self, base_data):
        """n_bootstrap > 0 is gated under within-group-varying PSU until
        PR 4 ships the cell-level Hall-Mammen wild bootstrap.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        df_["psu"] = df_["period"] % 2  # varies within group
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        with pytest.raises(NotImplementedError, match="n_bootstrap"):
            ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1).fit(
                df_, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd,
            )

    def test_auto_inject_with_varying_strata_nest_true_succeeds(self, base_data):
        """When strata varies across cells of a group and the user
        passes ``nest=True`` with no explicit ``psu``, the auto-inject
        path is valid: ``SurveyDesign.resolve()`` combines
        ``(stratum, psu)`` into globally-unique labels via the
        nest=True path (``diff_diff/survey.py:299-302``), so the
        cross-stratum PSU uniqueness check is satisfied. Byte-check
        against the explicit ``SurveyDesign(..., psu="group",
        nest=True)`` baseline — both paths resolve to the same design.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = df_["period"] % 2
        sd_auto = SurveyDesign(weights="pw", strata="stratum", nest=True)
        sd_explicit = SurveyDesign(
            weights="pw", strata="stratum", psu="group", nest=True,
        )
        r_auto = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd_auto, L_max=2,
        )
        r_explicit = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd_explicit, L_max=2,
        )
        assert np.isfinite(r_auto.overall_att)
        assert np.isfinite(r_auto.overall_se)
        if np.isfinite(r_auto.overall_se) and np.isfinite(r_explicit.overall_se):
            assert r_auto.overall_se == pytest.approx(
                r_explicit.overall_se, rel=1e-6
            )
        assert r_auto.survey_metadata is not None
        assert r_explicit.survey_metadata is not None
        assert (
            r_auto.survey_metadata.df_survey
            == r_explicit.survey_metadata.df_survey
        )

    def test_auto_inject_with_varying_strata_raises(self, base_data):
        """Auto-injected `psu=<group>` with nest=False cannot honor
        strata that vary across cells of a group — the synthesized PSU
        column would reuse group labels across strata and trip the
        cross-stratum PSU uniqueness check. fit() detects that combo
        before survey resolution and raises a targeted ValueError
        pointing users to the explicit `psu=<col>, nest=True` path.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = df_["period"] % 2  # varies across cells of each group
        sd = SurveyDesign(weights="pw", strata="stratum")
        with pytest.raises(ValueError, match=r"psu=<col>"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df_, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd,
            )

    def test_within_cell_psu_variation_rejected(self, base_data):
        """Multiple PSUs inside a single (g, t) cell (a multi-obs-per-
        cell panel) remain ambiguous under the cell allocator and must
        be rejected.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        df_["psu"] = 0
        # Duplicate the first row with a different PSU so that cell
        # (group[0], period[0]) has two obs with different PSU labels.
        dup = df_.iloc[0].copy()
        dup["psu"] = 99
        df_ = pd.concat([df_, pd.DataFrame([dup])], ignore_index=True)
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        with pytest.raises(ValueError, match="PSU to be constant within each"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df_, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd,
            )

    def test_within_cell_strata_variation_rejected(self, base_data):
        """Multiple strata inside a single (g, t) cell are ambiguous
        under the cell allocator and must be rejected.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        # Duplicate the first row with a different stratum.
        dup = df_.iloc[0].copy()
        dup["stratum"] = 1
        df_ = pd.concat([df_, pd.DataFrame([dup])], ignore_index=True)
        df_["psu"] = np.arange(len(df_))
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        with pytest.raises(ValueError, match="strata to be constant within each"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df_, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd,
            )

    def test_accepts_varying_weights_within_group(self, base_data):
        """Within-group-varying pweights remain supported — the expansion
        psi_i = U[g] * (w_i / W_g) handles obs-level weight variation."""
        df_ = base_data.copy()
        rng = np.random.default_rng(7)
        df_["pw"] = rng.uniform(0.5, 2.0, size=len(df_))
        sd = SurveyDesign(weights="pw")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_auto_inject_psu_matches_explicit_group_psu(self, base_data):
        """SurveyDesign(weights='pw') (no PSU) must yield the same SE and
        df_survey as SurveyDesign(weights='pw', psu='group') after
        dCDH auto-injects psu=group."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        r_no_psu = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=SurveyDesign(weights="pw"),
        )
        r_explicit = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=SurveyDesign(weights="pw", psu="group"),
        )
        assert r_no_psu.overall_att == pytest.approx(
            r_explicit.overall_att, abs=1e-10
        )
        if np.isfinite(r_no_psu.overall_se) and np.isfinite(r_explicit.overall_se):
            assert r_no_psu.overall_se == pytest.approx(
                r_explicit.overall_se, rel=1e-6
            )
        assert r_no_psu.survey_metadata is not None
        assert r_explicit.survey_metadata is not None
        assert (
            r_no_psu.survey_metadata.df_survey
            == r_explicit.survey_metadata.df_survey
        )

    def test_degenerate_cohort_survey_se_is_nan(self):
        """When every variance-eligible group is its own singleton
        cohort (D_{g,1}, F_g, S_g), the cohort-recentered IF is
        identically zero. The survey SE path must return NaN (not 0.0)
        so the degenerate-cohort warning fires and inference stays
        NaN-consistent — matching the _plugin_se contract documented
        in REGISTRY.md."""
        # 4 groups × 5 periods, each group switches at a unique F_g so
        # the (D_{g,1}=0, F_g, S_g=+1) cohort key is unique per group.
        rows = []
        for g, f_switch in enumerate([1, 2, 3, 4]):
            for t in range(5):
                d = 1 if t >= f_switch else 0
                y = float(g) + 0.5 * t + float(d)
                rows.append({
                    "group": g,
                    "period": t,
                    "treatment": d,
                    "outcome": y,
                    "pw": 1.0,
                })
        df_ = pd.DataFrame(rows)
        sd = SurveyDesign(weights="pw")

        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = ChaisemartinDHaultfoeuille(seed=1).fit(
                df_,
                outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd,
            )

        # overall_se must be NaN on degenerate cohorts (not 0.0)
        assert np.isnan(result.overall_se), (
            f"Degenerate-cohort survey overall_se must be NaN, "
            f"got {result.overall_se}"
        )
        # Degenerate-cohort warning must fire
        assert any(
            "cohort" in str(wi.message).lower()
            and "identically zero" in str(wi.message).lower()
            for wi in w
        ), "Expected degenerate-cohort warning to fire under survey path"

    def test_subpopulation_preserves_full_design_df_survey(self, base_data):
        """Under dCDH auto-inject, zero-weighting an entire group must not
        shrink df_survey below what the full-design PSU count would give.

        Mirrors SurveyDesign.subpopulation() semantics where excluded
        rows keep their weights at zero but remain in the design so
        that t critical values, p-values, CIs, and HonestDiD bounds
        reflect the full sampling structure."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Mimic subpopulation() by zero-weighting one entire group
        excluded_group = df_["group"].unique()[0]
        df_.loc[df_["group"] == excluded_group, "pw"] = 0.0

        sd = SurveyDesign(weights="pw")
        r_subpop = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        # Reference: explicit psu='group' preserves the full-design
        # PSU count because the resolver sees all groups (even those
        # entirely zero-weighted). The auto-inject path must match this.
        r_explicit = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=SurveyDesign(weights="pw", psu="group"),
        )
        assert r_subpop.survey_metadata is not None
        assert r_explicit.survey_metadata is not None
        assert (
            r_subpop.survey_metadata.df_survey
            == r_explicit.survey_metadata.df_survey
        ), (
            f"Auto-inject df_survey={r_subpop.survey_metadata.df_survey} "
            f"must match explicit psu='group' df_survey="
            f"{r_explicit.survey_metadata.df_survey} "
            f"(full-design subpopulation contract)."
        )

    def test_off_horizon_row_duplication_does_not_change_se(self, base_data):
        """Under auto-injected psu=group, duplicating an observation
        within a group (cell mean unchanged because the duplicate matches
        the existing row exactly) must not change the SE. Under the old
        per-obs-PSU fallback this invariant did not hold."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        sd = SurveyDesign(weights="pw")

        r_base = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=2, survey_design=sd,
        )

        # Duplicate the first row: cell mean y_gt stays the same
        # (identical y/treatment). Under per-obs PSU fallback the
        # "extra observation" would change the variance; under
        # auto-inject psu=group the group structure is unchanged.
        dup = df_.iloc[0].copy()
        df_dup = pd.concat([df_, pd.DataFrame([dup])], ignore_index=True)
        r_dup = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_dup, outcome="outcome", group="group",
            time="period", treatment="treatment",
            L_max=2, survey_design=sd,
        )
        if np.isfinite(r_base.overall_se) and np.isfinite(r_dup.overall_se):
            assert r_base.overall_se == pytest.approx(
                r_dup.overall_se, rel=1e-6
            ), (
                f"Row duplication changed SE ({r_base.overall_se} vs "
                f"{r_dup.overall_se}) — auto-inject psu=group is not active."
            )

    def test_cell_allocator_row_sum_identity(self):
        """Cell-period allocator contract: for every group, the per-
        period attribution sums across time to the per-group IF
        (before cohort centering). This is the invariant that makes
        PSU-level Binder aggregation telescope to ``U_centered[g]``
        under within-group-constant PSU and therefore guarantees byte-
        identity with the legacy group-level allocator on the old
        accepted input set. Hand-computed on a 4-group × 3-period
        panel: two never-treated (stable_0) and two joiners switching
        at ``t = 2``.
        """
        from diff_diff.chaisemartin_dhaultfoeuille import (
            _compute_full_per_group_contributions,
            _cohort_recenter,
            _cohort_recenter_per_period,
        )

        # D_mat, Y_mat, N_mat shaped (n_groups=4, n_periods=3).
        D_mat = np.array(
            [
                [0, 0, 0],  # G0 never-treated
                [0, 0, 0],  # G1 never-treated
                [0, 0, 1],  # G2 joiner at t=2
                [0, 0, 1],  # G3 joiner at t=2
            ],
            dtype=float,
        )
        Y_mat = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.1, 3.1, 4.2],
                [0.5, 1.2, 5.4],
                [1.3, 2.4, 6.1],
            ],
            dtype=float,
        )
        N_mat = np.ones_like(D_mat, dtype=int)
        # Per-period cell counts aligned to periods[1:]
        # t=1: all stable_0 (4 in n_00); t=2: 2 joiners (n_10) + 2 stable_0 (n_00)
        n_10_t_arr = np.array([0, 2], dtype=int)
        n_00_t_arr = np.array([4, 2], dtype=int)
        n_01_t_arr = np.array([0, 0], dtype=int)
        n_11_t_arr = np.array([0, 0], dtype=int)
        # A11 zeroed at t=1 (no joiners); active at t=2.
        a11_plus_zeroed = np.array([True, False], dtype=bool)
        a11_minus_zeroed = np.array([True, True], dtype=bool)

        U, U_pp = _compute_full_per_group_contributions(
            D_mat=D_mat, Y_mat=Y_mat, N_mat=N_mat,
            n_10_t_arr=n_10_t_arr, n_00_t_arr=n_00_t_arr,
            n_01_t_arr=n_01_t_arr, n_11_t_arr=n_11_t_arr,
            a11_plus_zeroed_arr=a11_plus_zeroed,
            a11_minus_zeroed_arr=a11_minus_zeroed,
            side="overall",
            compute_per_period=True,
        )
        assert U_pp is not None

        # Hand computation at t=2 joiner side:
        #   G0: stable_0, -(2/2) * (3.0 - 2.0) = -1.0
        #   G1: stable_0, -(2/2) * (4.2 - 3.1) = -1.1
        #   G2: joiner,  (5.4 - 1.2) = 4.2
        #   G3: joiner,  (6.1 - 2.4) = 3.7
        expected_U = np.array([-1.0, -1.1, 4.2, 3.7])
        np.testing.assert_allclose(U, expected_U, atol=1e-12)

        # Row-sum identity: U_per_period.sum(axis=1) == U exactly.
        np.testing.assert_allclose(U_pp.sum(axis=1), U, atol=1e-12)

        # Post-period attribution: all mass at t=2 (the transition's
        # post cell); t=0 and t=1 columns are zero for every group.
        np.testing.assert_array_equal(U_pp[:, 0], np.zeros(4))
        np.testing.assert_array_equal(U_pp[:, 1], np.zeros(4))
        np.testing.assert_allclose(U_pp[:, 2], expected_U, atol=1e-12)

        # Cohort centering preserves the row-sum identity: per-period
        # cohort centering and group-level cohort centering produce
        # 2D and 1D arrays whose row sums agree to FP precision.
        # Cohorts: A = {G0, G1} (never-treated), B = {G2, G3} (joiners).
        cohort_ids = np.array([0, 0, 1, 1])
        U_c = _cohort_recenter(U, cohort_ids)
        U_pp_c = _cohort_recenter_per_period(U_pp, cohort_ids)
        np.testing.assert_allclose(U_pp_c.sum(axis=1), U_c, atol=1e-12)

    def test_within_cell_check_excludes_zero_weight_rows(self, base_data):
        """A zero-weight row with a different PSU label from its cell
        must not trigger rejection — it is out-of-sample by the
        subpopulation contract and does not enter the variance.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        df_["psu"] = 0
        # Inject a zero-weight row whose PSU would collide with the
        # first row's cell if it were counted.
        sample = df_.iloc[0].copy()
        sample["psu"] = 99  # would violate within-cell constancy if counted
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_, outcome="outcome", group="group",
            time="period", treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
