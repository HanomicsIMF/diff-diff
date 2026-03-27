"""Tests for Phase 6 survey support: Subpopulation, DEFF, Replicate Weights."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DEFFDiagnostics,
    DifferenceInDifferences,
    SurveyDesign,
    compute_deff_diagnostics,
)


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def basic_did_data():
    """Simple 2x2 DiD panel with survey design columns."""
    np.random.seed(42)
    n_units = 80
    n_periods = 4
    rows = []
    for unit in range(n_units):
        treated = unit < 40
        ft = 3 if treated else 0
        stratum = unit // 20  # 4 strata
        psu = unit // 10  # 8 PSUs
        wt = 1.0 + 0.5 * (unit % 5)

        for t in range(1, n_periods + 1):
            y = 10.0 + unit * 0.05 + t * 0.3
            if treated and t >= 3:
                y += 2.0
            y += np.random.normal(0, 0.5)
            rows.append({
                "unit": unit, "time": t, "first_treat": ft,
                "outcome": y, "stratum": stratum, "psu": psu,
                "weight": wt, "fpc": 200.0,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Subpopulation Analysis
# =============================================================================


class TestSubpopulationAnalysis:
    """Tests for SurveyDesign.subpopulation()."""

    def test_basic_subpopulation(self, basic_did_data):
        """Subpopulation returns a new SurveyDesign and DataFrame."""
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        mask = basic_did_data["unit"] < 60  # Exclude last 20 units
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        assert new_sd.weights == "_subpop_weight"
        assert new_sd.strata == "stratum"
        assert new_sd.psu == "psu"
        assert "_subpop_weight" in new_data.columns

        # Excluded obs have zero weight
        excluded = new_data[~mask.values]
        assert (excluded["_subpop_weight"] == 0.0).all()

        # Included obs have original weights
        included = new_data[mask.values]
        np.testing.assert_array_equal(
            included["_subpop_weight"].values,
            basic_did_data.loc[mask.values, "weight"].values,
        )

    def test_subpopulation_preserves_design_structure(self, basic_did_data):
        """PSU/strata structure is preserved even when some PSUs are zeroed."""
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        # Exclude all units in stratum 0
        mask = basic_did_data["stratum"] != 0
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        # Should resolve without error — stratum 0 PSUs still exist in design
        resolved = new_sd.resolve(new_data)
        assert resolved.n_strata == 4  # All strata still present
        assert resolved.n_psu == 8  # All PSUs still present

    def test_subpopulation_with_did(self, basic_did_data):
        """Subpopulation works end-to-end with DifferenceInDifferences."""
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        mask = basic_did_data["unit"] < 60
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        # Add treated/post columns for the estimator
        new_data["treated"] = (new_data["first_treat"] > 0).astype(int)
        new_data["post"] = (new_data["time"] >= 3).astype(int)
        est = DifferenceInDifferences()
        result = est.fit(
            new_data, outcome="outcome", treatment="treated", time="post",
            survey_design=new_sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_subpopulation_preserves_more_psus_than_subset(self, basic_did_data):
        """Subpopulation retains all PSUs in the design; subset drops them."""
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        mask = basic_did_data["unit"] < 60

        # Subpopulation: preserves full design
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)
        resolved_subpop = new_sd.resolve(new_data)

        # Naive subset: loses PSUs in excluded observations
        subset = basic_did_data[mask.values].copy()
        resolved_subset = sd.resolve(subset)

        # Subpopulation preserves the full PSU count
        assert resolved_subpop.n_psu >= resolved_subset.n_psu

    def test_subpopulation_mask_as_column_name(self, basic_did_data):
        """Mask can be a column name string."""
        sd = SurveyDesign(weights="weight")
        data = basic_did_data.copy()
        data["include"] = data["unit"] < 60
        new_sd, new_data = sd.subpopulation(data, "include")
        assert (new_data.loc[~data["include"], "_subpop_weight"] == 0.0).all()

    def test_subpopulation_mask_as_callable(self, basic_did_data):
        """Mask can be a callable."""
        sd = SurveyDesign(weights="weight")
        new_sd, new_data = sd.subpopulation(
            basic_did_data, lambda df: df["unit"] < 60,
        )
        excluded = new_data[basic_did_data["unit"] >= 60]
        assert (excluded["_subpop_weight"] == 0.0).all()

    def test_subpopulation_no_weights_uses_uniform(self, basic_did_data):
        """When SurveyDesign has no weights, subpopulation uses 1.0."""
        sd = SurveyDesign(strata="stratum")
        mask = basic_did_data["unit"] < 60
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        included = new_data[mask.values]
        assert (included["_subpop_weight"] == 1.0).all()

    def test_subpopulation_all_excluded_raises(self, basic_did_data):
        """Excluding all obs should raise ValueError."""
        sd = SurveyDesign(weights="weight")
        mask = np.zeros(len(basic_did_data), dtype=bool)
        with pytest.raises(ValueError, match="excludes all"):
            sd.subpopulation(basic_did_data, mask)

    def test_subpopulation_none_excluded_identity(self, basic_did_data):
        """Including all obs should return weights unchanged."""
        sd = SurveyDesign(weights="weight")
        mask = np.ones(len(basic_did_data), dtype=bool)
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)
        np.testing.assert_array_equal(
            new_data["_subpop_weight"].values,
            basic_did_data["weight"].values,
        )

    def test_subpopulation_mask_length_mismatch_raises(self, basic_did_data):
        """Mask length must match data length."""
        sd = SurveyDesign(weights="weight")
        mask = np.ones(10, dtype=bool)
        with pytest.raises(ValueError, match="length"):
            sd.subpopulation(basic_did_data, mask)

    def test_zero_weights_allowed_in_resolve(self):
        """resolve() should accept zero weights (no longer strictly positive)."""
        data = pd.DataFrame({
            "w": [1.0, 2.0, 0.0, 1.5],
            "y": [1, 2, 3, 4],
        })
        sd = SurveyDesign(weights="w")
        resolved = sd.resolve(data)
        assert resolved.weights[2] == 0.0  # Zero preserved after normalization

    def test_negative_weights_rejected(self):
        """Negative weights should still be rejected."""
        data = pd.DataFrame({"w": [1.0, -1.0, 2.0], "y": [1, 2, 3]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="non-negative"):
            sd.resolve(data)


# =============================================================================
# DEFF Diagnostics
# =============================================================================


class TestDEFFDiagnostics:
    """Tests for per-coefficient design effect diagnostics."""

    @staticmethod
    def _fit_lr_with_survey(data, sd):
        """Fit a LinearRegression with survey design for testing DEFF."""
        from diff_diff.linalg import LinearRegression
        from diff_diff.survey import _resolve_survey_for_fit

        resolved, sw, swt, sm = _resolve_survey_for_fit(sd, data, "analytical")
        y = data["outcome"].values
        X = np.column_stack([
            np.ones(len(data)),
            data["treated"].values,
            data["post"].values,
            (data["treated"] * data["post"]).values,
        ])
        reg = LinearRegression(
            include_intercept=False, weights=sw,
            weight_type=swt, survey_design=resolved,
        )
        reg.fit(X, y)
        return reg

    def test_uniform_weights_deff_near_one(self, basic_did_data):
        """With uniform weights and no clustering, DEFF should be ~1."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)
        data["uniform_wt"] = 1.0

        sd = SurveyDesign(weights="uniform_wt")
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()
        assert isinstance(deff, DEFFDiagnostics)
        # With uniform weights, DEFF should be close to 1
        for d in deff.deff:
            if np.isfinite(d):
                assert d == pytest.approx(1.0, abs=0.15)

    def test_clustered_design_deff_gt_one(self, basic_did_data):
        """With clustered design, DEFF should be > 1 for at least some coefs."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()
        # At least one coefficient should have DEFF > 1 (design effect)
        assert np.any(deff.deff[np.isfinite(deff.deff)] > 1.0)

    def test_deff_srs_se_matches_robust_vcov(self, basic_did_data):
        """SRS SE in DEFF diagnostics should match compute_robust_vcov."""
        from diff_diff.linalg import compute_robust_vcov

        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()

        # Manually compute SRS SE
        srs_vcov = compute_robust_vcov(
            reg._X, reg.residuals_,
            cluster_ids=None, weights=reg.weights,
            weight_type=reg.weight_type,
        )
        expected_srs_se = np.sqrt(np.diag(srs_vcov))
        np.testing.assert_allclose(deff.srs_se, expected_srs_se, rtol=1e-10)

    def test_standalone_compute_deff_diagnostics(self, basic_did_data):
        """compute_deff_diagnostics() works as a standalone function."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        reg = self._fit_lr_with_survey(data, sd)
        deff = compute_deff_diagnostics(
            reg._X, reg.residuals_, reg.vcov_,
            reg.weights, weight_type=reg.weight_type,
            coefficient_names=["intercept", "treated", "post", "treated:post"],
        )
        assert deff.coefficient_names is not None
        assert len(deff.coefficient_names) == 4
        assert len(deff.deff) == 4

    def test_deff_effective_n(self, basic_did_data):
        """effective_n should be n / DEFF."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()
        n = reg.n_obs_
        for i in range(len(deff.deff)):
            if np.isfinite(deff.deff[i]) and deff.deff[i] > 0:
                assert deff.effective_n[i] == pytest.approx(
                    n / deff.deff[i], rel=1e-10
                )

    def test_compute_deff_no_survey_raises(self, basic_did_data):
        """compute_deff() without survey should raise."""
        from diff_diff.linalg import LinearRegression

        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        y = data["outcome"].values
        X = np.column_stack([np.ones(len(data)), data["treated"].values])
        reg = LinearRegression(include_intercept=False)
        reg.fit(X, y)
        with pytest.raises(ValueError, match="survey design"):
            reg.compute_deff()


# =============================================================================
# Replicate Weight Variance
# =============================================================================


@pytest.fixture
def replicate_data():
    """Data with full-sample + replicate weight columns for testing."""
    np.random.seed(123)
    n = 200
    k = 20  # number of replicates

    # Simple regression: y = 1 + 2*x + eps
    x = np.random.randn(n)
    eps = np.random.randn(n) * 0.5
    y = 1.0 + 2.0 * x + eps

    # Full-sample weights (non-uniform)
    w = 1.0 + np.random.exponential(0.5, n)

    data = pd.DataFrame({"x": x, "y": y, "weight": w})

    # Generate JK1 replicate weights (delete-1 jackknife on PSU clusters)
    # Each replicate drops one cluster and rescales remaining weights
    cluster_size = n // k
    for r in range(k):
        w_r = w.copy()
        start = r * cluster_size
        end = min((r + 1) * cluster_size, n)
        w_r[start:end] = 0.0
        # Rescale remaining: multiply by k/(k-1)
        w_r[w_r > 0] *= k / (k - 1)
        data[f"rep_{r}"] = w_r

    return data, [f"rep_{r}" for r in range(k)]


class TestReplicateWeightVariance:
    """Tests for replicate weight variance estimation."""

    def test_brr_validation(self):
        """BRR validation: method + weights required together."""
        with pytest.raises(ValueError, match="replicate_method must be provided"):
            SurveyDesign(weights="w", replicate_weights=["r1", "r2"])
        with pytest.raises(ValueError, match="replicate_weights must be provided"):
            SurveyDesign(weights="w", replicate_method="BRR")

    def test_fay_rho_validation(self):
        """Fay requires rho in (0, 1)."""
        with pytest.raises(ValueError, match="fay_rho"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="Fay", fay_rho=0.0,
            )
        with pytest.raises(ValueError, match="fay_rho"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="BRR", fay_rho=0.5,
            )

    def test_replicate_strata_mutual_exclusion(self):
        """Replicate weights cannot be combined with strata/psu/fpc."""
        with pytest.raises(ValueError, match="cannot be combined"):
            SurveyDesign(
                weights="w", strata="s",
                replicate_weights=["r1", "r2"], replicate_method="BRR",
            )

    def test_replicate_requires_weights(self):
        """Full-sample weights required alongside replicates."""
        with pytest.raises(ValueError, match="Full-sample weights"):
            SurveyDesign(
                replicate_weights=["r1", "r2"], replicate_method="BRR",
            )

    def test_jk1_resolve(self, replicate_data):
        """JK1 replicates resolve correctly."""
        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)
        assert resolved.uses_replicate_variance
        assert resolved.n_replicates == len(rep_cols)
        assert resolved.replicate_weights.shape == (len(data), len(rep_cols))
        assert resolved.strata is None  # Short-circuited
        assert resolved.df_survey == len(rep_cols) - 1

    def test_jk1_vcov_finite(self, replicate_data):
        """JK1 replicate vcov produces finite results."""
        from diff_diff.linalg import LinearRegression

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        reg = LinearRegression(
            include_intercept=False, weights=resolved.weights,
            weight_type="pweight", survey_design=resolved,
        )
        reg.fit(X, y)

        assert np.all(np.isfinite(reg.coefficients_))
        assert np.all(np.isfinite(np.diag(reg.vcov_)))
        # SE should be positive
        se = np.sqrt(np.diag(reg.vcov_))
        assert np.all(se > 0)

    def test_brr_vcov(self, replicate_data):
        """BRR uses 1/R factor."""
        from diff_diff.survey import compute_replicate_vcov

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved = sd.resolve(data)
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        from diff_diff.linalg import solve_ols
        coef, _, _ = solve_ols(
            X, y, weights=resolved.weights, weight_type="pweight",
        )

        vcov = compute_replicate_vcov(X, y, coef, resolved)
        assert np.all(np.isfinite(np.diag(vcov)))

    def test_fay_inflates_over_brr(self, replicate_data):
        """Fay's BRR with rho > 0 produces larger variance than BRR."""
        from diff_diff.survey import compute_replicate_vcov
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd_brr = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved_brr = sd_brr.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved_brr.weights)
        vcov_brr = compute_replicate_vcov(X, y, coef, resolved_brr)

        sd_fay = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="Fay", fay_rho=0.5,
        )
        resolved_fay = sd_fay.resolve(data)
        vcov_fay = compute_replicate_vcov(X, y, coef, resolved_fay)

        # Fay variance = BRR variance / (1-rho)^2 > BRR variance
        assert np.all(np.diag(vcov_fay) > np.diag(vcov_brr))

    def test_replicate_metadata(self, replicate_data):
        """SurveyMetadata should include replicate info."""
        from diff_diff.survey import compute_survey_metadata

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)
        sm = compute_survey_metadata(resolved, data["weight"].values)
        assert sm.replicate_method == "JK1"
        assert sm.n_replicates == len(rep_cols)
        assert sm.df_survey == len(rep_cols) - 1

    def test_replicate_with_did(self, replicate_data):
        """Replicate weights work end-to-end with DifferenceInDifferences."""
        data, rep_cols = replicate_data
        # Add DiD structure
        n = len(data)
        data["treated"] = (np.arange(n) < n // 2).astype(int)
        data["post"] = (np.arange(n) % 4 >= 2).astype(int)
        data["outcome"] = data["y"] + 1.5 * data["treated"] * data["post"]

        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        est = DifferenceInDifferences()
        result = est.fit(
            data, outcome="outcome", treatment="treated", time="post",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None
        assert result.survey_metadata.replicate_method == "JK1"

    def test_replicate_if_variance(self, replicate_data):
        """IF-based replicate variance produces finite results."""
        from diff_diff.survey import compute_replicate_if_variance

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)

        # Synthetic influence function
        psi = np.random.randn(len(data)) * 0.1
        var = compute_replicate_if_variance(psi, resolved)
        assert np.isfinite(var)
        assert var >= 0

    def test_min_2_replicates(self, replicate_data):
        """Need at least 2 replicate columns."""
        data, _ = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=["rep_0"],
            replicate_method="BRR",
        )
        with pytest.raises(ValueError, match="At least 2"):
            sd.resolve(data)

    def test_jkn_requires_replicate_strata(self):
        """JKn must have replicate_strata."""
        with pytest.raises(ValueError, match="replicate_strata is required"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="JKn",
            )

    def test_jkn_replicate_strata_length_mismatch(self):
        """replicate_strata length must match replicate_weights."""
        with pytest.raises(ValueError, match="length"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="JKn", replicate_strata=[0],
            )

    def test_jkn_variance(self, replicate_data):
        """JKn with per-stratum scaling produces finite results."""
        from diff_diff.survey import compute_replicate_vcov
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        # Assign first half of replicates to stratum 0, second half to stratum 1
        n_rep = len(rep_cols)
        strata = [0] * (n_rep // 2) + [1] * (n_rep - n_rep // 2)

        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JKn", replicate_strata=strata,
        )
        resolved = sd.resolve(data)
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        coef, _, _ = solve_ols(X, y, weights=resolved.weights)
        vcov = compute_replicate_vcov(X, y, coef, resolved)
        assert np.all(np.isfinite(np.diag(vcov)))
        assert np.all(np.diag(vcov) > 0)

    def test_replicate_if_scale_matches_analytical(self):
        """Replicate IF variance should match analytical sum(psi^2) for SRS."""
        from diff_diff.survey import compute_replicate_if_variance

        np.random.seed(42)
        n = 50
        psi = np.random.randn(n) * 0.1

        # Build JK1 replicates: delete-1 jackknife
        w_full = np.ones(n)
        rep_cols = []
        data = pd.DataFrame({"w": w_full})
        for r in range(n):
            w_r = np.ones(n) * n / (n - 1)
            w_r[r] = 0.0
            col = f"rep_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="w", replicate_weights=rep_cols, replicate_method="JK1",
        )
        resolved = sd.resolve(data)

        v_rep = compute_replicate_if_variance(psi, resolved)
        v_analytical = float(np.sum(psi**2))

        # JK1 gives (n-1)/n * sum(...) which should approximate sum(psi^2)
        assert v_rep == pytest.approx(v_analytical, rel=0.05), (
            f"Replicate IF variance {v_rep:.6f} does not match analytical "
            f"{v_analytical:.6f}"
        )

    def test_replicate_if_matches_survey_if_variance(self):
        """Replicate IF variance should approximate compute_survey_if_variance
        when JK1 replicates match the PSU structure."""
        from diff_diff.survey import (
            compute_replicate_if_variance,
            compute_survey_if_variance,
            ResolvedSurveyDesign,
        )

        np.random.seed(123)
        n = 60
        n_psu = 15
        psu = np.repeat(np.arange(n_psu), n // n_psu)
        psi = np.random.randn(n) * 0.1
        weights = np.ones(n)

        # TSL variance with PSU clustering
        resolved_tsl = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=psu, fpc=None,
            n_strata=0, n_psu=n_psu, lonely_psu="remove",
        )
        v_tsl = compute_survey_if_variance(psi, resolved_tsl)

        # JK1 replicates: delete one PSU at a time, rescale remaining
        rep_arr = np.zeros((n, n_psu))
        for r in range(n_psu):
            w_r = weights.copy()
            w_r[psu == r] = 0.0
            w_r[psu != r] *= n_psu / (n_psu - 1)
            rep_arr[:, r] = w_r
        # Normalize each column to sum=n (matching resolve() normalization)
        for c in range(n_psu):
            cs = rep_arr[:, c].sum()
            if cs > 0:
                rep_arr[:, c] *= n / cs

        resolved_rep = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1",
            n_replicates=n_psu,
        )
        v_rep = compute_replicate_if_variance(psi, resolved_rep)

        # Should be in the same ballpark (within 50% — different estimators
        # of the same quantity)
        assert v_rep > 0 and v_tsl > 0
        ratio = v_rep / v_tsl
        assert 0.5 < ratio < 2.0, (
            f"Replicate IF var ({v_rep:.6f}) vs TSL IF var ({v_tsl:.6f}) "
            f"ratio={ratio:.2f}, expected within [0.5, 2.0]"
        )

    def test_all_zero_weights_rejected_by_solve_ols(self):
        """solve_ols should reject all-zero weight vectors."""
        from diff_diff.linalg import solve_ols

        X = np.column_stack([np.ones(10), np.random.randn(10)])
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="sum to zero"):
            solve_ols(X, y, weights=np.zeros(10))


# =============================================================================
# Estimator-Level Replicate Weight Tests
# =============================================================================


class TestEstimatorReplicateWeights:
    """End-to-end tests for replicate weights with each estimator."""

    @staticmethod
    def _make_staggered_replicate_data():
        """Create staggered panel with replicate weight columns."""
        np.random.seed(42)
        n_units, n_periods = 60, 6
        n_rep = 15
        rows = []
        for unit in range(n_units):
            if unit < 20:
                ft = 3
            elif unit < 40:
                ft = 5
            else:
                ft = 0
            wt = 1.0 + 0.3 * (unit % 5)
            for t in range(1, n_periods + 1):
                y = 10.0 + unit * 0.05 + t * 0.2
                if ft > 0 and t >= ft:
                    y += 2.0
                y += np.random.normal(0, 0.5)
                rows.append({
                    "unit": unit, "time": t, "first_treat": ft,
                    "outcome": y, "weight": wt,
                })
        data = pd.DataFrame(rows)
        # Generate JK1 replicates (delete-cluster jackknife)
        cluster_size = n_units // n_rep
        rep_cols = []
        for r in range(n_rep):
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n_units)
            w_r = data["weight"].values.copy()
            mask = (data["unit"] >= start) & (data["unit"] < end)
            w_r[mask] = 0.0
            w_r[~mask] *= n_rep / (n_rep - 1)
            col = f"rep_{r}"
            data[col] = w_r
            rep_cols.append(col)
        return data, rep_cols

    def test_callaway_santanna_replicate(self):
        """CallawaySantAnna with replicate weights produces finite results."""
        from diff_diff import CallawaySantAnna

        data, rep_cols = self._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        result = CallawaySantAnna(estimation_method="reg", n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_efficient_did_replicate(self):
        """EfficientDiD with replicate weights produces finite results."""
        from diff_diff import EfficientDiD

        data, rep_cols = self._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        result = EfficientDiD(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_subpopulation_preserves_replicate_metadata(self):
        """subpopulation() should preserve replicate design fields."""
        data, rep_cols = self._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        mask = data["unit"] < 40
        new_sd, new_data = sd.subpopulation(data, mask)

        assert new_sd.replicate_method == "JK1"
        assert new_sd.replicate_weights == rep_cols
        # Excluded obs have zero replicate weights
        excluded = new_data[~mask.values]
        for col in rep_cols:
            assert (excluded[col] == 0.0).all()
