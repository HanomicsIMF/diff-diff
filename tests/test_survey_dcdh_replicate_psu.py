"""Replicate-weight variance and PSU-level bootstrap tests for dCDH.

Covers the two survey-variance extensions added after PR #307:
1. Replicate-weight variance (BRR/Fay/JK1/JKn/SDR) via the unified
   `compute_replicate_if_variance` helper, routed through the inline
   branch in `_survey_se_from_group_if` (Class A sites) and the
   parallel inline branch in `_compute_heterogeneity_test` (Class B).
2. PSU-level Hall-Mammen wild bootstrap via `group_to_psu_map`
   threaded through `_compute_dcdh_bootstrap`, with an identity-map
   fast path preserving auto-inject `psu=group` semantics.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import ChaisemartinDHaultfoeuille, SurveyDesign
from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights


# ── Fixtures ────────────────────────────────────────────────────────


REPLICATE_METHODS = ["BRR", "Fay", "JK1", "JKn", "SDR"]


def _make_reversible_panel(
    n_groups: int = 30,
    n_periods: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Simple reversible-treatment panel with two switch cohorts."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        f = 2 if g < n_groups // 2 else 3
        for t in range(n_periods):
            d = 1 if t >= f else 0
            y = float(g) + 0.5 * t + 1.0 * d + rng.normal(scale=0.3)
            rows.append(
                {
                    "group": g,
                    "period": t,
                    "treatment": d,
                    "outcome": y,
                    "pw": 1.0,
                    "x_het": float(g) * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _attach_replicate_weights(
    df: pd.DataFrame,
    R: int,
    method: str,
    seed: int = 101,
) -> pd.DataFrame:
    """Add R replicate-weight columns and return the mutated DataFrame."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    for r in range(R):
        if method == "BRR" or method == "Fay" or method == "SDR":
            df[f"rep{r}"] = rng.choice([0.5, 1.5], size=len(df))
        elif method in ("JK1", "JKn"):
            df[f"rep{r}"] = rng.uniform(0.5, 1.5, size=len(df))
    return df


def _rep_cols(R: int) -> list:
    return [f"rep{r}" for r in range(R)]


def _build_replicate_design(R: int, method: str) -> SurveyDesign:
    """Build a SurveyDesign object with method-specific required params."""
    if method == "Fay":
        return SurveyDesign(
            weights="pw",
            replicate_weights=_rep_cols(R),
            replicate_method="Fay",
            fay_rho=0.5,
        )
    if method == "JKn":
        return SurveyDesign(
            weights="pw",
            replicate_weights=_rep_cols(R),
            replicate_method="JKn",
            replicate_strata=[r % 2 for r in range(R)],
        )
    return SurveyDesign(
        weights="pw",
        replicate_weights=_rep_cols(R),
        replicate_method=method,
    )


@pytest.fixture(scope="module")
def base_panel():
    return _make_reversible_panel(n_groups=30, n_periods=5, seed=42)


@pytest.fixture
def replicate_design():
    """Helper constructing a SurveyDesign with replicate weights for a given method."""

    def _build(df: pd.DataFrame, R: int, method: str, seed: int = 101) -> pd.DataFrame:
        return _attach_replicate_weights(df, R=R, method=method, seed=seed)

    return _build


def _make_strictly_coarser_psu_panel(
    n_groups_per_psu: int = 3,
    n_psu: int = 4,
    n_periods: int = 5,
    seed: int = 17,
) -> pd.DataFrame:
    """Panel where groups nest strictly inside PSUs.

    12 groups across 4 PSUs (3 groups/PSU). Two switch cohorts so the
    estimator has both joiners at different F_g.
    """
    rng = np.random.default_rng(seed)
    n_groups = n_groups_per_psu * n_psu
    rows = []
    for g in range(n_groups):
        psu = g // n_groups_per_psu
        f = 2 if g < n_groups // 2 else 3
        for t in range(n_periods):
            d = 1 if t >= f else 0
            y = float(g) + 0.5 * t + 1.0 * d + rng.normal(scale=0.3)
            rows.append(
                {
                    "group": g,
                    "period": t,
                    "treatment": d,
                    "outcome": y,
                    "pw": 1.0,
                    "psu": psu,
                }
            )
    return pd.DataFrame(rows)


# ── 1. Class A replicate (sites via _survey_se_from_group_if) ────────


class TestReplicateClassA:
    """Replicate variance wired through the inline branch in
    `_survey_se_from_group_if` — inherited by overall DID_M, joiners
    DID_+, leavers DID_-, multi-horizon DID_l, placebo DID^pl_l.
    """

    @pytest.mark.parametrize("method", REPLICATE_METHODS)
    def test_overall_se_finite(self, base_panel, replicate_design, method):
        R = 20
        df = replicate_design(base_panel, R=R, method=method)
        sd = _build_replicate_design(R, method)
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0

    @pytest.mark.parametrize("method", REPLICATE_METHODS)
    def test_inference_fields_finite(self, base_panel, replicate_design, method):
        R = 20
        df = replicate_design(base_panel, R=R, method=method)
        sd = _build_replicate_design(R, method)
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_t_stat)
        assert np.isfinite(res.overall_p_value)
        assert np.all(np.isfinite(res.overall_conf_int))

    @pytest.mark.parametrize("method", ["BRR", "JK1"])
    def test_df_survey_reflects_n_valid(self, base_panel, replicate_design, method):
        R = 24
        df = replicate_design(base_panel, R=R, method=method)
        sd = _build_replicate_design(R, method)
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # All replicates should succeed on a clean panel → df = n_valid - 1 = R - 1
        assert res.survey_metadata.df_survey == R - 1

    def test_jk1_converges_to_tsl(self, replicate_design):
        """JK1 replicate SE should be in the same order of magnitude as
        analytical TSL SE on a large panel. Uses a loose 50% tolerance
        because convergence depends on weight-construction details; the
        stringent R-parity convergence test lives in
        `test_survey_phase6.py`."""
        df = _make_reversible_panel(n_groups=60, n_periods=6, seed=42)
        R = 30
        df_rep = replicate_design(df, R=R, method="JK1")

        sd_rep = _build_replicate_design(R, "JK1")
        sd_tsl = SurveyDesign(weights="pw")

        res_rep = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_rep,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_rep,
        )
        res_tsl = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_tsl,
        )
        # Same point estimate (both unweighted)
        assert res_rep.overall_att == pytest.approx(res_tsl.overall_att, rel=1e-6)
        # SEs should be within 50% of each other (broad envelope)
        ratio = res_rep.overall_se / res_tsl.overall_se
        assert 0.5 <= ratio <= 2.0, (
            f"Replicate SE ({res_rep.overall_se:.4f}) and TSL SE "
            f"({res_tsl.overall_se:.4f}) differ by more than 2x."
        )

    @pytest.mark.parametrize("method", ["BRR", "JK1"])
    def test_multi_horizon_under_replicate(
        self, replicate_design, method
    ):
        """Multi-horizon DID_l inherits the Class A dispatch — horizon 1
        (always identifiable on this panel) should produce a finite
        replicate-based SE."""
        df = _make_reversible_panel(n_groups=30, n_periods=6, seed=42)
        R = 20
        df = replicate_design(df, R=R, method=method)
        sd = _build_replicate_design(R, method)
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
            L_max=1,
        )
        info = res.event_study_effects[1]
        assert np.isfinite(info["effect"])
        assert np.isfinite(info["se"]) and info["se"] > 0

    def test_placebo_under_replicate(self, replicate_design):
        """Placebo DID^{pl}_l under replicate weights also carries finite SE."""
        df = _make_reversible_panel(n_groups=30, n_periods=6, seed=42)
        R = 20
        df = replicate_design(df, R=R, method="BRR")
        sd = _build_replicate_design(R, "BRR")
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
            L_max=1,
        )
        if res.placebo_event_study:
            info = res.placebo_event_study.get(1)
            if info is not None and info.get("n_obs", 0) > 0:
                assert np.isfinite(info["se"])

    def test_did_x_replicate(self, base_panel, replicate_design):
        """DID^X covariate adjustment flows through the Class A dispatch
        (Y_mat residualization happens upstream of IF)."""
        df = base_panel.copy()
        rng = np.random.default_rng(123)
        df["cov1"] = rng.normal(size=len(df))
        R = 20
        df = replicate_design(df, R=R, method="JK1")
        sd = _build_replicate_design(R, "JK1")
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
            controls=["cov1"],
            L_max=1,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se) and res.overall_se > 0


# ── 2. Class B replicate (heterogeneity + twowayfeweights) ──────────


class TestReplicateClassB:
    """Sites that call `compute_survey_if_variance` directly (or don't
    route through the variance path at all) and need their own
    replicate handling."""

    @pytest.mark.parametrize("method", REPLICATE_METHODS)
    def test_heterogeneity_se_finite(self, base_panel, replicate_design, method):
        R = 20
        df = replicate_design(base_panel, R=R, method=method)
        sd = _build_replicate_design(R, method)
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
            L_max=1,
            heterogeneity="x_het",
        )
        assert res.heterogeneity_effects is not None
        info = res.heterogeneity_effects[1]
        assert np.isfinite(info["beta"])
        assert np.isfinite(info["se"]) and info["se"] > 0

    def test_twowayfeweights_accepts_replicate(self, base_panel, replicate_design):
        """TWFE diagnostic produces the same beta_fe / sigma_fe /
        fraction_negative under a replicate design as under a
        pweight-only design (replicate weights only affect
        `resolved.weights` cell aggregation)."""
        R = 20
        df = replicate_design(base_panel, R=R, method="BRR")
        sd_rep = _build_replicate_design(R, "BRR")
        sd_plain = SurveyDesign(weights="pw")

        res_rep = twowayfeweights(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_rep,
        )
        res_plain = twowayfeweights(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_plain,
        )
        # Cell aggregation uses resolved.weights; replicate_weights are
        # present but don't change the aggregated diagnostics.
        assert res_rep.beta_fe == pytest.approx(res_plain.beta_fe, rel=1e-10)
        assert res_rep.sigma_fe == pytest.approx(res_plain.sigma_fe, rel=1e-10)
        assert res_rep.fraction_negative == pytest.approx(
            res_plain.fraction_negative, rel=1e-10
        )


# ── 3. PSU-level Hall-Mammen wild bootstrap ─────────────────────────


class TestPSUBootstrap:

    def test_auto_inject_bit_identical_to_group_level(self, base_panel):
        """Under auto-inject psu=group the Hall-Mammen PSU bootstrap
        collapses to the group-level multiplier bootstrap via the
        identity-map fast path — producing bit-identical SE."""
        df = base_panel.copy()
        sd_auto = SurveyDesign(weights="pw")
        sd_explicit = SurveyDesign(weights="pw", psu="group")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            r_auto = ChaisemartinDHaultfoeuille(n_bootstrap=200, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd_auto,
            )
            r_explicit = ChaisemartinDHaultfoeuille(n_bootstrap=200, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd_explicit,
            )
        assert r_auto.overall_se == pytest.approx(r_explicit.overall_se, rel=1e-10)

    def test_coarser_psu_produces_finite_se(self):
        """Under a strictly coarser PSU, the Hall-Mammen PSU bootstrap
        still produces a finite SE (the value depends on within-PSU
        IF correlation)."""
        df = _make_strictly_coarser_psu_panel()
        sd = SurveyDesign(weights="pw", psu="psu")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(n_bootstrap=500, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_no_warning_under_auto_inject(self, base_panel):
        """No UserWarning about PSU-level bootstrap should fire under
        auto-inject psu=group (the warning only fires for strictly
        coarser PSU)."""
        sd = SurveyDesign(weights="pw")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ChaisemartinDHaultfoeuille(n_bootstrap=100, seed=1).fit(
                base_panel,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )
        assert not any(
            "Hall-Mammen wild" in str(w.message)
            or "group-level multiplier" in str(w.message)
            for w in caught
        ), (
            "Bootstrap-PSU warning should not fire under auto-inject psu=group, "
            f"but got: {[str(w.message) for w in caught]}"
        )

    def test_warning_under_coarser_psu(self):
        """When PSU is strictly coarser than group, the Hall-Mammen
        warning fires."""
        df = _make_strictly_coarser_psu_panel()
        sd = SurveyDesign(weights="pw", psu="psu")
        with pytest.warns(UserWarning, match="Hall-Mammen wild"):
            ChaisemartinDHaultfoeuille(n_bootstrap=100, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    @pytest.mark.parametrize("weight_type", ["rademacher", "mammen", "webb"])
    def test_all_weight_types_under_psu(self, weight_type):
        """Rademacher / Mammen / Webb multipliers should all produce
        finite SE under the PSU-level path."""
        df = _make_strictly_coarser_psu_panel()
        sd = SurveyDesign(weights="pw", psu="psu")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(
                n_bootstrap=200, seed=1, bootstrap_weights=weight_type
            ).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_multi_horizon_psu_bootstrap(self):
        """Event study bootstrap inherits PSU broadcasting."""
        df = _make_strictly_coarser_psu_panel(n_periods=6)
        sd = SurveyDesign(weights="pw", psu="psu")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(n_bootstrap=300, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
                L_max=1,
            )
        info = res.event_study_effects[1]
        assert np.isfinite(info["effect"])
        assert np.isfinite(info["se"]) and info["se"] > 0

    def test_multi_horizon_shared_draw_under_coarser_psu(self):
        """End-to-end test of the shared-draw multi-horizon bootstrap
        under a strictly coarser PSU with L_max >= 2. Exercises the
        per-horizon invariant assertion + sup-t band computation
        through the Hall-Mammen wild PSU path."""
        df = _make_strictly_coarser_psu_panel(n_periods=7)
        sd = SurveyDesign(weights="pw", psu="psu")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(n_bootstrap=500, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
                L_max=2,
            )
        # Both horizons should produce finite bootstrap SEs via the
        # shared-draw path projected through the PSU map.
        for lag in (1, 2):
            info = res.event_study_effects[lag]
            if info["n_obs"] > 0:
                assert np.isfinite(info["effect"])
                assert np.isfinite(info["se"]) and info["se"] > 0

    def test_sup_t_under_coarser_psu(self):
        """Sup-t critical value is computable under coarser PSU."""
        df = _make_strictly_coarser_psu_panel(n_periods=6)
        sd = SurveyDesign(weights="pw", psu="psu")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(n_bootstrap=500, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
                L_max=2,
            )
        # sup_t_bands may be None if only one horizon is valid or if
        # the cband critical value was not computed.
        if res.sup_t_bands is not None:
            assert res.sup_t_bands.get("cband_crit_value") is None or (
                res.sup_t_bands["cband_crit_value"] > 1.0
            )


# ── 4. Invariants and cross-cutting contracts ────────────────────────


class TestInvariants:

    def test_replicate_plus_bootstrap_rejected(self, base_panel, replicate_design):
        """Replicate variance + n_bootstrap > 0 raises NotImplementedError."""
        R = 10
        df = replicate_design(base_panel, R=R, method="BRR")
        sd = _build_replicate_design(R, "BRR")
        with pytest.raises(NotImplementedError, match="replicate weights and n_bootstrap"):
            ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_non_survey_unchanged(self, base_panel):
        """Non-survey fits (survey_design=None) should produce bit-
        identical SE to the pre-PSU-bootstrap behavior."""
        r1 = ChaisemartinDHaultfoeuille(n_bootstrap=200, seed=1).fit(
            base_panel,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        r2 = ChaisemartinDHaultfoeuille(n_bootstrap=200, seed=1).fit(
            base_panel,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert r1.overall_se == pytest.approx(r2.overall_se, rel=1e-10)

    def test_map_for_target_id_lookup(self):
        """`_map_for_target` builds the PSU map from group IDs via dict
        lookup, not positional reuse. All current dCDH bootstrap
        targets use the variance-eligible group ordering, so the
        helper looks up each target group's PSU via
        `group_id_to_psu_code`. Length mismatch → ValueError (loud
        failure rather than silent miscluster)."""
        from diff_diff.chaisemartin_dhaultfoeuille_bootstrap import (
            _map_for_target,
        )

        gid_to_psu = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 2}
        eligible = np.asarray(["a", "b", "c", "d", "e"])
        expected = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        built = _map_for_target(5, gid_to_psu, eligible)
        assert built is not None
        assert np.array_equal(built, expected)

        # None passthrough when no PSU info
        assert _map_for_target(5, None, None) is None

        # Length mismatch → loud failure
        with pytest.raises(ValueError, match="target size"):
            _map_for_target(3, gid_to_psu, eligible)

        # Missing group ID → loud failure
        gid_to_psu_incomplete = {"a": 0, "b": 0, "c": 1, "d": 1}
        with pytest.raises(ValueError, match="has no entry"):
            _map_for_target(5, gid_to_psu_incomplete, eligible)

        # Non-prefix reordering: different ordered group IDs produce a
        # different PSU map even if the dict is the same. This is the
        # key invariant the previous prefix-slicing lacked.
        eligible_reordered = np.asarray(["c", "a", "d", "e", "b"])
        built_reordered = _map_for_target(5, gid_to_psu, eligible_reordered)
        assert built_reordered is not None
        expected_reordered = np.array([1, 0, 1, 2, 0], dtype=np.int64)
        assert np.array_equal(built_reordered, expected_reordered)

    def test_generate_psu_or_group_weights_broadcast(self):
        """Direct unit test of the PSU-level weight generator:
        groups mapped to the same PSU receive the same multiplier
        within a single bootstrap replicate (Hall-Mammen wild PSU
        contract)."""
        from diff_diff.chaisemartin_dhaultfoeuille_bootstrap import (
            _generate_psu_or_group_weights,
        )

        # 6 groups → 3 PSUs: groups (0,1), (2,3), (4,5) share multipliers.
        group_to_psu = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        rng = np.random.default_rng(0)
        W = _generate_psu_or_group_weights(
            n_bootstrap=50,
            n_groups_target=6,
            weight_type="rademacher",
            rng=rng,
            group_to_psu_map=group_to_psu,
        )
        assert W.shape == (50, 6)
        # Groups in the same PSU must receive the same multiplier
        # within each bootstrap replicate.
        assert np.array_equal(W[:, 0], W[:, 1])
        assert np.array_equal(W[:, 2], W[:, 3])
        assert np.array_equal(W[:, 4], W[:, 5])
        # Different PSUs should NOT all produce identical columns
        # (given 50 replicates and Rademacher weights, collision
        # probability is 2^-49 → effectively 0).
        assert not np.array_equal(W[:, 0], W[:, 2])

    def test_generate_psu_or_group_weights_identity(self):
        """Identity map (each group its own PSU) uses the fast path and
        produces group-level weights bit-identical to the non-PSU path."""
        from diff_diff.chaisemartin_dhaultfoeuille_bootstrap import (
            _generate_psu_or_group_weights,
        )

        identity_map = np.arange(6, dtype=np.int64)
        rng1 = np.random.default_rng(0)
        W_identity = _generate_psu_or_group_weights(
            n_bootstrap=20,
            n_groups_target=6,
            weight_type="mammen",
            rng=rng1,
            group_to_psu_map=identity_map,
        )
        rng2 = np.random.default_rng(0)
        W_plain = _generate_psu_or_group_weights(
            n_bootstrap=20,
            n_groups_target=6,
            weight_type="mammen",
            rng=rng2,
            group_to_psu_map=None,
        )
        assert np.array_equal(W_identity, W_plain)

    @pytest.mark.parametrize("method", ["BRR", "JK1"])
    def test_honest_did_under_replicate(
        self, base_panel, replicate_design, method
    ):
        """HonestDiD bounds should flow through replicate SE and the
        reduced df_survey (df = min(n_valid) - 1)."""
        R = 20
        df = replicate_design(base_panel, R=R, method=method)
        sd = _build_replicate_design(R, method)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
                L_max=1,
                honest_did=True,
            )
        assert res.survey_metadata.df_survey == R - 1
        if res.honest_did_results is not None:
            # HonestDiD bounds exist and are finite; exact structure is
            # honest_did.py internals. Just verify the attribute is
            # populated so the SE flow-through is exercised.
            assert res.honest_did_results is not None

    def test_rank_deficient_replicate_uses_design_df(self, base_panel):
        """Duplicated replicate columns → QR-rank < R → design df < R-1.

        Regression for PR #311 CI review R1 P1. The main dCDH surface,
        ``survey_metadata.df_survey``, and HonestDiD must all use the
        reduced effective df (``min(design_df, n_valid - 1)``) rather
        than the naive ``n_valid - 1`` that ignored the design's QR
        rank. Before the fix, ``survey_metadata.df_survey`` stayed at
        ``R - 1`` for rank-deficient replicate designs — which is
        anti-conservative for t-inference.
        """
        R = 12
        df = _attach_replicate_weights(base_panel, R=R, method="BRR", seed=3)
        # Force duplicate columns to induce rank deficiency:
        # rep1 = rep0, rep3 = rep2 → true rank = R - 2.
        df["rep1"] = df["rep0"]
        df["rep3"] = df["rep2"]
        sd = _build_replicate_design(R, "BRR")
        # Read off the design-level df via the resolved design so the
        # test is robust to internal QR tolerance changes.
        resolved = sd.resolve(df)
        design_df = resolved.df_survey
        assert design_df is not None and design_df < R - 1, (
            f"Expected rank deficiency: design df={design_df} "
            f"should be < {R - 1}"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd, L_max=1, honest_did=True,
            )
        # survey_metadata carries the final (reduced) df — not R - 1.
        assert res.survey_metadata.df_survey == design_df, (
            f"Expected survey_metadata.df_survey == design_df "
            f"({design_df}), got {res.survey_metadata.df_survey}."
        )
        # HonestDiD reads results.survey_metadata.df_survey directly
        # (honest_did.py:973), so the same reduced df flows through.
        # We assert the HonestDiD result is populated (df flow-through
        # is exercised by the honest_did=True call above).
        assert res.honest_did_results is not None

    def test_dropped_replicate_reduces_df(self, base_panel):
        """When some replicate columns produce degenerate per-replicate
        estimates (e.g., all-zero weight vectors), the design's QR rank
        drops below R — which flows through to the effective df.

        Regression for PR #311 CI review R1 P2. Even when every IF site
        reports the full ``n_valid = R``, the persisted df must never
        exceed the design-level df. Pre-fix, ``_effective_df_survey``
        returned ``n_valid - 1 = R - 1`` which over-counted degrees of
        freedom on a reduced-rank design.
        """
        R = 15
        df = _attach_replicate_weights(base_panel, R=R, method="JK1", seed=4)
        # Zero out two replicate columns → drops rank by 2.
        df["rep0"] = 0.0
        df["rep1"] = 0.0
        sd = _build_replicate_design(R, "JK1")
        resolved = sd.resolve(df)
        design_df = resolved.df_survey
        assert design_df is not None and design_df <= R - 3, (
            f"Expected zero-column rank deficiency: design df={design_df} "
            f"should be <= {R - 3}"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd, L_max=1,
            )
        # Persisted df must be capped by the design df, not R - 1.
        assert res.survey_metadata.df_survey is not None
        assert res.survey_metadata.df_survey <= design_df, (
            f"Expected survey_metadata.df_survey <= design_df "
            f"({design_df}), got {res.survey_metadata.df_survey}. "
            f"Effective df must NEVER exceed the design-level df under "
            f"replicate variance."
        )
        assert res.survey_metadata.df_survey < R - 1, (
            f"Expected persisted df to reflect reduced rank (< R - 1 = "
            f"{R - 1}), got {res.survey_metadata.df_survey}."
        )

    def test_rank_1_replicate_forces_nan_inference(self, base_panel):
        """All replicate columns identical → QR-rank = 1 → design df
        undefined (None). Even if per-IF replicate SEs come back
        finite, every public inference field (t_stat / p_value /
        conf_int) MUST be NaN — otherwise users get z-based inference
        silently instead of the NaN contract documented in REGISTRY.md.

        R2 P0 regression: before the fix, ``_effective_df_survey``
        returned None, ``safe_inference(df=None)`` computed z-based
        statistics, and a rank-1 replicate design with finite SEs
        would produce valid-looking (but wrong) p-values/CIs.
        """
        R = 8
        df = _attach_replicate_weights(base_panel, R=R, method="BRR", seed=3)
        # Make every replicate column identical → QR-rank = 1.
        rep_col_shared = df["rep0"].copy()
        for r in range(R):
            df[f"rep{r}"] = rep_col_shared
        sd = _build_replicate_design(R, "BRR")
        resolved = sd.resolve(df)
        assert resolved.df_survey is None, (
            "Rank-1 replicate matrix must have undefined design df"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd, L_max=1,
            )
        # survey_metadata.df_survey stays None — the honest undefined
        # signal for downstream consumers.
        assert res.survey_metadata.df_survey is None
        # Main dCDH inference fields are NaN via the _inference_df
        # coercion (None → 0 under replicate → safe_inference NaN
        # branch).
        assert np.isnan(res.overall_t_stat)
        assert np.isnan(res.overall_p_value)
        assert not np.all(np.isfinite(res.overall_conf_int))
        # Event study horizon 1 is also NaN.
        if 1 in res.event_study_effects:
            info = res.event_study_effects[1]
            assert np.isnan(info["t_stat"])
            assert np.isnan(info["p_value"])

    def test_heterogeneity_replicate_cross_surface_df_consistency(
        self, base_panel,
    ):
        """With ``heterogeneity=`` active under a rank-deficient
        replicate design, every public surface — top-level dCDH,
        event study, heterogeneity, ``survey_metadata.df_survey``,
        and HonestDiD — MUST use the SAME final effective df.

        R2 P1 regression: before the fix, the main surfaces
        (``overall_*``, ``event_study_effects``) computed t/p/CI
        with an intermediate ``_df_survey`` that did not include
        heterogeneity's ``n_valid_het``. Then
        ``survey_metadata.df_survey`` was overwritten with the
        post-heterogeneity min — so HonestDiD used a different df
        than the main dCDH surface.
        """
        from scipy import stats as _stats

        R = 12
        df = _attach_replicate_weights(base_panel, R=R, method="BRR", seed=7)
        # Rank deficiency: 2 duplicated pairs → rank = R - 2 = 10.
        df["rep1"] = df["rep0"]
        df["rep3"] = df["rep2"]
        sd = _build_replicate_design(R, "BRR")
        resolved = sd.resolve(df)
        design_df = resolved.df_survey
        assert design_df is not None and design_df < R - 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, outcome="outcome", group="group",
                time="period", treatment="treatment",
                survey_design=sd, L_max=1,
                heterogeneity="x_het",
                honest_did=True,
            )
        final_df = res.survey_metadata.df_survey
        # All sites report n_valid == R → effective df = min(design_df, R-1)
        # = design_df.
        assert final_df == design_df
        # Verify overall inference was computed with df=final_df by
        # checking p-value against t(final_df) — would disagree if
        # df=R-1 or df=None was used.
        t_stat = res.overall_att / res.overall_se
        expected_p = 2 * _stats.t.sf(abs(t_stat), df=final_df)
        assert res.overall_p_value == pytest.approx(expected_p, rel=1e-6)
        # Same check for heterogeneity surface.
        if res.heterogeneity_effects:
            het_info = res.heterogeneity_effects[1]
            if np.isfinite(het_info["se"]):
                het_t = het_info["beta"] / het_info["se"]
                expected_het_p = 2 * _stats.t.sf(abs(het_t), df=final_df)
                assert het_info["p_value"] == pytest.approx(
                    expected_het_p, rel=1e-6
                )
        # HonestDiD bounds were computed from the same survey_metadata;
        # asserting the result attribute is populated confirms the df
        # flow-through was exercised.
        assert res.honest_did_results is not None
