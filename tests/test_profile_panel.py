"""Tests for ``diff_diff.profile_panel`` and the ``PanelProfile`` dataclass."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import pytest

from diff_diff import PanelProfile, profile_panel
from diff_diff.profile import Alert


def _make_panel(
    *,
    n_units: int,
    periods: Iterable[int],
    first_treat: Optional[Dict[int, int]] = None,
    outcome_fn: Any = None,
) -> pd.DataFrame:
    """Build a balanced long panel with optional per-unit first-treatment timing.

    ``first_treat`` maps unit -> first treatment period (inclusive). Units not
    in the mapping are never-treated.
    """
    first_treat = first_treat or {}
    rows = []
    rng = np.random.default_rng(0)
    for u in range(1, n_units + 1):
        for t in periods:
            tr = 1 if (u in first_treat and t >= first_treat[u]) else 0
            if outcome_fn is not None:
                y = outcome_fn(u, t, tr, rng)
            else:
                y = float(u) + 0.1 * t + 0.5 * tr
            rows.append({"u": u, "t": t, "tr": tr, "y": y})
    return pd.DataFrame(rows)


def _alert_codes(profile: PanelProfile) -> set[str]:
    return {a.code for a in profile.alerts}


def test_balanced_binary_2x2():
    first_treat = {u: 1 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=[0, 1], first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.is_staggered is False
    assert profile.has_never_treated is True
    assert profile.n_units == 20
    assert profile.n_periods == 2
    assert profile.is_balanced is True


def test_staggered_multi_cohort():
    first_treat: Dict[int, int] = {}
    first_treat.update({u: 3 for u in range(1, 11)})
    first_treat.update({u: 5 for u in range(11, 21)})
    first_treat.update({u: 7 for u in range(21, 31)})
    df = _make_panel(n_units=40, periods=range(1, 9), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.is_staggered is True
    assert profile.n_cohorts == 3
    assert profile.cohort_sizes == {3: 10, 5: 10, 7: 10}
    assert profile.first_treatment_period == 3
    assert profile.last_treatment_period == 7
    assert profile.has_never_treated is True


def test_binary_non_absorbing_switcher():
    rows = []
    rng = np.random.default_rng(0)
    for u in range(1, 21):
        treat_seq = [0, 1, 1, 0, 0] if u > 10 else [0, 0, 0, 0, 0]
        for t, tr in enumerate(treat_seq):
            rows.append({"u": u, "t": t, "tr": tr, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_non_absorbing"
    assert profile.cohort_sizes == {}
    assert profile.is_staggered is False
    assert profile.has_never_treated is True


def test_continuous_treatment():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 41):
        dose = float(rng.uniform(0, 5))
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "continuous"
    assert profile.cohort_sizes == {}
    assert profile.is_staggered is False


def test_categorical_treatment_object_dtype():
    rows = []
    for u in range(1, 11):
        arm = "A" if u <= 5 else "B"
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": arm, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "categorical"
    assert profile.has_never_treated is False
    assert profile.has_always_treated is False


def test_no_never_treated_alert():
    first_treat = {u: 2 for u in range(1, 21)}
    df = _make_panel(n_units=20, periods=range(0, 5), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.has_never_treated is False
    codes = _alert_codes(profile)
    assert "no_never_treated" in codes


def test_has_always_treated_alert():
    rows = []
    for u in range(1, 21):
        for t in range(5):
            tr = 1 if u <= 5 else (1 if t >= 3 else 0)
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.has_always_treated is True
    codes = _alert_codes(profile)
    assert "has_always_treated_units" in codes


def test_unbalanced_panel_below_threshold():
    first_treat = {u: 3 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 5), first_treat=first_treat)
    df = df.iloc[::3].reset_index(drop=True)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.is_balanced is False
    assert profile.observation_coverage < 0.70
    codes = _alert_codes(profile)
    assert "panel_highly_unbalanced" in codes


def test_binary_outcome_float_dtype_alert():
    first_treat = {u: 2 for u in range(11, 31)}
    df = _make_panel(
        n_units=30,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, rng: float(tr),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.outcome_is_binary is True
    assert profile.outcome_dtype == "float64"
    codes = _alert_codes(profile)
    assert "outcome_looks_binary_but_dtype_float" in codes


def test_outcome_missing_fraction_computed():
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df.loc[0:9, "y"] = np.nan
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert 0.0 < profile.outcome_missing_fraction < 1.0
    assert profile.outcome_missing_fraction == pytest.approx(10 / len(df))


def test_short_pre_panel_alert():
    first_treat = {u: 1 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=[0, 1, 2, 3], first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.min_pre_periods == 1
    codes = _alert_codes(profile)
    assert "short_pre_panel" in codes


def test_missing_column_raises_value_error():
    df = pd.DataFrame({"u": [1, 2], "t": [0, 1], "y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="treatment"):
        profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")


def test_panel_profile_is_frozen():
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.n_units = 999  # type: ignore[misc]


def test_to_dict_is_json_serializable():
    first_treat = {u: 3 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 6), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    payload = profile.to_dict()
    as_json = json.dumps(payload)
    roundtripped = json.loads(as_json)
    assert roundtripped["treatment_type"] == "binary_absorbing"
    assert set(roundtripped.keys()) >= {
        "n_units",
        "n_periods",
        "n_obs",
        "is_balanced",
        "observation_coverage",
        "treatment_type",
        "is_staggered",
        "n_cohorts",
        "cohort_sizes",
        "has_never_treated",
        "has_always_treated",
        "first_treatment_period",
        "last_treatment_period",
        "min_pre_periods",
        "min_post_periods",
        "outcome_dtype",
        "outcome_is_binary",
        "outcome_has_zeros",
        "outcome_has_negatives",
        "outcome_missing_fraction",
        "outcome_summary",
        "alerts",
    }


def test_alerts_are_factual_no_recommender_language():
    first_treat = {u: 1 for u in range(11, 21)}
    df = _make_panel(n_units=12, periods=[0, 1, 2, 3], first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    forbidden_substrings = (
        "recommend",
        "should use",
        "use estimator",
        "we suggest",
        "you should",
    )
    for alert in profile.alerts:
        lowered = alert.message.lower()
        for phrase in forbidden_substrings:
            assert phrase not in lowered, (
                f"alert {alert.code!r} contains recommender-adjacent phrase "
                f"{phrase!r} in message: {alert.message!r}"
            )


def test_alert_dataclass_is_frozen():
    a = Alert(code="x", severity="info", message="m", observed=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.code = "y"  # type: ignore[misc]


def test_all_zero_treatment_is_binary_absorbing():
    """Degenerate binary: no unit is ever treated. Must classify as binary,
    not continuous, so the documented taxonomy matches the implementation."""
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=None)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.has_never_treated is True
    assert profile.has_always_treated is False
    assert profile.cohort_sizes == {}
    assert profile.n_cohorts == 0


def test_all_one_treatment_is_binary_absorbing_always_treated():
    """Degenerate binary: every unit treated in every period. Must classify as
    binary_absorbing with has_always_treated=True."""
    rows = []
    for u in range(1, 21):
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": 1, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.has_never_treated is False
    assert profile.has_always_treated is True
    codes = _alert_codes(profile)
    assert "has_always_treated_units" in codes


def test_binary_with_nans_only_zeros_observed_is_binary():
    """Binary panel with some NaNs and only 0 observed among non-NaN values —
    still classify as binary, not continuous."""
    rows = []
    for u in range(1, 11):
        for t in range(4):
            tr = 0 if (u + t) % 2 == 0 else np.nan
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"


def test_all_nan_treatment_is_categorical():
    """Treatment column entirely NaN — classify as categorical (no info)."""
    rows = []
    for u in range(1, 11):
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": np.nan, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "categorical"


def test_top_level_import_surface():
    """profile_panel, PanelProfile, and Alert must be importable from the
    top-level namespace so `help(diff_diff)` points at real symbols."""
    import diff_diff

    assert callable(diff_diff.profile_panel)
    assert diff_diff.PanelProfile.__name__ == "PanelProfile"
    assert diff_diff.Alert.__name__ == "Alert"
    for name in ("profile_panel", "PanelProfile", "Alert"):
        assert name in diff_diff.__all__, f"{name} missing from __all__"


def test_duplicate_unit_time_rows_do_not_inflate_coverage():
    """Duplicate (unit, time) rows must not make a panel look balanced.
    observation_coverage must stay in [0, 1] and derive from the unique
    (unit, time) support, and the duplicate_unit_time_rows alert fires."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df_dup = pd.concat([df, df.iloc[:5].copy()], ignore_index=True)
    profile = profile_panel(df_dup, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.is_balanced is True
    assert 0.0 <= profile.observation_coverage <= 1.0
    assert "duplicate_unit_time_rows" in _alert_codes(profile)

    df_missing_cell = df.drop(df.index[0]).reset_index(drop=True)
    df_dup_missing = pd.concat(
        [df_missing_cell, df_missing_cell.iloc[:5].copy()], ignore_index=True
    )
    profile2 = profile_panel(df_dup_missing, unit="u", time="t", treatment="tr", outcome="y")
    assert profile2.is_balanced is False
    assert profile2.observation_coverage < 1.0
    assert "duplicate_unit_time_rows" in _alert_codes(profile2)


def test_reversal_through_nan_is_binary_non_absorbing():
    """A 0 -> 1 -> NaN -> 0 path must be detected as non-absorbing: the
    observed non-NaN subsequence violates weak monotonicity. Previously a
    NaN-inclusive diff could report False monotonicity violation."""
    rows = []
    for u in range(1, 11):
        treat_seq = [0, 1, np.nan, 0]
        for t, tr in enumerate(treat_seq):
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_non_absorbing"


def test_continuous_zero_dose_controls_flag_has_never_treated():
    """Continuous treatment with some zero-dose units must flag
    has_never_treated=True. Previously continuous panels hardcoded
    has_never_treated=False regardless of control availability."""
    rows = []
    rng = np.random.default_rng(0)
    for u in range(1, 21):
        dose = 0.0 if u <= 5 else float(rng.uniform(0.5, 3.0))
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "continuous"
    assert profile.has_never_treated is True
    assert profile.has_always_treated is True


def test_guide_api_strings_resolve_against_public_api():
    """Sanity-check that every estimator referenced in the autonomous guide
    exists in the public API, plus the `hausman_pretest` classmethod location
    and the `not_yet_treated` control-group string. Guards against guide
    drift that the CI reviewer has previously flagged."""
    import diff_diff
    from diff_diff import get_llm_guide

    text = get_llm_guide("autonomous")

    for name in (
        "DifferenceInDifferences",
        "MultiPeriodDiD",
        "TwoWayFixedEffects",
        "CallawaySantAnna",
        "SunAbraham",
        "ChaisemartinDHaultfoeuille",
        "ImputationDiD",
        "TwoStageDiD",
        "StackedDiD",
        "WooldridgeDiD",
        "EfficientDiD",
        "SyntheticDiD",
        "TROP",
        "TripleDifference",
        "StaggeredTripleDifference",
        "ContinuousDiD",
        "HeterogeneousAdoptionDiD",
    ):
        assert name in text, f"estimator {name!r} missing from guide"
        assert hasattr(diff_diff, name), f"{name!r} in guide but not exported"

    assert hasattr(
        diff_diff.EfficientDiD, "hausman_pretest"
    ), "EfficientDiD.hausman_pretest classmethod missing from the public API"

    assert "EfficientDiD.hausman_pretest" in text
    assert "Hausman.hausman_pretest" not in text

    assert 'control_group="not_yet_treated"' in text
    assert "notyettreated" not in text

    # HAD targets WAS / WAS_d_lower, not ATT; event-study is per-event-
    # time, not per-cohort. Guard against the guide drifting back to
    # ATT-shaped / per-cohort phrasing.
    assert "Weighted Average Slope (WAS)" in text
    assert "WAS_d_lower" in text
    assert "per-cohort Pierce-Schott" not in text

    # EfficientDiD has three paths when no never-treated exists:
    # PT-Post, PT-All, or control_group="last_cohort". The guide must
    # mention last_cohort in the no-never-treated section so agents do
    # not rule out the supported path.
    assert 'control_group="last_cohort"' in text

    # SunAbraham requires a never-treated cohort; the fit path raises a
    # ValueError when none exists. Guard the matrix / prose contract so
    # the guide cannot drift back to claiming SunAbraham is optional.
    sun_abraham_row = next(
        line for line in text.splitlines() if "`SunAbraham`" in line and "|" in line
    )
    cells = [cell.strip() for cell in sun_abraham_row.strip("|").split("|")]
    # Column order: estimator, binary_absorbing, staggered, continuous,
    # triple-diff, never-treated-required, covariate, few-treated,
    # heterogeneous-adoption, clustered-SE.
    assert cells[5] == "✓", (
        "SunAbraham matrix row must mark never-treated-required=✓ " f"(row: {sun_abraham_row!r})"
    )

    # HAD Assumption 3 is not testable per REGISTRY.md; the guide must
    # not claim otherwise.
    assert "Assumption 3" in text  # mentioned as untestable, not as validated
    assert "validate Assumptions 3 and 7" not in text
    assert "not testable" in text

    # EfficientDiD requires never-treated under BOTH assumption="PT-All"
    # and assumption="PT-Post" — PT-Post is not a "drop the requirement"
    # escape hatch. Only control_group="last_cohort" admits all-treated
    # panels. Guard against guide drift back to the incorrect wording.
    assert "PT-Post is the weaker" in text or "both" in text.lower()
    # The old claim "switch to `assumption=\"PT-Post\"` to drop" must
    # not reappear in any form.
    assert 'switch to `assumption="PT-Post"` to drop' not in text

    # Matrix covariate cells: SyntheticDiD accepts fit(covariates=...)
    # and residualizes the outcome; ContinuousDiD.fit has no covariate
    # surface. Guard the matrix rows against drift.
    sdid_row = next(line for line in text.splitlines() if "`SyntheticDiD`" in line and "|" in line)
    sdid_cells = [c.strip() for c in sdid_row.strip("|").split("|")]
    assert sdid_cells[6] in ("✓", "partial"), (
        "SyntheticDiD covariate-adjustment cell must be ✓ or partial "
        f"(residualization path exists); got {sdid_cells[6]!r}"
    )
    cdid_row = next(line for line in text.splitlines() if "`ContinuousDiD`" in line and "|" in line)
    cdid_cells = [c.strip() for c in cdid_row.strip("|").split("|")]
    assert cdid_cells[6] == "✗", (
        "ContinuousDiD covariate-adjustment cell must be ✗ "
        f"(no covariate surface on fit()); got {cdid_cells[6]!r}"
    )

    # §5 API signatures: compute_pretrends_power takes a fitted results
    # object (not df), plot_sensitivity takes SensitivityResults,
    # plot_honest_event_study takes HonestDiDResults. Guard against
    # drift back to the df-first / results-only signatures.
    assert "`compute_pretrends_power(results" in text
    assert "`plot_sensitivity(sensitivity_results" in text
    assert "`plot_honest_event_study(honest_results" in text


def test_min_pre_post_use_per_unit_observed_support():
    """On an unbalanced panel where one treated unit is missing its
    earliest pre-period, min_pre_periods must reflect that unit's actual
    observed support. Previously _compute_pre_post used the global period
    set, which could hide short-panel cases and suppress the short_pre_panel
    alert."""
    rows = []
    for u in range(1, 21):
        first_treat = 3
        for t in range(0, 6):
            if u == 1 and t <= 1:
                continue
            tr = 1 if t >= first_treat else 0
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.min_pre_periods == 1
    assert "short_pre_panel" in _alert_codes(profile)


def test_missing_unit_or_time_ids_are_dropped_consistently():
    """NaN values in unit or time must not push observation_coverage above
    1.0. `nunique()` drops NaN while `drop_duplicates()` keeps NaN as a
    distinct key, which previously produced coverage > 1 silently. The
    fix drops NaN-id rows up front, emits the missing_id_rows_dropped
    alert, and computes all structural facts on the non-missing subset."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df_with_missing = df.copy()
    df_with_missing.loc[[0, 1, 2], "u"] = np.nan
    df_with_missing.loc[[5, 6], "t"] = np.nan
    profile = profile_panel(df_with_missing, unit="u", time="t", treatment="tr", outcome="y")
    assert 0.0 <= profile.observation_coverage <= 1.0
    codes = _alert_codes(profile)
    assert "missing_id_rows_dropped" in codes
    drop_alert = next(a for a in profile.alerts if a.code == "missing_id_rows_dropped")
    assert drop_alert.observed == 5


def test_row_with_both_ids_missing_counted_once():
    """A row with BOTH unit and time NaN must count as one dropped row,
    not two. Previously `isna().sum()` summed the two columns and
    double-counted rows missing both identifiers."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df_both_missing = df.copy()
    df_both_missing.loc[0, "u"] = np.nan
    df_both_missing.loc[0, "t"] = np.nan
    profile = profile_panel(df_both_missing, unit="u", time="t", treatment="tr", outcome="y")
    drop_alert = next(a for a in profile.alerts if a.code == "missing_id_rows_dropped")
    assert drop_alert.observed == 1


def test_empty_dataframe_raises_value_error():
    """Direct empty input must raise, not silently return a 'balanced'
    profile with zero units/periods."""
    df = pd.DataFrame({"u": [], "t": [], "tr": [], "y": []})
    with pytest.raises(ValueError, match="empty"):
        profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")


def test_empty_after_id_drop_raises_value_error():
    """If every row has a missing unit or time identifier, the panel is
    empty after the drop; raise rather than returning is_balanced=True
    on zero rows."""
    df = pd.DataFrame(
        {
            "u": [np.nan, np.nan],
            "t": [0, 1],
            "tr": [0, 1],
            "y": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="no rows remain"):
        profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
