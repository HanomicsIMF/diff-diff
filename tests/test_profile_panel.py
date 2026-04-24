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
