"""Tests for the bundled LLM guide accessor."""

import importlib.resources

import pytest

from diff_diff import get_llm_guide
from diff_diff._guides_api import _VARIANT_TO_FILE


@pytest.mark.parametrize("variant", ["concise", "full", "practitioner", "autonomous"])
def test_all_variants_load(variant):
    text = get_llm_guide(variant)
    assert isinstance(text, str)
    assert len(text) > 1000


def test_default_is_concise():
    assert get_llm_guide() == get_llm_guide("concise")


def test_full_is_largest():
    lengths = {v: len(get_llm_guide(v)) for v in ("concise", "full", "practitioner", "autonomous")}
    assert lengths["full"] > lengths["concise"]
    assert lengths["full"] > lengths["practitioner"]
    assert lengths["full"] > lengths["autonomous"]


def test_content_stability_practitioner_workflow():
    assert "8-step" in get_llm_guide("practitioner").lower()


def test_content_stability_self_reference_after_rewrite():
    assert "get_llm_guide" in get_llm_guide("concise")


def test_content_stability_autonomous_fingerprints():
    text = get_llm_guide("autonomous")
    assert "profile_panel" in text
    assert "estimator-support matrix" in text.lower()
    # Wave 2 additions: outcome / dose shape field references.
    assert "outcome_shape" in text
    assert "treatment_dose" in text
    assert "is_count_like" in text
    # has_never_treated is the authoritative ContinuousDiD gate;
    # treatment_dose fields are descriptive only.
    assert "has_never_treated" in text
    # The ContinuousDiD prerequisite summary must continue to mention
    # the duplicate-row hard stop alongside the field-based gates -
    # `_precompute_structures()` silently resolves duplicate cells via
    # last-row-wins, so a reader treating the summary as exhaustive
    # could route duplicate-containing panels into a silent-overwrite
    # path. Guard against that wording regression.
    assert "duplicate_unit_time_rows" in text, (
        "ContinuousDiD prerequisite summary must mention the "
        "`duplicate_unit_time_rows` alert: the precompute path resolves "
        "duplicate (unit, time) cells via last-row-wins, so duplicates "
        "must be removed before fitting."
    )


def test_autonomous_contains_worked_examples_section():
    """The §5 worked-examples section walks an agent through three
    end-to-end PanelProfile -> reasoning -> validation flows. Each
    example carries a unique fingerprint phrase keyed off its
    PanelProfile -> estimator path; these regressions guard the
    examples from accidental deletion or scope drift."""
    text = get_llm_guide("autonomous")
    assert "## §5. Worked examples" in text
    # §5.1: binary staggered with never-treated -> CallawaySantAnna
    assert "§5.1 Binary staggered panel with never-treated controls" in text
    assert 'control_group="never_treated"' in text
    # §5.2: continuous dose -> ContinuousDiD prerequisites via treatment_dose
    assert "§5.2 Continuous-dose panel with zero baseline" in text
    assert "TreatmentDoseShape(" in text
    # §5.3: count-shaped outcome -> WooldridgeDiD QMLE
    assert "§5.3 Count-shaped outcome" in text
    assert 'WooldridgeDiD(method="poisson")' in text
    assert 'WooldridgeDiD(family="poisson")' not in text, (
        "WooldridgeDiD takes `method=` not `family=`; the wrong kwarg "
        "in the autonomous guide would produce runtime errors when an "
        "agent follows the worked example."
    )


def test_autonomous_worked_examples_avoid_recommender_language():
    """Worked examples must mirror the rest of the guide's discipline:
    no prescriptive language in the example reasoning. Multiple paths
    must remain explicit."""
    text = get_llm_guide("autonomous")
    # Locate the §5 block; check its body for forbidden phrasing.
    start = text.index("## §5. Worked examples")
    end = text.index("## §6. Post-fit validation utilities")
    section_5 = text[start:end].lower()
    forbidden = (
        "you should always",
        "always pick",
        "we recommend",
        "the best estimator is",
    )
    for phrase in forbidden:
        assert phrase not in section_5, (
            f"§5 worked examples contain prescriptive phrase {phrase!r}; "
            "the guide must keep multiple paths explicit."
        )


def test_autonomous_contains_intact_estimator_matrix():
    # Section 3 is a markdown table with 10 data columns + the estimator
    # name column -> rows have at least 11 pipe characters. This guards
    # against the matrix being accidentally deleted or truncated.
    text = get_llm_guide("autonomous")
    assert any(
        line.count("|") >= 11 for line in text.splitlines()
    ), "Section 3 estimator-support matrix appears to be missing or truncated."


def test_wheel_content_matches_package_resource():
    for variant, filename in _VARIANT_TO_FILE.items():
        on_disk = (
            importlib.resources.files("diff_diff.guides")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
        assert get_llm_guide(variant) == on_disk


def test_utf8_encoding_preserved():
    # llms-full.txt contains the non-ASCII ligature '\u0153' (oe, from
    # "D'Haultfoeuille"); verify UTF-8 roundtrips through the packaged guide.
    text = get_llm_guide("full")
    assert "\u0153" in text


@pytest.mark.parametrize("bad", ["bogus", "", "CONCISE", None, 0, True, ["x"]])
def test_unknown_variant_raises(bad):
    with pytest.raises(ValueError, match="Unknown guide variant"):
        get_llm_guide(bad)


def test_exported_in_namespace():
    import diff_diff

    assert "get_llm_guide" in diff_diff.__all__
    assert callable(diff_diff.get_llm_guide)


def test_module_docstring_mentions_helper():
    import diff_diff

    assert "get_llm_guide" in diff_diff.__doc__
