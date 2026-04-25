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
    """`llms-full.txt` is the API-docs roll-up; it should remain larger
    than the short `concise` summary and the workflow-prose
    `practitioner` guide. The `autonomous` reference guide is
    deliberately excluded from this comparison: it serves a different
    audience (LLM agents reasoning about estimator choice) and has
    grown organically through Wave 1 + Wave 2 review rounds with
    estimator-matrix detail, worked examples, and contract citations
    that don't have a counterpart in `llms-full.txt`'s API roll-up.
    Either of the two can be larger without violating any user-facing
    invariant."""
    lengths = {v: len(get_llm_guide(v)) for v in ("concise", "full", "practitioner")}
    assert lengths["full"] > lengths["concise"]
    assert lengths["full"] > lengths["practitioner"]


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
    # ContinuousDiD also requires strictly positive treated doses
    # (`continuous_did.py:287-294` raises on negative dose support).
    # The autonomous guide must list `dose_min > 0` so an agent reading
    # `treatment_dose.dose_min == -1.5` knows to route the panel away
    # from ContinuousDiD before paying for the failed fit.
    assert "dose_min > 0" in text, (
        "ContinuousDiD prerequisite summary must mention "
        "`dose_min > 0`: the estimator hard-rejects negative treated "
        "dose support at line 287-294 of continuous_did.py."
    )
    # The five profile-side screening checks are necessary but not
    # sufficient: `ContinuousDiD.fit()` takes a separate `first_treat`
    # column (which `profile_panel` does not see) and applies
    # additional validation. The autonomous guide must explicitly
    # mention the `first_treat` validation surface so an agent
    # passing the profile-side screen still knows to validate the
    # `first_treat` column they will supply to `fit()`.
    assert "first_treat" in text, (
        "ContinuousDiD documentation must mention the separate "
        "`first_treat` column that `ContinuousDiD.fit()` validates "
        "(NaN/inf/negative rejection, dose=0 unit drops, force-zero "
        "coercion). The five profile-side screening checks alone are "
        "necessary but not sufficient for fit-time success."
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
    assert "§5.2 Continuous-dose panel with zero-dose controls" in text
    assert "TreatmentDoseShape(" in text
    # §5.3: count-shaped outcome -> WooldridgeDiD QMLE
    assert "§5.3 Count-shaped outcome" in text
    assert 'WooldridgeDiD(method="poisson")' in text
    assert 'WooldridgeDiD(family="poisson")' not in text, (
        "WooldridgeDiD takes `method=` not `family=`; the wrong kwarg "
        "in the autonomous guide would produce runtime errors when an "
        "agent follows the worked example."
    )


def test_autonomous_count_outcome_uses_asf_outcome_scale_estimand():
    """§4.11 and §5.3 must describe `WooldridgeDiD(method="poisson")`'s
    `overall_att` as an ASF-based outcome-scale difference (matching the
    estimator at `wooldridge.py:1225` and the reporting helper at
    `_reporting_helpers.py:262-281`), NOT as a multiplicative /
    proportional / log-link effect. An agent following an example that
    described the headline as "multiplicative" would misreport the
    scalar - the library's reported `overall_att` is `E[exp(η_1)] -
    E[exp(η_0)]`, a difference on the natural outcome scale.

    Guards against regressing the wording back to "multiplicative
    effect" / "proportional change" framing. Multiplicative
    interpretations may appear in the guide as a clearly-marked
    derived post-hoc reading, but never as the description of the
    estimator's reported `overall_att`."""
    text = get_llm_guide("autonomous")
    # Locate §4.11 and §5.3 blocks; check that within them the Poisson
    # path is described with ASF / outcome-scale wording, NOT as the
    # estimator's reported scalar being multiplicative or proportional.
    sec_4_11_start = text.index("### §4.11 Outcome-shape considerations")
    sec_4_11_end = text.index("## §5. Worked examples")
    sec_4_11 = text[sec_4_11_start:sec_4_11_end]

    sec_5_3_start = text.index("### §5.3 Count-shaped outcome")
    sec_5_3_end = text.index("## §6. Post-fit validation utilities")
    sec_5_3 = text[sec_5_3_start:sec_5_3_end]

    forbidden_phrases = (
        "multiplicative effect under qmle",
        "estimates the multiplicative effect",
        "multiplicative (log-link) effect",
        "report the multiplicative effect",
        "report the multiplicative",
    )
    for section_name, body in (("§4.11", sec_4_11), ("§5.3", sec_5_3)):
        lowered = body.lower()
        for phrase in forbidden_phrases:
            assert phrase not in lowered, (
                f"{section_name} of the autonomous guide describes the "
                f"WooldridgeDiD Poisson `overall_att` with the phrase "
                f"{phrase!r}; the estimator returns an ASF-based "
                f"outcome-scale difference (`E[exp(η_1)] - E[exp(η_0)]`), "
                f"not a multiplicative ratio. See `wooldridge.py:1225` "
                f"and `_reporting_helpers.py:262-281`."
            )

    # Positive: each block must explicitly anchor the estimand to the
    # ASF / outcome-scale framing so future edits can't silently weaken
    # the description.
    assert "ASF" in sec_5_3, "§5.3 must reference the ASF interpretation"
    assert "outcome scale" in sec_5_3.lower(), (
        "§5.3 must label the WooldridgeDiD `overall_att` as an "
        "outcome-scale quantity to prevent multiplicative-ratio drift."
    )


def test_autonomous_negative_dose_path_does_not_route_to_had():
    """The §5.2 negative-dose counter-example must not present
    `HeterogeneousAdoptionDiD` as a direct routing alternative
    when `dose_min < 0`. HAD's contract requires non-negative
    dose support and raises on negative post-period dose
    (`had.py:1450-1459`, paper Section 2). Routing to HAD on a
    negative-dose panel without re-encoding would steer the agent
    into an unsupported estimator path. Guards against the wording
    regressing back to a too-broad "HAD as fallback" framing on
    this branch."""
    text = get_llm_guide("autonomous")
    # Locate counter-example #5 (negative-dose path) within §5.2.
    sec_5_2_start = text.index("### §5.2 Continuous-dose panel")
    sec_5_3_start = text.index("### §5.3 Count-shaped outcome")
    sec_5_2 = text[sec_5_2_start:sec_5_3_start]
    # The negative-dose paragraph must explicitly state HAD is NOT a
    # routing alternative on this branch. We assert the disqualifying
    # phrase is present; we do not forbid `HeterogeneousAdoptionDiD`
    # entirely because the section may legitimately mention it as a
    # candidate AFTER re-encoding.
    assert "HAD" in sec_5_2 or "HeterogeneousAdoptionDiD" in sec_5_2, (
        "§5.2 must mention HAD by name on the negative-dose branch "
        "so its non-applicability can be explicitly called out."
    )
    assert "had.py:1450-1459" in sec_5_2, (
        "§5.2 must cite `had.py:1450-1459` on the negative-dose "
        "branch to anchor HAD's non-negative-dose contract (HAD "
        "raises on negative post-period dose, paper Section 2). "
        "Without this citation, the agent could route a "
        "negative-dose panel directly to HAD and hit a fit-time "
        "error."
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
