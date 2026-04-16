"""Tests for the bundled LLM guide accessor."""
import importlib.resources

import pytest

from diff_diff import get_llm_guide
from diff_diff._guides_api import _VARIANT_TO_FILE


@pytest.mark.parametrize("variant", ["concise", "full", "practitioner"])
def test_all_variants_load(variant):
    text = get_llm_guide(variant)
    assert isinstance(text, str)
    assert len(text) > 1000


def test_default_is_concise():
    assert get_llm_guide() == get_llm_guide("concise")


def test_full_is_largest():
    lengths = {v: len(get_llm_guide(v)) for v in ("concise", "full", "practitioner")}
    assert lengths["full"] > lengths["concise"]
    assert lengths["full"] > lengths["practitioner"]


def test_content_stability_practitioner_workflow():
    assert "8-step" in get_llm_guide("practitioner").lower()


def test_content_stability_self_reference_after_rewrite():
    assert "get_llm_guide" in get_llm_guide("concise")


def test_wheel_content_matches_package_resource():
    for variant, filename in _VARIANT_TO_FILE.items():
        on_disk = (
            importlib.resources.files("diff_diff.guides")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
        assert get_llm_guide(variant) == on_disk


def test_utf8_encoding_preserved():
    # llms-full.txt contains the em-dash '\u2014'; verify it roundtrips.
    text = get_llm_guide("full")
    assert "\u2014" in text


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
