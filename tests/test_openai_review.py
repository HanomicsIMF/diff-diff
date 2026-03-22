"""Tests for .claude/scripts/openai_review.py — local AI review script.

These tests are skipped in CI when the script is not available (e.g., when
the package is installed via pip into a temp directory). They run locally
where the repo checkout includes .claude/scripts/.
"""

import importlib.util
import pathlib
import subprocess

import pytest

# ---------------------------------------------------------------------------
# Import the script as a module (it's not in a package)
# ---------------------------------------------------------------------------


def _find_script() -> "pathlib.Path | None":
    """Find openai_review.py relative to the repo root."""
    # Method 1: relative to this test file (works in local checkout)
    candidate = (
        pathlib.Path(__file__).resolve().parent.parent
        / ".claude"
        / "scripts"
        / "openai_review.py"
    )
    if candidate.exists():
        return candidate

    # Method 2: relative to git repo root (works in worktrees)
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        candidate = pathlib.Path(root) / ".claude" / "scripts" / "openai_review.py"
        if candidate.exists():
            return candidate
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


_SCRIPT_PATH = _find_script()

# Skip entire module if the script isn't available (e.g., CI pip-install)
pytestmark = pytest.mark.skipif(
    _SCRIPT_PATH is None,
    reason="openai_review.py not found (not in repo checkout)",
)


@pytest.fixture(scope="module")
def review_mod():
    """Import openai_review.py as a module."""
    assert _SCRIPT_PATH is not None
    spec = importlib.util.spec_from_file_location("openai_review", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# _sections_for_file
# ---------------------------------------------------------------------------


class TestSectionsForFile:
    def test_direct_match(self, review_mod):
        assert "BaconDecomposition" in review_mod._sections_for_file("bacon.py")

    def test_companion_file(self, review_mod):
        assert "SunAbraham" in review_mod._sections_for_file("sun_abraham_bootstrap.py")

    def test_no_match(self, review_mod):
        assert review_mod._sections_for_file("linalg.py") == []

    def test_staggered_maps_multiple(self, review_mod):
        sections = review_mod._sections_for_file("staggered.py")
        assert "CallawaySantAnna" in sections
        assert "SunAbraham" in sections

    def test_longest_prefix_wins(self, review_mod):
        # sun_abraham.py should match "sun_abraham" not "staggered"
        sections = review_mod._sections_for_file("sun_abraham.py")
        assert sections == ["SunAbraham"]


# ---------------------------------------------------------------------------
# _needed_sections
# ---------------------------------------------------------------------------


class TestNeededSections:
    def test_basic(self, review_mod):
        text = "M\tdiff_diff/bacon.py"
        assert "BaconDecomposition" in review_mod._needed_sections(text)

    def test_visualization_submodule(self, review_mod):
        text = "M\tdiff_diff/visualization/_event_study.py"
        assert "Event Study Plotting" in review_mod._needed_sections(text)

    def test_visualization_multiple_files(self, review_mod):
        """All visualization/ submodule files map via directory to Event Study Plotting."""
        text = (
            "M\tdiff_diff/visualization/_event_study.py\n"
            "M\tdiff_diff/visualization/_diagnostic.py"
        )
        sections = review_mod._needed_sections(text)
        assert "Event Study Plotting" in sections

    def test_non_diff_diff_paths_ignored(self, review_mod):
        text = "M\ttests/test_bacon.py\nM\tCLAUDE.md"
        assert review_mod._needed_sections(text) == set()

    def test_utility_files_no_sections(self, review_mod):
        text = "M\tdiff_diff/linalg.py\nM\tdiff_diff/utils.py"
        assert review_mod._needed_sections(text) == set()

    def test_mixed_files(self, review_mod):
        text = (
            "M\tdiff_diff/bacon.py\n"
            "M\tdiff_diff/linalg.py\n"
            "M\ttests/test_bacon.py"
        )
        sections = review_mod._needed_sections(text)
        assert sections == {"BaconDecomposition"}

    def test_empty_input(self, review_mod):
        assert review_mod._needed_sections("") == set()


# ---------------------------------------------------------------------------
# extract_registry_sections
# ---------------------------------------------------------------------------


class TestExtractRegistrySections:
    SAMPLE_REGISTRY = (
        "# Registry\n\n"
        "## Table of Contents\nTOC content\n\n"
        "## BaconDecomposition\nBacon content line 1\nBacon content line 2\n\n"
        "## SunAbraham\nSA content\n\n"
        "## Event Study Plotting (`plot_event_study`)\nPlotting content\n"
    )

    def test_extract_single_section(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"BaconDecomposition"}
        )
        assert "Bacon content line 1" in result
        assert "SA content" not in result

    def test_extract_multiple_sections(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"BaconDecomposition", "SunAbraham"}
        )
        assert "Bacon content" in result
        assert "SA content" in result

    def test_prefix_match_for_headings_with_parens(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"Event Study Plotting"}
        )
        assert "Plotting content" in result

    def test_empty_section_names(self, review_mod):
        assert review_mod.extract_registry_sections(self.SAMPLE_REGISTRY, set()) == ""

    def test_nonexistent_section(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"NonExistent"}
        )
        assert result == ""


# ---------------------------------------------------------------------------
# _adapt_review_criteria
# ---------------------------------------------------------------------------


class TestAdaptReviewCriteria:
    def test_replaces_opening_line(self, review_mod):
        source = "You are an automated PR reviewer for a causal inference library."
        result = review_mod._adapt_review_criteria(source)
        assert "automated PR reviewer" not in result
        assert "code reviewer" in result

    def test_replaces_pr_language(self, review_mod):
        source = "If the PR changes an estimator"
        result = review_mod._adapt_review_criteria(source)
        assert "If the changes affect an estimator" in result

    def test_warns_on_missing_substitution(self, review_mod, capsys):
        # A text that doesn't contain any of the expected patterns
        result = review_mod._adapt_review_criteria("Totally different text")
        captured = capsys.readouterr()
        assert "Warning: prompt substitution did not match" in captured.err

    def test_all_substitutions_apply_to_real_prompt(self, review_mod, capsys):
        """Verify all substitutions match the actual pr_review.md file."""
        assert _SCRIPT_PATH is not None
        repo_root = _SCRIPT_PATH.parent.parent.parent
        prompt_path = repo_root / ".github" / "codex" / "prompts" / "pr_review.md"
        if not prompt_path.exists():
            pytest.skip("pr_review.md not found")
        source = prompt_path.read_text()
        review_mod._adapt_review_criteria(source)
        captured = capsys.readouterr()
        assert "Warning: prompt substitution did not match" not in captured.err


# ---------------------------------------------------------------------------
# compile_prompt
# ---------------------------------------------------------------------------


class TestCompilePrompt:
    def test_basic_structure(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="Review criteria here.",
            registry_content="Registry content.",
            diff_text="diff --git a/foo.py",
            changed_files_text="M\tfoo.py",
            branch_info="feature/test",
            previous_review=None,
        )
        assert "Review criteria here." in result
        assert "Registry content." in result
        assert "diff --git a/foo.py" in result
        assert "Branch: feature/test" in result
        assert "previous-review-output" not in result

    def test_includes_previous_review(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="Criteria.",
            registry_content="Registry.",
            diff_text="diff content",
            changed_files_text="M\tfoo.py",
            branch_info="main",
            previous_review="Previous review findings here.",
        )
        assert "<previous-review-output>" in result
        assert "Previous review findings here." in result
        assert "follow-up review" in result

    def test_no_previous_review_block_when_none(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="C.",
            registry_content="R.",
            diff_text="D.",
            changed_files_text="M\tf.py",
            branch_info="b",
            previous_review=None,
        )
        assert "<previous-review-output>" not in result


# ---------------------------------------------------------------------------
# PREFIX_TO_SECTIONS mapping coverage
# ---------------------------------------------------------------------------


class TestPrefixMappingCoverage:
    """Validate that known estimator modules have PREFIX_TO_SECTIONS entries."""

    # Core estimator files that MUST have a mapping
    EXPECTED_MAPPED = [
        "estimators.py",
        "twfe.py",
        "staggered.py",
        "sun_abraham.py",
        "imputation.py",
        "two_stage.py",
        "stacked_did.py",
        "synthetic_did.py",
        "triple_diff.py",
        "trop.py",
        "bacon.py",
        "honest_did.py",
        "power.py",
        "pretrends.py",
        "diagnostics.py",
        "visualization.py",
        "continuous_did.py",
        "efficient_did.py",
        "survey.py",
    ]

    # Utility files that intentionally have NO mapping
    EXPECTED_UNMAPPED = [
        "linalg.py",
        "utils.py",
        "results.py",
        "prep.py",
        "prep_dgp.py",
        "datasets.py",
        "_backend.py",
        "bootstrap_utils.py",
        "__init__.py",
    ]

    def test_all_estimator_files_have_mapping(self, review_mod):
        for filename in self.EXPECTED_MAPPED:
            sections = review_mod._sections_for_file(filename)
            assert sections, f"{filename} has no PREFIX_TO_SECTIONS mapping"

    def test_utility_files_have_no_mapping(self, review_mod):
        for filename in self.EXPECTED_UNMAPPED:
            sections = review_mod._sections_for_file(filename)
            assert sections == [], f"{filename} unexpectedly has a mapping: {sections}"

    def test_visualization_submodule_maps_correctly(self, review_mod):
        """Ensure visualization/ subdirectory files map via directory name."""
        text = "M\tdiff_diff/visualization/_event_study.py"
        assert "Event Study Plotting" in review_mod._needed_sections(text)

        # _diagnostic.py inside visualization/ maps to Event Study Plotting
        # (via directory), NOT PlaceboTests (which is diagnostics.py at top level)
        text = "M\tdiff_diff/visualization/_diagnostic.py"
        sections = review_mod._needed_sections(text)
        assert "Event Study Plotting" in sections


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_rough_estimate(self, review_mod):
        # 400 chars -> ~100 tokens
        text = "a" * 400
        assert review_mod.estimate_tokens(text) == 100

    def test_empty_string(self, review_mod):
        assert review_mod.estimate_tokens("") == 0
