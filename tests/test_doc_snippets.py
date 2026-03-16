"""
Smoke tests for Python code blocks in RST documentation.

Extracts ``.. code-block:: python`` snippets from RST files and executes them
in isolated namespaces with synthetic data available. Catches TypeError and
AttributeError from wrong kwargs or non-existent parameters.
"""

import re
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# RST files to validate (the ones that had review findings + key user-facing)
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

RST_FILES = [
    "choosing_estimator.rst",
    "troubleshooting.rst",
    "quickstart.rst",
    "index.rst",
    "api/datasets.rst",
    "api/diagnostics.rst",
    "api/utils.rst",
    "api/prep.rst",
    "api/two_stage.rst",
    "api/bacon.rst",
    "api/visualization.rst",
]

# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(
    r"^\.\.\s+code-block::\s+python\s*$\n"  # directive line
    r"(?:\s*:\w[^:]*:.*\n)*"  # optional directive options
    r"\n"  # blank separator
    r"((?:[ \t]+\S.*\n|[ \t]*\n)+)",  # indented body
    re.MULTILINE,
)


def _extract_snippets(rst_path: Path) -> List[Tuple[int, str]]:
    """Return list of (block_index, dedented_code) from an RST file."""
    text = rst_path.read_text()
    snippets = []
    for i, m in enumerate(_CODE_BLOCK_RE.finditer(text)):
        code = textwrap.dedent(m.group(1))
        snippets.append((i, code))
    return snippets


# ---------------------------------------------------------------------------
# Skip heuristics
# ---------------------------------------------------------------------------
_SKIP_PATTERNS = [
    r"%matplotlib",  # Jupyter magics
    r"plt\.show\(\)",  # interactive display
    r"^\s*fig\s*$",  # bare variable display in Jupyter
    r"load_card_krueger|load_castle_doctrine|load_divorce_laws|load_mpdta",  # network
    r"load_dataset\(",  # network
    r"maturin\s+develop",  # shell commands in python block
    r"pip\s+install",
    r"wild_bootstrap_se\(X,",  # low-level array API pseudo-code
    r"wide_to_long\(",  # references undefined wide_data variable
]


def _should_skip(code: str) -> Optional[str]:
    """Return a reason string if the snippet should be skipped, else None."""
    for pat in _SKIP_PATTERNS:
        if re.search(pat, code, re.MULTILINE):
            return f"matches skip pattern: {pat}"
    # Skip if no actual Python statements (just comments / blank)
    lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return "no executable statements"
    return None


# ---------------------------------------------------------------------------
# Build parameterized test cases
# ---------------------------------------------------------------------------
def _collect_cases() -> List[Tuple[str, str, Optional[str]]]:
    """Collect (test_id, code, skip_reason) triples."""
    cases = []
    for rel in RST_FILES:
        rst_path = DOCS_DIR / rel
        if not rst_path.exists():
            continue
        label = rel.replace("/", "_").removesuffix(".rst")
        for idx, code in _extract_snippets(rst_path):
            test_id = f"{label}:block{idx}"
            skip = _should_skip(code)
            cases.append((test_id, code, skip))
    return cases


_CASES = _collect_cases()

# Snippets that reference context variables from prior code blocks (e.g. a
# ``results`` object from a previous fit) and cannot run in isolation.  These
# aren't API mismatches — they're inherently context-dependent.
_KNOWN_FAILURES = set()  # type: set[str]


# ---------------------------------------------------------------------------
# Shared namespace builder
# ---------------------------------------------------------------------------
def _build_namespace() -> dict:
    """
    Build an exec namespace with diff_diff imports and synthetic data.

    Provides ``data`` (staggered panel) and ``balanced`` (same ref) so that
    most snippets that reference ``data`` can execute.
    """
    import diff_diff

    ns: dict = {"__builtins__": __builtins__}

    # Make all public diff_diff names available
    for name in dir(diff_diff):
        if not name.startswith("_"):
            ns[name] = getattr(diff_diff, name)

    ns["diff_diff"] = diff_diff

    # Remove 'results' module — it shadows the common variable name that
    # context-dependent snippets use for fit() return values.
    ns.pop("results", None)

    # Synthetic datasets that doc snippets commonly reference
    rng = np.random.default_rng(42)
    staggered = diff_diff.generate_staggered_data(
        n_units=60, n_periods=8, seed=42
    )
    # Add alias columns that doc snippets expect
    staggered["post"] = (
        staggered["period"] >= staggered["first_treat"].replace(0, 9999)
    ).astype(int)
    staggered["treatment"] = staggered["treated"]
    staggered["y"] = staggered["outcome"]
    staggered["unit_id"] = staggered["unit"]
    staggered["x1"] = rng.normal(size=len(staggered))
    staggered["x2"] = rng.normal(size=len(staggered))
    staggered["x3"] = rng.normal(size=len(staggered))
    staggered["state"] = staggered["unit_id"]
    staggered["ever_treated"] = staggered["treated"]
    staggered["group"] = np.where(staggered["treated"] == 1, "treatment", "control")

    ns["data"] = staggered
    ns["balanced"] = staggered.copy()
    ns["df"] = staggered

    # numpy / pandas always handy
    ns["np"] = np
    ns["pd"] = pd

    # matplotlib stub so plot calls don't actually render
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns["plt"] = plt
        ns["matplotlib"] = matplotlib
    except ImportError:
        pass

    return ns


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_id, code, skip_reason",
    [pytest.param(tid, c, s, id=tid) for tid, c, s in _CASES],
)
def test_doc_snippet(test_id: str, code: str, skip_reason: Optional[str]):
    """Execute a documentation code snippet and assert no TypeError/AttributeError."""
    if skip_reason:
        pytest.skip(skip_reason)
    if test_id in _KNOWN_FAILURES:
        pytest.xfail(f"Pre-existing doc bug: {test_id}")

    ns = _build_namespace()
    try:
        exec(compile(code, f"<{test_id}>", "exec"), ns)
    except (TypeError, AttributeError) as exc:
        pytest.fail(
            f"Snippet {test_id} raised {type(exc).__name__}: {exc}\n\n"
            f"Code:\n{textwrap.indent(code, '  ')}"
        )
    except Exception:
        # Other errors (ValueError, data issues) are not the focus of this
        # test — we only catch wrong kwargs (TypeError) and missing
        # attributes (AttributeError).
        pass
