"""Tests for CIParams bootstrap scaling in conftest.py."""

import math

import tests.conftest as conftest_module
from tests.conftest import CIParams


class TestCIParamsBootstrap:
    def test_min_n_in_pure_python_mode(self, monkeypatch):
        """min_n raises the floor in pure Python mode."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(499, min_n=199) == 199

    def test_min_n_passthrough_in_rust_mode(self, monkeypatch):
        """min_n has no effect when Rust backend is available."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", False)
        assert CIParams.bootstrap(499, min_n=199) == 499

    def test_min_n_capped_at_original_request(self, monkeypatch):
        """min_n never exceeds the original n."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(100, min_n=199) == 100

    def test_n_lte_10_ignores_min_n(self, monkeypatch):
        """n <= 10 always returns n regardless of min_n or mode."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(10, min_n=199) == 10

    def test_default_min_n_preserves_existing_behavior(self, monkeypatch):
        """Default min_n=11 matches pre-change behavior."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(499) == max(11, int(math.sqrt(499) * 1.6))  # 35
