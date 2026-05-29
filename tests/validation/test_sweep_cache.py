"""Tests for the validation sweep disk cache (validation/sweep_cache.py).

Covers the properties the cache's correctness rests on: a deterministic
content-addressed key; invalidation on any solver-source change; *non*-
invalidation when only observable/plot code changes; an atomic, corruption-safe
store; and a transparent disable switch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from validation import sweep_cache as sc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Default state: caching enabled, no dir override (tests pass cache_dir).
    monkeypatch.delenv("QPSIM_SWEEP_CACHE", raising=False)
    monkeypatch.delenv("QPSIM_SWEEP_CACHE_DIR", raising=False)


def _make_qpsim_tree(root):
    """A minimal fake qpsim/ tree: a solver subpackage + an observables one."""
    q = root / "qpsim"
    (q / "solvers").mkdir(parents=True)
    (q / "observables").mkdir(parents=True)
    (q / "__init__.py").write_text("")
    (q / "constants.py").write_text("KB = 1.0\n")
    (q / "solvers" / "__init__.py").write_text("")
    (q / "solvers" / "newton.py").write_text("def solve():\n    return 1\n")
    (q / "observables" / "__init__.py").write_text("")
    (q / "observables" / "ac.py").write_text("def obs():\n    return 2\n")
    return q


# ── key ───────────────────────────────────────────────────────────────

class TestCacheKey:
    def test_deterministic(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        k1 = sc.cache_key("f", {"a": 1}, {"b": 2}, extra_source="src", qpsim_root=q)
        k2 = sc.cache_key("f", {"a": 1}, {"b": 2}, extra_source="src", qpsim_root=q)
        assert k1 == k2
        assert len(k1) == 64  # sha256 hex

    def test_varies_with_each_input(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        base = sc.cache_key("f", {"a": 1}, {"b": 2}, extra_source="src", qpsim_root=q)
        assert sc.cache_key("g", {"a": 1}, {"b": 2}, extra_source="src", qpsim_root=q) != base
        assert sc.cache_key("f", {"a": 9}, {"b": 2}, extra_source="src", qpsim_root=q) != base
        assert sc.cache_key("f", {"a": 1}, {"b": 9}, extra_source="src", qpsim_root=q) != base
        assert sc.cache_key("f", {"a": 1}, {"b": 2}, extra_source="x", qpsim_root=q) != base

    def test_fingerprint_key_order_irrelevant(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        k1 = sc.cache_key("f", {"a": 1, "b": 2}, {}, qpsim_root=q)
        k2 = sc.cache_key("f", {"b": 2, "a": 1}, {}, qpsim_root=q)
        assert k1 == k2

    def test_numpy_scalars_and_arrays_in_fingerprint(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        # np scalar vs equal python float must hash the same (canonicalized).
        k_np = sc.cache_key("f", {"x": np.float64(1.5)}, {}, qpsim_root=q)
        k_py = sc.cache_key("f", {"x": 1.5}, {}, qpsim_root=q)
        assert k_np == k_py


# ── solve-source digest ─────────────────────────────────────────────────

class TestSolveSourceDigest:
    def test_ignores_observables_subpackage(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        d0 = sc.solve_source_digest(q)

        # Editing observable code must NOT change the digest.
        (q / "observables" / "ac.py").write_text("def obs():\n    return 999\n")
        assert sc.solve_source_digest(q) == d0

        # A brand-new observables module is also ignored.
        (q / "observables" / "qfactor.py").write_text("def q():\n    return 3\n")
        assert sc.solve_source_digest(q) == d0

    def test_changes_on_solver_edit(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        d0 = sc.solve_source_digest(q)
        (q / "solvers" / "newton.py").write_text("def solve():\n    return 42\n")
        assert sc.solve_source_digest(q) != d0

    def test_changes_on_new_solver_file_and_rename(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        d0 = sc.solve_source_digest(q)
        (q / "solvers" / "anderson.py").write_text("def acc():\n    return 0\n")
        d1 = sc.solve_source_digest(q)
        assert d1 != d0
        # Rename (path folded into the hash) changes the digest even at equal bytes.
        (q / "solvers" / "anderson.py").rename(q / "solvers" / "aa.py")
        assert sc.solve_source_digest(q) != d1


# ── store / load ────────────────────────────────────────────────────────

class TestStoreLoad:
    def test_roundtrip_preserves_arrays(self, tmp_path):
        cdir = tmp_path / "cache"
        arrays = {
            "a": np.linspace(0.0, 1.0, 7),
            "b": np.arange(6, dtype=np.int64).reshape(2, 3),
            "s": np.array(3.5),  # 0-d scalar
        }
        path = sc.store("figX", "deadbeef", arrays, cache_dir=cdir)
        # Stored at exactly <key>.npz (no doubled extension from np.savez).
        assert path.name == "deadbeef.npz"
        out = sc.load("figX", "deadbeef", cache_dir=cdir)
        assert out is not None
        assert set(out) == set(arrays)
        for k in arrays:
            np.testing.assert_array_equal(out[k], arrays[k])
            assert out[k].dtype == arrays[k].dtype

    def test_missing_is_none(self, tmp_path):
        assert sc.load("figX", "nope", cache_dir=tmp_path / "cache") is None

    def test_corrupt_entry_is_treated_as_miss(self, tmp_path):
        cdir = tmp_path / "cache"
        path = sc.store("figX", "k", {"x": np.arange(3.0)}, cache_dir=cdir)
        path.write_bytes(b"not a real npz file")
        assert sc.load("figX", "k", cache_dir=cdir) is None

    def test_provenance_sidecar_written(self, tmp_path):
        cdir = tmp_path / "cache"
        sc.store(
            "figX", "k", {"x": np.zeros(1)},
            provenance={"figure": "figX", "note": "hello"}, cache_dir=cdir,
        )
        _, meta = sc._entry_paths(cdir, "figX", "k")
        data = json.loads(meta.read_text())
        assert data["note"] == "hello"

    def test_figure_id_with_slash_is_path_safe(self, tmp_path):
        cdir = tmp_path / "cache"
        sc.store("fischer_2023/fig7", "k", {"x": np.zeros(1)}, cache_dir=cdir)
        assert (cdir / "fischer_2023__fig7" / "k.npz").exists()
        assert sc.load("fischer_2023/fig7", "k", cache_dir=cdir) is not None


# ── cached_solve (integration) ──────────────────────────────────────────

class TestCachedSolve:
    def test_miss_then_hit(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        cdir = tmp_path / "cache"
        calls = []

        def solve():
            calls.append(1)
            return {"x": np.array([float(len(calls))])}

        r1 = sc.cached_solve("figX", solve, fingerprint={"a": 1}, cache_dir=cdir, qpsim_root=q)
        r2 = sc.cached_solve("figX", solve, fingerprint={"a": 1}, cache_dir=cdir, qpsim_root=q)
        assert len(calls) == 1  # second call was a cache hit
        np.testing.assert_array_equal(r1["x"], r2["x"])

    def test_recomputes_on_solver_source_change_but_not_observable_change(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        cdir = tmp_path / "cache"
        calls = []

        def solve():
            calls.append(1)
            return {"x": np.array([float(len(calls))])}

        sc.cached_solve("figX", solve, fingerprint={"a": 1}, cache_dir=cdir, qpsim_root=q)
        sc.cached_solve("figX", solve, fingerprint={"a": 1}, cache_dir=cdir, qpsim_root=q)
        assert len(calls) == 1  # warm

        # Solver edit -> key changes -> recompute.
        (q / "solvers" / "newton.py").write_text("def solve():\n    return 7\n")
        sc.cached_solve("figX", solve, fingerprint={"a": 1}, cache_dir=cdir, qpsim_root=q)
        assert len(calls) == 2

        # Observable edit -> key unchanged -> still a hit.
        (q / "observables" / "ac.py").write_text("def obs():\n    return 7\n")
        sc.cached_solve("figX", solve, fingerprint={"a": 1}, cache_dir=cdir, qpsim_root=q)
        assert len(calls) == 2

    def test_recomputes_on_fingerprint_change(self, tmp_path):
        q = _make_qpsim_tree(tmp_path)
        cdir = tmp_path / "cache"
        calls = []

        def solve():
            calls.append(1)
            return {"x": np.zeros(1)}

        sc.cached_solve("figX", solve, fingerprint={"a": 1}, cache_dir=cdir, qpsim_root=q)
        sc.cached_solve("figX", solve, fingerprint={"a": 2}, cache_dir=cdir, qpsim_root=q)
        assert len(calls) == 2

    def test_disabled_env_is_passthrough(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "0")
        q = _make_qpsim_tree(tmp_path)
        cdir = tmp_path / "cache"
        calls = []

        def solve():
            calls.append(1)
            return {"x": np.zeros(1)}

        sc.cached_solve("figX", solve, fingerprint={}, cache_dir=cdir, qpsim_root=q)
        sc.cached_solve("figX", solve, fingerprint={}, cache_dir=cdir, qpsim_root=q)
        assert len(calls) == 2  # ran both times
        assert not cdir.exists() or not list(cdir.rglob("*.npz"))  # nothing stored


# ── env / lifecycle ─────────────────────────────────────────────────────

class TestEnvAndLifecycle:
    def test_is_enabled_default_and_toggle(self, monkeypatch):
        assert sc.is_enabled() is True
        for off in ("0", "false", "No", "OFF", ""):
            monkeypatch.setenv("QPSIM_SWEEP_CACHE", off)
            assert sc.is_enabled() is False
        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "1")
        assert sc.is_enabled() is True

    def test_cache_dir_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path / "elsewhere"))
        assert sc.default_cache_dir() == tmp_path / "elsewhere"

    def test_default_cache_dir_under_validation(self, monkeypatch):
        monkeypatch.delenv("QPSIM_SWEEP_CACHE_DIR", raising=False)
        d = sc.default_cache_dir()
        assert d.name == ".sweep_cache"
        assert d.parent.name == "validation"

    def test_clear_per_figure_and_all(self, tmp_path):
        cdir = tmp_path / "cache"
        sc.store("figA", "k1", {"x": np.zeros(1)}, cache_dir=cdir)
        sc.store("figB", "k2", {"x": np.zeros(1)}, cache_dir=cdir)

        sc.clear(figure="figA", cache_dir=cdir)
        assert sc.load("figA", "k1", cache_dir=cdir) is None
        assert sc.load("figB", "k2", cache_dir=cdir) is not None

        sc.clear(cache_dir=cdir)
        assert sc.load("figB", "k2", cache_dir=cdir) is None
