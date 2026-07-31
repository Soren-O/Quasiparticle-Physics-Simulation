from __future__ import annotations

import numpy as np
import pytest
from validation.fischer_2023.fig6_author_component_parity import (
    _comparison_metrics,
    _embed_author_phonons,
)
from validation.fischer_2023.fig6_author_frozen_state import (
    FrozenStateDiagnosticError,
)


def test_author_phonons_embed_at_identical_frequency_and_zero_elsewhere() -> None:
    qpsim_omega = np.arange(0.0, 9.0)
    author_omega = np.arange(1.0, 5.0)
    author_n = np.array([0.4, 0.3, 0.2, 0.1])

    full, indices = _embed_author_phonons(
        author_omega,
        author_n,
        qpsim_omega,
        spacing_uev=1.0,
    )

    np.testing.assert_array_equal(indices, np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(full[indices], author_n)
    np.testing.assert_array_equal(full[[0, 5, 6, 7, 8]], 0.0)


def test_author_phonon_embedding_rejects_shifted_coordinates() -> None:
    with pytest.raises(FrozenStateDiagnosticError, match="h, 2h"):
        _embed_author_phonons(
            np.array([1.5, 2.5]),
            np.array([0.2, 0.1]),
            np.arange(0.0, 5.0, 0.5),
            spacing_uev=1.0,
        )


def test_component_metric_is_scale_aware_and_weighted() -> None:
    author = np.array([1.0e-30, -2.0e-30])
    qpsim = np.array([2.0e-30, -1.0e-30])
    metrics = _comparison_metrics(
        author,
        qpsim,
        weights=np.array([2.0, 3.0]),
    )

    assert metrics["l1_symmetric_relative_error"] == pytest.approx(1.0 / 3.0)
    assert metrics["max_absolute_error_s_inv"] == pytest.approx(1.0e-30)
    assert metrics["author_weighted_sum"] == pytest.approx(-4.0e-30)
    assert metrics["qpsim_weighted_sum"] == pytest.approx(1.0e-30)
    assert metrics["weighted_sum_signed_error"] == pytest.approx(5.0e-30)
