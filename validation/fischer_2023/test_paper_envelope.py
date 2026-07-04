"""Fast pins for the analytic envelope helpers in :mod:`_paper_envelope`.

Guards the F&C 2023 Eq. (26) Airy-argument convention (review item 4,
``REVIEW-2026-07-02-code-health.md``): the intermediate-regime envelope is

    f(x) = 3 b0 Ai(x^2 / 4^{1/3})

with the cube root applying to the 4 only. The plausible misreading
``Ai((x^2/4)^{1/3})`` coincides with the correct form at x = 1 but differs
by ~9x at x = 2 and ~4000x at x = 3, so the pins below sit at x > 1.
Argument convention verified against the published text (arXiv:2212.08155,
Eq. (26)) on 2026-07-03.

No solves; runs in the default suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023._paper_envelope import f_intermediate, f_low

# 3 * Ai(x^2 / 4^{1/3}) at b0 = 1, computed with scipy.special.airy
# (scipy 1.x, machine precision) after verifying the argument convention
# against the published Eq. (26).
_EQ26_PINS = {
    0.9: 6.881710433849e-01,
    1.5: 2.404983504962e-01,
    2.0: 4.563817619646e-02,
    3.0: 6.718834688133e-05,
}

# Same points evaluated with the misread argument (x^2/4)^{1/3}; kept to
# assert the implementation stays far from the wrong convention.
_MISREAD_VALUES = {
    2.0: 4.058772489386e-01,
    3.0: 2.766997278578e-01,
}


@pytest.mark.parametrize("x", sorted(_EQ26_PINS))
def test_f_intermediate_pins_eq26_airy_argument(x: float) -> None:
    got = f_intermediate(np.array([x]), b0=1.0)[0]
    assert got == pytest.approx(_EQ26_PINS[x], rel=1e-9)


@pytest.mark.parametrize("x", sorted(_MISREAD_VALUES))
def test_f_intermediate_is_not_the_cube_root_misreading(x: float) -> None:
    got = f_intermediate(np.array([x]), b0=1.0)[0]
    assert got != pytest.approx(_MISREAD_VALUES[x], rel=0.5)


def test_f_low_eq25a_coefficients() -> None:
    # Eq. (25a): f(1) = b0 (1 - 0.564 + 0.119).
    got = f_low(np.array([1.0]), b0=1.0)[0]
    assert got == pytest.approx(0.555, rel=1e-12)


def test_low_to_intermediate_crossover_is_continuous_ish() -> None:
    # The envelope switches from Eq. (25a) to Eq. (26) at x = 0.8; the two
    # asymptotic forms should agree there to within ~10% (they are envelopes,
    # not a matched expansion).
    x = np.array([0.8])
    lo = f_low(x, b0=1.0)[0]
    mid = f_intermediate(x, b0=1.0)[0]
    assert mid == pytest.approx(lo, rel=0.12)
