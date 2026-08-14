"""What the expression whitelist admits, and what it must refuse."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.fields.safe_eval import SafeExpressionError, compile_expression


def _f(source: str, variables=("x",)):
    return compile_expression(source, variables=variables)


class TestWhatItAdmits:
    def test_arithmetic_over_a_declared_variable(self):
        assert _f("2 * x + 1")(x=3.0) == 7.0

    def test_vectorises_over_an_array(self):
        got = _f("np.exp(-x**2)")(x=np.array([0.0, 1.0]))
        np.testing.assert_allclose(got, [1.0, np.exp(-1.0)])

    def test_a_leading_return_is_accepted(self):
        """The box looks like a function body, so people type `return`."""
        assert _f("return x * 2")(x=2.0) == 4.0

    def test_an_empty_expression_is_zero(self):
        assert _f("")(x=1.0) == 0.0

    def test_params_supply_constants_without_editing_the_expression(self):
        fn = _f("params.get('a', 1.0) * x")
        assert fn(x=2.0) == 2.0
        assert fn(x=2.0, params={"a": 10.0}) == 20.0

    def test_a_conditional_makes_a_pulse(self):
        fn = _f("np.where((t > 1.0) & (t < 2.0), 1.0, 0.0)", variables=("t",))
        np.testing.assert_array_equal(
            fn(t=np.array([0.5, 1.5, 2.5])), [0.0, 1.0, 0.0]
        )

    def test_several_variables_together(self):
        fn = _f("E * x + t", variables=("E", "x", "t"))
        assert fn(E=2.0, x=3.0, t=1.0) == 7.0


class TestWhatItRefuses:
    @pytest.mark.parametrize("source", [
        "__import__('os').system('echo hi')",
        "().__class__.__bases__",
        "x.__class__",
        "open('/etc/passwd')",
        "eval('1')",
        "exec('x=1')",
        "globals()",
        "[i for i in range(3)]",
        "lambda y: y",
        "np.__loader__",
        "np['exp']",
        "x.tobytes()",
        "np.linalg.norm(x)",
    ])
    def test_escape_attempts_are_rejected(self, source):
        with pytest.raises(SafeExpressionError):
            _f(source)

    def test_an_undeclared_variable_is_named_in_the_error(self):
        """Reported when the setup is validated, not mid-run."""
        with pytest.raises(SafeExpressionError, match="Unknown name 'y'"):
            _f("y * 2")

    def test_the_error_lists_what_is_in_scope(self):
        with pytest.raises(SafeExpressionError, match="E, t, x"):
            _f("q", variables=("x", "t", "E"))

    def test_a_statement_is_not_an_expression(self):
        with pytest.raises(SafeExpressionError, match="single expression"):
            _f("x = 1")

    def test_a_huge_integer_power_is_refused_not_evaluated(self):
        """`2 ** 10**9` hangs the interpreter rather than returning a field."""
        with pytest.raises(SafeExpressionError, match="Exponent"):
            _f("2 ** 100000")

    def test_an_overlong_source_is_refused_before_parsing(self):
        with pytest.raises(SafeExpressionError, match="past the"):
            _f("x + " * 2000 + "x")

    def test_a_missing_variable_at_call_time_is_reported(self):
        with pytest.raises(SafeExpressionError, match="needs x"):
            _f("x")()
