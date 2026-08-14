"""Compile a user-typed expression to a callable, by AST whitelist.

Prescribed fields — a drive's shape in energy, space or time, an initial
occupation, a gap map — are ultimately arbitrary functions, and presets only
ever cover the cases somebody anticipated. This compiles one typed expression
into something callable, admitting only nodes and names on an explicit list.

**What this is and is not.** It is a guard against a typo or a stray import
reaching ``eval`` in the user's own process — the expression comes from the
person running the simulation, on their own machine, about their own device.
It is *not* a security boundary against a hostile author: do not use it on
expressions arriving from somewhere else without deciding separately that the
source is trusted. The whitelist is tight (no imports, no statements, no
comprehensions, no lambdas, no dunder names or attributes, no attribute
chains, no subscripting a module, no method calls except ``params.get``), and
it is deliberately easier to widen than to work around.

The vocabulary is deliberately small: ``np`` and ``math`` for the usual
elementary functions, a handful of builtins, whichever variables the caller
declares, and ``params`` for a dict of the author's own constants — so a
shape can be written once and re-tuned without editing the expression.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

_SAFE_CALLABLES: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "len": len,
    "float": float,
    "int": int,
    "bool": bool,
}

_SAFE_NUMPY_FUNCTIONS = {
    "abs", "sqrt", "exp", "expm1", "log", "log1p", "log10",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh",
    "where", "maximum", "minimum", "clip", "power", "heaviside", "sign",
    "arange", "linspace", "zeros_like", "ones_like", "full_like",
    "floor", "ceil", "round", "mod",
}
_SAFE_NUMPY_CONSTANTS = {"pi", "e", "inf", "nan"}

_SAFE_MATH_FUNCTIONS = {
    "sqrt", "exp", "log", "log10", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "tanh", "floor", "ceil",
}
_SAFE_MATH_CONSTANTS = {"pi", "e", "tau", "inf", "nan"}

_SAFE_NUMPY_ATTRS = _SAFE_NUMPY_FUNCTIONS | _SAFE_NUMPY_CONSTANTS
_SAFE_MATH_ATTRS = _SAFE_MATH_FUNCTIONS | _SAFE_MATH_CONSTANTS
# Shape and size only. Anything reachable from a numpy array's richer
# attribute surface is a way out of the whitelist.
_SAFE_VALUE_ATTRS = {"size", "shape", "ndim"}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.IfExp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
)

# A constant exponent big enough to hang the process on integers. `2 ** 10**9`
# is a plausible typo and takes the interpreter out with it; there is no
# legitimate prescribed field that needs it.
_MAX_CONSTANT_EXPONENT = 1024

# Long enough for any real profile, short enough that a paste accident is
# rejected before it is parsed.
_MAX_SOURCE_CHARS = 4000


class SafeExpressionError(ValueError):
    """A prescribed expression that the whitelist refuses to compile."""


class _Validator(ast.NodeVisitor):
    def __init__(self, allowed_variables: Iterable[str]) -> None:
        self.allowed_variables = set(allowed_variables)
        self.allowed_names = (
            self.allowed_variables | set(_SAFE_CALLABLES) | {"np", "math", "params"}
        )

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(
            node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)
        ):
            return
        if not isinstance(node, _ALLOWED_NODES):
            raise SafeExpressionError(
                f"Unsupported syntax in a prescribed expression: "
                f"{type(node).__name__}."
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise SafeExpressionError("Dunder names are not allowed.")
        if node.id not in self.allowed_names:
            allowed = ", ".join(sorted(self.allowed_variables))
            raise SafeExpressionError(
                f"Unknown name {node.id!r}. Variables in scope here: {allowed}."
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise SafeExpressionError("Dunder attribute access is not allowed.")
        if not isinstance(node.value, ast.Name):
            raise SafeExpressionError("Chained attribute access is not allowed.")
        base = node.value.id
        if base == "np":
            if node.attr not in _SAFE_NUMPY_ATTRS:
                raise SafeExpressionError(f"np.{node.attr} is not available here.")
        elif base == "math":
            if node.attr not in _SAFE_MATH_ATTRS:
                raise SafeExpressionError(f"math.{node.attr} is not available here.")
        elif base == "params":
            if node.attr != "get":
                raise SafeExpressionError(
                    "Only params.get(name, default) is available on params."
                )
        elif base in self.allowed_variables:
            if node.attr not in _SAFE_VALUE_ATTRS:
                raise SafeExpressionError(
                    f"{base}.{node.attr} is not available; only "
                    f"{', '.join(sorted(_SAFE_VALUE_ATTRS))}."
                )
        else:
            raise SafeExpressionError(f"Unknown attribute base {base!r}.")
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in {"np", "math"}:
            raise SafeExpressionError("Modules cannot be subscripted.")
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            exponent = node.right.value
            if isinstance(exponent, int) and abs(exponent) > _MAX_CONSTANT_EXPONENT:
                raise SafeExpressionError(
                    f"Exponent {exponent} exceeds {_MAX_CONSTANT_EXPONENT}: an "
                    "integer power that large hangs the process rather than "
                    "returning a field."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg is None:
                raise SafeExpressionError("Starred keyword arguments are not allowed.")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise SafeExpressionError("Starred arguments are not allowed.")

        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in _SAFE_CALLABLES:
                raise SafeExpressionError(f"{func.id!r} is not callable here.")
        elif isinstance(func, ast.Attribute):
            if not isinstance(func.value, ast.Name):
                raise SafeExpressionError("Chained attribute calls are not allowed.")
            base = func.value.id
            if base == "np":
                if func.attr not in _SAFE_NUMPY_FUNCTIONS:
                    raise SafeExpressionError(f"np.{func.attr} is not callable here.")
            elif base == "math":
                if func.attr not in _SAFE_MATH_FUNCTIONS:
                    raise SafeExpressionError(f"math.{func.attr} is not callable here.")
            elif base == "params":
                if func.attr != "get":
                    raise SafeExpressionError("Only params.get(...) is available.")
            else:
                raise SafeExpressionError("Method calls are not allowed.")
        else:
            raise SafeExpressionError("Unsupported call target.")

        self.visit(func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)


def _normalise(source: str) -> str:
    """Accept ``return <expr>`` as well as a bare expression.

    People reach for ``return`` because the box looks like a function body.
    Accepting it costs nothing and removes a confusing first failure.
    """
    text = str(source or "").strip()
    if not text:
        return "0.0"
    if "\n" not in text and text.startswith("return "):
        text = text[len("return "):].strip()
    return text


def compile_expression(
    source: str, *, variables: Iterable[str]
) -> Callable[..., Any]:
    """Compile ``source`` to ``f(**variables, params=...) -> value``.

    ``variables`` names what the expression may refer to. Everything else is
    rejected at compile time, so a mistyped variable is reported when the
    setup is validated rather than in the middle of a run.
    """
    text = _normalise(source)
    if len(text) > _MAX_SOURCE_CHARS:
        raise SafeExpressionError(
            f"Expression is {len(text)} characters, past the "
            f"{_MAX_SOURCE_CHARS} limit."
        )
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError(
            f"Not a single expression (optionally prefixed by 'return '): {exc.msg}."
        ) from exc

    names = tuple(variables)
    _Validator(names).visit(parsed)
    code = compile(parsed, "<prescribed-expression>", "eval")

    def evaluate(*, params: dict[str, Any] | None = None, **values: Any) -> Any:
        missing = [name for name in names if name not in values]
        if missing:
            raise SafeExpressionError(
                f"Expression needs {', '.join(missing)}, which was not supplied."
            )
        env: dict[str, Any] = {
            "__builtins__": {},
            "np": np,
            "math": math,
            "params": dict(params or {}),
            **_SAFE_CALLABLES,
            **values,
        }
        # Whitelisted AST with empty builtins; see the module docstring for
        # exactly what that guarantees and what it does not.
        return eval(code, env, {})

    evaluate.__doc__ = f"Prescribed expression over {', '.join(names)}: {text}"
    return evaluate
