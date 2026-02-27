from __future__ import annotations

import ast
import math
from typing import Any, Callable, Iterable

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
    "abs",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "where",
    "maximum",
    "minimum",
    "clip",
    "power",
    "heaviside",
    "arange",
    "zeros_like",
    "ones_like",
    "full_like",
}
_SAFE_NUMPY_CONSTANTS = {"pi", "e", "inf", "nan", "float64", "float32", "int64", "int32", "bool_"}

_SAFE_MATH_FUNCTIONS = {
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
}
_SAFE_MATH_CONSTANTS = {"pi", "e", "tau", "inf", "nan"}

_SAFE_NUMPY_ATTRS = _SAFE_NUMPY_FUNCTIONS | _SAFE_NUMPY_CONSTANTS
_SAFE_MATH_ATTRS = _SAFE_MATH_FUNCTIONS | _SAFE_MATH_CONSTANTS
_SAFE_VALUE_ATTRS = {"size", "shape"}

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


class _SafeExpressionValidator(ast.NodeVisitor):
    def __init__(self, allowed_variables: Iterable[str]) -> None:
        self.allowed_variables = set(allowed_variables)
        self.allowed_names = set(self.allowed_variables) | set(_SAFE_CALLABLES) | {"np", "math"}

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)):
            return
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax in custom expression: {type(node).__name__}.")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise ValueError("Dunder names are not allowed in custom expressions.")
        if node.id not in self.allowed_names:
            raise ValueError(f"Unsupported name in custom expression: {node.id!r}.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed in custom expressions.")
        if not isinstance(node.value, ast.Name):
            raise ValueError("Nested attribute access is not allowed in custom expressions.")
        base = node.value.id
        if base == "np":
            if node.attr not in _SAFE_NUMPY_ATTRS:
                raise ValueError(f"Unsupported numpy attribute in custom expression: np.{node.attr}.")
        elif base == "math":
            if node.attr not in _SAFE_MATH_ATTRS:
                raise ValueError(f"Unsupported math attribute in custom expression: math.{node.attr}.")
        elif base == "params":
            if node.attr != "get":
                raise ValueError(f"Unsupported params attribute in custom expression: params.{node.attr}.")
        elif base in self.allowed_variables:
            if node.attr not in _SAFE_VALUE_ATTRS:
                raise ValueError(f"Unsupported attribute in custom expression: {base}.{node.attr}.")
        else:
            raise ValueError(f"Unsupported attribute base in custom expression: {base!r}.")
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in {"np", "math"}:
            raise ValueError("Subscript access on modules is not allowed in custom expressions.")
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Call(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Starred keyword arguments are not allowed in custom expressions.")

        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in _SAFE_CALLABLES:
                raise ValueError(f"Unsupported function in custom expression: {func.id!r}.")
        elif isinstance(func, ast.Attribute):
            if not isinstance(func.value, ast.Name):
                raise ValueError("Nested attribute calls are not allowed in custom expressions.")
            base = func.value.id
            if base == "np":
                if func.attr not in _SAFE_NUMPY_FUNCTIONS:
                    raise ValueError(f"Unsupported numpy function in custom expression: np.{func.attr}.")
            elif base == "math":
                if func.attr not in _SAFE_MATH_FUNCTIONS:
                    raise ValueError(f"Unsupported math function in custom expression: math.{func.attr}.")
            elif base == "params":
                if func.attr != "get":
                    raise ValueError(f"Unsupported params method in custom expression: params.{func.attr}.")
            else:
                raise ValueError("Method calls are not allowed in custom expressions.")
        else:
            raise ValueError("Unsupported call target in custom expressions.")

        self.visit(func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)


def _normalize_expression_source(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return "0.0"
    if "\n" not in text and text.startswith("return "):
        text = text[len("return "):].strip()
    return text


def compile_safe_expression(source: str, *, variable_names: Iterable[str]) -> Callable[..., Any]:
    expression_source = _normalize_expression_source(source)
    try:
        parsed = ast.parse(expression_source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            "Custom expressions must be a single expression (optionally prefixed by 'return ')."
        ) from exc

    _SafeExpressionValidator(variable_names).visit(parsed)
    code = compile(parsed, "<custom-expression>", "eval")
    required = tuple(variable_names)

    def evaluate(**variables: Any) -> Any:
        missing = [name for name in required if name not in variables]
        if missing:
            missing_csv = ", ".join(missing)
            raise ValueError(f"Missing variables for custom expression evaluation: {missing_csv}.")
        env = {
            "__builtins__": {},
            "np": np,
            "math": math,
            **_SAFE_CALLABLES,
            **variables,
        }
        return eval(code, env, {})

    return evaluate
