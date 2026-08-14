"""User-prescribed fields over energy, space and time.

A drive, an initial occupation and a gap map are all the same kind of object:
a function the user prescribes, evaluated onto the solver's own grid. This
package holds the vocabulary for stating one — presets for the shapes people
actually reach for, and a compiled expression for everything else.
"""

from qpsim.fields.safe_eval import SafeExpressionError, compile_expression

__all__ = ["SafeExpressionError", "compile_expression"]
