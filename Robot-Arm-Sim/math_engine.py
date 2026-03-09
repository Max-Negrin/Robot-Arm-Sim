"""
Safe mathematical expression engine for the robot arm simulator.

Allows users to enter SymPy expressions for specific joints, parsed and
evaluated in a sandboxed environment. Defense-in-depth:
1. Raw string validation (reject dangerous substrings)
2. Restricted sympy.sympify locals (only whitelisted names)
3. AST walk to verify all expression nodes are safe types
"""

import math
import sympy
from sympy import (
    Symbol, pi as sym_pi, sin, cos, tan, asin, acos, atan, atan2,
    sqrt, Abs, exp, log, Float, Integer, Rational,
)
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.numbers import Number, NegativeOne, One, Zero, Half
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------

_MAX_EXPR_LEN = 200

_BLOCKED_SUBSTRINGS = [
    "__", "import", "eval", "exec", "open", "os.", "sys.", "subprocess",
    "getattr", "setattr", "delattr", "globals", "locals", "compile",
    "breakpoint", "input",
]

# Symbols available to user expressions
_VARIABLE_NAMES = set()
for i in range(1, 11):
    _VARIABLE_NAMES.add(f"theta_{i}")
    _VARIABLE_NAMES.add(f"L{i}")
_VARIABLE_NAMES.update(["x", "y", "z", "r", "target_x", "target_y", "target_z", "target_r", "t"])

# Build the restricted locals dict for sympify
_SYMPIFY_LOCALS: Dict[str, object] = {}
for name in _VARIABLE_NAMES:
    _SYMPIFY_LOCALS[name] = Symbol(name)
_SYMPIFY_LOCALS["pi"] = sym_pi
_SYMPIFY_LOCALS["sin"] = sin
_SYMPIFY_LOCALS["cos"] = cos
_SYMPIFY_LOCALS["tan"] = tan
_SYMPIFY_LOCALS["asin"] = asin
_SYMPIFY_LOCALS["acos"] = acos
_SYMPIFY_LOCALS["atan"] = atan
_SYMPIFY_LOCALS["atan2"] = atan2
_SYMPIFY_LOCALS["sqrt"] = sqrt
_SYMPIFY_LOCALS["abs"] = Abs
_SYMPIFY_LOCALS["exp"] = exp
_SYMPIFY_LOCALS["log"] = log

# Safe AST node types
_SAFE_TYPES = (
    Symbol, Number, Float, Integer, Rational, NegativeOne, One, Zero, Half,
    Add, Mul, Pow, sym_pi.__class__,
    sin, cos, tan, asin, acos, atan, atan2, sqrt, Abs, exp, log,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MathEngine:
    """
    Safely parse and evaluate SymPy expressions entered by the user.

    Usage::

        engine = MathEngine()
        err = engine.set_expression(2, "pi/4 - theta_1")
        if not err:
            val = engine.evaluate(2, {"theta_1": 0.5})
    """

    def __init__(self) -> None:
        self.expressions: Dict[int, sympy.Expr] = {}
        self.raw_strings: Dict[int, str] = {}

    def set_expression(self, joint_index: int, expr_string: str) -> str:
        """
        Parse and store an expression. Returns empty string on success,
        or an error message on failure.
        """
        expr_string = expr_string.strip()

        # Length check
        if len(expr_string) > _MAX_EXPR_LEN:
            return f"Expression too long (max {_MAX_EXPR_LEN} chars)"
        if not expr_string:
            return "Empty expression"

        # Blocked substring check
        lower = expr_string.lower()
        for blocked in _BLOCKED_SUBSTRINGS:
            if blocked in lower:
                return f"Forbidden token: '{blocked}'"

        # Parse
        try:
            expr = sympy.sympify(expr_string, locals=_SYMPIFY_LOCALS)
        except (sympy.SympifyError, SyntaxError, TypeError) as e:
            return f"Parse error: {str(e)[:80]}"

        # AST safety walk
        if not self._validate_expression(expr):
            return "Expression contains disallowed operations"

        self.expressions[joint_index] = expr
        self.raw_strings[joint_index] = expr_string
        return ""

    def evaluate(self, joint_index: int, context: Dict[str, float]) -> Optional[float]:
        """
        Evaluate a stored expression with the given variable bindings.
        Returns float result, or None if evaluation fails.
        """
        if joint_index not in self.expressions:
            return None

        expr = self.expressions[joint_index]
        try:
            subs = {}
            for name, value in context.items():
                if name in _SYMPIFY_LOCALS and isinstance(_SYMPIFY_LOCALS[name], Symbol):
                    subs[_SYMPIFY_LOCALS[name]] = value
                elif name == "pi":
                    continue
                else:
                    subs[Symbol(name)] = value

            result = expr.subs(subs)
            return float(result.evalf())
        except (TypeError, ValueError, AttributeError, ZeroDivisionError):
            return None

    def clear(self, joint_index: int) -> None:
        self.expressions.pop(joint_index, None)
        self.raw_strings.pop(joint_index, None)

    def has_expression(self, joint_index: int) -> bool:
        return joint_index in self.expressions

    def get_display_string(self, joint_index: int) -> str:
        return self.raw_strings.get(joint_index, "")

    @staticmethod
    def _validate_expression(expr: sympy.Expr) -> bool:
        """Walk the SymPy expression tree and reject anything unsafe."""
        for node in sympy.preorder_traversal(expr):
            if isinstance(node, _SAFE_TYPES):
                continue
            # Allow function applications of whitelisted functions
            if hasattr(node, "func") and node.func in (
                sin, cos, tan, asin, acos, atan, atan2, sqrt, Abs, exp, log,
            ):
                continue
            # Reject unknown types
            return False
        return True
