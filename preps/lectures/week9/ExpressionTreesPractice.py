from typing import Any, Union


class Expr:
    """An abstract class representing a Python expression.
    """
    def evaluate(self) -> Any:
        """Return the *value* of this expression.

        The returned value should be the result of how this expression would be
        evaluated by the Python interpreter.
        """
        raise NotImplementedError

class Num(Expr):
    """An numeric constant literal.

    === Attributes ===
    n: the value of the constant
    """
    n: Union[int, float]

    def __init__(self, number: Union[int, float]) -> None:
        """Initialize a new numeric constant."""
        self.n = number

    def evaluate(self) -> Any:
        """Return the *value* of this expression.

        The returned value should be the result of how this expression would be
        evaluated by the Python interpreter.

        >>> number = Num(10.5)
        >>> number.evaluate()
        10.5
        """
        return self.n  # Simply return the value itself!

class BinOp(Expr):
    """An arithmetic binary operation.

        === Attributes ===
        left: the left operand
        op: the name of the operator
        right: the right operand

        === Representation Invariants ===
        - self.op == '+' or self.op == '*'
        """
    left: Expr
    op: str
    right: Expr

    def __init__(self, left: Expr, op: str, right: Expr) -> None:
        """Initialize a new binary operation expression.

                Precondition: <op> is the string '+' or '*'.
                """
        self.left = left
        self.op = op
        self.right = right

    def evaluate(self) -> Any:
        """Return the *value* of this expression.
        """

        left_val = self.left.evaluate()
        right_val = self.right.evaluate()

        if self.op == '+':

            return left_val + right_val

        elif self.op  == '*':

            return left_val * right_val

        else:

            raise ValueError

class Statement:
    def evaluate(self, env: dict[str, Any]) -> Any:
        """Return the *value* of this expression, in given environment"""