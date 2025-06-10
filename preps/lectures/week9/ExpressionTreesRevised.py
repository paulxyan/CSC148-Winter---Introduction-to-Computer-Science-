from typing import Any, Optional


class Statement:
    """Representing a statement in python"""

    def evaluate(self, env: dict[str, Any]) -> Any:

        """Return result of the statement"""

        raise NotImplementedError

class Expr(Statement):

    """An expression in Python language"""


class Name(Expr):

    """Represents a variable in python"""

    id: str

    def __init__(self, id: str) -> None:

        self.id = id

    def evaluate(self, env: dict[str, Any]) -> Any:

        if self.id not in env:
            return ValueError

        else:
            return env[self.id]

class Num(Expr):

    """A number literal in python"""

    def __init__(self, val: int):

        self.val = val

    def evaluate(self, env: dict[str, Any]) -> Any:
        return self.val

class Bool(Expr):

    """Represents a boolean expression in python"""

    val: bool

    def __init__(self, val: bool) -> None:
        self.val = val

    def evaluate(self, env: dict) -> Any:

        return self.val

class Assign(Statement):

    def __init__(self, target: str, value: Expr) -> None:

        self.target = target
        self.value = value

    def evaluate(self, env: dict) -> Any:

        env[self.target] = self.value.evaluate(env)

class If(Statement):

    """An if statement.
    === Attributes ===
    test: The condition expression of this if statement.
    body: A sequence of statements to evaluate if the condition is true.
    orelse: A sequence of statements to evaluate if the condition is false.
    (This would be empty in the case that there is no `else` block.)
    """
    test: Expr
    body: list[Statement]
    orelse: list[Statement]


    def __init__(self, condition: Expr, body: list[Statement], orelse:
                 list[Statement]):

        self.test = condition
        self.body = body
        self.orelse = orelse

    def evaluate(self, env: dict[str, Any]) -> Optional[Any]:
        """Evaluate this statement.
        >>> stmt = If(Bool(True),
        ... [Assign('x', Num(1))],
        ... [Assign('y', Num(0))])
        ...
        >>> env = {}
        >>> stmt.evaluate(env)
        >>> env
        {'x': 1}
        """

        if bool(self.test.evaluate(env)):

            for stmt in self.body:
                stmt.evaluate(env)

        else:

            for stmt in self.orelse:
                stmt.evaluate(env)

class ForRange(Statement):
    """A for loop that loops over a range of numbers.
    for <target> in range(<start>, <stop>):
    <body>
    === Attributes ===
    target: The loop variable.
    start: The start for the range (inclusive).
    stop: The end of the range (this is *exclusive*, so <stop> is not included in the loop).
    body: The statements to execute in the loop body.
    """

    def evaluate(self, env: dict[str, Any]) -> Optional[Any]:
        """Evaluate this statement.
        Raise a TypeError if either the start or stop expressions do *not*
        evaluate to integers. (This is technically a bit stricter than real Python.)
        >>> statement = ForRange('x', Num(1), BinOp(Num(2), '+', Num(3)),
        ... [Print(Name('x'))])
        >>> statement.evaluate({})
        1
        2
        3
        4
        """

        pass

