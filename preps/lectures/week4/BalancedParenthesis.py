from StacksAndQueuePractise import Stack


def is_balanced(line: str) -> bool:
    """Return whether <line> contains balanced parentheses.
    Ignore square and curly brackets.
    >>> is_balanced('(a * (3 + b))')
    True
    >>> is_balanced('(a * (3 + b]]') # Note that the two ']'s don't matter.
    False
    >>> is_balanced('1 + 2(x - y)}') # Note that the '}' doesn't matter.
    False
    >>> is_balanced('(2133{233}{[]})')
    True
    >>> is_balanced('3 - (x')
    False
    """


    stack1 = Stack()

    for char in line:

        if char == '{' or char == '[' or char == '(':
            stack1.push(char)

        elif char == '}' or char == ']' or char == ')':

            if stack1.is_empty():
                return False

            val = stack1.pop()

            if (char == ')' and val != '('):
                return False

            elif char == '}' and val != '{':
                return False

            elif char == ']' and val != '[':
                return False


    return stack1.is_empty()