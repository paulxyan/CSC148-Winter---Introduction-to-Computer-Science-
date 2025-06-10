from typing import Union, Optional
from time import time

def sum_nested(obj: Union[int, list]) -> int:
    """Return the sum of the numbers in a nested list <obj>"""

    if isinstance(obj, int):
        return obj

    else:

        s = 0
        for sublist in obj:

            s += sum_nested(sublist)

def nested_list_contains(obj: Union[int, list], item: int) -> bool:
    """Return whether the given item appears in <obj>.


    >>> nested_list_contains(10 , 10)
    True

    >>> nested_list_contains([10, [5, [3, 1]], 2], 1)
    True
    """

    if isinstance(obj, int):

        return obj == item

    else:

        for sublist in obj:

            if nested_list_contains(sublist, item):

                return nested_list_contains(sublist, item)

        return False

def first_at_depth(obj: Union[int, list], d: int) -> Optional[int]:
    """Return the first (leftmost) item in <obj> at depth <d>
    >>> first_at_depth(100, 1) is None
    True
    >>> first_at_depth(100, 0)
    100
    >>> first_at_depth([10, [20, 30], 20], 2)
    20
    """

    if isinstance(obj, int):

        return obj if d == 0 else None

    else:

        for sublist in obj:

            if first_at_depth(sublist, d-1):

                return first_at_depth(sublist, d-1)

        return None

def add_one(obj: Union[int, list]) -> None:
    """Add one to every number stored in <obj>. Do nothing if <obj> is an int.
    If <obj> is a list, *mutate* it to change the numbers stored.
    >>> lst0 = 1
    >>> add_one(lst0)
    >>> lst0
    1
    >>> lst1 = []
    >>> add_one(lst1)
    >>> lst1
    []
    >>> lst2 = [1, [2, 3], [[[5]]]]
    >>> add_one(lst2)
    >>> lst2
    [2, [3, 4], [[[6]]]]
    """
    if isinstance(obj, int):
        return

    else:

        for i in range(len(obj)):

            if isinstance(obj[i], int):
                obj[i] += 1

            else:

                add_one(obj[i])

def fib(n: int) -> int:

    """Returns the n-th fibonacci number.
    """

    if n < 2:

        return 1

    else:

        return fib(n-1) + fib(n-2)


def fibonacci(n: int) -> int:
    """ Return the <n>th fibonacci number, that is n if n < 2,
    or fibonacci(n-2) + fibonacci(n-1) otherwise.

    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(3)
    2
    >>> fibonacci(30)
    832040
    """

    if n < 2:
        return n

    else:
        return fibonacci(n-2) + fibonacci(n-1)

def fib_memo(n: int, seen: dict[int, int]) -> int:
    """ Return the <n>th fibonacci number reasonably quickly, using a dictionary
        to remember previously computed fibonacci terms.

        >>> fib_memo(0, {})
        0
        >>> fib_memo(1, {})
        1
        >>> fib_memo(3, {})
        2
        >>> fib_memo(30, {})
        832040
        """

    if n not in seen:

        seen[n] = (n if n < 2 else fib_memo(n-1, seen) + fib_memo(n-2, seen))

    return seen[n]




if __name__ == '__main__':

    import doctest
    doctest.testmod()

    N = 35

    start = time()

    print(fib_memo(N, {}))
    print("Run time 1: {}".format(time() - start))

    start = time()

    print(fibonacci(N))
    print("Classic Fibonacci run time: {}".format(time() - start))