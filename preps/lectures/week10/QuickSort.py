from typing import Any


def quicksort(lst: list) -> list:
    """Return a sorted list with the same elements as <lst>.

        This is a *non-mutating* version of quicksort; it does not mutate the
        input list.
    """

    if len(lst) < 2:

        return lst[:]

    else:

        pivot = lst[0]

        smaller, bigger = _partition(lst[1:], pivot)

        smaller_sorted = quicksort(smaller)

        bigger_sorted = quicksort(bigger)

        return smaller_sorted + [pivot] + bigger_sorted

def _partition(lst: list, pivot: Any) -> tuple[list, list]:
    """Return a partition of <lst> with the chosen pivot.

    Return two lists, where the first contains the items in <lst>
    that are <= pivot, and the second is the items in <lst> that are > pivot.
    """

    smaller = []
    bigger = []

    for item in lst:
        if item <= pivot:
            smaller.append(item)
        else:
            bigger.append(item)

    return smaller, bigger