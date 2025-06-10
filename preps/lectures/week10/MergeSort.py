def mergesort(lst: list) -> list:
    """Return a sorted list with the same elements as <lst>.
        This is a *non-mutating* version of mergesort; it does not mutate the
        input list.
    """

    if len(lst) < 2:

        return lst[:]

    else:

        mid = len(lst) // 2
        left_sorted = mergesort(lst[:mid])
        right_sorted = mergesort(lst[mid:])

        return _merge(left_sorted, right_sorted)

def _merge(lst1: list, lst2: list) -> list:
    """Return a sorted list with the elements in <lst1> and <lst2>.

    Precondition: <lst1> and <lst2> are sorted.
    """

    index1 = 0
    index2 = 0
    merged = []

    while index1 < len(lst1) and index2  < len(lst2):

        if lst1[index1] <= lst2[index2]:
            merged.append(lst1[index1])
            index1 += 1
        else:

            merged.append(lst2[index2])
            index2 += 1

    return merged + lst1[index1:] + lst2[index2:]