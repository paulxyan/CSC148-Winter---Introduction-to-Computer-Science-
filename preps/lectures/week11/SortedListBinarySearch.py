"""sorted list class"""
from time import time

class SortedList(list):

    """list in non-decreasing order"""

    def __contains__(self, v: int, start=None, stop=None) -> bool:
        """ Return whether this SortedList contains <v>.
           <start> is an int, the index in the list where we start looking
           <stop> is an int, the index in the list where we stop looking

           >>> lst = SortedList()
           >>> 15 in lst
           False
           >>> lst = SortedList([17])
           >>> 17 in lst
           True
           >>> lst = SortedList([5, 10, 15, 20])
           >>> 15 in lst
           True
           >>> 17 in lst
           False
        """

        if start is None:
            start, stop = 0, len(self) - 1

        if start > stop:
            return False

        else:
            mid = (stop + start) // 2

            if self[mid] == v:
                return True
            if self[mid] < v:
                return self.__contains__(v, mid+1, stop)

            else:
                return self.__contains__(v, start, mid-1)


def timing_tests(n: int) -> None:

    lst = list(range(n))
    start = time()

    found = n // 2 in lst

    print('Searched in Python list L[{:8} elements]: {:8.5f} sec, result={}'.
          format(n, time() - start, found))

    lst = SortedList(lst)
    start = time()
    found = n // 2 in lst
    print('Searched in SortedList L[{:8} elements] : {:8.5f} sec, result={}\n'.format(n, time()-start, found))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    n = 1000000
    for _ in range(6):
        timing_tests(n)
        n *= 2

