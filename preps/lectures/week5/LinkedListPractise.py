#For an Array-based list: inserting and deleting at the front of the
# built-in list takes time proportional to the length of the list,
# because every item in the list needs to be shifted
#By one spot

from __future__ import annotations
from typing import Any, Optional


class _Node:

    """

    === Attributes ===
    item:
        The data stored in this node.

    next:
        The next node in the list

    """

    item: Any
    next: Optional[_Node]

    def __init__(self, item: Any) -> None:

        self.item = item
        self.next = None


class LinkedList:

    _first: Optional[_Node]

    def __init__(self, elem: list) -> None:

        self._first = None

        for _ in elem:

            self.append(_)


#Traversing through a linked list

    def append(self, item: Any) -> None:

        """Add the given ite, to the end of this linked list."""
        curr = self._first

        if curr is None:
            new_node = _Node(item)
            self._first = new_node

        else:
            while curr.next is not None:
                curr = curr.next

            new_node = _Node(item)
            curr.next = new_node

    def insert(self, index: int, item: Any) -> None:
        """Insert a new node containing item at position <index>.

        Precondition: index >= 0.

        Raise IndexError if index > len(self).

        Note: if index == len(self), this method adds the item to the end
        of the linked list, which is the same as LinkedList.append.

        >>> lst = LinkedList([1, 2, 10, 200])
        >>> lst.insert(2, 300)
        >>> str(lst)
        '[1 -> 2 -> 300 -> 10 -> 200]'
        >>> lst.insert(5, -1)
        >>> str(lst)
        '[1 -> 2 -> 300 -> 10 -> 200 -> -1]'
        """


        if index == 0:

            new_node = _Node(item)

            self._first, new_node.next = new_node, self._first

        else:

            curr = self._first
            curr_index = 0


            while curr is not None and curr_index < index - 1:
                curr = curr.next
                curr_index += 1

            if curr is None:

                raise IndexError

            else:

                new_node = _Node(item)

                curr.next, new_node.next = new_node, curr.next

    def __str__(self) -> str:

        """String representation of a linked-list implementation of List ADT

        >>> lst = LinkedList([1, 2, 10, 200])
        >>> str(lst)
        '[1 -> 2 -> 300 -> 10 -> 200]'

        """

        curr = self._first

        if curr is None:

            return "[]"

        else:

            string = "["

            while curr.next is not None:

                string += f"{curr.item} -> "
                curr = curr.next

            string += f"{curr.item}]"

            return string

if __name__ == '__main__':

    lst = LinkedList([1, 2, 10, 200])
    print(lst)
