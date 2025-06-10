"""Lab 5: Linked List Exercises

=== CSC148 Winter 2025 ===
Department of Mathematical and Computational Sciences,
University of Toronto Mississauga

=== Module Description ===
This module contains the code for a linked list implementation with two classes,
LinkedList and _Node.

All of the code from lecture is here, as well as some exercises to work on.
"""
from __future__ import annotations
from typing import Any, Optional, Set


class _Node:
    """A node in a linked list.

    Note that this is considered a "private class", one which is only meant
    to be used in this module by the LinkedList class, but not by client code.

    === Attributes ===
    item:
        The data stored in this node.
    next:
        The next node in the list, or None if there are no more nodes.
    """
    item: Any
    next: Optional[_Node]

    def __init__(self, item: Any) -> None:
        """Initialize a new node storing <item>, with no next node.
        """
        self.item = item
        self.next = None  # Initially pointing to nothing


class LinkedList:
    """A linked list implementation of the List ADT.
    """
    # === Private Attributes ===
    # _first:
    #     The first node in the linked list, or None if the list is empty.
    _first: Optional[_Node]

    def __init__(self, items: list) -> None:
        """Initialize a new empty linked list containing the given items.
        """
        self._first = None

        curr = self._first

        for item in items:

            if self._first is None:

                new_node = _Node(item)
                self._first = new_node
                curr = new_node
            else:
                new_node = _Node(item)
                curr.next = new_node
                curr = new_node



    # ------------------------------------------------------------------------
    # Methods from lecture/readings
    # ------------------------------------------------------------------------
    def is_empty(self) -> bool:
        """Return whether this linked list is empty.

        # >>> LinkedList([]).is_empty()
        # True
        # >>> LinkedList([1, 2, 3]).is_empty()
        # False
        """
        return self._first is None

    def __str__(self) -> str:
        """Return a string representation of this list in the form
        '[item1 -> item2 -> ... -> item-n]'.

        >>> str(LinkedList([1, 2, 3]))
        '[1 -> 2 -> 3]'
        >>> str(LinkedList([]))
        '[]'
        """
        items = []
        curr = self._first
        while curr is not None:
            items.append(str(curr.item))
            curr = curr.next
        return '[' + ' -> '.join(items) + ']'

    def __getitem__(self, index: int) -> Any:
        """Return the item at position <index> in this list.

        Raise IndexError if <index> is >= the length of this list.
        """
        curr = self._first
        curr_index = 0

        while curr is not None and curr_index < index:
            curr = curr.next
            curr_index += 1

        assert curr is None or curr_index == index

        if curr is None:
            raise IndexError
        else:
            return curr.item

    def insert(self, index: int, item: Any) -> None:
        """Insert the given item at the given index in this list.

        Raise IndexError if index > len(self) or index < 0.
        Note that adding to the end of the list is okay.

        >>> lst = LinkedList([1, 2, 10, 200])
        >>> lst.insert(2, 300)
        >>> str(lst)
        '[1 -> 2 -> 300 -> 10 -> 200]'
        >>> lst.insert(5, -1)
        >>> str(lst)
        '[1 -> 2 -> 300 -> 10 -> 200 -> -1]'
        >>> lst.insert(100, 2)
        Traceback (most recent call last):
        IndexError
        """
        # Create new node containing the item
        new_node = _Node(item)

        if index == 0:
            self._first, new_node.next = new_node, self._first
        else:
            # Iterate to (index-1)-th node.
            curr = self._first
            curr_index = 0
            while curr is not None and curr_index < index - 1:
                curr = curr.next
                curr_index += 1

            if curr is None:
                raise IndexError
            else:
                # Update links to insert new node
                curr.next, new_node.next = new_node, curr.next

    def delete(self, index: int) -> None:
        """Deleting from a linked list at the give index
        Raise index error if index >= len(self) or index < 0

        >>> lst = LinkedList([1, 2, 3, 4, 5])
        >>> lst.delete(4)
        >>> str(lst)
        '[1 -> 2 -> 3 -> 4]'
        >>> lst.delete(0)
        >>> str(lst)
        '[2 -> 3 -> 4]'
        """


        if self._first is None:
            raise IndexError

        elif index == 0:

            self._first = self._first.next


        else:

            curr = self._first
            curr_index = 0
            prev = None

            while curr is not None and curr_index < index:

                prev = curr
                curr = curr.next
                curr_index += 1

            if curr is None:
                raise IndexError

            else:

                prev.next = curr.next

    def negative_index_LL(self, index: int) -> Any:
        """Returns item at <index> using negative indexing
        Given a negative index, returns the value at that index in the LinkedList

        Precondition: -1 >= index

        """

        if self._first is None:
            return -1
        fast = self._first
        slow = None

        for _ in range(abs(index)-1):

            try:
                fast = fast.next

            except AttributeError:

                return -1

        while fast is not None:

            if slow is None:
                slow = self._first
            else:
                slow = slow.next
            fast = fast.next

        return slow.item





    # ------------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of elements in this list.

    # Lab Task 1
    # ------------------------------------------------------------------------
    # TODO: implement this method
        # >>> lst = LinkedList([])
        # >>> len(lst)              # Equivalent to lst.__len__()
        # 0
        # >>> lst = LinkedList([1, 2, 3])
        # >>> len(lst)
        # 3
        """

        length = 0
        curr = self._first

        while curr is not None:

            length += 1
            curr = curr.next

        return length

    # TODO: implement this method
    def count(self, item: Any) -> int:
        """Return the number of times <item> occurs in this list.

        Use == to compare items.

        # >>> lst = LinkedList([1, 2, 1, 3, 2, 1])
        # >>> lst.count(1)
        # 3
        # >>> lst.count(2)
        # 2
        # >>> lst.count(3)
        # 1
        """

        occur = 0

        curr = self._first

        while curr is not None:

            if curr.item == item:
                occur += 1

            curr = curr.next

        return occur

    # TODO: implement this method
    def index(self, item: Any) -> int:
        """Return the index of the first occurrence of <item> in this list.

        Raise ValueError if the <item> is not present.

        Use == to compare items.

        # >>> lst = LinkedList([1, 2, 1, 3, 2, 1])
        # >>> lst.index(1)
        # 0
        # >>> lst.index(3)
        # 3
        # >>> lst.index(148)
        # Traceback (most recent call last):
        # ValueError
        """


        curr = self._first
        i = 0

        while curr is not None:

            if curr.item == item:
                return i

            i += 1
            curr = curr.next

        raise ValueError

    # TODO: implement this method
    def __setitem__(self, index: int, item: Any) -> None:
        """Store item at position <index> in this list.

        Raise IndexError if index >= len(self).

        # >>> lst = LinkedList([1, 2, 3])
        # >>> lst[0] = 100  # Equivalent to lst.__setitem__(0, 100)
        # >>> lst[1] = 200
        # >>> lst[2] = 300
        # >>> str(lst)
        # '[100 -> 200 -> 300]'
        """

        if index  >= len(self):

            raise IndexError

        else:

            curr = self._first

            i = 0

            while curr is not None and i < index:

                curr += curr.next
                i += 1

            if curr is None:
                raise IndexError

            else:

                curr.item = item

    def reverse(self) -> None:

        """
        Reverses the order of the elements in the LinkedList

        """


        curr = self._first
        tmp = None

        while curr is not None:

            if curr.next is None:
                self._first = curr

            tracker = curr.next
            curr.next, tmp = tmp, curr
            curr = tracker


    def find_center(self) -> Any:
        """
        Finds the center lilypad of the LinkedList chain

        Returns:
            The value of the center lilypad

        Precondition:
            The list is non-empty
        """

        slow = self._first
        fast = self._first

        while fast and fast.next:


            if fast.next is None or fast.next.next is None:

                return slow.item

            fast = fast.next.next

            slow = slow.next



        if fast is not None:
            return slow.item

        else:
            return -1

    def reverse_nodes(self, i: int) -> None:
        """Reverse the nodes at index i and i + 1 by changing their next references
        (not by changing their items).
        Precondition: Both i and i + 1 are valid indexes in the list.
        >>> lst = LinkedList([5, 10, 15, 20, 25, 30])
        >>> print(lst)
        [5 -> 10 -> 15 -> 20 -> 25 -> 30]
        >>> lst.reverse_nodes(1)
        >>> print(lst)
        [5 -> 15 -> 10 -> 20 -> 25 -> 30]
        >>> lst = LinkedList([5, 10, 15, 20, 25, 30])
        >>> lst.reverse_nodes(0)
        >>> print(lst)
        [10 -> 5 -> 15 -> 20 -> 25 -> 30]
        >>> lst = LinkedList([5, 10, 15, 20, 25, 30])
        >>> lst.reverse_nodes(4)
        >>> print(lst)
        [5 -> 10 -> 15 -> 20 -> 30 -> 25]
        """

        if i == 0:
            temp = self._first
            self._first = temp.next
            temp.next = self._first
            self._first.next = temp

        else:

            curr = self._first

            for unused_ in range(i-1):
                curr = curr.next

            temp = curr.next
            curr.next = curr.next.next
            temp.next = curr.next.next
            curr.next.next = temp

class CircularLinkedList(LinkedList):
    head: _Node

    def __init__(self):
        self.head = None

    def is_circular(self) -> bool:
        """


        """
        head = self.head
        all_nodes = set()
        while head:

            if head.next is self.head:
                return True

            if head in all_nodes:
                return False

            all_nodes.add(head)
            head = head.next

        return False

    def __str__(self) -> str:
        """Return a string representation of this list in the form
        '[item1 -> item2 -> ... -> item-n]'.

        >>> str(LinkedList([1, 2, 3]))
        '[1 -> 2 -> 3]'
        >>> str(LinkedList([]))
        '[]'
        """
        items = []
        curr = self.head
        while curr.next is not self.head:
            items.append(str(curr.item))
            curr = curr.next

        items.append(str(curr.item))
        return '[' + ' -> '.join(items) + ']'

class _DLLNode:
    item: Any
    next: _DLLNode
    prev: _DLLNode

    def __init__(self, value: Any):
        self.item, self.next, self.prev = value, None, None

class DoublyLinkedList(LinkedList):
    _first: _DLLNode
    tail: _DLLNode

    def __init__(self) -> None:
        self._first, self.tail = None, None

    def reverse(self) -> None:

        """
        Reverses the order of the elements in a DoublyLinkedList

        """

        curr = self._first
        tmp = None

        while curr is not None:

            tmp = curr.next

            curr.next, curr.prev = curr.prev, curr.next

            if tmp is None:

                self._first = curr

            curr = tmp







if __name__ == '__main__':
    # import python_ta
    # python_ta.check_all()
    import doctest
    doctest.testmod()

    lst1 = LinkedList([1, 2, 3, 4, 5])

    print(lst1)
    print(lst1.find_center())