from typing import Any

from LinkedListPractise import LinkedList


class Stack:

    """Implementation Stack ADT LIFO approach using a linked list to
    optimize for efficiency; front of the list is now the top of the stack

    """

    # Private Attributes
    # _items: items contained in the stack

    _items: LinkedList

    def __init__(self):
        self._items = LinkedList([])

    def is_empty(self) -> bool:

        return self._items.is_empty()

    def pop(self) -> Any:
        """Remove and return the element at the top of this stack.

        Raise an EmptyStackError if this stack is empty.

        >>> s = Stack()
        >>> s.push('hello')
        >>> s.push('goodbye')
        >>> s.pop()
        'goodbye'
        """

        if self.is_empty():

            return None

        else:

            item = self._items[0]
            self._items.delete(0)
            return item#Contant O(1) runtime

    def push(self, item: Any) -> None:

        """Add a new element to the top of the stack"""
        self._items.insert(0, item) #Runtime efficiency: O(1)


class Queue:

    """Implementation of Queue ADT using linked list implementation
    back of linked list represents top of the queue


    """


    def __init__(self) -> None:

        self._items = LinkedList([])

    def is_empty(self) -> bool:

        return self._items.is_empty()

    def enqueue(self, item: Any) -> None:

        self._items.insert(len(self._items), item)

    def dequeue(self) -> Any:


        if self.is_empty():
            return None

        else:

            val = self._items[0]
            self._items.delete(0)

            return val


