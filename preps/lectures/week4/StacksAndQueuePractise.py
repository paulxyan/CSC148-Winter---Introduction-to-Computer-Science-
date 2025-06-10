from typing import Any


class Stack:

    """Represents the stack ADT; client code is not allowed to access the items besides the top element

    #Private Attributes

    # _items: the items in the stack, represented as a list

    """

    _items: list[Any]

    def __init__(self):

        self._items = []


    def is_empty(self):
        """Check if stack is empty"""

        return self._items == []

    def pop(self) -> Any:

        """Return the most recently added item according to LIFO approach"""


        if not self.is_empty():

            return self._items.pop()

    def push(self, item: Any) -> None:

        """Add an item to the stack implementation of the stack ADT"""


        self._items.append(item)



