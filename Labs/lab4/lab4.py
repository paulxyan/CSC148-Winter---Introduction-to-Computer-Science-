"""CSC148 Lab 4: Abstract Data Types

=== CSC148 Winter 2025 ===
Department of Mathematical and Computational Sciences,
University of Toronto Mississauga

=== Module Description ===
In this module, you will write two different functions that operate on a Stack.
Pay attention to whether or not the stack should be modified.
"""

from typing import Any, Optional


###############################################################################
# Task 1: Practice with stacks
###############################################################################
class Stack:
    """A last-in-first-out (LIFO) stack of items.

    Stores data in a last-in, first-out order. When removing an item from the
    stack, the most recently-added item is the one that is removed.
    """
    # === Private Attributes ===
    # _items:
    #     The items stored in this stack. The end of the list represents
    #     the top of the stack.
    _items: list

    def __init__(self) -> None:
        """Initialize a new empty stack."""
        self._items = []

    def is_empty(self) -> bool:
        """Return whether this stack contains no items.

        >>> s = Stack()
        >>> s.is_empty()
        True
        >>> s.push('hello')
        >>> s.is_empty()
        False
        """
        return self._items == []

    def push(self, item: Any) -> None:
        """Add a new element to the top of this stack."""
        self._items.append(item)

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
            raise EmptyStackError
        else:
            return self._items.pop()

    def __str__(self) -> str:

        h = [str(items) for items in self._items]
        return ', '.join(h)


class EmptyStackError(Exception):
    """Exception raised when an error occurs."""
    pass


def size(s: Stack) -> int:
    """Return the number of items in s.

    >>> s = Stack()
    >>> size(s)
    0
    >>> s.push('hi')
    >>> s.push('more')
    >>> s.push('stuff')
    >>> size(s)
    3
    """
    side_stack = Stack()
    count = 0
    # Pop everything off <s> and onto <side_stack>, counting as we go.
    while not s.is_empty():
        side_stack.push(s.pop())
        count += 1
    # Now pop everything off <side_stack> and back onto <s>.
    while not side_stack.is_empty():
        s.push(side_stack.pop())
    # <s> is restored to its state at the start of the function call.
    # We consider that it was not mutated.
    return count


# TODO: implement this function!
def remove_big(s: Stack) -> None:
    """Remove the items in <stack> that are greater than 5.

    Do not change the relative order of the other items.

    >>> s = Stack()
    >>> s.push(1)
    >>> s.push(29)
    >>> s.push(8)
    >>> s.push(4)
    >>> remove_big(s)
    >>> s.pop()
    4
    >>> s.pop()
    1
    >>> s.is_empty()
    True
    """

    temp_stack = Stack()
    temp = None

    while not s.is_empty():

        temp = s.pop()
        if temp <= 5:
            temp_stack.push(temp)

    while not temp_stack.is_empty():
        s.push(temp_stack.pop())


# TODO: implement this function!
def double_stack(s: Stack) -> Stack:
    """Return a new stack that contains two copies of every item in <stack>.

    We'll leave it up to you to decide what order to put the copies into in
    the new stack.

    >>> s = Stack()
    >>> s.push(1)
    >>> s.push(29)
    >>> new_stack = double_stack(s)
    >>> s.pop()  # s should be unchanged.
    29
    >>> s.pop()
    1
    >>> s.is_empty()
    True
    >>> new_items = []
    >>> new_items.append(new_stack.pop())
    >>> new_items.append(new_stack.pop())
    >>> new_items.append(new_stack.pop())
    >>> new_items.append(new_stack.pop())
    >>> sorted(new_items)
    [1, 1, 29, 29]

    Note: Cannot modify the original Stack
    """

    temp_stack1 =  Stack()
    double_stack = Stack()
    temp = None

    while not s.is_empty():

        temp = s.pop()
        temp_stack1.push(temp)
        double_stack.push(temp)
        double_stack.push(temp)

    while not temp_stack1.is_empty():
        s.push(temp_stack1.pop())

    return double_stack


# TODO: implement this class! Note that you'll need at least one private
# attribute to store items.
class Queue:
    """A first-in-first-out (FIFO) queue of items.

    Stores data in a first-in, first-out order. When removing an item from the
    queue, the most recently-added item is the one that is removed.
    """
    _items: list
    def __init__(self) -> None:
        """Initialize a new empty queue."""

        self._items = []

    def is_empty(self) -> bool:
        """Return whether this queue contains no items.

        >>> q = Queue()
        >>> q.is_empty()
        True
        >>> q.enqueue('hello')
        >>> q.is_empty()
        False
        """

        return self._items == []

    def enqueue(self, item: Any) -> None:
        """Add <item> to the back of this queue.
        """

        self._items.append(item)

    def dequeue(self) -> Optional[Any]:
        """Remove and return the item at the front of this queue.

        Return None if this Queue is empty.
        (We illustrate a different mechanism for handling an erroneous case.)

        >>> q = Queue()
        >>> q.enqueue('hello')
        >>> q.enqueue('goodbye')
        >>> q.dequeue()
        'hello'
        """

        if not self.is_empty():
            return self._items.pop(0)

        else:

            return None

    def __str__(self) -> str:

        h = [str(items) for items in self._items]
        return ', '.join(h)


def product(integer_queue: Queue) -> int:
    """Return the product of integers in the queue.

    Remove all items from the queue.

    Precondition: integer_queue contains only integers.

    >>> q = Queue()
    >>> q.enqueue(2)
    >>> q.enqueue(4)
    >>> q.enqueue(6)
    >>> product(q)
    48
    >>> q.is_empty()
    True
    """

    total = 1
    temp = None

    while not integer_queue.is_empty():
        temp = integer_queue.dequeue()
        total *= temp

    return total



def product_star(integer_queue: Queue) -> int:
    """Return the product of integers in the queue.

    Precondition: integer_queue contains only integers.

    >>> primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    >>> prime_line = Queue()
    >>> for prime in primes:
    ...     prime_line.enqueue(prime)
    ...
    >>> product_star(prime_line)
    6469693230
    >>> prime_line.is_empty()
    False
    """

    total = 1
    temp = None
    temp_queue = Queue()

    while not integer_queue.is_empty():
        temp = integer_queue.dequeue()
        temp_queue.enqueue(temp)
        total *= temp

    while not temp_queue.is_empty():
        integer_queue.enqueue(temp_queue.dequeue())

    return total

def filter_queue(queue: Queue, condition: callable) -> None:
    """
    Given a Queue and a condition, removes all elements from the Queue that do not satisfy the condition.
    >>> queue = Queue()
    >>> queue.enqueue(1)
    >>> queue.enqueue(2)
    >>> queue.enqueue(3)
    >>> queue.enqueue(4)
    >>> filter_queue(queue, lambda x: x % 2 == 0)
    >>> str(queue)
    '2, 4'
    """

    temp_queue = Queue()
    temp = None
    val = None

    while not queue.is_empty():

        temp = queue.dequeue()

        try:

            val = condition(temp)

        except TypeError:

            return None

        if not isinstance(val, bool):
            return None

        elif val:

            temp_queue.enqueue(temp)

    while not temp_queue.is_empty():
        queue.enqueue(temp_queue.dequeue())

def filter_stack(stack: Stack, condition: callable) -> None:
    """
      Given a Stack and a condition, removes all elements from the Stack that do not satisfy the condition.
      >>> stack = Stack()
      >>> stack.push(1)
      >>> stack.push(2)
      >>> stack.push(3)
      >>> stack.push(4)
      >>> filter_stack(stack, lambda x: x % 2 == 0)
      >>> str(stack)
      '4, 2'
      """

    temp_queue = Queue()
    temp = None
    val = None

    while not stack.is_empty():

        temp = stack.pop()

        try:

            val = condition(temp)

        except TypeError:

            return None

        if not isinstance(val, bool):
            return None

        elif val:

            temp_queue.enqueue(temp)

    while not temp_queue.is_empty():
        stack.push(temp_queue.dequeue())


def rev_queue(queue: Queue) -> None:
    """
    Given a Queue, reverses the order of the elements in the Queue.
    >>> queue = Queue()
    >>> queue.enqueue(1)
    >>> queue.enqueue(2)
    >>> queue.enqueue(3)
    >>> queue.enqueue(4)
    >>> rev_queue(queue)
    >>> str(queue)
    '4, 3, 2, 1'
    """

    temp_stack = Stack()

    while not queue.is_empty():

        temp_stack.push(queue.dequeue())

    while not temp_stack.is_empty():

        queue.enqueue(temp_stack.pop())

def rev_stack(stack: Stack) -> None:

    """
    Given a stack, reverses the order of the elements in the Stack.
    >>> stack = Stack()
    >>> stack.push(1)
    >>> stack.push(2)
    >>> stack.push(3)
    >>> stack.push(4)
    >>> rev_stack(stack)
    >>> str(stack)
    '4, 3, 2, 1'

    """

    temp_queue = Queue()

    while not stack.is_empty():

        temp_queue.enqueue(stack.pop())

    while not temp_queue.is_empty():

        stack.push(temp_queue.dequeue())


def reverse_four(stack: Stack) -> None:

    """
    Given a stack, reverses the last four elements of the stack. If there are less than four elements, the stack should remain unchanged.
    >>> stack = Stack()
    >>> stack.push(1)
    >>> stack.push(2)
    >>> stack.push(3)
    >>> stack.push(4)
    >>> stack.push(5)
    >>> reverse_four(stack)
    >>> str(stack)
    '4, 3, 2, 1, 5'
    """

    temp_stack = Stack()

    items = 0

    temp_queue = Queue()

    while not stack.is_empty():

        temp_queue.enqueue(stack.pop())
        items += 1

    if items < 4:

        while not temp_queue.is_empty():
            temp_stack.push(temp_queue.dequeue())

        while not temp_stack.is_empty():
            stack.push(temp_stack.pop())

    else:

        index = 0

        while index < items - 4:

            temp_stack.push(temp_queue.dequeue())

        while not temp_queue.is_empty():

            stack.push(temp_queue.dequeue())

        while not temp_stack.is_empty():
            stack.push(temp_stack.pop())



if __name__ == '__main__':
    import doctest
    doctest.testmod()

    q = Queue()
    q.enqueue(1)
    q.enqueue(2)

    print(q)