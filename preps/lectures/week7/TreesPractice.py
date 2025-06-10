from __future__ import annotations

from collections import deque
from typing import*

class Tree:

    """
    A recursive tree data structure
    """

    # === Private Attributes ===
    # The item stored at this tree's root, or None if the tree is empty.
    _root: Optional[Any]
    # The list of all subtrees of this tree.
    _subtrees: list[Tree]

    # === Representation Invariants ===
    # - If self._root is None then self._subtrees is an empty list.
    #   This setting of attributes represents an empty tree.
    #
    #   Note: self._subtrees may be empty when self._root is not None.
    #   This setting of attributes represents a tree consisting of just one
    #   node.

    def __init__(self, root: Optional[Any], subtrees: list[Tree]) -> None:
        """Initialize a new tree with the given root value and subtrees.

        If <root> is None, this tree is empty.
        Precondition: if <root> is None, then <subtrees> is empty.
        """
        self._root = root
        self._subtrees = subtrees

    def is_empty(self) -> bool:
        """Return whether this tree is empty.

        >>> t1 = Tree(None, [])
        >>> t1.is_empty()
        True
        >>> t2 = Tree(3, [])
        >>> t2.is_empty()
        False
        """
        return self._root is None

    def __len__(self) -> int:
        """Return the number of items contained in this tree.

        >>> t1 = Tree(None, [])
        >>> len(t1)
        0
        >>> t2 = Tree(3, [Tree(4, []), Tree(1, [])])
        >>> len(t2)
        3
        """
        if self.is_empty():
            return 0
        else:
            size = 1  # count the root
            for subtree in self._subtrees:
                size += subtree.__len__()  # could also do len(subtree) here
            return size


    def _str_indented(self, depth: int=0) -> str:

        """Return an indented string represntation of tree"""

        if self.is_empty():
            return ''

        else:

            s = ' ' * depth  + str(self._root) + '\n'
            for subtree in self._subtrees:

                s += subtree._str_indented(depth + 1)

            return s

    def __str__(self) -> str:

        """Return a string representation of this tree.

        """

        return self._str_indented() #Have not changed API of str method

    def _delete_root(self):
        """Remove the root of this tree"""

        if self._subtrees == []:

            self._root = None

        else:

            chosen_one = self._subtrees.pop()

            self._root = chosen_one._root

            self._subtrees.extend(chosen_one._subtrees)

    def _delete_root2(self):

        """to delete the root, find the leftmost leaf of the
        tree, and move that leaf so that it becomes the new root value. """

        if self._subtrees == []:

            self._root = None

        else:

            val = self._subtrees[0]._replace_left()
            if self._subtrees[0].is_empty(): self._subtrees.pop(0)

            self._root = val

    def _replace_left(self) -> int:

        """Return the left-most child value in this tree"""

        if self._subtrees == []:

            val1 = self._root

            self._root = None

            return val1

        else:

            val = self._subtrees[0]._replace_left()

            if self._subtrees[0].is_empty():
                self._subtrees.pop(0)

            return val




    def delete_item(self, item: Any) -> bool:

        """Delete *one* occurrence of <item> from this tree.

            Return True if <item> was deleted, and False otherwise.
            Do not modify this tree if it does not contain <item>.
        """

        if self.is_empty():

            return False

        elif self._root == item:

            self._delete_root2()

            return True

        else:

            for subtree in self._subtrees:

                deleted = subtree.delete_item(item)

                if deleted and subtree.is_empty(): #Remember that the tree structure is recursively defined

                    self._subtrees.remove(subtree)

                    return True

                elif deleted:

                    return True


            return False

    def leaves(self) -> list:

        """Return a list of all of the leaf items in the tree

        >>> t = Tree(10, [Tree(1, []), Tree(2, []), Tree(3, [])])
        >>>
        """

        if self.is_empty():

            return []

        elif self._subtrees == []:

            return [self._root]

        else:

            total = []

            for subtree in self._subtrees:

                total.extend(subtree.leaves())

            return total

    def average(self) -> float:
        """Return the average of all the values in this tree.

        Return 0.0 if this tree is empty.

        Precondition: this is a tree of numbers
        """

        if self.is_empty():

            return 0.0

        sum_, nodes = self.get_add_info()

        return sum_ / nodes


    def get_add_info(self) -> tuple[float, int]:


        #Base case is when tree has no leaves

        sum_ = self._root
        nodes = 1

        for subtree in self._subtrees:

            curr = subtree.get_add_info()
            sum_ += curr[0]
            nodes += curr[1]

        return sum_, nodes

    def delete_item_all(self, item):

        """Deletes all instances of <item>."""

        if self.is_empty():

            return

        else:

            for subtree in self._subtrees:

                subtree.delete_item_all()

            if self._root == item:

                self._delete_root()

    def _sum(self) -> int:
        """Return the sum of al values in the tree"""

        if self.is_empty():

            return 0

        else:

            temp = self._root

            for subtree in self._subtrees:

                temp += subtree._sum()

            return temp

    def _check_uniqueness(self, table: set) -> bool:

        """check if every item in the tree is unique

        >>> t1 = Tree(3, [Tree(4, []), Tree(6, [])])
        >>> t1._check_uniqueness(set())
        True
        >>> t2 = Tree(4, [Tree(4, []), Tree(6, [])])
        >>> t2._check_uniqueness(set())
        False

        """

        if self.is_empty():

            return False

        elif self._subtrees == []:

            table.add(self._root)

            return True

        else:


            for subtree in self._subtrees:

                if not (subtree._check_uniqueness(table)):

                    return False

            if self._root in table:

                return False

            table.add(self._root)

            return True

    def is_happy_tree(self) -> bool:
        """
        Returns whether or not the tree is a Happy Little Tree.
        >>> t = Tree(3, [Tree(4, []), Tree(2, [])])
        >>> t.is_happy_tree()
        True
        """

        if self.is_empty():

            return False

        else:

            total = self._sum()

            unique = self._check_uniqueness(set())
            if not (len(self._subtrees) % 2 == 0):

                return False

            if not(total % 2 == 1):

                return False

            if not unique:

                return False

            return True



    def preorder_visit(self, act: Callable[[Tree], Any]) -> None:
        """ Visit each node of this Tree in preorder, and perform an action
        on the nodes as they are visited, using the function <act>.
        >>> lt = Tree(2, [Tree(4, []), Tree(5, [])])
        >>> rt = Tree(13, [Tree(16, []), Tree(17, [])])
        >>> t = Tree(10, [lt, rt])
        >>> def act(tree): print(tree._root)
        >>> t.preorder_visit(act)
        10
        2
        4
        5
        13
        16
        17
        """

        if self.is_empty():

            return

        else:

            act(self)

            for subtree in self._subtrees:

                subtree.preorder_visit(act)

    def postorder_visit(self, act: Callable[[Tree], Any]) -> None:
        """ Visit each node of this Tree in postorder, and perform an action
        on the nodes as they are visited, using the function <act>.
        >>> lt = Tree(2, [Tree(4, []), Tree(5, [])])
        >>> rt = Tree(13, [Tree(16, []), Tree(17, [])])
        >>> t = Tree(10, [lt, rt])
        >>> def act(tree): print(tree._root)
        >>> t.postorder_visit(act)
        4
        5
        2
        16
        17
        13
        10
        """

        if self.is_empty():

            return

        else:

            for subtree in self._subtrees:

                subtree.postorder_visit(act)

            act(self)

    def levelorder_visit(self, act: Callable[[Tree], Any]) -> None:
        """ Visit each node of this Tree in level order, and perform an action
        on the nodes as they are visited, using the function <act>.
        >>> lt = Tree(2, [Tree(4, []), Tree(5, [])])
        >>> rt = Tree(13, [Tree(16, []), Tree(17, [])])
        >>> t = Tree(10, [lt, rt])
        >>> def act(tree): print(tree._root)
        >>> t.levelorder_visit(act)
        10
        2
        13
        4
        5
        16
        17
        """

        if self.is_empty():

            return


        act(self)
        queue = deque()

        queue.extend(self._subtrees)

        while not len(queue) == 0:

            curr = queue.popleft()

            act(curr)

            queue.extend(curr._subtrees)




if __name__ == '__main__':



    t4 = Tree(8, [Tree(10, []), Tree(20, [])])
    t5 = Tree(6, [Tree(7, []), t4])
    t6 = Tree(30, [Tree(1, []), Tree(2, []), Tree(3, [])])
    t3 = Tree(5, [Tree(4, [Tree(100, [])]), t5, t6])

    print(t3)

    t3.delete_item(5)

    print(t3)







