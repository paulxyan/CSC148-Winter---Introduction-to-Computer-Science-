from __future__ import annotations
from typing import Optional, Any

class BinarySearchTree:
    """Binary Search Tree class.

    This class represents a binary tree satisfying the Binary Search Tree
    property: for every node, its value is >= all items stored in its left
    subtree, and <= all items stored in its right subtree.

    # === Representation Invariants ===
    # - If _root is None, then so are _left and _right.
    #   This represents an empty BST.
    # - If _root is not None, then _left and _right are BinarySearchTrees
    # - (BST Property) All items in _left are <= _root,
    #   and all items in _right are >= root.
    """
    # === Private Attributes ===
    # The item stored at the root of the tree, or None if the tree is empty.
    _root: Optional[Any]
    # The left subtree, or None if the tree is empty.
    _left: Optional[BinarySearchTree]
    # The right subtree, or None if the tree is empty.
    _right: Optional[BinarySearchTree]

    def __init__(self, root: Optional[Any]) -> None:
        """Initialize a new BST containing only the given root value.

        If <root> is None, initialize an empty BST.
        """
        if root is None:
            self._root = None
            self._left = None
            self._right = None
        else:
            self._root = root
            self._left = BinarySearchTree(None)  # self._left is an empty BST
            self._right = BinarySearchTree(None)  # self._right is an empty BST

    def is_empty(self) -> bool:
        """Return whether this BST is empty.
        """
        return self._root is None

    def __contains__(self, item: Any) -> bool:
        """Return whether <item> is in this BST.
        """

        if self.is_empty():
            return False
        else:

            if item == self._root:
                return True

            elif item < self._root:
                return item in self._left

            else:
                return item in self._right

    def items(self) -> list[int]:
        """Return all the items in a BST"""

        if self.is_empty():
            return []

        else:

            return self._left.items() + [self._root] + self._right.items()

    def delete(self, item: Any) -> None:
        """Remove *one* ocurrence of <item> from this BST.

        Do nothing if <item> is not in the BST
        """

        if self.is_empty():

            pass

        elif self._root == item:

            self.delete_root()

        elif item < self._root:
            self._left.delete(item)

        else:

            self._right.delete(item)

    def delete_root(self) -> None:

        """Remove the root of this tree

        Precondition: this tree is *non-empty*.
        """

        if self._left.is_empty() and self._right.is_empty():

            self._root = None
            self._left = None
            self._right = None

        elif self._left.is_empty():

            self._root, self._left, self._right = \
                self._right._root, self._right._left, self._right._right

        elif self._right.is_empty():
            self._root, self._left, self._right = \
                self._left._root, self._left._left, self._left._right

        else:

            self._root = self._left.extract_max()

    def extract_max(self) -> Any:
        """Remove and return the maximum item stored in this tree.

        Precondition: this tree is *non-empty*
        """
        if self._right.is_empty():

            max_item = self._root
            self._root, self._left, self._right = \
                self._left._root, self._left._left, self._left._right

            return max_item

        else:

            return self._right.extract_max()

