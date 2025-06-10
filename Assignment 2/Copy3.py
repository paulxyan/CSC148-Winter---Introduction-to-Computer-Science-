"""
Assignment 2 starter code
CSC148, Winter 2025
Instructors: Bogdan Simion, Rutwa Engineer, Marc De Benedetti, Romina Piunno

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time
from typing import Optional

from huffman import HuffmanTree
from utils import *


def _merge_symbol_and_tree(pair2: tuple[int, int], trees2: list[HuffmanTree], _update_merge: callable) -> Optional[
    HuffmanTree]:
    """ Merge a pair consisting of a symbol and a tree, update the lists, and return the new HuffmanTree. """

    # Iterate through the trees to find which one matches the symbol in the pair
    for t in trees2:
        if pair2[0] == t.symbol:
            trees2.remove(t)  # Remove the tree that we are merging with
            if pair2[1] > t.symbol:
                current2 = HuffmanTree(pair2[0] + pair2[1], t, HuffmanTree(pair2[1]))  # Create a new tree
            else:
                current2 = HuffmanTree(pair2[0] + pair2[1], HuffmanTree(pair2[1]), t)  # Create a new tree

            # Update the merge information
            return current2  # Break the loop and return as soon as we find a match

        elif pair2[1] == t.symbol:
            trees2.remove(t)  # Remove the tree that we are merging with
            if pair2[0] <= t.symbol:
                current2 = HuffmanTree(pair2[0] + t.symbol, HuffmanTree(pair2[0]), t)  # Create a new tree
            else:
                current2 = HuffmanTree(pair2[0] + t.symbol, t, HuffmanTree(pair2[0]))  # Create a new tree
            # Update the merge information
            return current2  # Break the loop and return as soon as we find a match

def _update_merge(a: int, b: int, new_tree: HuffmanTree, trees: list, orig: list, values: list) -> None:
    """Inline helper to update our tracking lists after merging."""
    values.remove(a)
    values.remove(b)
    if a in orig:
        orig.remove(a)
    if b in _check_duplicate(a, b, orig):
        orig.remove(b)
    values.append(a + b)
    trees.append(new_tree)

    return

def _check_duplicate(a: int, b: int, to_check: list) -> list:
    """Returns the list with the values of <a> removed from original to check"""
    copy = to_check.copy()
    if a in copy: copy.remove(a)
    return copy



def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True

    >>> h = build_frequency_dict(bytes("hello", "ascii"))
    >>> h == {104: 1, 101: 1, 108: 2, 111: 1}
    True
    >>> tree = build_frequency_dict(b"helloworld")
    >>> d = {104: 1, 101: 1, 108: 3, 111: 2, 119: 1, 114: 1, 100: 1}
    >>> tree == d
    True
    """

    frequency_table = {}

    for byte in text:

        frequency_table.setdefault(byte, 0)
        frequency_table[byte] += 1

    return frequency_table


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    tree = build_huffman_tree({104: 1, 101: 1, 108: 3, 111: 2, 119: 1, 114: 1, 100: 1})
    test_tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(111), HuffmanTree(None, HuffmanTree(104), HuffmanTree(101))), HuffmanTree(None, HuffmanTree(108), HuffmanTree(None, HuffmanTree(119), HuffmanTree(None, HuffmanTree(114), HuffmanTree(100)))))
    tree == test_tree
    True
    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if not freq_dict:
        return HuffmanTree()

    # Special case: only one symbol.
    if len(freq_dict) == 1:
        symbol, _ = next(iter(freq_dict.items()))
        return HuffmanTree(None, HuffmanTree(symbol), HuffmanTree((symbol + 1) % 256))

    # Initialize lists to track available frequencies and already built trees.
    values = list(freq_dict.values())
    orig = values.copy()  # frequencies from original symbols
    trees = []  # holds merged tree nodes


    # Start by merging the two smallest values.
    sorted_vals = sorted(values)
    pair = (sorted_vals[0], sorted_vals[1])
    current = HuffmanTree(pair[0] + pair[1], HuffmanTree(pair[0]), HuffmanTree(pair[1]))
    _update_merge(pair[0], pair[1], current, trees, orig, values)


    while len(values) >= 2:
        sorted_vals = sorted(values)
        pair = (sorted_vals[0], sorted_vals[1])

        # If both minimums come from original symbols, merge them.
        if pair[0] in orig and pair[1] in _check_duplicate(pair[0], pair[1], orig):
            current = HuffmanTree(pair[0] + pair[1], HuffmanTree(pair[0]), HuffmanTree(pair[1]))
            _update_merge(pair[0], pair[1], current, trees, orig, values)
            continue

        # If both come from merged trees (and not from orig), take the earliest two trees.
        tree_vals = [t.symbol for t in trees]

        if (pair[0] in tree_vals and pair[1] in _check_duplicate(pair[0], pair[1], tree_vals)and
                pair[0] not in orig and pair[1] not in orig and len(trees) >= 2):
            candidates = [t for t in trees if t.symbol == pair[0] or t.symbol == pair[1]]
            t1, t2 = candidates[0], candidates[1]
            if t1.symbol < t2.symbol:
                current = HuffmanTree(pair[0] + pair[1], t1, t2)
            else:
                current = HuffmanTree(pair[0] + pair[1], t2, t1)
            trees = [t for t in trees if t is not t1 and t is not t2]
            _update_merge(pair[0], pair[1], current, trees, orig, values)


        else:
            # Otherwise, one value is from an original symbol and the other from a tree.
            current = _merge_symbol_and_tree(pair, trees, _update_merge)
            _update_merge(pair[0], pair[1], current, trees, orig, values)


    # Inline version of _fix_symbols to reassign correct symbols to the leaves.
    def _fix_symbols(tree: HuffmanTree, mapping: dict[int, int]) -> None:
        if tree.is_leaf():
            for key in list(mapping.keys()):
                if tree.symbol == mapping[key]:
                    tree.symbol = key
                    mapping.pop(key)
                    break
        else:
            tree.symbol = None
            _fix_symbols(tree.left, mapping)
            _fix_symbols(tree.right, mapping)

    _fix_symbols(current, freq_dict.copy())
    return current


def _find_lowest_two(table: list[int]) -> Optional[tuple[int, int]]:

    """Return the lowest two values in <table>, returning tuple with the smaller value in 0th index,
    returns None if len(table) < 2
    """

    if len(table) < 2:

        return None

    sorted_items = sorted(table)

    return sorted_items[0], sorted_items[1]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    possible_paths = _all_binary_paths(tree, [])

    all_symbols = []

    for path in possible_paths:

        symbol = _get_symbol_from_path(tree, path)
        all_symbols.append(symbol)

    result_dict = dict(zip(all_symbols, possible_paths))

    return result_dict


def _get_symbol_from_path(tree: HuffmanTree, path: str) -> int:

    """Returns the corresponding symbol at the end of the path in a huffman tree,
    eventually check for the edge case where the path doesn't exist

    >>> tree2 = HuffmanTree(None, HuffmanTree(2), \
    HuffmanTree(None, HuffmanTree(7), HuffmanTree(20)))

    >>> d = '10'
    >>> e = _get_symbol_from_path(tree2, d)
    >>> e == 7
    True

    """

    if tree.is_leaf():

        if path == '':

            return tree.symbol

        else:

            raise ValueError(f"Invalid path '{path}' for a leaf node.")

    if not path:

        raise ValueError("Path to short to reach a leaf node.")

    if path[0] == '0':

        return _get_symbol_from_path(tree.left, path[1:])

    elif path[0] == '1':

        return _get_symbol_from_path(tree.right, path[1:])

    else:

        raise ValueError("Invalid character in path")


def _all_binary_paths(tree: HuffmanTree, container: list) -> list[str]:
    """Return all the paths represented as binary in <tree>

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = _all_binary_paths(tree, [])
    >>> d == ['0', '1']
    True

    >>> tree = HuffmanTree(None, HuffmanTree(2), HuffmanTree(None, HuffmanTree(7), HuffmanTree(20)))
    >>> h = _all_binary_paths(tree, [])
    >>> h == ['0', '10', '11']
    True

    >>> tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(3), HuffmanTree(4)), HuffmanTree(None, HuffmanTree(7), HuffmanTree(20)))
    >>> g = _all_binary_paths(tree, [])
    >>> g == ['00', '01', '10', '11']
    True
    """

    if tree.is_leaf():

        return container

    else:

        if not container:

            potential_paths1 = _all_binary_paths(tree.left, ['0'])
            potential_paths2 = _all_binary_paths(tree.right, ['1'])

        else:

            potential_paths1 = _all_binary_paths(tree.left, [path + '0' for path in container])
            potential_paths2 = _all_binary_paths(tree.right, [path + '1' for path in container])

        return potential_paths1 + potential_paths2


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """

    total = _count_internal(tree)

    temp = 0

    if total == 0: return

    postorder_trees = _trees_in_postorder(tree)

    for tree in postorder_trees:

        tree.number = temp

        temp += 1


def _trees_in_postorder(tree: HuffmanTree) -> list[HuffmanTree]:
    """returns lists of internal nodes in <tree> in Postorder Traversal"""

    if tree.is_leaf(): return []

    else:

        first = _trees_in_postorder(tree.left)
        second = _trees_in_postorder(tree.right)

        return first + second + [tree]


def _count_internal(tree: HuffmanTree) -> int:

    """Return the amount of internal nodes in <tree>
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> _count_internal(tree) == 3
    True
    """

    if tree.is_leaf(): return 0

    else:

        total = 1

        total += _count_internal(tree.left)
        total += _count_internal(tree.right)

        return total


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """

    if freq_dict == {}: return 0.0

    else:

        denominator = sum(list(freq_dict.values()))

        if denominator == 0.0:

            return 0.0

        codes = get_codes(tree)

        sorted_codes = sorted(list(codes.items()), key=lambda x: x[0])

        sorted_freq = sorted(list(freq_dict.items()), key=lambda x: x[0])

        numerator_list = [(len(sorted_codes[i][1]) * sorted_freq[i][1]) for i in range(0, len(sorted_freq))]

        return (sum(numerator_list)) / denominator


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """

    string_bytes = []

    temp = ''

    for integer in text:

        if integer not in codes:

            raise ValueError(f"Symbol {integer} not in codes.")

        else:

            symbol_code = codes[integer]

            temp += symbol_code

            if len(temp) > 8:

                string_bytes.append(temp[0:8])
                temp = temp[8:]

            elif len(temp) == 8:

                string_bytes.append(temp)
                temp = ''

    while len(temp) > 8:
        string_bytes.append(temp[0:8])
        temp = temp[8:]

    if temp != '' and len(temp) < 8: string_bytes.append(temp.ljust(8, '0'))

    new_list = [bits_to_byte(x) for x in string_bytes]

    return bytes(new_list)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108, 1, 3, 1, 2, 1, 4]
    """

    if tree.is_leaf():

        return bytes()

    bytes_list = []

    bytes_list.extend(list(tree_to_bytes(tree.left)))
    bytes_list.extend(list(tree_to_bytes(tree.right)))

    if tree.left.is_leaf():

        bytes_list.append(0)
        bytes_list.append(tree.left.symbol)

    else:

        bytes_list.append(1)
        bytes_list.append(tree.left.number)

    if tree.right.is_leaf():

        bytes_list.append(0)
        bytes_list.append(tree.right.symbol)

    else:

        bytes_list.append(1)
        bytes_list.append(tree.right.number)

    return bytes(bytes_list)

def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    # Note:the number of each node in the list corresponds to the index of the node in the list.
    # That is, the node at index 0 in the list has node-number 0,
    # the node at index 1 in the list has node-number 1, etc.

    #Base case

    root_node = node_lst[root_index]

    if root_node.l_type == 0 and root_node.r_type  == 0:

        return HuffmanTree(None, HuffmanTree(root_node.l_data), HuffmanTree(root_node.r_data))

    elif root_node.l_type == 1 and root_node.r_type == 0:

        return HuffmanTree(None, generate_tree_general(node_lst, root_node.l_data), HuffmanTree(root_node.r_data))

    elif root_node.l_type == 0 and root_node.r_type == 1:

        return HuffmanTree(None, HuffmanTree(root_node.l_data), generate_tree_general(node_lst, root_node.r_data))

    else:

        return HuffmanTree(None, generate_tree_general(node_lst, root_node.l_data), generate_tree_general(node_lst, root_node.r_data))



def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """

    root_node = node_lst[root_index]

    if root_node.l_type == 0 and root_node.r_type == 0:
        return HuffmanTree(None, HuffmanTree(root_node.l_data), HuffmanTree(root_node.r_data))

    elif root_node.l_type == 1 and root_node.r_type == 0:

        return HuffmanTree(None, generate_tree_postorder(node_lst, root_index - 1), HuffmanTree(root_node.r_data))

    elif root_node.l_type == 0 and root_node.r_type == 1:

        return HuffmanTree(None, HuffmanTree(root_node.l_data), generate_tree_postorder(node_lst, root_index - 1))

    else:

        return HuffmanTree(None, generate_tree_postorder(node_lst, root_index - 2), generate_tree_postorder(node_lst, root_index - 1))


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """


    all_bits = ''.join([byte_to_bits(byte) for byte in text])

    curr_node = tree

    i = 0

    total_bytes = []

    while len(total_bytes) < size:

        if all_bits[i] == '0': curr_node = curr_node.left

        elif all_bits[i] == '1': curr_node = curr_node.right

        if curr_node.is_leaf():

            total_bytes.append(curr_node.symbol)
            curr_node = tree

        i += 1

    return bytes(total_bytes)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

        We say that the Huffman tree is suboptimal for the given frequencies if it is
     possible to swap a higher-frequency symbol lower in the tree with a
     lower-frequency symbol higher in the tree.

        you should solve this function only by swapping symbols via valid swaps (as described above),
     and not doing anything else like building another huffman tree and copying over symbols.


    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """

    # Step 1, get a sorted dict from highest freq to lowest:

    # dict_copy = freq_dict.copy()

    #  = sorted(list(dict_copy.items()), key=lambda x: x[1], reverse=True)


    sorted_tuple = sorted(list(freq_dict.items()), key=lambda x:x[1], reverse=False)
    all_items = [x[0] for x in sorted_tuple]


    all_levels = _get_level_order_leaves(tree)

    for level in sorted(all_levels.keys()):

        all_levels[level].reverse()
        levels_list = all_levels[level]
        current_nodes = levels_list

        for node in current_nodes:

            if all_items:

                node.symbol = all_items.pop()

            else:

                raise IndexError

def _get_level_order_leaves(tree: HuffmanTree) -> dict[int, list[HuffmanTree]]:
    """Return the nodes in <tree> HuffmanTree by level order"""

    """Scan the Huffman tree in level order while keeping track of levels.

        Returns a dictionary where keys are levels, and values are lists of node symbols.
    """

    if tree is None:

        return {}

    queue = [(tree, 0)]  # Queue stores (node, level)
    level_dict = {}  # Dictionary to store nodes by level

    while queue:
        node, level = queue.pop(0)  # Dequeue a node and its level

        # Store node's symbol at the corresponding level
        if node.is_leaf():
            if level in level_dict:
                level_dict[level].append(node)
            else:
                level_dict[level] = [node]

        # Enqueue children with the next level
        if node.left:
            queue.append((node.left, level + 1))
        if node.right:
            queue.append((node.right, level + 1))

    return level_dict


if __name__ == "__main__":


    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
