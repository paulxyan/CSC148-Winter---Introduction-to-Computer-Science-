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

    def update_merge(a: int, b: int, new_tree: HuffmanTree) -> None:
        """Inline helper to update our tracking lists after merging."""
        values.remove(a)
        values.remove(b)
        if a in orig:
            orig.remove(a)
        if b in orig:
            orig.remove(b)
        values.append(a + b)
        trees.append(new_tree)

    # Start by merging the two smallest values.
    sorted_vals = sorted(values)
    pair = (sorted_vals[0], sorted_vals[1])
    current = HuffmanTree(pair[0] + pair[1], HuffmanTree(pair[0]), HuffmanTree(pair[1]))
    update_merge(pair[0], pair[1], current)

    while len(values) >= 2:
        sorted_vals = sorted(values)
        pair = (sorted_vals[0], sorted_vals[1])

        # If both minimums come from original symbols, merge them.
        if pair[0] in orig and pair[1] in orig:
            current = HuffmanTree(pair[0] + pair[1], HuffmanTree(pair[0]), HuffmanTree(pair[1]))
            update_merge(pair[0], pair[1], current)
            continue

        # If both come from merged trees (and not from orig), take the earliest two trees.
        tree_vals = [t.symbol for t in trees]
        if (pair[0] in tree_vals and pair[1] in tree_vals and 
                pair[0] not in orig and pair[1] not in orig and len(trees) >= 2):
            candidates = [t for t in trees if t.symbol == pair[0] or t.symbol == pair[1]]
            t1, t2 = candidates[0], candidates[1]
            if t1.symbol < t2.symbol:
                current = HuffmanTree(pair[0] + pair[1], t1, t2)
            else:
                current = HuffmanTree(pair[0] + pair[1], t2, t1)
            trees = [t for t in trees if t not in (t1, t2)]
            update_merge(pair[0], pair[1], current)
        else:
            # Otherwise, one value is from an original symbol and the other from a tree.
            for t in trees:
                if pair[0] == t.symbol:
                    trees.remove(t)
                    if pair[1] > t.symbol:
                        current = HuffmanTree(pair[0] + pair[1], t, HuffmanTree(pair[1]))
                    else:
                        current = HuffmanTree(pair[0] + pair[1], HuffmanTree(pair[1]), t)
                    update_merge(pair[0], pair[1], current)
                    break
                elif pair[1] == t.symbol:
                    trees.remove(t)
                    if pair[0] <= t.symbol:
                        current = HuffmanTree(pair[0] + t.symbol, HuffmanTree(pair[0]), t)
                    else:
                        current = HuffmanTree(pair[0] + t.symbol, t, HuffmanTree(pair[0]))
                    update_merge(pair[0], pair[1], current)
                    break

    # Inline version of _fix_symbols to reassign correct symbols to the leaves.
    def fix_symbols(tree: HuffmanTree, mapping: dict[int, int]) -> None:
        if tree.is_leaf():
            for key in list(mapping.keys()):
                if tree.symbol == mapping[key]:
                    tree.symbol = key
                    mapping.pop(key)
                    break
        else:
            tree.symbol = None
            fix_symbols(tree.left, mapping)
            fix_symbols(tree.right, mapping)

    fix_symbols(current, freq_dict.copy())
    return current