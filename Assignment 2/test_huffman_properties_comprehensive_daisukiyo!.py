from __future__ import annotations

from random import shuffle
import random

import pytest
from hypothesis import given, assume, settings, strategies as st, HealthCheck
from hypothesis.strategies import binary, integers, dictionaries, text
from utils import byte_to_bits, bits_to_byte
from compress import *

settings.register_profile("norand", settings(derandomize=True, max_examples=200))
settings.load_profile("norand")


# === Test Byte Utilities ===
# Technically, these utilities are given to you in the starter code, so these
# first 3 tests below are just intended as a sanity check to make sure that you
# did not modify these methods and are therefore using them incorrectly.
# You will not be submitting utils.py anyway, so these first three tests are
# solely for your own benefit, as a sanity check.

@given(integers(0, 255))
def test_byte_to_bits(b: int) -> None:
    """ Test that byte_to_bits produces binary strings of length 8."""
    assert set(byte_to_bits(b)).issubset({"0", "1"})
    assert len(byte_to_bits(b)) == 8


@given(text(["0", "1"], min_size=0, max_size=8))
def test_bits_to_byte(s: str) -> None:
    """ Test that bits_to_byte produces a byte."""
    b = bits_to_byte(s)
    assert isinstance(b, int)
    assert 0 <= b <= 255


@given(integers(0, 255), integers(0, 7))
def test_get_bit(byte: int, bit_pos: int) -> None:
    """ Test that get_bit(byte, bit) produces  bit values."""
    b = get_bit(byte, bit_pos)
    assert isinstance(b, int)
    assert 0 <= b <= 1


# === Test the compression code ===

@given(binary(min_size=0, max_size=1000))
def test_build_frequency_dict(byte_list: bytes) -> None:
    """ Test that build_frequency_dict returns dictionary whose values sum up
    to the number of bytes consumed.
    """
    # creates a copy of byte_list, just in case your implementation of
    # build_frequency_dict modifies the byte_list
    b, d = byte_list, build_frequency_dict(byte_list)
    assert isinstance(d, dict)
    assert sum(d.values()) == len(b)


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_build_huffman_tree(d: dict[int, int]) -> None:
    """ Test that build_huffman_tree returns a non-leaf HuffmanTree."""
    t = build_huffman_tree(d)
    assert isinstance(t, HuffmanTree)
    assert not t.is_leaf()


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_get_codes(d: dict[int, int]) -> None:
    """ Test that the sum of len(code) * freq_dict[code] is optimal, so it
    must be invariant under permutation of the dictionary.
    Note: This also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    c1 = get_codes(t)
    d2 = list(d.items())
    shuffle(d2)
    d2 = dict(d2)
    t2 = build_huffman_tree(d2)
    c2 = get_codes(t2)
    assert sum([d[k] * len(c1[k]) for k in d]) == \
           sum([d2[k] * len(c2[k]) for k in d2])


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_number_nodes(d: dict[int, int]) -> None:
    """ If the root is an interior node, it must be numbered two less than the
    number of symbols, since a complete tree has one fewer interior nodes than
    it has leaves, and we are numbering from 0.
    Note: this also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    assume(not t.is_leaf())
    count = len(d)
    number_nodes(t)
    assert count == t.number + 2


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_avg_length(d: dict[int, int]) -> None:
    """ Test that avg_length returns a float in the interval [0, 8], if the max
    number of symbols is 256.
    """
    t = build_huffman_tree(d)
    f = avg_length(t, d)
    assert isinstance(f, float)
    assert 0 <= f <= 8.0


@given(binary(min_size=2, max_size=1000))
def test_compress_bytes(b: bytes) -> None:
    """ Test that compress_bytes returns a bytes object that is no longer
    than the input bytes. Also, the size of the compressed object should be
    invariant under permuting the input.
    Note: this also indirectly tests build_frequency_dict, build_huffman_tree,
    and get_codes.
    """
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed = compress_bytes(b, c)
    assert isinstance(compressed, bytes)
    assert len(compressed) <= len(b)
    lst = list(b)
    shuffle(lst)
    b = bytes(lst)
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed2 = compress_bytes(b, c)
    assert len(compressed2) == len(compressed)


@given(binary(min_size=2, max_size=1000))
def test_tree_to_bytes(b: bytes) -> None:
    """ Test that tree_to_bytes generates a bytes representation of a postorder
    traversal of a tree's internal nodes.
    Since each internal node requires 4 bytes to represent, and there are
    1 fewer internal nodes than distinct symbols, the length of the bytes
    produced should be 4 times the length of the frequency dictionary, minus 4.
    Note: also indirectly tests build_frequency_dict, build_huffman_tree, and
    number_nodes.
    """
    d = build_frequency_dict(b)
    assume(len(d) > 1)
    t = build_huffman_tree(d)
    number_nodes(t)
    output_bytes = tree_to_bytes(t)
    dictionary_length = len(d)
    leaf_count = dictionary_length
    assert (4 * (leaf_count - 1)) == len(output_bytes)


# === Test a roundtrip conversion

@given(binary(min_size=1, max_size=1000))
def test_round_trip_compress_bytes(b: bytes) -> None:
    """ Test that applying compress_bytes and then decompress_bytes
    will produce the original text.
    """
    text = b
    freq = build_frequency_dict(text)
    assume(len(freq) > 1)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(text, codes)
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert text == decompressed

# ==================== BASIC FUNCTIONALITY TESTS ====================

def test_empty_text_gives_empty_dict():
    assert build_frequency_dict(b'') == {}

def test_single_byte_text():
    assert build_frequency_dict(b'A') == {65: 1}

def test_multiple_occurrences():
    data = b'abracadabra'
    freq = build_frequency_dict(data)
    expected = {ord('a'): 5, ord('b'): 2, ord('r'): 2, ord('c'): 1, ord('d'): 1}
    assert freq == expected

def test_manual_tree_to_codes():
    tree = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    codes = get_codes(tree)
    assert codes == {1: '0', 2: '1'}

# ==================== ROUND TRIP TESTS ====================

@given(st.binary(min_size=1, max_size=500))
def test_compress_then_decompress_returns_original(data):
    freq = build_frequency_dict(data)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    compressed = compress_bytes(data, codes)
    result = decompress_bytes(tree, compressed, len(data))
    assert result == data

# ==================== TREE STRUCTURE TESTS ====================

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=2, max_size=50))
def test_tree_contains_all_symbols(symbols):
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    for s in freq:
        assert s in codes

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=2, max_size=50))
def test_prefix_free(symbols):
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    code_list = list(codes.values())
    for i in range(len(code_list)):
        for j in range(len(code_list)):
            if i != j:
                assert not code_list[i].startswith(code_list[j])

# ==================== AVG LENGTH TESTS ====================

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=2, max_size=200))
def test_avg_length_reasonable(symbols):
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    tree = build_huffman_tree(freq)
    avg = avg_length(tree, freq)
    assert avg >= 1

# ==================== IMPROVE TREE TESTS ====================

def test_improve_tree_preserves_shape():
    left = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    right = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    tree = HuffmanTree(None, left, right)
    import copy
    original = copy.deepcopy(tree)
    freq = {1: 5, 2: 5, 3: 5, 4: 5}
    improve_tree(tree, freq)
    def structure(t):
        if t.is_leaf():
            return 'leaf'
        return structure(t.left), structure(t.right)
    assert structure(tree) == structure(original)

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=2, max_size=50))
def test_improve_tree_reduces_avg_or_equal(symbols):
    print("="*40)
    print(f"Symbols: {symbols}")

    # Build frequency dictionary directly
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1

    print(f"Frequency dict: {freq}")

    # Build tree BEFORE improve_tree
    tree = build_huffman_tree(freq)
    print("Tree before improve_tree:")
    print(tree)

    # Calculate avg length BEFORE
    before = avg_length(tree, freq)
    print(f"Average length before improve_tree: {before}")

    # Apply improve_tree
    improve_tree(tree, freq)

    # Tree AFTER improve_tree
    print("Tree after improve_tree:")
    print(tree)

    # Calculate avg length AFTER
    after = avg_length(tree, freq)
    print(f"Average length after improve_tree: {after}")

    # Assert
    print(f"Property check: after <= before? {after <= before}")
    assert after <= before

    print("="*40 + "\n")



# ==================== PATHOLOGICAL CASES ====================

def test_perfectly_balanced_tree():
    """Test a balanced tree (powers of two symbols)."""
    freq = {i: 1 for i in range(8)}  # 8 symbols
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    # All code lengths should be equal
    lengths = set(len(code) for code in codes.values())
    assert len(lengths) == 1

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=2, max_size=50))
def test_highly_skewed_frequencies(symbols):
    """Test where one symbol is extremely frequent."""
    assume(len(set(symbols)) > 1)
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    # Boost one symbol massively
    heavy = list(freq.keys())[0]
    freq[heavy] *= 1000
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    # Heavy symbol gets short code
    assert len(codes[heavy]) == min(len(code) for code in codes.values())

def test_prefix_free_codes_in_pathological_tree():
    """Test that codes generated from a hand-crafted, pathological Huffman tree are prefix-free."""
    # Construct a tree with an intentionally unbalanced structure:
    #      *
    #     / \
    #    1   *
    #       / \
    #      2   *
    #         / \
    #        3   4
    #
    # This tree structure ensures that the code lengths increase down the right subtree.

    tree = HuffmanTree(None,
                       HuffmanTree(1),
                       HuffmanTree(None,
                                   HuffmanTree(2),
                                   HuffmanTree(None,
                                               HuffmanTree(3),
                                               HuffmanTree(4))))

    # Generate codes from this pathological tree:
    codes = get_codes(tree)

    # Extract all the codes into a list:
    code_list = list(codes.values())

    # Verify that no code is a prefix of another:
    for i in range(len(code_list)):
        for j in range(len(code_list)):
            if i != j:
                assert not code_list[i].startswith(code_list[j]), \
                    f"Code {code_list[i]} is a prefix of {code_list[j]}"


# ==================== IMPROVE TREE CORNER TESTS ====================

def test_improve_tree_reduces_or_preserves_avg_length():
    """Explicit improve_tree test with unbalanced initial tree."""
    freq = {i: i+1 for i in range(6)}
    tree = build_huffman_tree(freq)
    avg_before = avg_length(tree, freq)
    improve_tree(tree, freq)
    avg_after = avg_length(tree, freq)
    assert avg_after <= avg_before

# ==================== ROUND-TRIP WITH RANDOMIZED OPTIMIZATION ====================

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=10, max_size=300))
def test_round_trip_with_improve(symbols):
    """Ensure round trip works even after improve_tree runs."""
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    tree = build_huffman_tree(freq)
    improve_tree(tree, freq)
    codes = get_codes(tree)
    data = bytes(symbols)
    compressed = compress_bytes(data, codes)
    result = decompress_bytes(tree, compressed, len(data))
    assert result == data


# ==================== POSTORDER vs GENERAL RECONSTRUCTION ====================

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=5,
                max_size=50))
def test_postorder_general_roundtrip(symbols):
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1

    tree = build_huffman_tree(freq)
    data = bytes(symbols)

    # Original encode-decode
    codes = get_codes(tree)
    compressed = compress_bytes(data, codes)
    decompressed = decompress_bytes(tree, compressed, len(data))
    assert decompressed == data

    # POSTORDER
    number_nodes(tree)
    nodes_bytes = tree_to_bytes(tree)
    node_list = bytes_to_nodes(nodes_bytes)
    tree_postorder = generate_tree_postorder(node_list, len(node_list) - 1)
    codes_postorder = get_codes(tree_postorder)
    compressed_postorder = compress_bytes(data, codes_postorder)
    decompressed_postorder = decompress_bytes(tree_postorder, compressed_postorder, len(data))
    assert decompressed_postorder == data

    # GENERAL
    tree_general = generate_tree_general(node_list, len(node_list) - 1)
    codes_general = get_codes(tree_general)
    compressed_general = compress_bytes(data, codes_general)
    decompressed_general = decompress_bytes(tree_general, compressed_general, len(data))
    assert decompressed_general == data


# ==================== SYMBOL DISTRIBUTION PROPERTIES ====================

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=50, max_size=200))
def test_frequent_symbols_get_shorter_codes(symbols):
    """Check that more frequent symbols have shorter or equal codes."""
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1

    tree = build_huffman_tree(freq)
    codes = get_codes(tree)

    pairs = [(s1, s2) for s1 in freq for s2 in freq if freq[s1] > freq[s2]]
    for s1, s2 in pairs:
        assert len(codes[s1]) <= len(codes[s2])

# ==================== HIGHLY UNBALANCED TREE ====================

def test_extreme_skew():
    """Test a tree where one symbol dominates and others are rare."""
    freq = {i: 1 for i in range(1, 256)}
    freq[0] = 10**6

    tree = build_huffman_tree(freq)
    codes = get_codes(tree)

    # The dominating symbol gets the shortest possible code
    assert len(codes[0]) == 1

# ==================== SINGLETON SYMBOL SPECIAL CASE ====================

def test_singleton_symbol():
    """Singleton input should still round-trip correctly."""
    data = bytes([42] * 500)
    freq = build_frequency_dict(data)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(data, codes)
    decompressed = decompress_bytes(tree, compressed, len(data))
    assert decompressed == data


# ==================== BMP-LIKE PATTERN ====================

def test_bmp_header_like_pattern():
    """Simulate typical bitmap-like header followed by uniform noise."""
    header = bytes([66, 77] + [0] * 50)  # 'BM' header + padding
    noise = bytes([random.randint(0, 255) for _ in range(1000)])
    data = header + noise
    freq = build_frequency_dict(data)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(data, codes)
    decompressed = decompress_bytes(tree, compressed, len(data))
    assert decompressed == data

# ==================== TEXT-LIKE PATTERN ====================

def test_text_with_spaces():
    """Simulate ASCII text with high space frequency."""
    text = (b'This is a test sentence with many spaces. ' * 50)
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(text, codes)
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert decompressed == text

# ==================== RUN-LENGTH-LIKE PATTERN ====================

@given(st.integers(min_value=1, max_value=255))
def test_long_repeats(byte):
    """Simulate data with long runs of the same byte (e.g., silence in audio)."""
    data = bytes([byte] * 10000)
    freq = build_frequency_dict(data)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(data, codes)
    decompressed = decompress_bytes(tree, compressed, len(data))
    assert decompressed == data

# ==================== RANDOMIZED WAVE PATTERN ====================

def test_waveform_like():
    """Simulate low-frequency varying data like a digitized sine wave."""
    data = bytes([(i % 256) for i in range(10000)])
    freq = build_frequency_dict(data)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(data, codes)
    decompressed = decompress_bytes(tree, compressed, len(data))
    assert decompressed == data

# ==================== EDGE CASES ====================

def test_single_symbol_tree():
    """Test that build_huffman_tree correctly handles the single-symbol case."""
    symbol = 42  # any valid symbol
    freq = {symbol: 10}  # only one symbol with any positive frequency

    tree = build_huffman_tree(freq)

    # The tree must not be a leaf
    assert not tree.is_leaf(), "Tree should not be a leaf when there's only one symbol"

    # The tree must have exactly two leaves
    assert tree.left.is_leaf() and tree.right.is_leaf(), "Both children should be leaves"

    # One of them should be the actual symbol, the other the dummy
    dummy_symbol = (symbol + 1) % 256
    symbols_in_tree = {tree.left.symbol, tree.right.symbol}
    assert {symbol, dummy_symbol} == symbols_in_tree, "Symbols mismatch"

    # The codebook must assign codes to both symbols
    codes = get_codes(tree)
    assert set(codes.keys()) == {symbol, dummy_symbol}, "Codebook missing expected symbols"
    assert all(code != '' for code in codes.values()), "Assigned codes should not be empty"

    print("test_single_symbol_tree passed!")


@given(st.lists(st.integers(min_value=0, max_value=255), min_size=255, max_size=256))
def test_max_symbols_case(symbols):
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    assert len(codes) <= 256


# ==================== FINAL TESTS ====================
@given(dictionaries(integers(min_value=0, max_value=255),
                    integers(min_value=1, max_value=1000), min_size=2,
                    max_size=256))
def test_number_nodes_full_range(d):
    tree = build_huffman_tree(d)
    number_nodes(tree)
    internal_nodes = []

    def collect(node):
        if not node.is_leaf():
            internal_nodes.append(node.number)
            collect(node.left)
            collect(node.right)

    collect(tree)
    assert sorted(internal_nodes) == list(range(len(internal_nodes)))


def test_padding_edge_case():
    # Create data where code lengths sum to a multiple of 8
    freq = {i: 1 for i in range(8)}
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)

    # Construct text that compresses to exactly 8 bits
    text = bytes([list(freq.keys())[0]])
    compressed = compress_bytes(text, codes)

    # Should produce exactly 1 byte even if no padding is needed
    assert len(compressed) == 1


def test_compress_decompress_file(tmp_path):
    in_file = tmp_path / "infile.bin"
    out_file = tmp_path / "outfile.bin"

    data = b"Hello Huffman!"
    in_file.write_bytes(data)

    compress_file(str(in_file), str(in_file) + ".huf")
    decompress_file(str(in_file) + ".huf", str(out_file))

    assert out_file.read_bytes() == data


@given(st.lists(st.integers(min_value=0, max_value=255), min_size=5, max_size=50))
def test_postorder_vs_general_structure(symbols):
    freq = {s: symbols.count(s) for s in set(symbols)}
    tree = build_huffman_tree(freq)
    number_nodes(tree)
    nodes_bytes = tree_to_bytes(tree)
    node_list = bytes_to_nodes(nodes_bytes)

    tree_postorder = generate_tree_postorder(node_list, len(node_list) - 1)
    tree_general = generate_tree_general(node_list, len(node_list) - 1)

    # Check that they produce the same codes for all symbols
    codes_postorder = get_codes(tree_postorder)
    codes_general = get_codes(tree_general)

    assert codes_postorder == codes_general


@given(st.lists(st.integers(min_value=0, max_value=255), min_size=2, max_size=200))
def test_improve_tree_monotonicity(symbols):
    freq = {s: symbols.count(s) for s in set(symbols)}
    tree = build_huffman_tree(freq)
    avg_before = avg_length(tree, freq)
    improve_tree(tree, freq)
    avg_after = avg_length(tree, freq)
    assert avg_after <= avg_before


# ==================== STRESS TESTS ====================

@given(st.binary(min_size=1000, max_size=5000))
def test_large_compress_decompress(data):
    """Stress test with large random binary data."""
    freq = build_frequency_dict(data)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    compressed = compress_bytes(data, codes)
    result = decompress_bytes(tree, compressed, len(data))
    assert result == data

@given(st.lists(st.integers(min_value=0, max_value=255), min_size=500, max_size=1000))
def test_large_symbol_set(symbols):
    """Stress test with up to 1000 symbols."""
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    # Confirm that all symbols received codes
    for s in freq:
        assert s in codes

# ==================== TORTURE ROUND TRIP ====================

@settings(suppress_health_check=[HealthCheck.large_base_example], deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=255), min_size=2000, max_size=5000))
def test_torture_round_trip(symbols):
    """Test extremely large input for full compression and decompression pipeline."""
    data = bytes(symbols)
    freq = build_frequency_dict(data)
    tree = build_huffman_tree(freq)
    improve_tree(tree, freq)

    codes = get_codes(tree)
    compressed = compress_bytes(data, codes)
    decompressed = decompress_bytes(tree, compressed, len(data))

    assert decompressed == data


if __name__ == "__main__":
    pytest.main(["test_huffman_properties_comprehensive.py"])
