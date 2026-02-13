from compress import *
from utils import *
import huffman


def test_generate_tree_postorder():

    tree2 = HuffmanTree(None, HuffmanTree(3, None, None), \
                       HuffmanTree(2, None, None))
    number_nodes(tree2)

    bytes2_ = tree_to_bytes(tree2)

    read_nodes2 = bytes_to_nodes(bytes2_)

    print(read_nodes2)

    assert generate_tree_postorder(read_nodes2, len(read_nodes2) - 1) == tree2

def test_generate_tree_postorder2():
    left3 = HuffmanTree(None, HuffmanTree(3, None, None), \
                       HuffmanTree(2, None, None))
    right3 = HuffmanTree(5)
    tree3 = HuffmanTree(None, left3, right3)

    number_nodes(tree3)

    bytes3_ = tree_to_bytes(tree3)

    read_nodes3 = bytes_to_nodes(bytes3_)

    assert generate_tree_postorder(read_nodes3, len(read_nodes3) - 1) == tree3


def test_get_codes_empty():

    tree1 = HuffmanTree()

    assert get_codes(tree1) == {}


def test_get_codes_general():

    left_tree = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))

    right_tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))

    left_tree2 = HuffmanTree(None, HuffmanTree(5), left_tree)

    merge_tree = HuffmanTree(None, left_tree2, right_tree)

    left_tree3 = HuffmanTree(None, HuffmanTree(6), merge_tree)

    next_tree = HuffmanTree(None, HuffmanTree(7), HuffmanTree(8))

    merge_tree2 = HuffmanTree(None, HuffmanTree(9), next_tree)

    final_tree = HuffmanTree(None, left_tree3, merge_tree2)

    result = {6: '00', 4: '0111', 3: '0110', 5:'0100', 2: '01011',
              1:'01010', 9:'10', 7:'110', 8:'111'}

    assert result == get_codes(final_tree)


def test_number_nodes():

    left_tree = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))

    right_tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))

    left_tree2 = HuffmanTree(None, HuffmanTree(5), left_tree)

    merge_tree = HuffmanTree(None, left_tree2, right_tree)

    left_tree3 = HuffmanTree(None, HuffmanTree(6), merge_tree)

    next_tree = HuffmanTree(None, HuffmanTree(7), HuffmanTree(8))

    merge_tree2 = HuffmanTree(None, HuffmanTree(9), next_tree)

    final_tree = HuffmanTree(None, left_tree3, merge_tree2)

    number_nodes(final_tree)

    assert final_tree.number == 7

    assert left_tree.number == 0

    assert right_tree.number == 2











if __name__ == '__main__':

    import pytest

    pytest.main(['PaulYansTest.py'])
    left = HuffmanTree(None, HuffmanTree(3, None, None), \
                       HuffmanTree(2, None, None))
    right = HuffmanTree(5)
    tree = HuffmanTree(None, left, right)

    number_nodes(tree)

    bytes_ = tree_to_bytes(tree)

    read_nodes = bytes_to_nodes(bytes_)

    print(list(bytes_))
    print(read_nodes)










