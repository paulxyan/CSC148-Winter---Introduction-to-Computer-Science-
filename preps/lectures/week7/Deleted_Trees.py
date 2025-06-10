from TreesPractice import Tree

def test_remove_empty_tree():


    t1 = Tree(10, [Tree(1, []), Tree(2, []), Tree(3, [])])

    t1.delete_item(1)

    t1.delete_item(2)

    assert t1._subtrees.__len__() == 1

def test_remove_root():

    t2 = Tree(1, [Tree(2, []), Tree(3, []), Tree(4, [])])

    t2.delete_item(1)


    assert t2.__len__() == 3

def test_leaves():

    t = Tree(10, [Tree(1, []), Tree(2, []), Tree(3, [])])

    actual = t.leaves()

    assert actual == [1, 2, 3]


def test_delete_replaceLeftmost():


    t4 = Tree(8, [Tree(10, []), Tree(20, [])])
    t5 = Tree(6, [Tree(7, []), t4])
    t6 = Tree(30, [Tree(1, []), Tree(2, []), Tree(3, [])])
    t3 = Tree(5, [Tree(4, [Tree(100, [])]), t5, t6])

    print(t3)







if __name__ == '__main__':

    import pytest

    pytest.main(['Deleted_Trees.py'])