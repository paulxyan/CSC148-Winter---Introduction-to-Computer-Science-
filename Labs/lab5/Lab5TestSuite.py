from LinkedListPractise import _Node, CircularLinkedList, LinkedList

def test_circular_linked_list():

    lst1 = CircularLinkedList()
    Head1 = _Node(2)
    lst1.head = Head1
    lst1.head.next =  _Node(3)
    lst1.head.next.next = _Node(4)
    lst1.head.next.next.next  = Head1

    assert lst1.is_circular() == True

def test_negative_indicing():

    lst1 = LinkedList([1, 2, 3, 4, 5])

    assert lst1.negative_index_LL(-5) == 1

    lst2 = LinkedList([])

    assert lst2.negative_index_LL(-1) == -1

def test_circular_linked_list_wrong_link():

    lst2 = CircularLinkedList()

    WrongNode = _Node(2)

    lst2.head = _Node(1)
    lst2.head.next = WrongNode
    lst2.head.next.next = _Node(3)
    lst2.head.next.next.next = WrongNode

    assert lst2.is_circular() ==  False

def test_test():

    assert 1 + 1 == 2


if __name__ == '__main__':

    import pytest
    pytest.main(['Lab5TestSuite.py'])