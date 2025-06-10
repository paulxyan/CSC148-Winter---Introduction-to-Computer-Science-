from typing import*

def sum_nested(obj: Union[int, list]) -> int:
    """Return the sum of the numbers in <obj> (or 0 if there are no numbers)."""
    if isinstance(obj, int):

        return obj

    else:

        sum1 = [sum_nested(nested_list) for nested_list in obj]

        return sum(sum1)

def flatten(obj: Union[int, list]) -> list[int]:
    """Return a (non-nested) list of the integers in <obj>."""
    if isinstance(obj, int):
        return [obj]

    else:

        answer = [flatten(flattened_list) for flattened_list in obj]

        return sum(answer, [])

def nested_list_contains(obj: Union[int, list], item: int) -> bool:
    if isinstance(obj, int): #Base Case
        return obj == item

    else:

        answers = [nested_list_contains(elem, item) for elem in obj]

        return any(answers)

def semi_homogeneous(obj: Union[int, list]) -> bool:
    """Return whether the given nested list is semi-homogeneous.
    A single integer and empty list are semi-homogeneous.
    In general, a list is semi-homogeneous if and only if:
    - all of its sub-nested-lists are integers, or all of them are lists
    - all of its sub-nested-lists are semi-homogeneous
    """

    if isinstance(obj, int) or obj == []:

        return True

    else:

        int_check = all([isinstance(elem, int) for elem in obj])
        list_check = all([isinstance(elem, list) for elem in obj])

        if not int_check and not list_check: return False

        else:

            return all([semi_homogeneous(elem) for elem in obj])




if __name__ == '__main__':

    print(sum_nested([1, 1, [1, 20], 1]))