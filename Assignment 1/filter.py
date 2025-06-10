"""
CSC148, Winter 2025
Assignment 1

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 Bogdan Simion, Diane Horton, Jacqueline Smith
"""
import time
import datetime
from call import Call
from customer import Customer


class Filter:
    """ A class for filtering customer data on some criterion. A filter is
    applied to a set of calls.

    This is an abstract class. Only subclasses should be instantiated.
    """
    def __init__(self) -> None:
        pass

    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all calls from <data>, which match the filter
        specified in <filter_string>.

        The <filter_string> is provided by the user through the visual prompt,
        after selecting this filter.
        The <customers> is a list of all customers from the input dataset.

         If the filter has
        no effect or the <filter_string> is invalid then return the same calls
        from the <data> input.

        Note that the order of the output matters, and the output of a filter
        should have calls ordered in the same manner as they were given, except
        for calls which have been removed.

        Precondition:
        - <customers> contains the list of all customers from the input dataset
        - all calls included in <data> are valid calls from the input dataset
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        raise NotImplementedError


class ResetFilter(Filter):
    """
    A class for resetting all previously applied filters, if any.
    """
    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Reset all of the applied filters. Return a List containing all the
        calls corresponding to <customers>.
        The <data> and <filter_string> arguments for this type of filter are
        ignored.

        Precondition:
        - <customers> contains the list of all customers from the input dataset
        """
        filtered_calls = []
        for c in customers:
            customer_history = c.get_history()
            # only take outgoing calls, we don't want to include calls twice
            filtered_calls.extend(customer_history[0])
        return filtered_calls

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Reset all of the filters applied so far, if any"


class CustomerFilter(Filter):
    """
    A class for selecting only the calls from a given customer.
    """
    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all unique calls from <data> made or
        received by the customer with the id specified in <filter_string>.

        The <customers> list contains all customers from the input dataset.

        The filter string is valid if and only if it contains a valid
        customer ID.
        - If the filter string is invalid, return the original list <data>
        - If the filter string is invalid, your code must not crash, as
        specified in the handout.

        Do not mutate any of the function arguments!
        """

        if len(filter_string) != 4:

            return data

        customer_dict = {customer.get_id(): customer for customer in customers}

        customer_calls = []

        try:

            check_id = int(filter_string)

        except ValueError:

            return data

        if check_id in customer_dict:

            customer_wanted = customer_dict[check_id]

            customer_history = customer_wanted.get_history()

            for call in data:

                if call in customer_history[0] or call in customer_history[1]:

                    customer_calls.append(call)

        return customer_calls if len(customer_calls) > 0 else data

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Filter events based on customer ID"


class DurationFilter(Filter):
    """
    A class for selecting only the calls lasting either over or under a
    specified duration.
    """
    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all unique calls from <data> with a duration
        of under or over the time indicated in the <filter_string>.

        The <customers> list contains all customers from the input dataset.

        The filter string is valid if and only if it contains the following
        input format: either "Lxxx" or "Gxxx", indicating to filter calls less
        than xxx or greater than xxx seconds, respectively.
        - If the filter string is invalid, return the original list <data>
        - If the filter string is invalid, your code must not crash, as
        specified in the handout.

        Do not mutate any of the function arguments!
        """

        if not is_valid_filter(filter_string):

            return data

        value_check = int(filter_string[1:])

        calls_return = []

        if filter_string[0] == "L":

            calls_return.extend([call for call in
                                 data if call.duration < value_check])

        elif filter_string[0] == "G":

            calls_return.extend([call for call in data if
                                 call.duration > value_check])

        return calls_return

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Filter calls based on duration; " \
               "L### returns calls less than specified length, G### for greater"


class LocationFilter(Filter):
    """
    A class for selecting only the calls that took place within a specific area
    """
    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all unique calls from <data>, which took
        place within a location specified by the <filter_string>
        (at least the source or the destination of the event was
        in the range of coordinates from the <filter_string>).

        The <customers> list contains all customers from the input dataset.

        The filter string is valid if and only if it contains four valid
        coordinates within the map boundaries.
        These coordinates represent the location of the lower left corner
        and the upper right corner of the search location rectangle,
        as 2 pairs of longitude/latitude coordinates, each separated by
        a comma and a space:
          lowerLong, lowerLat, upperLong, upperLat
        Calls that fall exactly on the boundary of this rectangle are
        considered a match as well.
        - If the filter string is invalid, return the original list <data>
        - If the filter string is invalid, your code must not crash, as
        specified in the handout.

        Do not mutate any of the function arguments!
        """

        if not is_valid_location_filter(filter_string):

            return data

        coordinates = filter_string.split(", ")

        lower_long, lower_lat, upper_long, upper_lat = map(float, coordinates)

        left_point = (lower_long, lower_lat)
        right_point = (upper_long, upper_lat)

        call_items = []

        for call in data:

            if (is_point_in_rectangle(left_point,
                                      right_point, call.src_loc)
                    or is_point_in_rectangle(
                    left_point, right_point, call.dst_loc)):
                call_items.append(call)

        return call_items

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Filter calls made or received in a given rectangular area. " \
               "Format: \"lowerLong, lowerLat, " \
               "upperLong, upperLat\" (e.g., -79.6, 43.6, -79.3, 43.7)"


def is_valid_filter(filter_string: str) -> bool:
    """Checks if the duration filter is valid"""
    # Check if the filter string is too short (at least 2 characters required)
    if not len(filter_string) == 4:
        return False

    # Check if the first character is either 'L' or 'G'
    if filter_string[0] not in {'L', 'G'}:
        return False

    # Check if the rest of the string is a valid integer
    if not filter_string[1:].isdigit():
        return False

    try:

        val = int(filter_string[1:])

    except ValueError:

        return False

    if val < 0:

        return False

    return True


def is_valid_location_filter(filter_string: str) -> bool:
    """Checks if location is valid"""
    if filter_string.count(",") != 3 or filter_string.count(" ") != 3:

        return False

    try:
        # Split the string by commas and whitespace
        coordinates = filter_string.split(", ")

        if " " in coordinates[0]:

            return False

        # Check if there are exactly 4 parts
        if len(coordinates) != 4:
            return False

        # Convert each coordinate to a float

        try:
            (lower_long, lower_lat,
             upper_long, upper_lat) = map(float, coordinates)

        except ValueError:

            return False

        # Validate longitude and latitude ranges
        if not -79.697878 <= lower_long <= upper_long <= -79.196382:
            return False
        if not 43.576959 <= lower_lat <= upper_lat <= 43.799568:
            return False

        # Ensure the rectangle coordinates are valid
        if lower_long > upper_long or lower_lat > upper_lat:
            return False

        return True

    except (ValueError, TypeError):
        # If conversion to float fails, or an invalid string format is provided
        return False


def is_point_in_rectangle(lower_left: tuple[float, float],
                          upper_right: tuple[float, float],
                          point: tuple[float, float]) -> bool:
    """
    checks if point is within rectangle
    """
    lower_long, lower_lat = lower_left
    upper_long, upper_lat = upper_right
    point_long, point_lat = point

    # Check if the point falls within the
    # rectangle boundaries (including the boundary itself)
    return (lower_long <= point_long <= upper_long) \
        and (lower_lat <= point_lat <= upper_lat)


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'allowed-import-modules': [
            'python_ta', 'typing', 'time', 'datetime', 'call', 'customer'
        ],
        'max-nested-blocks': 4,
        'allowed-io': ['apply', '__str__'],
        'disable': ['W0611', 'W0703'],
        'generated-members': 'pygame.*'
    })
