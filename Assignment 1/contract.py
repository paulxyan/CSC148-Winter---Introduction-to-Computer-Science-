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
import datetime
from math import ceil
from typing import Optional
from bill import Bill
from call import Call

# Constants for the month-to-month contract monthly fee and term deposit
MTM_MONTHLY_FEE = 50.00
TERM_MONTHLY_FEE = 20.00
TERM_DEPOSIT = 300.00

# Constants for the included minutes and SMSs in the term contracts (per month)
TERM_MINS = 100

# Cost per minute and per SMS in the month-to-month contract
MTM_MINS_COST = 0.05

# Cost per minute and per SMS in the term contract
TERM_MINS_COST = 0.1

# Cost per minute and per SMS in the prepaid contract
PREPAID_MINS_COST = 0.025


class Contract:
    """ A contract for a phone line

    This is an abstract class and should not be directly instantiated.

    Only subclasses should be instantiated.

    === Public Attributes ===
    start:
         starting date for the contract
    bill:
         bill for this contract for the last month of call records loaded from
         the input dataset
    """
    start: datetime.date
    bill: Optional[Bill]

    def __init__(self, start: datetime.date) -> None:
        """ Create a new Contract with the <start> date, starts as inactive
        """
        self.start = start
        self.bill = None

    def new_month(self, month: int, year: int, bill: Bill) -> None:
        """ Advance to a new month in the contract, corresponding to <month> and
        <year>. This may be the first month of the contract.
        Store the <bill> argument in this contract and set the appropriate rate
        per minute and fixed cost.

        DO NOT CHANGE THIS METHOD
        """
        raise NotImplementedError

    def bill_call(self, call: Call) -> None:
        """ Add the <call> to the bill.

        Precondition:
        - a bill has already been created for the month+year when the <call>
        was made. In other words, you can safely assume that self.bill has been
        already advanced to the right month+year.
        """
        self.bill.add_billed_minutes(ceil(call.duration / 60.0))

    def cancel_contract(self) -> float:
        """ Return the amount owed in order to close the phone line associated
        with this contract.

        Precondition:
        - a bill has already been created for the month+year when this contract
        is being cancelled. In other words, you can safely assume that self.bill
        exists for the right month+year when the cancelation is requested.
        """
        self.start = None
        return self.bill.get_cost()


class TermContract(Contract):
    """A term contract consisting of instance of contract which
    has a start date and end date, requires a large term deposit,
    has a set monthly cost, lower calling rates, and free minutes each month

    === Public Attributes ===

    end:
        end date for the contract

    curr:
        current month of billing for the TermContract
        to keep track of

    free_mins:

        amount of free minutes which a bill hasn't used
        for the current month, resets after each new month is called


    === Representation Invariants ===

    end date cannot be before the start date
    free_mins >= 0


    """

    end: datetime.date
    curr: datetime.date
    free_mins: int

    def __init__(self, start: datetime.date, end: datetime.date) -> None:
        """Create a new TermContract
        Precondition: Assume start date happens before end date
        """

        super().__init__(start)
        self.end = end
        self.curr = start
        self.free_mins = TERM_MINS

    def bill_call(self, call: Call) -> None:
        """ Add the <call> to the bill.

        Precondition:
        - a bill has already been created for the month+year when the <call>
        was made. In other words, you can safely assume that self.bill has been
        already advanced to the right month+year.
        """

        minutes_called = ceil(call.duration / 60.0)

        if self.free_mins >= minutes_called:

            self.free_mins -= minutes_called
            self.bill.add_free_minutes(minutes_called)

        elif minutes_called > self.free_mins:
            self.bill.add_billed_minutes(minutes_called - self.free_mins)
            self.bill.add_free_minutes(self.free_mins)
            self.free_mins = 0

    def new_month(self, month: int, year: int, bill: Bill) -> None:
        """ Advance to a new month in the contract, corresponding to <month> and
        <year>. This may be the first month of the contract.
        Store the <bill> argument in this contract and set the appropriate rate
        per minute and fixed cost.

        DO NOT CHANGE THIS METHOD

        OverRiding The method to account for TermContract parameters
        """

        self.free_mins = TERM_MINS
        bill.set_rates("TERM", TERM_MINS_COST)
        bill.add_fixed_cost(TERM_MONTHLY_FEE)

        # Making sure that term deposit is set at the beginning of the contract
        if month == self.start.month and year == self.start.year:

            bill.add_fixed_cost(TERM_DEPOSIT)

        self.bill = bill
        self.curr = datetime.date(year, month, 1)

    def cancel_contract(self) -> float:
        """ Return the amount owed in order to close the phone line associated
        with this contract.

        Precondition:
        - a bill has already been created for the month+year when this contract
        is being cancelled. In other words, you can safely assume that self.bill
        exists for the right month+year when the cancelation is requested.
        """

        self.start = None

        if self.curr >= self.end:
            return max(self.bill.get_cost() - TERM_DEPOSIT, 0)

        return self.bill.get_cost()


class MTMContract(Contract):
    """A subclass of contract, has no initial deposit,
     nor free minutes and no end date """

    def __init__(self, start: datetime.date) -> None:
        """ Create a new Contract with the <start> date, starts as inactive
        """

        super().__init__(start)

    def new_month(self, month: int, year: int, bill: Bill) -> None:
        """ Advance to a new month in the contract, corresponding to <month> and
        <year>. This may be the first month of the contract.
        Store the <bill> argument in this contract and set the appropriate rate
        per minute and fixed cost.

        DO NOT CHANGE THIS METHOD
        """

        bill.set_rates(contract_type="MTM", min_cost=MTM_MINS_COST)
        bill.add_fixed_cost(MTM_MONTHLY_FEE)
        self.bill = bill


class PrepaidContract(Contract):
    """Implementation of prepaid contract,
    type of contract and has no free minutes,
     but balance carries over month-to-month

    === Private Attributes ===

    _balance: balance of this users contract

    """

    _balance: float

    def __init__(self, start: datetime.date, balance: float) -> None:
        """ Create a new Contract with the <start> date, starts as inactive
        """
        super().__init__(start)
        self._balance = -balance

    def new_month(self, month: int, year: int, bill: Bill) -> None:

        # checking if the date is valid
        if self.bill is not None:
            # Adding the previous months bill to the total_balance
            self._balance = self.bill.get_cost()

        bill.set_rates(min_cost=PREPAID_MINS_COST, contract_type="PREPAID")
        bill.add_fixed_cost(self._balance)

        if self._balance > -10:
            self._balance -= 25

        self.bill = bill

    def cancel_contract(self) -> float:
        """ Return the amount owed in order to close the phone line associated
        with this contract.

        Precondition:
        - a bill has already been created for the month+year when this contract
        is being cancelled. In other words, you can safely assume that self.bill
        exists for the right month+year when the cancelation is requested.
        """

        self.start = None

        if self._balance > 0:

            return self._balance  # Only needs to return the recorded balance

        else:

            return 0.0


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'allowed-import-modules': [
            'python_ta', 'typing', 'datetime', 'bill', 'call', 'math'
        ],
        'disable': ['R0902', 'R0913'],
        'generated-members': 'pygame.*'
    })
