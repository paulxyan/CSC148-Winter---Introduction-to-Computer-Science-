import datetime

import pytest
import random

from callhistory import CallHistory
from application import create_customers, process_event_history, import_data
from contract import TermContract, MTMContract, PrepaidContract, Contract
from customer import Customer
from filter import DurationFilter, CustomerFilter, LocationFilter, ResetFilter
from phoneline import PhoneLine


def gen_term_contract() -> TermContract:
    return TermContract(datetime.date(2020, 1, 1),
                        datetime.date(2024, 1, 1))


def gen_mtm_contract() -> MTMContract:
    return MTMContract(datetime.date(2020, 1, 1))


def gen_prepay_contract(pay: int) -> PrepaidContract:
    return PrepaidContract(datetime.date(2020, 1, 1), pay)


def gen_contract(i: int) -> Contract:
    if i % 3 == 0:
        return gen_term_contract()
    elif i % 3 == 1:
        return gen_mtm_contract()
    return gen_prepay_contract(100)


def gen_phone_number(number: int) -> str:
    return str(number)


def gen_customer(call_amount: int, prev: int) -> Customer:
    c = Customer(prev)
    num = prev + 1
    for i in range(call_amount):
        c.add_phone_line(PhoneLine(gen_phone_number(i + num),
                                   gen_contract(i)))

    return c


def gen_dataset(cust_amount: int, numbers: int) -> list[Customer]:
    l = []
    for i in range(cust_amount):
        l.append(gen_customer(numbers, numbers * i))
    return l


def gen_connections(data: list[Customer], num_calls: int, month: int, year:int) -> dict:
    min_number = int(data[0].get_phone_numbers()[0])
    max_number = int(data[-1].get_phone_numbers()[-1])

    con = {}
    con['events'] = []

    for i in range(num_calls):
        mini = {}
        mini['type'] = 'call'
        mini['src_number'] = str(random.randint(min_number, max_number))
        mini['dst_number'] = str(random.randint(min_number, max_number))
        mini['time'] = '{0}-{1}-01 03:17:57'.format(year, month)
        mini['duration'] = random.randint(3, 600)
        mini['src_loc'] = (0, 0,)
        mini['dst_loc'] = (0, 0,)
        con['events'].append(mini)

    return con

def add_call(data: Customer,src: int, dst: int, durr: int, month: int, year:int) -> dict:
    con = {}
    con['events'] = []

    mini = {}
    mini['type'] = 'call'
    mini['src_number'] = str(src)
    mini['dst_number'] = str(dst)
    mini['time'] = '{0}-{1}-01 03:17:57'.format(year, month)
    mini['duration'] = durr
    mini['src_loc'] = (0, 0,)
    mini['dst_loc'] = (0, 0,)
    con['events'].append(mini)

    return con


def test_customer_creation():
    data = gen_dataset(10, 1)
    process_event_history(gen_connections(data[:1], 10, 1, 2020), data)
    process_event_history(gen_connections(data[1:2], 10, 1, 2021), data)

    history = data[0].get_call_history('1')
    history2 = data[1].get_call_history('2')

    assert len(data) == 10
    assert len(data[0].get_phone_numbers()) == 1
    assert data[0].get_phone_numbers()[0] == '1'
    assert data[0].get_id() == 0
    assert len(history) == 1
    assert len(history[0].incoming_calls) == 1
    assert len(history[0].outgoing_calls) == 1
    assert len(history[0].incoming_calls[(1, 2020)]) == 10
    assert len(history2) == 1
    assert len(history2[0].incoming_calls) == 1
    assert len(history2[0].outgoing_calls) == 1

    try:
        assert len(history2[0].incoming_calls[(1, 2020)]) == 0
    except KeyError:
        assert 1 == 1

    assert len(history2[0].incoming_calls[(1, 2021)]) == 10

    data = gen_dataset(2, 47)
    assert len(data) == 2
    assert len(data[0].get_phone_numbers()) == 47
    assert data[0].get_phone_numbers()[40] == '41'
    assert data[0].get_phone_numbers()[46] == '47'
    assert data[1].get_phone_numbers()[0] == '48'
    assert data[1].get_phone_numbers()[46] == '94'


def test_term_contract():
    customers = gen_dataset(100, 1)

    # Validate term contract creation
    customers[0].new_month(1, 2020)
    assert customers[0].generate_bill(1, 2020)[1] == pytest.approx(320)

    #Check to make sure free minutes are added first
    process_event_history(add_call(customers[0], 1, 2, 600, 1, 2020), customers)
    assert customers[0].generate_bill(1, 2020)[2][0]['free_mins'] == 10

    #Test free overflow
    process_event_history(add_call(customers[0], 1, 2, 6000, 1, 2020), customers)
    assert customers[0].generate_bill(1, 2020)[2][0]['free_mins'] == 100
    assert customers[0].generate_bill(1, 2020)[2][0]['billed_mins'] == 10



if __name__ == '__main__':
    pytest.main(['my_a1_tests.py'])