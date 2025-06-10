from datetime import date


class SickDay:

    """Represents an employee sick day"""

    date_num = {}



    def __init__(self, pay_date: date):

        self.pay_date = pay_date
        SickDay.date_num.setdefault(self.pay_date.year, 0)
        SickDay.date_num[self.pay_date.year] += 1




    def get_month(self) -> int:

        return self.pay_date.month

    def get_year(self) -> int:

        return self.pay_date.year




class Employee:

    """An employee of a company.
    This is an abstract class. Only subclasses should be instantiated.
    === Attributes ===
    id_: This employee's ID number.
    name: This employee's name.
    self.
    """

    id_: int
    name: str
    sick_days: list[SickDay]

    def __init__(self, id_: int, name: str) -> None:
        """Initialize this employee.
        Note: This initializer is meant for internal use only;
        Employee is an abstract class and should not be instantiated directly.
        """

        self.id_ = id_
        self.name = name
        self.sick_days = []

    def get_monthly_payment(self) -> float:
        """Return the amount that this Employee should be paid in one month.
        Round the amount to the nearest cent.
        """
        raise NotImplementedError

    def pay(self, pay_date: date) -> None:
        """Pay this Employee on the given date and record the payment.
        (Assume this is called once per month.)
        """
        payment = self.get_monthly_payment()
        print(f'An employee was paid {payment} on {pay_date}.')

    def record_sickday(self, time: date) -> None:

        if time.year in SickDay.date_num:
            if SickDay.date_num[time.year] == 10:
                return None

        else:

            self.sick_days.append(SickDay(time))

class SalariedEmployee(Employee):
    """An employee whose pay is computed based on an annual salary.
    === Attributes ===
    salary: This employee's annual salary
    """
    salary: float

    def __init__(self, id_: int, name: str, salary: float) -> None:
        """Initialize this salaried Employee."""
        Employee.__init__(self, id_, name)
        self.salary = salary

    def get_monthly_payment(self) -> float:
        """Return the amount that this Employee should be paid in one month.
        Round the amount to the nearest cent.
        """
        return round(self.salary / 12, 2)

class HourlyEmployee(Employee):
    """An employee whose pay is computed based on an hourly rate.
    === Attributes ===
    hourly_wage:
    This employee's hourly rate of pay.
    hours_per_month:
    The number of hours this employee works each month.
    """
    hourly_wage: float
    hours_per_month: float

    def __init__(self, id_: int, name: str, hourly_wage: float,
        hours_per_month: float) -> None:
        """Initialize this HourlyEmployee.
        """
        Employee.__init__(self, id_, name)
        self.hourly_wage = hourly_wage
        self.hours_per_month = hours_per_month

    def get_monthly_payment(self) -> float:
        """Return the amount that this Employee should be paid in one month.
        Round the amount to the nearest cent.
        """


        return self.hours_per_month * self.hourly_wage


    def pay(self, pay_date: date) -> None:

        payment = self.get_monthly_payment()

        num_days = 0

        for sick_days in self.sick_days:

            if sick_days.get_year() == pay_date.year and sick_days.get_month() == pay_date.month:

                num_days += 1

        #Use max to account for the fact that we can have a negative value of payment
        #If the original payment was small enough
        new_payment = max(0, payment - num_days * 8 * self.hourly_wage)

        print(f'This employee was paid {new_payment}')


class Company:
    """A company with employees.

    We use this class mainly as a client for the various Employee classes
    we defined in employee.

    === Public Attributes ===
    employees: the employees in the company.
    """
    employees: list[Employee]

    def __init__(self, employees: list[Employee]) -> None:
        self.employees = employees

    def pay_all(self, pay_date: date) -> None:
        """Pay all employees at this company."""
        for employee in self.employees:
            employee.pay(pay_date)



if __name__ == '__main__':
    
    import python_ta
    python_ta.check_all()

    # Illustrate a small company.
    my_corp = Company([SalariedEmployee(14, 'Fred Flintstone', 5200.0),
                       HourlyEmployee(23, 'Barney Rubble', 1.25, 50.0 ),
                       SalariedEmployee(99, 'Mr Slate', 120000.0)])

    my_corp.pay_all(date(2017, 8, 31))
    my_corp.pay_all(date(2017, 9, 30))









