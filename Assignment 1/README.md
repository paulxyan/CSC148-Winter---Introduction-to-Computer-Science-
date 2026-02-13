# CSC148 A1 — MewbileTech Phone Company

A Python application that loads historical customer call logs for a fictional Toronto phone carrier, records per-customer call history, computes monthly bills under multiple contract types, and visualizes call activity on a Toronto map with interactive filters.

---

## What the program does

- Imports a JSON dataset of **customers** and **events** (calls and SMS).
- **Ignores SMS** events for this assignment and processes **call** events.
- Builds an in-memory model of:
  - Customers, each with one or more phone lines
  - Phone lines, each with a contract, call history, and monthly bills
  - Calls, including timestamps, durations, and geographic locations
- Computes bills using contract rules for:
  - Month-to-month contracts
  - Term contracts (deposit, included minutes)
  - Prepaid contracts (balance carry-over, top-ups)
- Visualizes calls on a map and supports filtering and bill display.

---

## Quick start

### Requirements
- Python 3
- `pygame`

### Install dependencies
```bash
pip install pygame
```

### Run
```bash
python application.py
```

If a window opens showing a Toronto map, your environment is set up correctly.

---

## Visualizer controls

- `c` filter by customer id (show calls to/from that customer’s phone lines)
- `d` filter by duration (at least or at most a value, depending on the prompt)
- `l` filter by location
- `r` reset all filters
- `m` display a customer’s monthly bill for a chosen month

> Note: The first time you use a filter, a welcome pop-up may appear due to a known starter-code quirk. Close it and continue.

---

## Where to look (review guide)

### Main entry point
- **`application.py`**
  - loads the JSON log
  - creates `Customer` objects and their `PhoneLine`s
  - processes events and advances monthly billing when a new month is encountered

### Core domain model
- **`customer.py`** — manages a customer with multiple phone lines
- **`phoneline.py`** — per-line call handling and integration with billing and history
- **`callhistory.py`** — records incoming and outgoing calls for a line
- **`call.py`** — call event object (time, duration, source/destination locations)

### Billing and contracts
- **`bill.py`** — monthly bill object (rates, fixed costs, billed minutes)
- **`contract.py`** — contract behavior and subclasses:
  - `MTMContract`
  - `TermContract`
  - `PrepaidContract`

### Filtering
- **`filter.py`** — filter classes used by the visualizer (customer, duration, location)

### Data
- **`dataset.json`** — generated call log used by the program
- **`data.py`** — smaller, easier-to-read dataset format for understanding structure

### Tests
- **`sample_tests.py`** — starter tests provided with the assignment
- **`my_a1_tests.py`** — additional tests for edge cases and correctness

---

## What I implemented

This assignment includes starter code. My work focuses on implementing missing logic and connecting components:

- event ingestion and month-to-month progression
- recording incoming and outgoing calls in the correct customer and phone line histories
- contract billing logic for month-to-month, term, and prepaid contracts
- robust filter behavior that handles malformed input without crashing
- pytest coverage for correctness and edge cases

---

## Suggested repo structure (portfolio friendly)

This layout matches your current files and makes the project easy to review:

```text
mewbiletech-phone-company/
  README.md
  src/
    application.py
    bill.py
    call.py
    callhistory.py
    contract.py
    customer.py
    data.py
    filter.py
    phoneline.py
    visualizer.py
  data/
    dataset.json
  docs/
    StarterCodeArchitecture.pdf
    Documentation.pdf
  tests/
    sample_tests.py
    my_a1_tests.py
```

### Minimal-change alternative (keep the current folder, just add structure)
If you want the smallest diff from your existing setup, you can keep everything in the root and add folders later. At minimum, keep:

- `application.py` at the top level (so `python application.py` works)
- `README.md` at the top level
- tests separated into a `tests/` folder when you’re ready

---

## Skills demonstrated

- object-oriented design using composition and inheritance
- reading and extending a provided codebase from specifications and docstrings
- JSON parsing and transformation into domain objects
- stateful billing logic and monthly rollovers
- defensive programming and input validation
- automated testing with pytest
